"""Train the unified BEV driver from scratch with Lightning and native ROCm operators."""
import argparse
import json
import random
import time

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from pytorch_lightning.callbacks import ModelCheckpoint,Callback
from pytorch_lightning.loggers import CSVLogger
from torchdata.stateful_dataloader import StatefulDataLoader

from v5_qwen_drive_world.checkpoint_io import VerifiedCheckpointIO
from v5_qwen_drive_world.runtime import math_attention
from .fast_model import UnifiedDriver
from .unified_dataset import UnifiedData,BalancedBatches
from .unified_data import ART
from .sensors import LOG,write_json


def augment(ipm,lidar,nav,cmd,xy,occ,mask):
    b=len(ipm)
    draws=torch.rand((b,5),device=ipm.device)
    theta=(draws[:,0]*2-1)*np.deg2rad(1.5)
    c,s=theta.cos(),theta.sin()
    affine=torch.stack((c,-s,(draws[:,1]*4-2)/64,s,c,(draws[:,2]*4-2)/64),1).reshape(-1,2,3)
    rgb=F.grid_sample(ipm,F.affine_grid(affine,ipm.shape,align_corners=False),align_corners=False)*mask
    # One pixel across the road is 0.4 m. Only translate the navigation channel.
    shift=torch.randn(b,device=ipm.device)/64.
    ones,zeros=torch.ones_like(shift),torch.zeros_like(shift)
    a=torch.stack((ones,zeros,zeros,zeros,ones,shift),1).reshape(-1,2,3)
    nav=F.grid_sample(nav,F.affine_grid(a,nav.shape,align_corners=False),align_corners=False)
    x=torch.cat((rgb,lidar,nav),1)
    flip=draws[:,3]<.5
    return (torch.where(flip[:,None,None,None],x.flip((-2,)),x),
            torch.where(flip[:,None],cmd[:,[0,2,1]],cmd),
            torch.stack((xy[:,:,0],xy[:,:,1]*(1.-2.*flip.float())[:,None]),2),
            torch.where(flip[:,None,None,None],occ.flip((-2,)),occ))


class DriverTask(pl.LightningModule):
    def __init__(self,identity,epochs):
        super().__init__()
        self.driver=UnifiedDriver()
        self.identity=identity
        self.epochs=epochs
        self.register_buffer("weights",torch.tensor([1.]*20+[.25]*20).view(1,40,1))
        self.started=time.perf_counter()
        self.verified=False

    def training_step(self,batch,batch_idx):
        rgb,lidar,nav,v,cmd,truth,occ,valid,plane=batch
        with torch.autocast("cuda",enabled=False):
            projected,visible=self.driver.ipm.sample(rgb,plane)
            x,cmd,truth,occ=augment(projected,lidar.float()/255.,nav.float()/255.,cmd,truth,occ,visible)
        with math_attention():
            xy,logits=self.driver.plan(x,v,cmd)
        regression=(F.smooth_l1_loss(xy,truth,beta=.25,reduction="none")*self.weights).mean((0,1))
        gt=occ.float();p=logits.float().sigmoid()
        bce=F.binary_cross_entropy_with_logits(logits.float(),gt,reduction="none")
        pt=torch.where(gt.bool(),p,1-p)
        focal=((1.-pt).square()*bce*torch.where(gt.bool(),.75,.25)).mean((1,2,3))
        dice=(1.-(2*(p*gt).sum((-2,-1))+1.)/(p.sum((-2,-1))+gt.sum((-2,-1))+1.)).mean(1)
        loss=regression[0]+4*regression[1]+((dice+focal)*valid).sum()/valid.sum().clamp_min(1.)
        if not torch.isfinite(loss):raise ValueError("Non-finite training loss")
        self.log("train_loss",loss,on_step=True,batch_size=len(v))
        return loss

    def on_before_optimizer_step(self,opt):
        if self.verified:return
        norms=torch.stack(torch._foreach_norm([p.grad for p in self.parameters()]))
        if not norms.isfinite().all() or not norms.max()>0:raise RuntimeError("Nonfinite or missing gradients")
        write_json(ART/"operators.json",dict(torch=torch.__version__,device=torch.cuda.get_device_name(),bf16=True,
                   fp32_master=all(p.dtype==torch.float32 for p in self.parameters()),foreach_adamw=True,
                   sdpa_backend="MATH; convolutional network has no attention layers",parameters=sum(p.numel() for p in self.parameters())))
        self.verified=True

    def on_train_epoch_start(self):
        self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch)

    def on_train_batch_end(self,outputs,batch,batch_idx):
        if self.global_step%25==0 or self.global_step==1:
            write_json(ART/"status.json",dict(epoch=self.current_epoch+1,step=self.global_step,
                       loss=float(outputs["loss"]),elapsed_s=time.perf_counter()-self.started))

    def on_validation_epoch_start(self):
        self.near=torch.zeros(2,device=self.device)

    def validation_step(self,batch,batch_idx):
        rgb,lidar,nav,v,cmd,truth,occ,valid,plane=batch
        xy,logits=self.driver(rgb,lidar,nav,v,cmd,plane)
        keep=cmd[:,0]
        self.near[0]+=((xy[:,:15,1]-truth[:,:15,1]).abs().mean(1)*keep).sum()
        self.near[1]+=keep.sum()

    def on_validation_epoch_end(self):
        score=self.near[0]/self.near[1].clamp_min(1.)
        self.log("val_keep_near",score)
        with (ART/"validation.jsonl").open("a") as f:
            f.write(json.dumps(dict(epoch=self.current_epoch+1,step=self.global_step,keep=int(self.near[1]),mae=float(score)))+"\n")

    def configure_optimizers(self):
        opt=torch.optim.AdamW(self.parameters(),lr=2e-4,weight_decay=.01,foreach=True)
        return [opt],[torch.optim.lr_scheduler.CosineAnnealingLR(opt,self.epochs,eta_min=1e-5)]

    def on_save_checkpoint(self,checkpoint):
        checkpoint["task_state"]=dict(identity=self.identity,architecture="unified-ipm",epochs=self.epochs,
           python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),gpu=torch.cuda.get_rng_state_all())

    def on_load_checkpoint(self,checkpoint):
        s=checkpoint["task_state"]
        if s["identity"]!=self.identity or s["epochs"]!=self.epochs:raise ValueError("Dataset or training configuration changed")
        random.setstate(s["python"]);np.random.set_state(s["numpy"]);torch.set_rng_state(s["cpu"]);torch.cuda.set_rng_state_all(s["gpu"])


class RollingLast(Callback):
    def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
        if trainer.global_step%250==0:
            trainer.save_checkpoint(ART/"last.ckpt")


def main(resume=False):
    ART.mkdir(parents=True,exist_ok=True)
    alignment=json.loads((ART/"alignment/geometry-check.json").read_bytes())
    if not alignment["passed"]:raise ValueError("Alignment check requires resolution before training")
    pl.seed_everything(641,workers=True)
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(3)
    train,val=UnifiedData("train"),UnifiedData("validation")
    batches=BalancedBatches(train)
    train_dl=StatefulDataLoader(train,batch_sampler=batches,num_workers=3,persistent_workers=True)
    val_dl=StatefulDataLoader(val,batch_size=40,num_workers=2,persistent_workers=True)
    ckpt=ART/"last.ckpt"
    if ckpt.exists()!=resume:raise ValueError("Specify --resume for an existing run")
    task=DriverTask(train.identity+val.identity,30)
    callback=ModelCheckpoint(dirpath=ART,filename="best",monitor="val_keep_near",mode="min",save_top_k=1,
                             save_last=True,enable_version_counter=False)
    trainer=pl.Trainer(accelerator="gpu",devices=1,precision="bf16-mixed",max_epochs=30,
         callbacks=[callback,RollingLast()],plugins=[VerifiedCheckpointIO()],num_sanity_val_steps=0,
         logger=CSVLogger(str(LOG),name="training",version="unified-ipm"),gradient_clip_val=1.,
         enable_model_summary=False,enable_progress_bar=False)
    trainer.fit(task,train_dl,val_dl,ckpt_path=str(ckpt) if resume else None,weights_only=False)
    if trainer.global_step!=30*len(batches):raise RuntimeError("Incomplete optimizer updates")
    write_json(ART/"completed.json",dict(epochs=30,updates=trainer.global_step,batch=40,counts=[len(g) for g in train.groups],
               fractions=[.4,.2,.2,.2],best_score=float(callback.best_model_score),validation_windows=len(val)))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--resume",action="store_true")
    main(parser.parse_args().resume)
