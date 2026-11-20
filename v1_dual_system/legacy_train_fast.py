"""One native Batch-32 Lightning run. Training owns the GPU while CARLA is stopped."""
import argparse
import json
import math
import random
import time

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torchdata.stateful_dataloader import StatefulDataLoader

from v5_qwen_drive_world.checkpoint_io import VerifiedCheckpointIO
from .dataset import DaggerMix, FastData, NavigationData, SpatialData, FocusSampler, ForkSampler
from .legacy_fast_model import FastDriver, losses
from .sensors import ROOT, LOG, DATA, CAL_ID, write_json
from .deployment import load_driver

ART=ROOT/"artifacts/v1-dual-system"

def phase_folder(corrective,lane_supervision,road_roi=False):
    if road_roi:return ART/"road-roi"
    if lane_supervision:return ART/"lane-supervised"
    return ART/"corrective" if corrective else ART


class DriverTask(pl.LightningModule):
    def __init__(self,identity,epochs,corrective=False,resume=False,lane_identity=None,road_roi=False,dagger_identity=None,navigation=False,spatial=False,film=False):
        super().__init__()
        self.corrective=corrective
        self.lane_identity=lane_identity
        self.road_roi=road_roi
        self.dagger_identity=dagger_identity
        self.navigation=navigation
        self.spatial=spatial
        self.film=film
        self.folder=ART/"fork-film" if film else ART/"spatial" if spatial else ART/"navigation" if navigation else ART/"dagger" if dagger_identity else phase_folder(corrective,lane_identity is not None,road_roi)
        if corrective and not resume:
            parent=ART/"spatial" if film else ART/"navigation" if spatial else ART/"dagger" if navigation else ART/"road-roi" if dagger_identity else ART/"lane-supervised" if road_roi else phase_folder(bool(lane_identity),False)
            driver,source_step,_=load_driver(parent/"best.ckpt")
            if film:
                if source_step!=6413:raise ValueError("Expected epoch 11 spatial parent")
                driver.enable_film()
            elif spatial:
                if source_step!=9636:raise ValueError("Expected epoch 22 parent")
                driver.enable_spatial()
            elif navigation:
                driver.enable_navigation()
            elif road_roi and not dagger_identity:
                driver.enable_road_roi()
            elif lane_identity and not road_roi:
                driver.enable_lane_supervision()
            elif not road_roi:
                driver.enable_metric_fusion()
            driver.train()
            completed=json.loads((parent/"completed.json").read_bytes())
            write_json(self.folder/"lineage.json",dict(source=str(parent/"best.ckpt"),step=source_step,
                       parent_completed_updates=completed["updates"],additional_epochs=epochs,lane_identity=lane_identity,
                       dagger_identity=dagger_identity))
        else:
            driver=FastDriver(corrective=corrective,lane_supervision=lane_identity is not None,road_roi=road_roi,navigation=navigation,spatial=spatial,film=film)
        self.layout_name="channels_last"
        if corrective:
            selected=json.loads((ART/"corrective"/"layout.json").read_bytes())
            if (selected["identity"],selected["device"],selected["torch"])!=(identity,torch.cuda.get_device_name(),torch.__version__):
                raise ValueError("Training layout measurement is from a different run configuration")
            self.layout_name=selected["selected"]
        driver.layout=torch.channels_last if self.layout_name=="channels_last" else torch.contiguous_format
        self.driver=driver.to(memory_format=driver.layout)
        if corrective:
            counts=torch.tensor(json.loads((DATA/"audit-corrective.json").read_bytes())["splits"]["train"]["intents"],dtype=torch.float)
            if dagger_identity:
                extra=1991 if navigation else 2000
                counts[0]+=extra
            if lane_identity:
                counts[1:]=counts[1:].mean().item()
            self.register_buffer("class_weights",(counts.sum()/(3*counts)).sqrt(),persistent=False)
        self.identity,self.epochs=identity,epochs
        self.started=time.perf_counter()
        if resume and (self.folder/"status.json").is_file():
            self.started-=json.loads((self.folder/"status.json").read_bytes()).get("elapsed_s",0.)
        self.verified=False

    def training_step(self,batch,batch_idx):
        inputs=batch[:4]
        if self.film:
            # Real and counterfactual observations share one BatchNorm update.
            chosen=batch[8].abs()>4.
            paired=[v[chosen] for v in inputs]
            paired[2][:,2::2]*=-1.
            inputs=tuple(torch.cat((obs,other)) for obs,other in zip(inputs,paired))
        outputs=self.driver(*inputs,return_lane=self.lane_identity is not None,return_residual=self.navigation)
        if self.film:
            n=len(batch[0])
            counter=outputs[0][n:,14:30,1]
            outputs=tuple(v[:n] for v in outputs)
            separation=(outputs[0][chosen,14:30,1]-counter)*batch[8][chosen,None].sign()
            sensitivity=F.relu(3.-separation).sum()/max(separation.numel(),1)
        pred,occ=outputs[:2]
        emphasis=(batch[2][:,0],batch[3],self.class_weights) if self.corrective else None
        loss=losses(pred,occ,*batch[4:6],emphasis,near_lateral=self.road_roi,navigation=self.navigation)
        if self.navigation:
            keep=batch[3][:,0]
            if self.film:
                # Emphasize clean near-range KEEP tracking alongside the branch hinge.
                near=(pred[:,:15,1]-batch[4][:,:15,1]).abs().mean(1)
                loss=loss+2.*(near*keep).sum()/keep.sum().clamp_min(1.)+sensitivity
                keep=keep*batch[7]
            loss=loss+(outputs[3][:,:5].square().mean(1)*keep).sum()/keep.sum().clamp_min(1.)
        if self.lane_identity:
            # Regress offset in metres and heading as its lateral effect at 10 m.
            target=batch[6]
            lateral=F.smooth_l1_loss(outputs[2][:,0],target[:,0],beta=.25)
            heading=F.smooth_l1_loss(outputs[2][:,1],10*target[:,1],beta=.25)
            loss=loss+lateral+heading
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite joint loss")
        self.log("train_loss",loss,on_step=True,on_epoch=True,batch_size=len(pred))
        return loss

    def on_before_optimizer_step(self,optimizer):
        if self.verified: return
        gradients=[p.grad for p in self.parameters() if p.requires_grad]
        if any(g is None for g in gradients):
            raise RuntimeError("A trainable parameter has no gradient")
        norms=torch.stack(torch._foreach_norm(gradients))
        if not norms.isfinite().all() or not norms.max()>0:
            raise FloatingPointError("Nonfinite or entirely zero gradients")
        nav_grad=None
        if self.spatial:
            nav_grad=float(self.driver.bev[0].weight.grad[:,3].norm())
            if not nav_grad>0:raise RuntimeError("No gradient reached the new navigation channel")
        self.verified=True
        write_json(self.folder/"operators.json",dict(torch=torch.__version__,lightning=pl.__version__,
                   device=torch.cuda.get_device_name(),precision="bf16-mixed",trainable_tensors=len(gradients),
                   fp32_master=all(p.dtype==torch.float32 for p in self.parameters()),
                   sdpa="MATH",adamw="foreach",navigation_channel_grad_norm=nav_grad,memory_format=self.layout_name,
                   parameters=sum(p.numel() for p in self.parameters())))

    def on_train_batch_end(self,outputs,batch,batch_idx):
        if self.global_step%20==0 or self.global_step<3:
            write_json(self.folder/"status.json",dict(epoch=self.current_epoch+1,step=self.global_step,
                training_loss=float(outputs["loss"]),physical_batch=len(batch[0]),accumulation=1,
                elapsed_s=time.perf_counter()-self.started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))

    def on_validation_epoch_start(self):
        if self.corrective:
            self.keep_error=torch.zeros((),device=self.device)
            self.tracking_sum=torch.zeros(3,device=self.device)
            self.tracking_count=torch.zeros(3,device=self.device)

    def validation_step(self,batch,batch_idx):
        outputs=self.driver(*batch[:4],return_lane=self.lane_identity is not None)
        xy,occ=outputs[:2]
        err=xy-batch[4]
        val={"val_ade":err.norm(dim=-1).mean(),"val_fde":err[:,-1].norm(dim=-1).mean(),
             "val_lateral_mae":err[:,:,1].abs().mean(),"val_jerk":xy.diff(n=3,dim=1).norm(dim=-1).mean()*1000}
        if self.lane_identity:
            val["val_rgb_lane_offset_mae_m"]=(outputs[2][:,0]-batch[6][:,0]).abs().mean()
            val["val_rgb_lane_heading_mae_rad"]=(outputs[2][:,1]/10-batch[6][:,1]).abs().mean()
        if self.road_roi:
            val["val_near_lateral_mae"]=err[:,:15,1].abs().mean()
            val["val_far_lateral_mae"]=err[:,15:,1].abs().mean()
        fg=occ>0
        gt=batch[5].bool()
        val["val_occ_intersection"]=(fg&gt).sum().float()/len(xy)
        val["val_occ_union"]=(fg|gt).sum().float()/len(xy)
        self.log_dict(val,on_step=False,on_epoch=True,batch_size=len(xy))
        if self.corrective:
            look=(3.5+20.5*batch[2][:,0]/33.333).clamp(3.5,24.)
            near=(batch[4].norm(dim=-1)-look[:,None]).abs().argmin(1)
            sample=torch.arange(len(xy),device=xy.device)
            score=err[sample,near,1].abs()+.25*err[:,29,1].abs()+.05*err.norm(dim=-1).mean(1)
            intent=batch[3][:,:3].argmax(1)
            if self.navigation:self.keep_error+=(err[:,:15,1].abs().mean(1)*(intent==0)).sum()
            self.tracking_sum.scatter_add_(0,intent,score)
            self.tracking_count.scatter_add_(0,intent,torch.ones_like(score))

    def on_validation_epoch_end(self):
        if self.corrective:
            if (self.tracking_count==0).any():raise ValueError("Validation missing a command")
            if self.navigation:self.log("val_keep_near_mae",self.keep_error/self.tracking_count[0])
            per_command=self.tracking_sum/self.tracking_count
            self.log("val_tracking_macro",per_command.mean())
        stats={k:float(v) for k,v in self.trainer.callback_metrics.items() if k.startswith("val_")}
        if self.corrective:
            stats["val_tracking_macro"]=float(per_command.mean())
            stats["tracking_by_intent_m"]=per_command.cpu().tolist()
        if "val_occ_union" in stats:
            stats["val_occ_iou"]=stats["val_occ_intersection"]/max(stats["val_occ_union"],1.)
        with (self.folder/"validation.jsonl").open("a") as f:
            f.write(json.dumps(dict(epoch=self.current_epoch+1,step=self.global_step,**stats))+"\n")

    def configure_optimizers(self):
        lr=3e-5 if self.dagger_identity else 1e-4 if self.corrective else 2e-4
        params=self.parameters()
        if self.film:
            projections=[*self.driver.nav_encoder.parameters(),*self.driver.lane_film.parameters(),*self.driver.plan_film.parameters()]
            special={id(p) for p in projections}
            params=[{"params":[p for p in self.parameters() if id(p) not in special]},
                    {"params":projections,"lr":3e-4}]
        opt=torch.optim.AdamW(params,lr=lr,weight_decay=.01,foreach=True)
        return [opt],[torch.optim.lr_scheduler.CosineAnnealingLR(opt,self.epochs,eta_min=1e-5)]

    def on_save_checkpoint(self,ckpt):
        ckpt["task_state"]=dict(identity=self.identity,calibration=CAL_ID,batch=32,accumulate=1,epochs=self.epochs,corrective=self.corrective,
            lane_supervision=self.lane_identity is not None,lane_identity=self.lane_identity,
            road_roi=self.road_roi,navigation=self.navigation,spatial=self.spatial,film=self.film,dagger_identity=self.dagger_identity,
            python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),gpu=torch.cuda.get_rng_state_all())

    def on_load_checkpoint(self,ckpt):
        s=ckpt["task_state"]
        if s.get("film",False)!=self.film:raise ValueError("FiLM architecture mismatch")
        if s.get("spatial",False)!=self.spatial:raise ValueError("Spatial architecture mismatch")
        if s.get("navigation",False)!=self.navigation:raise ValueError("Navigation mode mismatch")
        if s.get("dagger_identity")!=self.dagger_identity:raise ValueError("DAgger training manifest mismatch")
        if s.get("lane_identity")!=self.lane_identity:raise ValueError("Lane supervision identity mismatch")
        if s.get("road_roi",False)!=self.road_roi:raise ValueError("ROI mode mismatch")
        if (s["identity"],s["calibration"],s["epochs"],s["batch"])!=(self.identity,CAL_ID,self.epochs,32):
            raise ValueError("Checkpoint/data/config mismatch")
        random.setstate(s["python"]);np.random.set_state(s["numpy"])
        torch.set_rng_state(s["cpu"]);torch.cuda.set_rng_state_all(s["gpu"])


class Latest(ModelCheckpoint):
    def on_train_epoch_end(self,trainer,pl_module):
        super().on_train_epoch_end(trainer,pl_module)
        if self._last_global_step_saved!=trainer.global_step:
            self._save_topk_checkpoint(trainer,self._monitor_candidates(trainer))


def start(epochs,resume,corrective,lane_supervision=False,road_roi=False,dagger=False,navigation=False,spatial=False,film=False):
    torch.backends.cudnn.benchmark=False
    torch.backends.mha.set_fastpath_enabled(False)
    # This offline trainer runs alone. The deployed pilot has its separate 0.15 cap.
    torch.cuda.set_per_process_memory_fraction(.80)
    torch.set_num_threads(4)
    pl.seed_everything(641,workers=True)
    train=SpatialData("train",film) if spatial else NavigationData("train") if navigation else DaggerMix() if dagger else FastData("train",corrective,lane_supervision=lane_supervision)
    val=SpatialData("validation",film) if spatial else NavigationData("validation") if navigation else FastData("validation",corrective,lane_supervision=lane_supervision)
    folder=ART/"fork-film" if film else ART/"spatial" if spatial else ART/"navigation" if navigation else ART/"dagger" if dagger else phase_folder(corrective,lane_supervision,road_roi)
    folder.mkdir(parents=True,exist_ok=True)
    latest_path=folder/"latest.ckpt"
    if latest_path.exists()!=resume: raise ValueError("Use --resume for an existing latest checkpoint")
    # Fixed, preassigned route split; Lightning serializes the StatefulDataLoader in its fit loop.
    sampler=ForkSampler(train) if film else FocusSampler(train) if spatial else None
    train_loader=StatefulDataLoader(train,batch_size=32,sampler=sampler,shuffle=sampler is None,num_workers=3,pin_memory=False,
                                    persistent_workers=True,prefetch_factor=2)
    val_loader=StatefulDataLoader(val,batch_size=32,shuffle=False,num_workers=2,pin_memory=False,
                                  persistent_workers=True,prefetch_factor=2)
    task=DriverTask(train.identity,epochs,corrective,resume,train.lane_identity if lane_supervision else None,road_roi,
                    train.training_identity if dagger or navigation else None,navigation,spatial,film)
    latest=Latest(dirpath=folder,filename="latest",every_n_train_steps=125,save_top_k=1,
                  enable_version_counter=False,save_on_train_epoch_end=True)
    metric="val_keep_near_mae" if navigation else "val_near_lateral_mae" if road_roi else "val_tracking_macro" if corrective else "val_ade"
    best=ModelCheckpoint(dirpath=folder,filename="best",monitor=metric,mode="min",save_top_k=1,
                         enable_version_counter=False,save_on_train_epoch_end=False)
    trainer=pl.Trainer(accelerator="gpu",devices=1,precision="bf16-mixed",max_epochs=epochs,
                       accumulate_grad_batches=1,gradient_clip_val=1.,benchmark=False,
                       callbacks=[latest,best],plugins=[VerifiedCheckpointIO()],num_sanity_val_steps=0,
                       logger=CSVLogger(str(LOG),name="training",version=folder.name if corrective else "main"),enable_progress_bar=False,
                       enable_model_summary=False,log_every_n_steps=20)
    trainer.fit(task,train_loader,val_loader,ckpt_path=str(latest_path) if resume else None,weights_only=False)
    per_epoch=len(sampler) if spatial else len(train)
    expected=epochs*math.ceil(per_epoch/32)
    if trainer.global_step!=expected:raise RuntimeError(f"{trainer.global_step}/{expected} optimizer updates")
    write_json(folder/"completed.json",dict(epochs=epochs,updates=expected,samples=epochs*per_epoch,
               metric=metric,best_score=float(best.best_model_score),manifest=train.identity,
               training_identity=train.training_identity if dagger or navigation else train.identity))
    final=json.loads((folder/"status.json").read_bytes())
    final.update(state="complete",epoch=epochs,step=expected,elapsed_s=time.perf_counter()-task.started)
    write_json(folder/"status.json",final)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--epochs",type=int)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--corrective",action="store_true",help="20 corrective epochs following the completed 30-epoch run")
    p.add_argument("--lane-supervised",action="store_true",help="Fine-tune with mirrored RGB lane supervision")
    p.add_argument("--road-roi",action="store_true",help="Fine-tune cropped multiscale RGB and near-term lateral tracking")
    p.add_argument("--dagger",action="store_true",help="Fine-tune epoch 26 on the mixed dataset for ten epochs")
    p.add_argument("--navigation",action="store_true")
    p.add_argument("--spatial",action="store_true")
    p.add_argument("--film",action="store_true")
    a=p.parse_args()
    if a.film:a.spatial=True
    if a.spatial:
        a.navigation=True
        if a.epochs not in (None,10 if a.film else 12):p.error("Expected 10 FiLM epochs or 12 spatial epochs")
    elif a.epochs is not None and not 30<=a.epochs<=50:p.error("Expected 30 to 50 epochs")
    if a.epochs is None:a.epochs=30
    roi=a.road_roi or a.dagger or a.navigation
    lane_supervision=a.lane_supervised or roi
    epochs=10 if a.film else 12 if a.spatial else 10 if a.dagger else 20 if a.corrective and not lane_supervision else a.epochs
    start(epochs,a.resume,a.corrective or lane_supervision,lane_supervision,roi,a.dagger,a.navigation,a.spatial,a.film)
