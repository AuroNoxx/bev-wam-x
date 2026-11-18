"""Fit the longitudinal readout using an internal route-based development split."""
import hashlib
import json
import math
from collections import defaultdict

import numpy as np
from .runtime import ROOT,configure

configure()
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchdata.stateful_dataloader import StatefulDataLoader
from .checkpoint_io import VerifiedCheckpointIO
from .geometry_repair_features import RUN,split_records
from .motion_profile import MotionProfile,profile_features
from .real_dataset import RealData
from .resume_disk_cached import UpdateBoundaryCheckpoint
from .sft_task import restore_rng,rng_state
from .train_geometry_repair import RepairData


DEST=RUN/'motion-profile'
CONFIG=json.loads((ROOT/'configs/v5-qwen-drive-world/motion-profile.json').read_text())


def prepare():
    data=RealData('subset2000')
    records=split_records(data)['train']
    cached=RepairData(records,'train')
    samples=[]
    hold_routes=set()
    routes=set()
    for i,r in enumerate(records):
        feat,label,_=cached[i]
        obs,base_v=profile_features({k:v[None] for k,v in feat.items()})
        tr=label['trajectory']
        d=float(tr.norm(dim=1).max())
        hold=d<.5
        launch=r['speed_kmh']<=3 and d>=3
        if hold:
            hold_routes.add(r['route'])
        routes.add(r['route'])
        samples.append((obs[0],base_v[0],feat['ctrl'],feat['v0'],tr,label['controls'],hold,launch))
    rng=np.random.default_rng(CONFIG['seed'])
    dev_routes=set(rng.permutation(sorted(hold_routes))[:2])
    others=rng.permutation(sorted(routes-hold_routes))
    dev_routes.update(others[:max(1,len(routes)//5-2)])
    dev=[i for i,r in enumerate(records) if r['route'] in dev_routes]
    fit=[int(i) for i in rng.permutation(len(records)) if records[i]['route'] not in dev_routes]
    fit=fit[:len(fit)//8*8]
    assert not ({records[i]['initial_condition_family'] for i in fit}&{records[i]['initial_condition_family'] for i in dev})
    assert any(samples[i][-2] for i in dev) and any(samples[i][-1] for i in dev)
    splits={'fit':fit,'development':dev}
    with (RUN/'latest.ckpt').open('rb') as source:
        frozen_world=hashlib.file_digest(source,'sha256').hexdigest()
    identity=dict(manifest=data.identity,source_geometry_checkpoint=frozen_world,config=CONFIG,
                  development_routes=sorted(dev_routes),indices=splits,
                  source_features=json.loads((RUN/'feature-identity.json').read_text()))
    return [[samples[i] for i in subset] for subset in (fit,dev)],identity


class FitMotion(pl.LightningModule):
    def __init__(self,identity):
        super().__init__()
        self.profile=MotionProfile()
        self.identity=identity
        self.pending_rng=None

    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(),lr=CONFIG['learning_rate'],foreach=True)
        steps=CONFIG['epochs']*len(self.identity['indices']['fit'])//8
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=CONFIG['learning_rate']*.08)
        return {'optimizer':optimizer,'lr_scheduler':{'scheduler':scheduler,'interval':'step'}}

    def training_step(self,batch,_):
        if self.pending_rng:
            restore_rng(self.pending_rng)
            self.pending_rng=None
        observation,base_v,raw,v0,truth,expert,hold,launch=batch
        control,tr,activity=self.profile(observation,base_v,raw,v0)
        with torch.autocast('cuda',enabled=False):
            delta=torch.diff(truth,dim=1,prepend=torch.zeros_like(truth[:,:1]))
            speed=delta.norm(dim=-1)/.1
            motion=(speed.reshape(-1,10,5).mean(2)/.8).clamp(0,1)
            event=F.binary_cross_entropy_with_logits(activity,motion)
            pos=F.smooth_l1_loss(tr[...,:2],truth)
            vel=F.smooth_l1_loss(tr[...,3],speed)
            accel=F.smooth_l1_loss(control[...,0],expert[...,0])
            smooth=F.smooth_l1_loss(control[:,1:,0],control[:,:-1,0])
            loss=(.5*pos+.7*vel+1.5*event+.15*accel+.08*smooth)*(1+.8*hold.float().mean())
        if not torch.isfinite(loss):
            raise ValueError('Non-finite longitudinal loss')
        self.log('train_loss',loss,on_step=True,batch_size=1)
        return loss

    def on_validation_epoch_start(self):
        self.errors=defaultdict(list)

    def validation_step(self,batch,_):
        observation,base_v,raw,v0,truth,_,hold,launch=batch
        _,tr,_=self.profile(observation,base_v,raw,v0)
        ade=float((tr[...,:2]-truth).norm(dim=-1).mean())
        self.errors['all'].append(ade)
        if hold.item():
            self.errors['hold'].append(ade)
        if launch.item():
            self.errors['launch'].append(ade)

    def on_validation_epoch_end(self):
        e={k:float(np.mean(v)) for k,v in self.errors.items()}
        score=.4*e['hold']+.3*e['launch']+.3*e['all']
        self.log('development_score',score)
        fields=dict(epoch=self.current_epoch+1,step=self.global_step,score=score,ade=e)
        with (DEST/'development.jsonl').open('a') as stream:
            stream.write(json.dumps(fields)+'\n')
        print('development',fields,flush=True)

    def on_save_checkpoint(self,c):
        c['task_state']={'identity':self.identity,'rng':rng_state(),'step':self.global_step}

    def on_load_checkpoint(self,c):
        if c['task_state']['identity']!=self.identity:
            raise ValueError('Longitudinal checkpoint source or development split changed')
        self.pending_rng=c['task_state']['rng']


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    DEST.mkdir(exist_ok=True)
    if (DEST/'latest.ckpt').exists() and not args.resume:
        raise ValueError('Use --resume for an existing longitudinal run')
    pl.seed_everything(CONFIG['seed'],workers=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    (fit,dev),identity=prepare()
    (DEST/'identity.json').write_text(json.dumps(identity,indent=2))
    print('training and development sizes',len(fit),len(dev),flush=True)
    loader=StatefulDataLoader(fit,batch_size=1,shuffle=True,generator=torch.Generator().manual_seed(CONFIG['seed']))
    val=StatefulDataLoader(dev,batch_size=1)
    best=ModelCheckpoint(dirpath=DEST,filename='best',monitor='development_score',mode='min',
                         save_on_train_epoch_end=False,enable_version_counter=False)
    latest=UpdateBoundaryCheckpoint(dirpath=DEST,filename='latest',every_n_train_steps=40,
                                    save_on_train_epoch_end=True,enable_version_counter=False)
    trainer=pl.Trainer(accelerator='gpu',devices=1,max_epochs=CONFIG['epochs'],precision='bf16-mixed',
                       accumulate_grad_batches=8,gradient_clip_val=2.,enable_progress_bar=False,num_sanity_val_steps=0,
                       callbacks=[best,latest],plugins=[VerifiedCheckpointIO()],
                       logger=pl.loggers.TensorBoardLogger(str(ROOT/'logs/v5-qwen-drive-world'),name='motion-profile'))
    trainer.fit(FitMotion(identity),loader,val,ckpt_path=str(DEST/'latest.ckpt') if args.resume else None,weights_only=False)
    assert trainer.global_step==len(fit)*CONFIG['epochs']//8


if __name__=='__main__':
    main()
