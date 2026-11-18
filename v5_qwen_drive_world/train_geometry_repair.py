"""Local head revision with fixed train split and matched native-size validation."""
import argparse
import json
import hashlib
import numpy as np
from .runtime import configure,ROOT,math_attention

configure()
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchdata.stateful_dataloader import StatefulDataLoader
from .checkpoint_io import VerifiedCheckpointIO
from .geometry_repair_features import RUN,BASE,CFG,split_records
from .components import load_perception
from .world2d import ActionConditionedWorld2D
from .geometry_repair import GeometryRepair
from .real_dataset import RealData
from .resume_disk_cached import UpdateBoundaryCheckpoint
from .sft_task import rng_state,restore_rng


class RepairData(torch.utils.data.Dataset):
    def __init__(self,records,split):
        self.records=records
        self.folder=RUN/'features'/split

    def __len__(self):
        return len(self.records)

    def __getitem__(self,i):
        record=self.records[i]
        feat=torch.load(self.folder/f'{i:04d}.pt',map_location='cpu',weights_only=True)
        if feat.pop('actor_path')!=record['actor_path']:
            raise ValueError('Actor cache has the wrong sample identity')
        with np.load(ROOT/record['label_path']) as data:
            keys=('trajectory','controls','current_support','support','current_occupancy','occupancy')
            labels={k:torch.from_numpy(data[k].copy()) for k in keys}
        return feat,labels,i


def world_loss(logits,target,support):
    prob=logits.float().sigmoid()
    weight=support.float()
    dims=(-2,-1)
    n=weight.sum(dims).clamp_min(1)
    positive=(target*weight).sum(dims)
    bce=F.binary_cross_entropy_with_logits(logits.float(),target,reduction='none')
    # Combine calibrated background error with overlap on rare occupied cells.
    overlap=(2*(prob*target*weight).sum(dims)+1)/(((prob+target)*weight).sum(dims)+1)
    spatial=((bce*(1+3*target)*weight).sum(dims)/n+
             2*((prob-target).square()*weight).sum(dims)/n+
             .5*(1-overlap)*(positive>0)).mean()
    common=support[:,1:] & support[:,:1]
    change=target[:,1:]-target[:,:1]
    changed=common & (change!=0)
    temporal=F.smooth_l1_loss(prob[:,1:]-prob[:,:1],change,reduction='none')
    return spatial+.15*((temporal*changed).sum(dims)/changed.sum(dims).clamp_min(1)).mean()


class RepairTask(pl.LightningModule):
    def __init__(self,net,records,identity):
        super().__init__()
        self.net=net
        self.records=records
        self.identity=identity
        self.pending_rng=None
        self.rows=[]
        self.frozen_training_logits={}

    def configure_optimizers(self):
        opt=torch.optim.AdamW((p for p in self.parameters() if p.requires_grad),lr=CFG['learning_rate'],foreach=True)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,CFG['epochs']*CFG['train_windows']//8,eta_min=CFG['learning_rate']*.08)
        return {'optimizer':opt,'lr_scheduler':{'scheduler':sched,'interval':'step'}}

    def step(self,batch):
        feat,label,idx=batch
        with math_attention():
            fixed=None
            if self.training:
                # A training index always carries the same observed World candidate.
                # The 640 detached maps occupy 160 MiB. The actor never sees them.
                i=int(idx.item())
                if i not in self.frozen_training_logits:
                    self.frozen_training_logits[i]=self.net.frozen_world_logits(
                        feat['latent'],label['trajectory'],feat['v0'])
                fixed=self.frozen_training_logits[i]
            ctrl,tr,logits=self.net(feat,label['trajectory'],fixed)
        with torch.autocast('cuda',enabled=False):
            truth=label['trajectory'].float()
            positions=F.smooth_l1_loss(tr[...,:2],truth)
            horizon=F.smooth_l1_loss(tr[:,[9,29,49],:2],truth[:,[9,29,49]])
            control=F.smooth_l1_loss(ctrl[...,0],label['controls'][...,0])
            jerk=F.smooth_l1_loss(ctrl[:,1:,0],ctrl[:,:-1,0])
            # Supervise whether the predicted motion actually persists over the full horizon.
            # This uses labels only during fitting and never locks a control at inference.
            displacement=truth.norm(dim=-1).amax(dim=1)
            slow=feat['v0'].flatten()<=3/3.6
            decisive=(displacement<.5)|(displacement>=3)
            predicted_distance=tr[...,:2].norm(dim=-1).amax(dim=1)
            transition=F.binary_cross_entropy_with_logits(
                (predicted_distance-1.75)/1.25,(displacement>=3).float(),reduction='none')
            transition=(transition*(slow&decisive).float()).mean()
            actual_v=(truth-torch.cat([torch.zeros_like(truth[:,:1]),truth[:,:-1]],1)).norm(dim=-1)/.1
            speed=F.smooth_l1_loss(tr[:,:,3],actual_v)
            target=torch.cat([label['current_occupancy'][:,None],label['occupancy']],1).float()
            support=torch.cat([label['current_support'][:,None],label['support']],1).bool()
            occupancy=world_loss(logits,target,support)
            loss=.30*positions+.15*horizon+.15*control+.08*jerk+.15*speed+.4*transition+3*occupancy
        return loss,tr,logits.sigmoid(),target,support

    def training_step(self,batch,batch_idx):
        if self.pending_rng:
            restore_rng(self.pending_rng);self.pending_rng=None
        self.net.world.eval()
        loss,*_=self.step(batch)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite repair loss')
        self.log('train_loss',loss,on_step=True,batch_size=1)
        return loss

    def on_save_checkpoint(self,c):
        c['task_state']={'feature_identity':self.identity,'rng':rng_state(),'epoch':self.current_epoch,
                         'step':self.global_step,'microbatch':1,'accumulation':8,'effective_batch':8}

    def on_train_epoch_end(self):
        print('epoch',self.current_epoch+1,'updates',self.global_step,flush=True)

    def on_load_checkpoint(self,c):
        if c['task_state']['feature_identity']!=self.identity:
            raise ValueError('Repair checkpoint input identity mismatch')
        self.pending_rng=c['task_state']['rng']

    def on_validation_epoch_start(self):
        self.rows=[]

    def validation_step(self,batch,batch_idx):
        feat,label,idx=batch
        _,tr,prob,target,support=self.step(batch)
        r=self.records[int(idx.item())]
        truth=label['trajectory']
        error=(tr[...,:2]-truth).norm(dim=-1)[0]
        common=support[:,1:] & support[:,:1]
        static=prob[:,:1].expand_as(prob[:,1:])
        future=prob[:,1:]
        change=common & (target[:,1:]!=target[:,:1])
        modes=(('world',future),('persistence',static))
        row=dict(route=r['route'],index=r['index'],speed_kmh=r['speed_kmh'],ade=float(error.mean()),
                 fde=[float(error[t]) for t in (9,29,49)],x5=float(tr[0,-1,0]),true_x=float(truth[0,-1,0]),
                 dmax=float(truth.norm(dim=-1).max()),jerk=float((tr[...,4:6].diff(dim=1)/.1).norm(dim=-1).max()))
        for name,field in modes:
            square=(field-target[:,1:]).square()
            row[name+'_brier']=float(square[common].mean())
            row[name+'_changed_sse']=[float(square[:,h][change[:,h]].sum()) for h in range(3)]
        row['changed']=[int(change[:,h].sum()) for h in range(3)]
        row['current_brier']=float((prob[:,0]-target[:,0]).square()[support[:,0]].mean())
        x=self.net.x.new_tensor(np.arange(128)+.5-16)
        y=self.net.y.new_tensor((np.arange(128)+.5)*.3-19.2)
        region=common & (x>0)&(x<80)&(y[:,None].abs()<=2)
        score=future[0,1][region[0,1]]
        positive=target[0,2][region[0,1]].bool()
        row['front_n_pos']=int(positive.sum())
        row['front_tp']=int((positive & (score>=.5)).sum())
        row['front_pred']=int((score>=.5).sum())
        row['front_hist']=[np.bincount(np.minimum((score[positive==c].float().cpu().numpy()*512).astype(int),511),
                                       minlength=512).tolist() for c in (0,1)]
        self.rows.append(row)
        self.log('validation_ade',error.mean(),batch_size=1)

    def on_validation_epoch_end(self):
        if len(self.rows)!=len(self.records):
            raise RuntimeError('Incomplete validation')
        out={'step':self.global_step,'epoch':self.current_epoch+1,'samples':self.rows}
        path=RUN/'validation.json'
        temp=path.with_suffix('.pending')
        temp.write_text(json.dumps(out));temp.replace(path)
        print('validation',self.global_step,'ADE',np.mean([r['ade'] for r in self.rows]),
              'Brier',np.mean([r['world_brier'] for r in self.rows]),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if (RUN/'latest.ckpt').exists() and not args.resume:
        raise ValueError('Use --resume for the existing repair run')
    pl.seed_everything(CFG['seed'],workers=True);torch.set_num_threads(6)
    torch.backends.cudnn.benchmark=False
    data=RealData('subset2000')
    records=split_records(data)
    identity=json.loads((RUN/'feature-identity.json').read_text())
    if data.identity!=identity['manifest'] or CFG!=identity['config']:
        raise ValueError('Training manifest or cache configuration changed')
    with (BASE/'latest.ckpt').open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=identity['base_sha256']:
            raise ValueError('Source checkpoint changed since feature extraction')
    perception,_=load_perception(json.loads((BASE/'config.json').read_text()),device='cpu')
    world=ActionConditionedWorld2D(perception)
    del perception
    world.load_state_dict(torch.load(RUN/'base-world.pt',map_location='cpu',weights_only=True))
    net=GeometryRepair(world)
    model=RepairTask(net,records['validation'],identity)
    train=StatefulDataLoader(RepairData(records['train'],'train'),batch_size=1,shuffle=True,
                             generator=torch.Generator().manual_seed(CFG['seed']))
    val=StatefulDataLoader(RepairData(records['validation'],'validation'),batch_size=1)
    latest=UpdateBoundaryCheckpoint(dirpath=RUN,filename='latest',every_n_train_steps=40,save_top_k=1,
                                    save_on_train_epoch_end=True,enable_version_counter=False)
    best=ModelCheckpoint(dirpath=RUN,filename='best',monitor='validation_ade',mode='min',
                         save_on_train_epoch_end=False,enable_version_counter=False)
    trainer=pl.Trainer(accelerator='gpu',devices=1,max_epochs=CFG['epochs'],precision='bf16-mixed',
                       accumulate_grad_batches=8,gradient_clip_val=2.,enable_progress_bar=False,
                       num_sanity_val_steps=0,check_val_every_n_epoch=4,callbacks=[latest,best],
                       logger=pl.loggers.TensorBoardLogger(str(ROOT/'logs/v5-qwen-drive-world'),name='geometry-repair'),
                       plugins=[VerifiedCheckpointIO()])
    trainer.fit(model,train,val,ckpt_path=str(RUN/'latest.ckpt') if args.resume else None,weights_only=False)
    assert trainer.global_step==CFG['epochs']*CFG['train_windows']//8


if __name__=='__main__':
    main()
