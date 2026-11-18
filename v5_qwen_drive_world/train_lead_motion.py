"""Fit the causal lead displacement readout and score untouched development routes."""
import argparse
import hashlib
import json
import numpy as np
from .lead_motion import CONFIG,DEST,ROOT,torch,F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchdata.stateful_dataloader import StatefulDataLoader
from .resume_disk_cached import UpdateBoundaryCheckpoint
from .checkpoint_io import VerifiedCheckpointIO
from .sft_task import rng_state,restore_rng


def metrics(rows):
    truth=np.array([r['target'] for r in rows])
    pred=np.array([r['prediction'] for r in rows])
    positive=truth>CONFIG['movement_m']
    called=pred>CONFIG['movement_m']
    tp=int((positive&called).sum())
    return dict(n=len(rows),positive=int(positive.sum()),predicted=int(called.sum()),tp=tp,
                precision=tp/max(1,int(called.sum())),recall=tp/max(1,int(positive.sum())),
                mae=float(np.abs(pred-truth).mean()),zero_mae=float(np.abs(truth).mean()))


class LeadReadout(pl.LightningModule):
    def __init__(self,x,identity,rows,dest=DEST):
        super().__init__()
        self.identity=identity
        self.rows=rows
        self.dest=dest
        self.pending_rng=None
        self.register_buffer('center',x.mean(0))
        self.register_buffer('scale',x.std(0).clamp_min(.02))
        self.readout=torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.LayerNorm(32),
                                         torch.nn.SiLU(),torch.nn.Linear(32,1))

    def forward(self,x):
        return self.readout(((x-self.center)/self.scale).clamp(-8,8)).flatten().float()

    def training_step(self,batch,_):
        if self.pending_rng:
            restore_rng(self.pending_rng)
            self.pending_rng=None
        x,y,_=batch
        p=self(x)
        positive=(y>CONFIG['movement_m']).float()
        loss=(1+2*positive)*(F.smooth_l1_loss(p,y,reduction='none')+
                             .15*F.binary_cross_entropy_with_logits((p-CONFIG['movement_m'])*3,positive,reduction='none'))
        return loss.mean()

    def configure_optimizers(self):
        opt=torch.optim.AdamW(self.parameters(),lr=CONFIG['learning_rate'],foreach=True)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,CONFIG['epochs'])
        return [opt],[sched]

    def on_save_checkpoint(self,c):
        c['task_state']={'identity':self.identity,'config':CONFIG,'rng':rng_state(),'step':self.global_step}

    def on_load_checkpoint(self,c):
        if c['task_state']['identity']!=self.identity:
            raise ValueError('Lead-motion features or split changed')
        self.pending_rng=c['task_state']['rng']

    def on_validation_epoch_start(self):
        self.predictions=[]

    def validation_step(self,batch,_):
        x,y,i=batch
        self.predictions.append({**self.rows[int(i.item())],'prediction':float(self(x).item())})

    def on_validation_epoch_end(self):
        stats=metrics(self.predictions)
        self.log('development_mae',stats['mae'])
        if self.current_epoch+1==CONFIG['epochs']:
            routes={r['route'] for r in self.predictions}
            by_route={route:metrics([r for r in self.predictions if r['route']==route]) for route in sorted(routes)}
            passed=(stats['precision']>=.75 and stats['recall']>=.75 and stats['mae']<stats['zero_mae']
                    and len(routes)>=2 and all(v['positive'] for v in by_route.values()))
            (self.dest/'result.json').write_text(json.dumps(dict(step=self.global_step,passed=bool(passed),
                overall=stats,by_route=by_route,windows=self.predictions),indent=2))
            print('held routes',stats,'passed',passed,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--full-pose',action='store_true')
    parser.add_argument('--directed',action='store_true')
    args=parser.parse_args()
    dest=DEST/'directed' if args.directed else DEST/'full-pose' if args.full_pose else DEST
    if (dest/'latest.ckpt').exists() and not args.resume:
        raise ValueError('Specify --resume to continue the existing run')
    x=torch.load(dest/'observations.pt',weights_only=True)
    doc=(dest/'labels-and-split.json').read_bytes()
    rows=json.loads(doc)['windows']
    assert len(rows)==len(x) and torch.isfinite(x).all()
    fit=np.flatnonzero([not r['dev'] for r in rows])
    dev=np.flatnonzero([r['dev'] for r in rows])
    fit=np.random.default_rng(CONFIG['seed']).permutation(fit)
    fit=fit[:len(fit)//8*8]
    assert not ({rows[i]['route'] for i in fit}&{rows[i]['route'] for i in dev})
    def dataset(ix):
        return [(x[i],torch.tensor(rows[i]['target'],dtype=torch.float32),int(i)) for i in ix]
    pl.seed_everything(CONFIG['seed'],workers=True)
    torch.backends.cudnn.benchmark=False
    train=StatefulDataLoader(dataset(fit),shuffle=True,batch_size=1,generator=torch.Generator().manual_seed(CONFIG['seed']))
    val=StatefulDataLoader(dataset(dev),batch_size=1)
    identity=hashlib.sha256(doc+x.numpy().tobytes()).hexdigest()
    task=LeadReadout(x[fit],identity,rows,dest)
    latest=UpdateBoundaryCheckpoint(dirpath=dest,filename='latest',every_n_train_steps=20,
                save_on_train_epoch_end=True,enable_version_counter=False)
    best=ModelCheckpoint(dirpath=dest,filename='best',monitor='development_mae',mode='min',
                save_on_train_epoch_end=False,enable_version_counter=False)
    trainer=pl.Trainer(accelerator='gpu',devices=1,max_epochs=CONFIG['epochs'],precision='bf16-mixed',
              accumulate_grad_batches=8,enable_progress_bar=False,num_sanity_val_steps=0,callbacks=[latest,best],
              plugins=[VerifiedCheckpointIO()],logger=pl.loggers.TensorBoardLogger(str(ROOT/'logs/v5-qwen-drive-world'),name=dest.name))
    trainer.fit(task,train,val,ckpt_path=str(dest/'latest.ckpt') if args.resume else None,weights_only=False)
    assert trainer.global_step==len(fit)*CONFIG['epochs']//8
    print('fitted',len(fit),'held out',len(dev),'updates',trainer.global_step,flush=True)


if __name__=='__main__':
    main()
