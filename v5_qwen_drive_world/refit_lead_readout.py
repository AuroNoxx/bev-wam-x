"""Fine-tune the existing directed lead readout and evaluate a locked final checkpoint."""
import argparse
import hashlib
import json

import numpy as np
from .lead_motion import ROOT,DEST,RealData,load_base,front_observation,torch,F
from .train_lead_motion import LeadReadout,metrics,restore_rng,pl,StatefulDataLoader,UpdateBoundaryCheckpoint,ModelCheckpoint,VerifiedCheckpointIO
from qwen_drive import CameraFrame

SOURCE=DEST/'directed'
RUN=SOURCE/'paired-refit'
DATA=ROOT/'data/v5-qwen-drive-world/lead-readout-refit'
CFG=json.loads((ROOT/'configs/v5-qwen-drive-world/lead-readout-refit.json').read_text())
OLD=ROOT/'data/v5-qwen-drive-world/lead-pairs-aligned/manifest.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Refit(LeadReadout):
    def training_step(self,batch,_):
        if self.pending_rng:
            restore_rng(self.pending_rng)
            self.pending_rng=None
        x,y,_=batch
        p=self(x)
        # Keep the loss in metres. A stationary false departure and a missed
        # departure have equal weight in the threshold term.
        regression=F.smooth_l1_loss(p,y,beta=.25)
        event=F.binary_cross_entropy_with_logits(4*(p-.3),(y>.3).float())
        return regression+.12*event

    def configure_optimizers(self):
        opt=torch.optim.AdamW(self.parameters(),lr=CFG['lr'],foreach=True)
        return [opt],[torch.optim.lr_scheduler.CosineAnnealingLR(opt,CFG['epochs'])]

    def validation_step(self,batch,_):
        x,y,_=batch
        self.log('development_mae',(self(x)-y).abs().mean(),batch_size=1)

    def on_validation_epoch_end(self):
        pass

    def on_save_checkpoint(self,c):
        super().on_save_checkpoint(c)
        c['task_state']['config']=CFG


def visual_features(base,rows):
    output=[]
    starts={}
    for r in rows:
        path=ROOT/r['actor']
        actor=json.loads(path.read_text())
        frames=actor['observations']
        physics=json.loads((path.parent/'physics.json').read_text())
        start=np.stack([physics[0][k] for k in ('ego','lead')])
        np.testing.assert_allclose(start,starts.setdefault(r['route'],start),atol=.001,rtol=0)
        transform=np.linalg.inv(physics[46]['ego'])
        assert abs((transform@(np.array(physics[46]['lead'])-physics[36]['lead']))[0,3]-r['target'])<1e-5
        assert len(frames)==4
        for f,i in zip(frames,(31,36,41,46)):
            record=physics[i]
            assert abs(record['time']-f['time'])<.005 and record['frame']==f['physics_frame']
            mount=np.linalg.solve(record['ego'],np.linalg.inv(f['camera']['world2cam']))
            assert np.linalg.norm(mount[:3,3]-[.8,0,1.6])<.002
            assert np.rad2deg(np.arccos(np.clip((np.trace(mount[:3,:3])-1)/2,-1,1)))<.1
        output.append(front_observation(base,[CameraFrame(ROOT/f['path']) for f in frames],actor,
                                        [f['camera'] for f in frames],directed=True))
    return torch.stack(output)


def score(head,x,rows):
    head.cuda().eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        out=[dict(r,prediction=float(head(v[None].cuda()).item())) for r,v in zip(rows,x,strict=True)]
    by_group={}
    for group in sorted({r.get('branch','all') for r in rows}):
        by_group[group]=metrics([r for r in out if r.get('branch','all')==group])
    return dict(overall=metrics(out),by_group=by_group,windows=out)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    RUN.mkdir(exist_ok=True)
    pl.seed_everything(CFG['seed'],workers=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    assert json.loads((DATA/'protocol.json').read_text())==CFG
    fit_rows=json.loads((DATA/'fit.json').read_text())
    old=json.loads(OLD.read_text())
    assert digest(SOURCE/'latest.ckpt')==old['config']['checkpoint_sha256']
    orig_x=torch.load(SOURCE/'observations.pt',weights_only=True)
    orig_rows=json.loads((SOURCE/'labels-and-split.json').read_text())['windows']
    fit_idx=np.random.default_rng(9147).permutation(np.flatnonzero([not r['dev'] for r in orig_rows]))
    fit_idx=fit_idx[:len(fit_idx)//8*8]
    chosen=np.random.default_rng(CFG['seed']).choice(fit_idx,len(fit_rows),replace=False)
    dev=np.flatnonzero([r['dev'] for r in orig_rows])
    base=load_base(RealData('subset2000'))
    x=visual_features(base,fit_rows)
    x=torch.cat((x,orig_x[chosen]))
    records=fit_rows+[orig_rows[i] for i in chosen]
    identity=hashlib.sha256(x.numpy().tobytes()+json.dumps([CFG,records,old['config']['checkpoint_sha256']],sort_keys=True).encode()).hexdigest()
    net=Refit(orig_x,identity,orig_rows,RUN)
    source=torch.load(SOURCE/'latest.ckpt',map_location='cpu',weights_only=False)
    net.load_state_dict(source['state_dict'])
    loader=StatefulDataLoader([(v,torch.tensor(r['target'],dtype=torch.float32),i) for i,(v,r) in enumerate(zip(x,records))],
                               batch_size=1,shuffle=True,generator=torch.Generator().manual_seed(CFG['seed']))
    val=StatefulDataLoader([(orig_x[i],torch.tensor(orig_rows[i]['target'],dtype=torch.float32),int(i)) for i in dev],batch_size=1)
    latest=UpdateBoundaryCheckpoint(dirpath=RUN,filename='latest',every_n_train_steps=24,save_on_train_epoch_end=True,enable_version_counter=False)
    best=ModelCheckpoint(dirpath=RUN,filename='best',monitor='development_mae',mode='min',save_on_train_epoch_end=False,enable_version_counter=False)
    steps=CFG['epochs']*len(x)//8
    ck=RUN/'latest.ckpt'
    if not ck.exists() or torch.load(ck,weights_only=False,map_location='cpu')['global_step']!=steps:
        if ck.exists() and not args.resume: raise RuntimeError('Specify --resume for an incomplete run')
        trainer=pl.Trainer(accelerator='gpu',devices=1,max_epochs=CFG['epochs'],precision='bf16-mixed',accumulate_grad_batches=8,
                           num_sanity_val_steps=0,enable_progress_bar=False,callbacks=[best,latest],plugins=[VerifiedCheckpointIO()],
                           logger=pl.loggers.TensorBoardLogger(str(ROOT/'logs/v5-qwen-drive-world'),name='paired-lead-refit'))
        trainer.fit(net,loader,val,ckpt_path=str(ck) if ck.exists() else None,weights_only=False)
        assert trainer.global_step==steps
    frozen=torch.load(ck,map_location='cpu',weights_only=False)
    assert frozen['task_state']['identity']==identity and frozen['global_step']==steps
    net.load_state_dict(frozen['state_dict'])
    old_net=LeadReadout(orig_x,{},[])
    old_net.load_state_dict(source['state_dict'])
    holdout=json.loads((DATA/'acceptance.json').read_text())
    train_routes={r['route'] for r in records}
    assert {r['route'] for r in holdout}.isdisjoint(train_routes|{r['route'] for r in old['windows']})
    assert all(r['route'].split('-')[0] in CFG['acceptance'] for r in holdout)
    results=dict(global_step=steps,checkpoint_sha256=digest(ck),fit_count=len(fit_rows),anchor_count=len(chosen))
    for key,feats,labels in [('acceptance',visual_features(base,holdout),holdout),('regression',visual_features(base,old['windows']),old['windows']),
                             ('internal',orig_x[dev],[orig_rows[i] for i in dev])]:
        results[key]={'before':score(old_net,feats,labels),'after':score(net,feats,labels)}
    s=results['acceptance']['after']['overall']
    results['passed']=s['precision']>=.75 and s['recall']>=.75 and s['mae']<s['zero_mae']
    (RUN/'results.json').write_text(json.dumps(results,indent=2))
    print('final step',steps,'gate',results['passed'],s,flush=True)


if __name__=='__main__':main()
