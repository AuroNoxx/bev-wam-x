"""Isolated Lightning small-fit experiment, with original verified checkpoint IO.

No test reads, downloads, simulation, or default full-dataset training.
"""
import argparse, copy, hashlib, json, math, os, statistics, time
from pathlib import Path
import numpy as np
import psutil
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_task import ROOT, configure_runtime, DrivingTask, VerifiedCheckpointIO, random_state, restore_random
from models_v1 import FlowPlanner
from joint_occ_model import JointOccFlow, representation_report
from stage1_objective import consistent_flow_loss
from diagnose_occ_frozen import summarize
from audit_joint_occ import write


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def resource_guard():
    conflicts=[]
    for p in psutil.process_iter(['pid','ppid','name','cmdline']):
        if p.pid==os.getpid() or p.pid in [x.pid for x in psutil.Process().parents()]:continue
        name=(p.info['name'] or '').lower();cmd=' '.join(p.info['cmdline'] or [])
        if 'carla' in name or (name.startswith('python') and any(x in cmd for x in ['train_lightning.py','train_v1.py','train_joint_occ.py','test_joint_occ.py','run_highway_occ.py','run_occ_closed_loop.py'])):
            conflicts.append({'pid':p.pid,'name':name,'command':cmd})
    if conflicts:raise RuntimeError(f'Active compute task: {conflicts}')
    if psutil.virtual_memory().available < 4*2**30:raise RuntimeError('Less than 4 GiB RAM available')


class MemorySamples(Dataset):
    def __init__(self,names):
        self.names=names;self.rows=[]
        for name in names:
            assert Path(name).parts[1] in ['train','validation']
            with np.load(ROOT/'data/occ_v3'/name) as f:
                assert int(f['preprocess_version'])==3
                assert np.array_equal(f['history_occ_ids'],f['history_pose_ids']-1)
                assert np.array_equal(f['future_occ_ids'],f['future_pose_ids']-1)
                self.rows.append({k:torch.from_numpy(f[k][:2].copy() if k=='future' else f[k].copy()) for k in ['history','state','trajectory','future']})
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[i]


class JointData(pl.LightningDataModule):
    def __init__(self,cfg,selection,seed):
        super().__init__();self.cfg=cfg;self.seed=seed
        self.datasets={s:MemorySamples(selection['splits'][s]) for s in ['train','validation']}
    def loader(self,split):
        return StatefulDataLoader(self.datasets[split],batch_size=self.cfg['batch_size'],shuffle=split=='train',num_workers=0,
            generator=torch.Generator().manual_seed(self.seed if split=='train' else 1907))
    def train_dataloader(self):return self.loader('train')
    def val_dataloader(self):return self.loader('validation')


class JointTask(DrivingTask):
    def __init__(self,variant,cfg,manifest_sha,selection_sha,seed):
        pl.LightningModule.__init__(self)
        self.mode='flow';self.variant=variant;self.settings=copy.deepcopy(cfg);self.settings.update(variant=variant,seed=seed,selection_sha=selection_sha)
        self.config=json.loads((ROOT/'configs/train_v1.json').read_text());self.manifest_sha=manifest_sha;self.pending_rng=None
        self.model=FlowPlanner() if variant=='original_flow' else JointOccFlow(variant,cfg['width'],len(cfg['future_seconds']))
        self.save_hyperparameters()
    def on_fit_start(self):pass
    def task_loss(self,batch,split):
        value,terms=consistent_flow_loss(self.model,batch,self.settings['objective'])
        if self.variant in ['auxiliary','fusion']:
            latent,var,sem=self.model.auxiliary_loss(batch)
            for name,term in [('latent',latent),('variance',var),('semantic',sem)]:
                value=value+self.settings['aux_weights'][name]*term;terms[name]=term
        self.log_dict({f'{split}/{k}':v for k,v in terms.items()},on_step=False,on_epoch=True,batch_size=len(batch['state']))
        return value
    def optimizer_step(self,*args,**kwargs):
        super().optimizer_step(*args,**kwargs)
        if self.variant in ['auxiliary','fusion']:self.model.update_teacher(self.settings['ema_momentum'])
    def validation_step(self,batch,batch_idx):
        with torch.random.fork_rng(devices=[self.device.index] if self.device.type=='cuda' else []):
            torch.manual_seed(1908+batch_idx)
            pred=self.model.sample({'history':batch['history'],'state':batch['state']},k=8,steps=16)
        err=((pred[...,:2]-batch['trajectory'][:,None,:,:2])*pred.new_tensor([50,10])).norm(dim=-1).mean()
        self.log('validation_meanADE_m',err,on_epoch=True,batch_size=len(pred))
    def configure_optimizers(self):
        opt=torch.optim.AdamW([p for p in self.parameters() if p.requires_grad],lr=self.settings['learning_rate'],weight_decay=self.settings['weight_decay'])
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=self.settings['max_steps'],eta_min=self.settings['learning_rate']*.1)
        return {'optimizer':opt,'lr_scheduler':{'scheduler':sched,'interval':'step'}}


class Measure(pl.Callback):
    def __init__(self):self.times=[];self.losses=[]
    def on_train_batch_start(self,trainer,module,batch,batch_idx):
        if torch.cuda.is_available():torch.cuda.synchronize()
        self.start=time.perf_counter()
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        if torch.cuda.is_available():torch.cuda.synchronize()
        if trainer.global_step>10:self.times.append(time.perf_counter()-self.start)
        self.losses.append(float(outputs['loss']))
        if trainer.global_step%64==0:print(json.dumps({'variant':module.variant,'seed':module.settings['seed'],'step':trainer.global_step,'recent_loss':statistics.mean(self.losses[-16:])}),flush=True)


def batch_to(rows,device):return {k:torch.stack([r[k] for r in rows]).to(device) for k in rows[0]}


@torch.no_grad()
def evaluate(model,ds,ablations=False):
    model.eval();device=next(model.parameters()).device;preds={};latency=[]
    modes=['normal','history_freeze_common','vehicles_to_unfilled','vehicles_to_unknown','mask_only'] if ablations else ['normal']
    for mode in modes:
        parts=[]
        for bi,start in enumerate(range(0,len(ds),8)):
            b=batch_to(ds.rows[start:start+8],device);h=b['history'].clone()
            if mode=='history_freeze_common':
                current=h[:,-1:].expand_as(h);common=(h!=255).all(1,keepdim=True).expand_as(h);h[common]=current[common]
            elif mode=='vehicles_to_unfilled':h[h==4]=0
            elif mode=='vehicles_to_unknown':h[h==4]=255
            elif mode=='mask_only':h[(h>0)&(h<8)]=1
            actor={'history':h,'state':b['state']}
            if device.type=='cuda':torch.cuda.synchronize()
            started=time.perf_counter()
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'),torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
                torch.manual_seed(1908+bi);p=model.sample(actor,k=8,steps=16)
            if device.type=='cuda':torch.cuda.synchronize()
            if mode=='normal':latency.append(time.perf_counter()-started)
            parts.append(p.cpu().numpy())
        preds[mode]=np.concatenate(parts)[None]
    state=np.stack([r['state'].numpy() for r in ds.rows]);target=np.stack([r['trajectory'].numpy() for r in ds.rows])
    result={mode:summarize(pred,state,target,np.asarray(ds.names),preds['normal'] if mode!='normal' else None) for mode,pred in preds.items()}
    xy=preds['normal'][...,:2]*[50,10]
    vel=np.diff(np.concatenate([np.zeros_like(xy[...,:1,:]),xy],-2),axis=-2)/.2
    result['physical_probes']={'mean_abs_acceleration_mps2':float(np.linalg.norm(np.diff(vel,axis=-2)/.2,axis=-1).mean()),
        'mean_abs_jerk_mps3':float(np.linalg.norm(np.diff(vel,n=2,axis=-2)/.04,axis=-1).mean()),
        'note':'raw numerical differences, not calibrated comfort/safety thresholds'}
    result['latency']={'batch8_k8_steps16_wall_seconds_mean':statistics.mean(latency),'wall_seconds_p95':float(np.quantile(latency,.95)),
                       'simulation_seconds':None,'scope':'model-only warm/cold mixed; excludes CPU data read and controller'}
    if isinstance(model,JointOccFlow) and model.variant in ['auxiliary','fusion']:
        b=batch_to(ds.rows[:32],device)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            result['representation']=representation_report(model,b)
    return result,preds


def make_trainer(folder,cfg,measure,max_steps=None,checkpoint=True,extra_callbacks=None):
    base={'dirpath':str(folder),'auto_insert_metric_name':False,'enable_version_counter':False,'save_top_k':1}
    callbacks=[measure]+(extra_callbacks or [])
    if checkpoint:callbacks += [ModelCheckpoint(**base,filename='best',monitor='validation_meanADE_m',mode='min',every_n_epochs=cfg['validation_every_epochs'],save_on_train_epoch_end=False),
        ModelCheckpoint(**base,filename='latest',every_n_train_steps=cfg['checkpoint_steps'],save_on_train_epoch_end=False),
        ModelCheckpoint(**base,filename='latest',every_n_epochs=1,save_on_train_epoch_end=True)]
    logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='joint-occ-v1',version=folder.name,default_hp_metric=False) if checkpoint else False
    return pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',devices=1,precision=cfg['precision'] if torch.cuda.is_available() else '32-true',
        max_steps=max_steps or cfg['max_steps'],max_epochs=-1,logger=logger,callbacks=callbacks,plugins=[VerifiedCheckpointIO()],
        enable_checkpointing=checkpoint,enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,
        check_val_every_n_epoch=cfg['validation_every_epochs'],limit_val_batches=1.0 if checkpoint else 0,gradient_clip_val=1.,log_every_n_steps=8,default_root_dir=str(folder))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/joint-occ-v1.json');p.add_argument('--out',default='artifacts/joint-occ-v1/runs')
    p.add_argument('--variant');p.add_argument('--seed',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args()
    resource_guard();configure_runtime();cfg=json.loads((ROOT/a.config).read_text());sp=ROOT/cfg['selection'];selection=json.loads(sp.read_text())
    assert digest(ROOT/cfg['manifest'])==selection['manifest_sha']
    if a.resume and (not a.variant or a.seed is None):raise ValueError('Resume requires explicit variant and seed')
    out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
    write(out/'experiment-config.json',cfg)
    for seed in [a.seed] if a.seed is not None else cfg['seeds']:
        data=JointData(cfg,selection,seed)
        for variant in [a.variant] if a.variant else cfg['variants']:
            folder=out/f'{variant}-seed{seed}'
            if not a.resume:folder.mkdir(exist_ok=False)
            pl.seed_everything(seed,workers=True);measure=Measure()
            task=JointTask(variant,cfg,selection['manifest_sha'],digest(sp),seed)
            device='cuda' if torch.cuda.is_available() else 'cpu';task.to(device)
            if a.resume:task.load_state_dict(torch.load(folder/'latest.ckpt',map_location=device,weights_only=False)['state_dict'])
            before,_=evaluate(task.model,data.datasets['train'])
            if not a.resume:write(folder/'before.json',before)
            if device=='cuda':torch.cuda.reset_peak_memory_stats()
            trainer=make_trainer(folder,cfg,measure);start=time.perf_counter()
            trainer.fit(task,datamodule=data,ckpt_path=str(folder/'latest.ckpt') if a.resume else None,weights_only=False)
            wall=time.perf_counter()-start
            # Lightning teardown returns parameters to CPU; use the measured GPU
            # for post-fit inference explicitly.
            task.to(device)
            resources={'torch':torch.__version__,'hip':torch.version.hip,'gpu':torch.cuda.get_device_name() if device=='cuda' else None,
                'parameters_total':sum(p.numel() for p in task.model.parameters()),'parameters_trainable':sum(p.numel() for p in task.model.parameters() if p.requires_grad),
                'training_wall_seconds':wall,'timed_steps':len(measure.times),'step_seconds_mean':statistics.mean(measure.times),
                'step_seconds_p95':float(np.quantile(measure.times,.95)),
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else None,
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30 if device=='cuda' else None,
                'process_rss_gib':psutil.Process().memory_info().rss/2**30,'global_step':trainer.global_step,
                'cost_note':'per-step excludes validation/checkpoint/data I/O; wall includes validation/checkpoints, in-memory 64-window dataset; allocator peaks are not total GPU memory',
                'initial_16_loss_mean':statistics.mean(measure.losses[:16]),'final_16_loss_mean':statistics.mean(measure.losses[-16:])}
            write(folder/'resources.json',resources)
            for split in ['train','validation']:
                result,preds=evaluate(task.model,data.datasets[split],ablations=split=='validation')
                write(folder/f'{split}.json',result)
                np.savez_compressed(folder/f'{split}-predictions.npz',names=data.datasets[split].names,**{k:v for k,v in preds.items()})
            write(folder/'completion.json',{'completed':True,'global_step':trainer.global_step,'test_read':False,'closed_loop':False,'checkpoint_files':[p.name for p in folder.glob('*.ckpt')]})
            print('COMPLETED '+json.dumps({'variant':variant,'seed':seed,**resources}),flush=True)
            del task,trainer
            if device=='cuda':torch.cuda.empty_cache()
    # Frozen 60-epoch v3 is a historical reference, not a matched-budget result.
    if not a.resume and not a.variant:
        model=FlowPlanner().to('cuda' if torch.cuda.is_available() else 'cpu')
        ck=torch.load(ROOT/'artifacts/lightning/flow-v3/best.ckpt',map_location='cpu',weights_only=False)
        model.load_state_dict({k.removeprefix('model.'):v for k,v in ck['state_dict'].items()})
        result,_=evaluate(model,data.datasets['validation'])
        write(out/'historical-v3-validation-reference.json',result)


if __name__=='__main__':main()
