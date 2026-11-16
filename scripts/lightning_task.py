"""Task adapters for Lightning; Trainer owns optimization, validation and resume."""
from pathlib import Path
import hashlib,json,math,os,random,time
import numpy as np
import psutil
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping,LearningRateMonitor,ThroughputMonitor
from lightning_fabric.plugins.io import TorchCheckpointIO
from models_v1 import MODELS,loss
from stage1_objective import consistent_flow_loss,trajectory_metrics

ROOT=Path(__file__).resolve().parents[1]

def configure_runtime():
    for key,name in [('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton'),('TEMP','temp'),('TMP','temp')]:
        path=ROOT/'.cache'/name;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
    os.environ['HIP_PATH']=str(ROOT/'.venv-train/Lib/site-packages/_rocm_sdk_core')
    os.environ['CC']=str(ROOT/'.venv-train/Lib/site-packages/triton/runtime/tcc/tcc.exe')
    torch.set_num_threads(6)

def random_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}

def restore_random(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])

class VerifiedCheckpointIO(TorchCheckpointIO):
    """The project's durability rule, implemented through Lightning's storage plugin."""
    def save_checkpoint(self,checkpoint,path,storage_options=None):
        path=Path(path);temporary=path.with_suffix('.pending')
        super().save_checkpoint(checkpoint,temporary,storage_options)
        check=torch.load(temporary,map_location='cpu',weights_only=False)
        assert {'state_dict','optimizer_states','lr_schedulers','loops','global_step','task_state'} <= check.keys()
        if not all(torch.isfinite(t).all() for t in check['state_dict'].values()):raise RuntimeError('Nonfinite checkpoint')
        del check
        with temporary.open('rb+') as f:os.fsync(f.fileno())
        os.replace(temporary,path)

class OccSamples(Dataset):
    def __init__(self,manifest,split,mode):
        self.version=manifest['preprocess_version']
        base=ROOT/manifest.get('cache_root','data/occ_v1')
        self.files=[base/x for x in manifest['splits'][split]]
        self.keys=['history','state','trajectory']+(['future'] if mode=='world' else [])
    def __len__(self):return len(self.files)
    def __getitem__(self,i):
        with np.load(self.files[i],allow_pickle=False) as a:
            if int(a['preprocess_version'])!=self.version:raise RuntimeError('Sample/manifest contract mismatch')
            if self.version==3:
                if int(a['occ_file_offset'])!=-1:raise RuntimeError('Invalid v3 OCC pairing')
                for prefix in ['history','future']:
                    if not np.array_equal(a[prefix+'_occ_ids'],a[prefix+'_pose_ids']-1):
                        raise RuntimeError('OCC/pose source IDs mismatch')
            return {k:torch.from_numpy(a[k].copy()) for k in self.keys}

class OccData(pl.LightningDataModule):
    def __init__(self,mode,settings):
        super().__init__();self.mode=mode;self.settings=settings
        self.config=json.loads((ROOT/'configs/train_v1.json').read_text())
        path=ROOT/settings.get('manifest','data/occ_v1/manifest.json');self.manifest=json.loads(path.read_text());self.sha=hashlib.sha256(path.read_bytes()).hexdigest()
        assert not self.manifest['pilot'] and self.manifest['preprocess_version']==settings.get('preprocess_version',2)
        if self.manifest['preprocess_version']==3:
            assert self.manifest['occ_file_offset']==-1 and self.manifest['cache_root']=='data/occ_v3'
        assert self.manifest['config_sha256']==hashlib.sha256((ROOT/'configs/train_v1.json').read_bytes()).hexdigest()
    def loader(self,split):
        workers=self.settings['workers'];train=split=='train'
        return StatefulDataLoader(OccSamples(self.manifest,split,self.mode),batch_size=self.config[self.mode]['batch_size'],
            shuffle=train,num_workers=workers,persistent_workers=workers>0,pin_memory=True,
            generator=torch.Generator().manual_seed(self.config['seed'] if train else 1907),
            **({'prefetch_factor':2} if workers else {}))
    def train_dataloader(self):return self.loader('train')
    def val_dataloader(self):return self.loader('validation')

class DrivingTask(pl.LightningModule):
    def __init__(self,mode,settings,manifest_sha,train_windows):
        super().__init__();self.mode=mode;self.settings=settings;self.manifest_sha=manifest_sha;self.train_windows=train_windows
        self.config=json.loads((ROOT/'configs/train_v1.json').read_text())
        if mode=='flow' and settings.get('trajectory_parameterization','absolute')!='absolute' and not settings.get('objective'):
            raise ValueError('Residual parameterization requires the physical-endpoint objective')
        self.model=MODELS[mode](parameterization=settings.get('trajectory_parameterization','absolute')) if mode=='flow' else MODELS[mode]()
        self.pending_rng=None
        if settings['channels_last']:
            torch.nn.utils.convert_conv2d_weight_memory_format(self.model,torch.channels_last);self.model.encoder.channels_last=True
        self.save_hyperparameters()
    def on_fit_start(self):
        if self.settings['compile']:self.model.compile(backend='inductor',dynamic=False)
    def training_step(self,batch,batch_idx):
        if self.pending_rng is not None:restore_random(self.pending_rng);self.pending_rng=None
        value=self.task_loss(batch,'train')
        self.log('train_loss',value,on_step=True,on_epoch=True,batch_size=len(batch['state']))
        return value
    def task_loss(self,batch,split):
        objective=self.settings.get('objective')
        if self.mode=='flow' and objective:
            value,terms=consistent_flow_loss(self.model,batch,objective)
            self.log_dict({f'{split}_objective/{k}':v for k,v in terms.items()},on_step=False,on_epoch=True,batch_size=len(batch['state']))
            return value
        return loss(self.mode,self.model,batch)
    def configure_gradient_clipping(self,optimizer,gradient_clip_val=None,gradient_clip_algorithm=None):
        # BF16 has no scaled gradients; use Lightning's documented clipping hook.
        if getattr(self.trainer.precision_plugin,'scaler',None) is not None:raise RuntimeError('This clipping hook requires BF16 without GradScaler')
        torch.nn.utils.clip_grad_norm_(self.parameters(),gradient_clip_val,error_if_nonfinite=True)
    def on_validation_epoch_start(self):
        self.validation_rng=random_state();pl.seed_everything(1907,workers=True,verbose=False)
    def validation_step(self,batch,batch_idx):
        self.log('validation_loss',self.task_loss(batch,'validation'),on_epoch=True,batch_size=len(batch['state']))
        if self.mode=='flow' and (self.settings.get('objective') or (self.current_epoch+1)%5==0):
            with torch.random.fork_rng(devices=[self.device.index]):
                torch.manual_seed(1908+batch_idx)
                pred=self.model.sample(batch,k=8,steps=16)
            if self.settings.get('objective'):
                metrics=trajectory_metrics(pred,batch)
                self.log_dict({f'trajectory/{k}':v.mean() for k,v in metrics.items()},batch_size=len(pred))
                manifest=self.trainer.datamodule.manifest
                families={r['route']:r['family'] for r in manifest['routes']}
                names=manifest['splits']['validation'][batch_idx*self.config['flow']['batch_size']:][:len(pred)]
                labels=[families[Path(n).parts[-2]] for n in names]
                for family in sorted(set(labels)):
                    select=torch.tensor([s==family for s in labels],device=self.device)
                    for key in ['meanADE8_m','minADE8_m','kinematic_residual_mps']:
                        self.log(f'family/{family}/{key}',metrics[key][select].mean(),batch_size=int(select.sum()))
            else:
                error=((pred[...,:2]-batch['trajectory'][:,None,:,:2])*pred.new_tensor([50,10])).norm(dim=-1)
                self.log('trajectory/minADE8_m',error.mean(-1).min(1).values.mean(),batch_size=len(batch['state']))
                self.log('trajectory/minFDE8_m',error[:,:,-1].min(1).values.mean(),batch_size=len(batch['state']))
    def on_validation_epoch_end(self):restore_random(self.validation_rng)
    def configure_optimizers(self):
        cfg={**self.config[self.mode],**self.settings.get('training',{})};optimizer=torch.optim.AdamW(self.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'],fused=self.settings['fused_adamw'])
        batches=math.ceil(self.train_windows/cfg['batch_size']);warm=batches*2;total=batches*cfg['max_epochs']
        def rate(step):return (step+1)/warm if step<warm else .1+.9*.5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,total-warm))))
        return {'optimizer':optimizer,'lr_scheduler':{'scheduler':torch.optim.lr_scheduler.LambdaLR(optimizer,rate),'interval':'step'}}
    def on_save_checkpoint(self,checkpoint):
        checkpoint['task_state']={'config':self.config,'manifest_sha':self.manifest_sha,'rng':random_state(),
            'resume_contract':'Lightning loop state + TorchData StatefulDataLoader + task RNG', 'settings':self.settings}
    def on_load_checkpoint(self,checkpoint):
        saved=checkpoint['task_state']
        if saved['config']!=self.config or saved['manifest_sha']!=self.manifest_sha:raise RuntimeError('Checkpoint data/config mismatch')
        if saved['settings']!=self.settings:raise RuntimeError('Checkpoint training settings mismatch; use an explicit weights-only initialization for a new experiment')
        self.pending_rng=saved['rng']

class RunStatus(pl.Callback):
    """Local observability only; no optimization or training-loop ownership."""
    def __init__(self,folder):self.folder=Path(folder);self.times=[]
    def on_train_start(self,trainer,module):self.started=time.monotonic()
    def on_train_epoch_start(self,trainer,module):self.epoch_start=time.monotonic()
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        if trainer.global_step%10==0:
            module.log_dict({'memory/allocated_gib':torch.cuda.memory_allocated()/2**30,'memory/reserved_gib':torch.cuda.memory_reserved()/2**30,
                'memory/available_ram_gib':psutil.virtual_memory().available/2**30,'elapsed_s':time.monotonic()-self.started})
    def on_train_epoch_end(self,trainer,module):
        self.times.append(time.monotonic()-self.epoch_start)
        row={'status':'running','framework':'pytorch-lightning','epoch':trainer.current_epoch+1,'step':trainer.global_step,
            'epoch_seconds':self.times[-1],'remaining_max_seconds':np.mean(self.times[-5:])*(trainer.max_epochs-trainer.current_epoch-1),
            **{k:float(v) for k,v in trainer.callback_metrics.items() if isinstance(v,torch.Tensor) and v.numel()==1}}
        row['trajectory_metric_epoch']=trainer.current_epoch+1 if module.settings.get('objective') else ((trainer.current_epoch+1)//5)*5
        self.folder.mkdir(parents=True,exist_ok=True);(self.folder/'status.json').write_text(json.dumps(row,indent=2))
        print('EPOCH '+json.dumps(row),flush=True)
    def on_train_end(self,trainer,module):
        path=self.folder/'status.json'
        row=json.loads(path.read_text()) if path.exists() else {};row.update(status='completed',step=trainer.global_step,remaining_max_seconds=0)
        path.write_text(json.dumps(row,indent=2))

def callbacks(folder,mode,settings=None):
    settings=settings or {}
    config=json.loads((ROOT/'configs/train_v1.json').read_text());cfg={**config[mode],**settings.get('training',{})}
    monitor=settings.get('monitor','validation_loss')
    base={'dirpath':str(folder),'auto_insert_metric_name':False,'enable_version_counter':False,'save_top_k':1}
    return [EarlyStopping(monitor,patience=cfg['patience'],min_delta=cfg['min_delta'],mode='min',check_finite=True),
        LearningRateMonitor(logging_interval='step'),ThroughputMonitor(batch_size_fn=lambda b:len(b['state']),window_size=30),RunStatus(folder),
        ModelCheckpoint(**base,filename='best',monitor=monitor,mode='min',every_n_epochs=1,save_on_train_epoch_end=True),
        ModelCheckpoint(**base,filename='latest',every_n_train_steps=config['checkpoint_steps'],save_on_train_epoch_end=False),
        ModelCheckpoint(**base,filename='latest',every_n_epochs=1,save_on_train_epoch_end=True)]
