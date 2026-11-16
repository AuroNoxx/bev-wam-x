"""Isolated Lightning training plumbing; formal training remains explicitly paused."""
import copy,json,hashlib,argparse,math,time
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchdata.stateful_dataloader import StatefulDataLoader
from b2d_native_occ import ROOT
from occ_qwen_native_data import NativeDataset,require_training_authorization
from occ_qwen_native_flow import NativeOccQwenJointFlow,load_text_backbone
from lightning_task import VerifiedCheckpointIO,random_state,restore_random

class NativeData(pl.LightningDataModule):
    def __init__(self,settings,dataset=None):
        super().__init__();self.settings=settings;self.dataset=dataset
    def train_dataloader(self):
        return StatefulDataLoader(self.dataset if self.dataset is not None else NativeDataset('train'),batch_size=self.settings['batch_size'],shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(self.settings['seed']))
    def val_dataloader(self):
        return StatefulDataLoader(NativeDataset('validation'),batch_size=self.settings['batch_size'],shuffle=False,num_workers=0)

class NativeTask(pl.LightningModule):
    def __init__(self,model,settings,manifest_sha,contract_test=False):
        super().__init__();self.model=model;self.settings=copy.deepcopy(settings);self.manifest_sha=manifest_sha
        self.contract_test=contract_test;self.pending_rng=None
    def on_fit_start(self):
        if self.contract_test:
            if not 0<self.trainer.max_steps<=4:
                raise RuntimeError('Contract tests are limited to four steps')
        else:require_training_authorization()
    def training_step(self,batch,batch_idx):
        if self.pending_rng is not None:restore_random(self.pending_rng);self.pending_rng=None
        loss,parts=self.model.loss(**batch)
        self.log_dict({'train/'+k:v for k,v in parts.items()},on_step=True,on_epoch=True,batch_size=len(batch['state']))
        self.log('train/loss',loss,on_step=True,on_epoch=True,batch_size=len(batch['state']))
        return loss
    def configure_optimizers(self):
        opt=torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],lr=self.settings['lr'],weight_decay=.01)
        total=math.ceil(5296/self.settings['batch_size'])*self.settings.get('max_epochs',20)
        warmup=min(100,max(1,total//20))
        def scale(step):
            if step<warmup:return (step+1)/warmup
            return .1+.9*.5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total-warmup))))
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,scale)
        return {'optimizer':opt,'lr_scheduler':{'scheduler':scheduler,'interval':'step'}}
    def optimizer_step(self,*args,**kwargs):
        super().optimizer_step(*args,**kwargs)
        self.model.update_teacher(self.settings.get('ema_momentum',.99))
    def on_validation_epoch_start(self):
        self.validation_rng=random_state();torch.manual_seed(1907)
    def validation_step(self,batch,batch_idx):
        loss,parts=self.model.loss(**batch)
        prediction=self.model.sample(batch['history'],batch['state'],steps=self.settings.get('sample_steps',8))['trajectory_xy']
        target=batch['trajectory'][...,:2]*prediction.new_tensor([50.,10.])
        ade=(prediction-target).norm(dim=-1).mean()
        self.log('val/action_ade_m',ade,on_epoch=True,batch_size=len(prediction))
        self.log('val/loss',loss,on_epoch=True,batch_size=len(prediction))
    def on_validation_epoch_end(self):restore_random(self.validation_rng)
    def on_save_checkpoint(self,checkpoint):
        checkpoint['task_state']={'settings':self.settings,'manifest_sha':self.manifest_sha,'rng':random_state(),'contract_test':self.contract_test,'schema':'native-occ-qwen-lightning-v1','backbone':'full frozen state included','ema':'model.teacher included','precision':'bf16-mixed production; no scaler required'}
    def on_load_checkpoint(self,checkpoint):
        s=checkpoint['task_state']
        if s['schema']!='native-occ-qwen-lightning-v1' or s['settings']!=self.settings or s['manifest_sha']!=self.manifest_sha or s['contract_test']!=self.contract_test:
            raise RuntimeError('Checkpoint architecture/data/settings/test contract mismatch')
        self.pending_rng=s['rng']

class TrainingStatus(pl.Callback):
    def __init__(self,folder):self.folder=Path(folder);self.started=None;self.step_start=0
    def write(self,trainer,module,phase,extra=None):
        self.folder.mkdir(parents=True,exist_ok=True)
        elapsed=time.monotonic()-self.started if self.started else 0
        done=max(0,trainer.global_step-self.step_start)
        total=math.ceil(5296/module.settings['batch_size'])*trainer.max_epochs
        metrics={k:float(v.detach().cpu()) for k,v in trainer.callback_metrics.items() if isinstance(v,torch.Tensor) and v.numel()==1}
        row={'phase':phase,'epoch':trainer.current_epoch+1,'global_step':trainer.global_step,'max_epochs':trainer.max_epochs,'elapsed_seconds':elapsed,'steps_per_second':done/max(elapsed,.001),'remaining_seconds':(total-trainer.global_step)*elapsed/max(done,1),'allocated_gb':torch.cuda.memory_allocated()/1e9,'reserved_gb':torch.cuda.memory_reserved()/1e9,'peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9,'metrics':metrics,**(extra or {})}
        target=self.folder/'status.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(target)
        print('STATUS '+json.dumps(row),flush=True)
    def on_train_start(self,trainer,module):self.started=time.monotonic();self.step_start=trainer.global_step;self.write(trainer,module,'training')
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        if trainer.global_step%10==0:self.write(trainer,module,'training')
    def on_validation_end(self,trainer,module):self.write(trainer,module,'validated')
    def on_train_end(self,trainer,module):self.write(trainer,module,'completed')
    def on_exception(self,trainer,module,exception):self.write(trainer,module,'failed',{'error':repr(exception)})

def settings():
    cfg=json.loads((ROOT/'configs/occ-qwen-native.json').read_text(encoding='utf-8-sig'))
    return {**cfg,'lr':cfg.get('lr',1e-4),'ema_momentum':.99,'sample_steps':8}

def make_callbacks(folder):
    common={'dirpath':str(folder),'auto_insert_metric_name':False,'enable_version_counter':False,'save_top_k':1}
    return [ModelCheckpoint(**common,filename='best',monitor='val/action_ade_m',mode='min',every_n_epochs=1,save_on_train_epoch_end=False),ModelCheckpoint(**common,filename='latest',every_n_train_steps=50,save_on_train_epoch_end=False),ModelCheckpoint(**common,filename='latest',every_n_epochs=1,save_on_train_epoch_end=True)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--fit',action='store_true');p.add_argument('--resume');a=p.parse_args()
    cfg=settings()
    if not a.fit:
        print(json.dumps({'formal_training_paused':cfg['formal_training_paused'],'train_windows':len(NativeDataset('train')),'validation_windows':len(NativeDataset('validation')),'action':'preflight only; no model or optimizer created'},indent=2));return
    require_training_authorization() # before loading weights or starting compute
    torch.set_num_threads(4)
    pl.seed_everything(cfg['seed'],workers=True,verbose=False)
    folder=ROOT/'artifacts/occ-qwen-native-training'
    model=NativeOccQwenJointFlow(load_text_backbone(ROOT/'models/Qwen3-VL-2B-Instruct'))
    sha=hashlib.sha256((ROOT/cfg['manifest']).read_bytes()).hexdigest()
    task=NativeTask(model,cfg,sha);data=NativeData(cfg)
    trainer=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_epochs=cfg.get('max_epochs',20),plugins=[VerifiedCheckpointIO()],callbacks=make_callbacks(folder)+[TrainingStatus(folder)],logger=TensorBoardLogger(ROOT/'logs/tensorboard',name='occ-qwen-native'),gradient_clip_val=1.,num_sanity_val_steps=0,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=10)
    trainer.fit(task,datamodule=data,ckpt_path=a.resume,weights_only=False)
if __name__=='__main__':main()


