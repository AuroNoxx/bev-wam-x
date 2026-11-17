"""Isolated v2 trainer: never resumes or rewrites v1 checkpoints."""
import argparse,copy,hashlib,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from b2d_native_occ import ROOT
from train_occ_qwen_native import NativeTask,TrainingStatus
from train_occ_qwen_native_optimized import CachedData,SafePause
from occ_qwen_native_v2 import NativeOccQwenV2,load_v2_backbone
from occ_qwen_performance import enable_channels_last
from lightning_task import VerifiedCheckpointIO,random_state

class NativeV2Task(NativeTask):
 def on_fit_start(self):
  if self.contract_test:
   if not 0<self.trainer.max_steps<=4:raise RuntimeError('Only bounded four-step contract tests')
  elif self.settings.get('formal_training_paused',True) or not self.settings.get('training_authorization'):raise RuntimeError('V2 training not authorized')
 def configure_optimizers(self):
  fast=[];heads=[]
  for name,p in self.model.named_parameters():
   if p.requires_grad:(fast if name.startswith(('encoder.','scene_in.','scene_out.','scene_norm.')) else heads).append(p)
  assert fast and heads
  opt=torch.optim.AdamW([{'params':fast,'lr':self.settings['encoder_lr'],'name':'occ_encoder_projectors'},{'params':heads,'lr':self.settings['lr'],'name':'action_world_adapters'}],weight_decay=.01,fused=False)
  total=math.ceil(5296/self.settings['batch_size'])*self.settings['max_epochs'];warmup=100
  def scale(step):
   if step<warmup:return (step+1)/warmup
   return .1+.45*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total-warmup))))
  return {'optimizer':opt,'lr_scheduler':{'scheduler':torch.optim.lr_scheduler.LambdaLR(opt,scale),'interval':'step'}}
 def configure_gradient_clipping(self,optimizer,gradient_clip_val=None,gradient_clip_algorithm=None):
  if getattr(self.trainer.precision_plugin,'scaler',None) is not None:raise RuntimeError('BF16 no scaler expected')
  torch.nn.utils.clip_grad_norm_([p for p in self.parameters() if p.requires_grad],gradient_clip_val,error_if_nonfinite=True)
 def on_train_start(self):
  steps=sorted({int(v['step']) for o in self.trainer.optimizers for v in o.state.values() if 'step' in v});sc=[x.scheduler.last_epoch for x in self.trainer.lr_scheduler_configs]
  assert (steps==[self.global_step] or self.global_step==0 and not steps) and sc==[self.global_step]
  print('V2_STATE '+json.dumps({'global_step':self.global_step,'optimizer_steps':steps,'scheduler_last_epoch':sc}),flush=True)
 def on_save_checkpoint(self,checkpoint):
  super().on_save_checkpoint(checkpoint);checkpoint['task_state']['schema']='native-occ-qwen-v2-lightning-v1'
 def on_load_checkpoint(self,checkpoint):
  s=checkpoint['task_state']
  if s['schema']!='native-occ-qwen-v2-lightning-v1' or s['settings']!=self.settings or s['manifest_sha']!=self.manifest_sha or s['contract_test']!=self.contract_test:raise RuntimeError('V2 resume contract mismatch')
  self.pending_rng=s['rng']
 def validation_step(self,batch,batch_idx):
  loss,_=self.model.loss(**batch)
  generator=torch.Generator(device=self.device).manual_seed(1907+batch_idx)
  out=self.model.sample(batch['history'],batch['state'],steps=8,generator=generator);pred=out['trajectory_xy'];target=batch['trajectory'][...,:2]*pred.new_tensor([50.,10.])
  ade=(pred-target).norm(dim=-1).mean();vel=out['residual_velocity'].float()*10+torch.stack([batch['state'][:,0]*30,torch.zeros_like(batch['state'][:,0])],-1)[:,None]
  jerk=torch.diff(vel,n=2,dim=1).norm(dim=-1)/.04
  target_world=self.model.world_target(batch['future']);base=self.model.persistence(batch['history'])
  known=((batch['future']>0)&(batch['future']<8)).float().mean(2);mask=F.adaptive_avg_pool2d(known.flatten(0,1)[:,None],(8,8)).reshape(len(pred),-1)>0
  metrics={'val/action_ade_m':ade,'val/loss':loss,'val/jerk_mean_mps3':jerk.mean(),'val/jerk_trajectory_over10':(jerk>10).any(1).float().mean()}
  for name,w in [('world',out['world_latent']),('persistence',base)]:metrics['val/'+name+'_mse']=((w-target_world).square().mean(-1)*mask).sum()/mask.sum().clamp_min(1)
  self.log_dict(metrics,on_epoch=True,batch_size=len(pred))

def config():return json.loads((ROOT/'configs/occ-qwen-native-v2.json').read_text(encoding='utf-8-sig'))
def make_model(cfg):
 model=NativeOccQwenV2(load_v2_backbone(ROOT/'models/Qwen3-VL-2B-Instruct',layers=cfg['qwen_layers']),accel_weight=cfg['accel_weight'],jerk_weight=cfg['jerk_weight'])
 if cfg['channels_last']:enable_channels_last(model)
 return model

def main():
 p=argparse.ArgumentParser();p.add_argument('--fit',action='store_true');p.add_argument('--resume');a=p.parse_args();cfg=config()
 if not a.fit:print(json.dumps(cfg,indent=2));return
 if cfg['formal_training_paused'] or not cfg['training_authorization']:raise RuntimeError('No v2 authorization')
 if json.loads((ROOT/'artifacts/occ-qwen-native-training/status.json').read_text())['phase']!='completed':raise RuntimeError('V1 has not completed')
 torch.set_num_threads(4);pl.seed_everything(cfg['seed'],workers=True,verbose=False)
 folder=ROOT/'artifacts/occ-qwen-native-v2-training';folder.mkdir(exist_ok=True)
 data=CachedData(cfg,ROOT/cfg['cache_index']);model=make_model(cfg)
 sha=hashlib.sha256((ROOT/cfg['manifest']).read_bytes()).hexdigest();task=NativeV2Task(model,cfg,sha)
 common={'dirpath':str(folder),'auto_insert_metric_name':False,'enable_version_counter':False,'save_top_k':1}
 callbacks=[ModelCheckpoint(**common,filename='best',monitor='val/action_ade_m',mode='min',every_n_epochs=1,save_on_train_epoch_end=False),ModelCheckpoint(**common,filename='latest',every_n_train_steps=100,save_on_train_epoch_end=False),ModelCheckpoint(**common,filename='latest',every_n_epochs=1,save_on_train_epoch_end=True),TrainingStatus(folder),SafePause(folder)]
 trainer=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_epochs=cfg['max_epochs'],plugins=[VerifiedCheckpointIO()],callbacks=callbacks,logger=TensorBoardLogger(ROOT/'logs/tensorboard',name='occ-qwen-native-v2'),gradient_clip_val=1.,num_sanity_val_steps=0,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=10)
 print('MODEL '+json.dumps({'total_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'validation':'per-batch independent generator1907+batch_idx; single candidate8steps'}),flush=True)
 trainer.fit(task,datamodule=data,ckpt_path=a.resume,weights_only=False)
if __name__=='__main__':main()
