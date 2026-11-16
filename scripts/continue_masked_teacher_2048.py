"""Authorized, versioned continuation; no automatic extension beyond step2048."""
import json,time,hashlib,copy
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from train_masked_teacher_bounded import Task,Data
from lightning_task import ROOT,configure_runtime,VerifiedCheckpointIO
from probe_teacher_history_matrix import idle
class ContinueTask(Task):
 def validation_step(self,b,idx):
  super().validation_step(b,idx)
  with torch.no_grad():
   _,z=self.model.forecast_batch(b);f,valid,frac=self.model.target_data(b);target=self.model.teacher(f).float();current=self.model.teacher(self.model.canonical_history(b)[:,-1:]).float().expand_as(target);w=valid.float();den=w.sum().clamp_min(1)
   self.log_dict({'world/latent_prediction_mse':((z.float()-target).square().mean(-1)*w).sum()/den,'world/persistence_mse':((current-target).square().mean(-1)*w).sum()/den},on_epoch=True,batch_size=len(b['state']))
class Monitor(pl.Callback):
 def __init__(self,out):self.out=out
 def on_train_start(self,t,m):self.start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 def on_train_batch_end(self,t,m,outputs,b,idx):
  if t.global_step%32==0 or t.global_step==257:
   torch.cuda.synchronize();n=t.global_step-256;elapsed=time.perf_counter()-self.start;r={'status':'running','step':t.global_step,'target':2048,'loss':float(outputs['loss']),'lr':t.optimizers[0].param_groups[0]['lr'],'elapsed_s':elapsed,'s_per_new_step':elapsed/n,'eta_s':elapsed/n*(2048-t.global_step),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'collision':None,'takeover':None,'automatic_astra_rl':False};(self.out/'status.json').write_text(json.dumps(r,indent=2));print('STATUS '+json.dumps(r),flush=True)
 def on_validation_end(self,t,m):
  r={'step':t.global_step,**{k:float(v) for k,v in t.callback_metrics.items() if isinstance(v,torch.Tensor) and v.numel()==1}}
  with (self.out/'validation.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
  print('VALIDATION '+json.dumps(r),flush=True)
def main():
 idle();configure_runtime();pl.seed_everything(1701,workers=True);src=ROOT/'artifacts/masked-teacher-155m-first256';out=ROOT/'artifacts/masked-teacher-155m-continue2048';out.mkdir(exist_ok=False)
 ck=torch.load(src/'best.ckpt',map_location='cpu',weights_only=False);assert ck['global_step']==256;assert ck['loops']['fit_loop']['state_dict']['combined_loader'][0]['_num_yielded']==256
 cfg=json.loads((src/'config.json').read_text());cfg.update(max_steps=2048,learning_rate=5e-5,experiment='masked-teacher-155m-versioned-continuation2048',checkpoint_steps=128)
 cfg['continuation']={'source':'masked-teacher-155m-first256/best.ckpt','initial_global_step':256,'new_updates':1792,'lr_schedule':'explicit restart at5e-5, cosine to5e-6 over1792 new steps','optimizer_moments':'retained','data_rng_cursor':'retained','not_exact_original_schedule_resume':True}
 sel=json.loads((src/'selection.json').read_text());task=ContinueTask(cfg,ck['task_state']['manifest_sha'],ck['task_state']['settings']['selection_sha']);task.train()
 # Explicitly version schedule while retaining model, optimizer moments, RNG and loader/loop cursor.
 ck['task_state']['settings']=copy.deepcopy(task.settings)
 for group in ck['optimizer_states'][0]['param_groups']:group.update(lr=5e-5,initial_lr=5e-5)
 sched=ck['lr_schedulers'][0];sched.update(T_max=1792,eta_min=5e-6,base_lrs=[5e-5],last_epoch=0,_step_count=1,_last_lr=[5e-5])
 ck['callbacks']={};VerifiedCheckpointIO().save_checkpoint(ck,str(out/'continuation-start.ckpt'));del ck
 (out/'config.json').write_text(json.dumps(cfg,indent=2));(out/'selection.json').write_text(json.dumps(sel));print('CONTINUATION_READY '+json.dumps(cfg['continuation']),flush=True)
 base=dict(dirpath=str(out),auto_insert_metric_name=False,enable_version_counter=False,save_top_k=1)
 tr=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_steps=2048,max_epochs=-1,max_time={'minutes':25},callbacks=[Monitor(out),ModelCheckpoint(**base,filename='latest',every_n_train_steps=128,save_on_train_epoch_end=False),ModelCheckpoint(**base,filename='best',monitor='validation_loss',mode='min',save_on_train_epoch_end=False)],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='masked-teacher',version='155m-continue2048'),num_sanity_val_steps=0,val_check_interval=256,check_val_every_n_epoch=None,gradient_clip_val=1.,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=16)
 tr.fit(task,datamodule=Data(sel),ckpt_path=str(out/'continuation-start.ckpt'),weights_only=False)
 r={'status':'completed','step':tr.global_step,'completed_budget':tr.global_step==2048,'test_read':False,'source_causality_certified':False,'validation':{k:float(v) for k,v in tr.callback_metrics.items() if isinstance(v,torch.Tensor) and v.numel()==1}};(out/'completion.json').write_text(json.dumps(r,indent=2));(out/'status.json').write_text(json.dumps(r,indent=2));print('DONE '+json.dumps(r),flush=True)
if __name__=='__main__':main()
