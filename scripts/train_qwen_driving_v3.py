"""Full valid-window frozen-Qwen driving-head training, per-epoch validation."""
from pathlib import Path
import json,math,time,hashlib
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from qwen_driving_v3_head import Head
from lightning_task import VerifiedCheckpointIO,random_state,restore_random
R=Path(__file__).resolve().parents[1];O=R/'artifacts/qwen-driving-v3';torch.set_num_threads(6)
class OwnCheckpointIO(VerifiedCheckpointIO):
 def load_checkpoint(self,path,map_location=None,weights_only=None):
  # Own local training checkpoint contains NumPy/Python RNG states.
  return super().load_checkpoint(path,map_location=map_location,weights_only=False)
class Driving(Head):
 def __init__(self):super().__init__('joint_fm');self.history=[];self.started=time.perf_counter()
 def on_save_checkpoint(self,c):
  c['task_state']=random_state()
  if hasattr(self,'train_generator'):c['task_state']['dataloader_generator']=self.train_generator.get_state()
 def on_load_checkpoint(self,c):
  restore_random(c['task_state'])
  if hasattr(self,'train_generator') and 'dataloader_generator' in c['task_state']:self.train_generator.set_state(c['task_state']['dataloader_generator'])
 def on_train_start(self):
  d=dict(restored_global_step=self.global_step,restored_epoch=self.current_epoch,optimizer_steps=sorted({float(v['step']) for v in self.trainer.optimizers[0].state.values() if 'step' in v}),note='Epoch1 recovery restored weights/optimizer/scheduler/loops; RNG and shuffle history were unavailable and reseeded, not bit-exact')
  (O/'resume_evidence.json').write_text(json.dumps(d,indent=2));print('RESUME_EVIDENCE',d,flush=True)
 def on_train_epoch_start(self):self.train_sum=0.;self.train_n=0
 def training_step(self,b,idx):
  f,st,y,wy=b;t=torch.rand(len(y),1,device=y.device);n=torch.randn_like(y);wn=torch.randn_like(wy);a,v=self(f,st,(1-t[:,:,None])*n+t[:,:,None]*y,t,(1-t[:,:,None])*wn+t[:,:,None]*wy);loss=(a-(y-n)).square().mean()+.1*(v-(wy-wn)).square().mean();self.log('train_loss',loss,on_step=True,on_epoch=True,batch_size=len(y));self.train_sum+=float(loss.detach())*len(y);self.train_n+=len(y)
  if (self.global_step+1)%50==0:
   d=dict(phase='training',epoch=self.current_epoch+1,step=self.global_step+1,batch_loss=float(loss.detach()),allocated_gib=torch.cuda.memory_allocated()/2**30,peak_gib=torch.cuda.max_memory_allocated()/2**30);(O/'live_progress.json').write_text(json.dumps(d));print('TRAIN',d,flush=True)
  return loss
 def configure_optimizers(self):
  opt=torch.optim.AdamW(self.parameters(),lr=1e-4,weight_decay=.01);steps=math.ceil(5296/64);total=steps*40
  def lr(s):
   if s<steps:return max(.01,(s+1)/steps)
   return .1+.9*.5*(1+math.cos(math.pi*min(1,(s-steps)/(total-steps))))
  return {'optimizer':opt,'lr_scheduler':{'scheduler':torch.optim.lr_scheduler.LambdaLR(opt,lr),'interval':'step'}}
 def on_validation_epoch_start(self):self.valrows=[]
 def validation_step(self,b,idx):
  f,st,y,wy=b;g=torch.Generator(device=self.device).manual_seed(9188+idx);x=torch.randn(y.shape,device=self.device,generator=g);w=torch.randn(wy.shape,device=self.device,generator=g)
  for j in range(16):a,v=self(f,st,x,torch.full((len(y),1),j/16,device=self.device),w);x=x+a/16;w=w+v/16
  ego=torch.stack([st[:,0]*30,torch.zeros(len(y),device=self.device)],-1)[:,None,:];vel=x*10+ego;truthv=y*10+ego;pos=vel.cumsum(1)*.2;truth=truthv.cumsum(1)*.2;err=torch.linalg.vector_norm(pos-truth,dim=-1);cv=ego*torch.arange(1,26,device=self.device)[None,:,None]*.2
  row=torch.stack([err.mean(1),err[:,4],err[:,9],err[:,14],err[:,24],abs(vel.norm(dim=-1)-truthv.norm(dim=-1)).mean(1),(torch.diff(vel,n=2,dim=1)/.04).norm(dim=-1).mean(1),(w-wy).square().mean((1,2)),(cv-truth).norm(dim=-1).mean(1)],1);self.valrows.append(row.detach().cpu())
 def on_validation_epoch_end(self):
  if not self.valrows:return
  v=torch.cat(self.valrows).mean(0).tolist();self.log('val_ADE_m',v[0],prog_bar=True);d=dict(epoch=self.current_epoch+1,step=self.global_step,train_loss=self.train_sum/max(1,self.train_n),val_ADE_m=v[0],endpoint_1_2_3_5s_m=v[1:5],speed_MAE_mps=v[5],jerk_mean_mps3=v[6],world_MSE=v[7],CV_ADE_m=v[8],elapsed_seconds=time.perf_counter()-self.started,peak_gib=torch.cuda.max_memory_allocated()/2**30,protocol='held-out Town13 offline; CV is kinematic reference NOT traditional driving score');self.history.append(d)
  with (O/'epoch_results.jsonl').open('a') as z:z.write(json.dumps(d)+'\n')
  (O/'live_progress.json').write_text(json.dumps(d,indent=2));print('EPOCH_RESULT',json.dumps(d),flush=True)
def main():
 rows=[]
 for p in sorted((O/'cache').glob('*.pt')):rows.extend(torch.load(p,weights_only=False))
 train=[r for r in rows if r['meta']['split']=='train'];val=[r for r in rows if r['meta']['split']=='validation'];assert len(train)==5296 and len(val)==1124;assert not {x['meta']['route'] for x in train}&{x['meta']['route'] for x in val};proj=torch.randn(2048,32,generator=torch.Generator().manual_seed(9101))/2048**.5
 def pack(rr):
  f=torch.stack([x['features'] for x in rr]);st=torch.stack([x['state'] for x in rr]);pos=torch.stack([x['trajectory'][:,:2]*torch.tensor([50.,10.]) for x in rr]);v=torch.diff(torch.cat([torch.zeros(len(rr),1,2),pos],1),dim=1)/.2;y=(v-torch.stack([st[:,0]*30,torch.zeros(len(rr))],-1)[:,None,:])/10;w=torch.stack([x['future_features'] for x in rr])@proj;return f,st,y,w
 tf,ts,ty,tw=pack(train);vf,vs,vy,vw=pack(val);mu=tw.mean((0,1));scale=tw.std((0,1)).clamp_min(.1);tw=(tw-mu)/scale;vw=(vw-mu)/scale;torch.save(dict(mu=mu,scale=scale,projection=proj),O/'world_normalization.pt');del rows,train,val
 pl.seed_everything(9101);task=Driving();config=dict(parameters_trainable=sum(p.numel() for p in task.parameters()),d_model=256,layers=4,heads=8,FFN=1024,batch=64,max_epochs=40,steps_per_epoch=83,max_steps=3320,lr=1e-4,weight_decay=.01,warmup_steps=83,precision='FP32 head; frozen BF16 Qwen cache',train_windows=5296,validation_windows=1124,train_town='Town12',validation_town='Town13',test='untouched',world_persistence_mse=float((((vf@proj-mu)/scale)-vw).square().mean()),caution='overlapping windows, not independent episodes; no language generation or backbone finetuning');(O/'config.json').write_text(json.dumps(config,indent=2));print('CONFIG',config,flush=True)
 task.train_generator=torch.Generator().manual_seed(9101);trainloader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tf,ts,ty,tw),batch_size=64,shuffle=True,num_workers=0,generator=task.train_generator);valloader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(vf,vs,vy,vw),batch_size=64,num_workers=0)
 cb=ModelCheckpoint(dirpath=O/'weights',filename='best',monitor='val_ADE_m',mode='min',save_last=True,save_top_k=1,enable_version_counter=False);tr=pl.Trainer(accelerator='gpu',devices=1,precision='32-true',max_epochs=40,callbacks=[cb],plugins=[OwnCheckpointIO()],logger=TensorBoardLogger(str(R/'logs/tensorboard'),name='qwen-driving-v3'),gradient_clip_val=1,num_sanity_val_steps=0,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=10);torch.cuda.reset_peak_memory_stats()
 with sdpa_kernel(SDPBackend.MATH):tr.fit(task,trainloader,valloader,ckpt_path=str(O/'weights/last.ckpt') if (O/'weights/last.ckpt').exists() else None)
 (O/'completed.json').write_text(json.dumps(dict(epochs=tr.current_epoch,steps=tr.global_step,best_score=float(cb.best_model_score),best_checkpoint=cb.best_model_path),indent=2));print('TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
