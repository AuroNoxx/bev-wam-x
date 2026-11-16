"""Bounded 155M masked teacher training; isolated outputs, no test access."""
import json,time,hashlib,argparse
from pathlib import Path
import torch,numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchdata.stateful_dataloader import StatefulDataLoader
from train_joint_occ import JointTask,MemorySamples
from lightning_task import ROOT,configure_runtime,VerifiedCheckpointIO
from masked_history_joint import MaskedHistoryJoint
from probe_teacher_history_matrix import idle
from stage1_objective import kinematic_residual
class DiskSamples(torch.utils.data.Dataset):
 def __init__(self,names):self.names=names
 def __len__(self):return len(self.names)
 def __getitem__(self,i):
  r=MemorySamples([self.names[i]])[0];r.update(history_valid=torch.ones(4,dtype=torch.bool),history_age_seconds=torch.tensor([.6,.4,.2,0.]),geometry_valid=r['history']!=255);return r
class Data(pl.LightningDataModule):
 def __init__(self,selection):super().__init__();self.selection=selection
 def loader(self,split):return StatefulDataLoader(DiskSamples(self.selection[split]),batch_size=8,shuffle=split=='train',num_workers=0,generator=torch.Generator().manual_seed(1701 if split=='train' else 1907))
 def train_dataloader(self):return self.loader('train')
 def val_dataloader(self):return self.loader('validation')
class Task(JointTask):
 def __init__(self,cfg,msha,ssha):
  super().__init__('fusion',cfg,msha,ssha,1701);self.model=MaskedHistoryJoint(768,15,12)
 def validation_step(self,b,batch_idx):
  with torch.random.fork_rng(devices=[self.device.index]):
   torch.manual_seed(1908+batch_idx);loss=self.task_loss(b,'validation');self.log('validation_loss',loss,on_epoch=True,batch_size=len(b['state']))
   p=self.model.sample(b,k=2,steps=8);ade=((p[...,:2]-b['trajectory'][:,None,:,:2])*p.new_tensor([50,10])).norm(dim=-1).mean();kin=kinematic_residual(p.flatten(0,1),b['state'].repeat_interleave(2,0)).norm(dim=-1).mean()
   self.log_dict({'diagnostic/ADE_m':ade,'diagnostic/kinematic_mps':kin},on_epoch=True,batch_size=len(p))
class Status(pl.Callback):
 def __init__(self,out):self.out=out
 def on_train_start(self,t,m):self.start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 def on_train_batch_end(self,t,m,outputs,b,idx):
  if t.global_step in [1,4,8] or t.global_step%16==0:
   torch.cuda.synchronize();elapsed=time.perf_counter()-self.start;r={'status':'running','step':t.global_step,'budget_steps':256,'loss':float(outputs['loss']),'elapsed_s':elapsed,'mean_s_per_step_including_io':elapsed/t.global_step,'remaining_train_s_estimate':elapsed/t.global_step*(256-t.global_step),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'collision':None,'takeover':None,'closed_loop':False,'astra_automatic_training':False};(self.out/'status.json').write_text(json.dumps(r,indent=2));print('STATUS '+json.dumps(r),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args();idle();configure_runtime();pl.seed_everything(1701,workers=True)
 out=ROOT/'artifacts/masked-teacher-155m-first256';out.mkdir(exist_ok=a.resume)
 cfg=json.loads((ROOT/'configs/joint-occ-v1.json').read_text());cfg.update(max_steps=256,learning_rate=1e-4,width=768,checkpoint_steps=64,validation_every_epochs=1,experiment='masked-teacher-155m-bounded-offline',seed=1701)
 cfg['limitations']='Offline supervised labels not uniquely correct driving. Four nominal history frames masked; source OCC causality NOT certified. No automatic Astra RL or learned CARLA closed loop. ADE diagnostic only.'
 train=sorted((ROOT/'data/occ_v3/cache/train').rglob('*.npz'));val=sorted((ROOT/'data/occ_v3/cache/validation').rglob('*.npz'))
 assert train and val
 # Fixed evenly spaced validation subset; no test enumeration.
 val=[val[i] for i in np.linspace(0,len(val)-1,64,dtype=int)]
 selection={k:[str(x.relative_to(ROOT/'data/occ_v3')) for x in v] for k,v in [('train',train),('validation',val)]};ss=json.dumps(selection,sort_keys=True);msha=hashlib.sha256((ROOT/'data/occ_v3/manifest.json').read_bytes()).hexdigest();task=Task(cfg,msha,hashlib.sha256(ss.encode()).hexdigest())
 if not a.resume:
  (out/'selection.json').write_text(ss);(out/'config.json').write_text(json.dumps(cfg,indent=2))
  print('CONFIG '+json.dumps({'train_windows':len(train),'validation_windows':len(val),'parameters_nonEMA':sum(p.numel() for p in task.model.parameters())-sum(p.numel() for p in task.model.teacher.parameters()),'width':768,'layers':15,'heads':12,'batch':8,'steps':256}),flush=True)
 # Verify mask-aware actor cannot read future labels before launching training.
 task.cuda().eval();b={k:v[None].cuda() for k,v in DiskSamples(selection['train'])[0].items()};xt=torch.randn(1,25,5,device='cuda');tt=torch.full((1,1),.3,device='cuda')
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  y=task.model(b,xt,tt);z=task.model({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.zeros_like(b['trajectory'])},xt,tt);assert torch.equal(y,z)
 del b,y,z;task.train();task.cpu();torch.cuda.empty_cache()
 base=dict(dirpath=str(out),auto_insert_metric_name=False,enable_version_counter=False,save_top_k=1)
 trainer=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_steps=256,max_epochs=-1,max_time={'minutes':20},callbacks=[Status(out),ModelCheckpoint(**base,filename='latest',every_n_train_steps=64,save_on_train_epoch_end=False),ModelCheckpoint(**base,filename='best',monitor='validation_loss',mode='min',save_on_train_epoch_end=False)],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='masked-teacher',version='155m-first256'),num_sanity_val_steps=0,val_check_interval=128,check_val_every_n_epoch=None,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=8)
 trainer.fit(task,datamodule=Data(selection),ckpt_path=str(out/'best.ckpt') if a.resume else None,weights_only=False)
 (out/'completion.json').write_text(json.dumps({'step':trainer.global_step,'completed_budget':trainer.global_step==256,'test_read':False,'source_causality_certified':False,'validation':{k:float(v) for k,v in trainer.callback_metrics.items() if isinstance(v,torch.Tensor) and v.numel()==1}},indent=2));print('DONE',trainer.global_step,flush=True)
if __name__=='__main__':main()
