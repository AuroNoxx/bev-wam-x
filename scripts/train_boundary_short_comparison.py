"""Isolated same-checkpoint, same-budget three-arm boundary training diagnostic."""
import json,time,copy,gc
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from train_masked_teacher_bounded import Task,Data,DiskSamples
from train_joint_occ import batch_to
from masked_history_joint import MaskedHistoryJoint
from lightning_task import ROOT,configure_runtime,VerifiedCheckpointIO
from probe_teacher_history_matrix import idle
from stage1_objective import consistent_flow_loss,kinematic_residual
from audit_candidate_boundary import audit
class IntegratedOutputJoint(MaskedHistoryJoint):
 def to_flow_space(self,b,p):return p
 def from_flow_space(self,b,p):
  p=p.float();direction=F.normalize(p[..., [3,2]],dim=-1,eps=1e-4);v=direction*p[...,4:5]*30;v0=torch.stack([b['state'][:,0]*30,torch.zeros_like(b['state'][:,0])],-1)[:,None];previous=torch.cat([v0,v[:,:-1]],1);xy=((v+previous)*.1).cumsum(1)/p.new_tensor([50.,10.]);return torch.cat([xy,direction[..., [1,0]],p[...,4:5]],-1)
 @torch.no_grad()
 def sample(self,b,k=4,steps=16):
  p=super().sample(b,k,steps);z=self.from_flow_space({'state':b['state'].repeat_interleave(k,0)},p.flatten(0,1));return z.reshape(len(p),k,25,5)
def extra_boundary(m,b):
 # Separate endpoint probe at fixed late bridge t=.8; not differentiable simulator.
 target=b['trajectory'].float();noise=torch.randn_like(target);t=torch.full((len(target),1),.8,device=target.device);xt=.2*noise+.8*target;endpoint=xt+.2*m(b,xt,t).float();res=kinematic_residual(endpoint,b['state'])[:,0]/2.;acc=(endpoint[:,0,4]-b['state'][:,0])*30/.2;excess=F.relu(acc.abs()-6)/6
 return F.smooth_l1_loss(res,torch.zeros_like(res))+.5*excess.square().mean()
class ProbeTask(Task):
 def __init__(self,cfg,msha,ssha,mode):
  super().__init__(cfg,msha,ssha);self.mode_variant=mode
  if mode=='integrated':self.model=IntegratedOutputJoint(768,15,12)
 def task_loss(self,b,split):
  v=super().task_loss(b,split)
  if self.mode_variant=='first_penalty':
   boundary=extra_boundary(self.model,b);v=v+.1*boundary;self.log(split+'/explicit_first_boundary',boundary,on_epoch=True,batch_size=len(b['state']))
  return v
class Progress(pl.Callback):
 def __init__(self,mode):self.mode=mode
 def on_train_batch_end(self,t,m,outputs,b,idx):
  if t.global_step%32==0:print('STEP '+json.dumps({'mode':self.mode,'step':t.global_step,'loss':float(outputs['loss'])}),flush=True)
@torch.no_grad()
def evaluate(m,names):
 m.cuda().eval();ds=DiskSamples(names);records=[]
 for start in range(0,len(ds),4):
  b=batch_to([ds[i] for i in range(start,start+4)],'cuda')
  with torch.random.fork_rng(devices=[0]),torch.autocast('cuda',dtype=torch.bfloat16):
   torch.manual_seed(8181+start);p=m.sample(b,k=4,steps=16).float();pred=(p*p.new_tensor([50.,10.,1,1,30.])).cpu().numpy()
  for j in range(len(pred)):
   for k,z in enumerate(pred[j]):
    q=audit(z,float(b['state'][j,0])*30);records.append({'window':start+j,'candidate':k,'first_accel_mps2':q['first_segment']['signed_speed_accel_mps2'],'first_residual_mps':q['first_segment']['kinematic_residual_norm_mps'],'kinematic_mps':q['full']['mean_kinematic_residual_mps'],'max_accel_mps2':q['full']['max_abs_scalar_accel_mps2'],'max_jerk_mps3':q['full']['max_abs_scalar_jerk_mps3'],'forward_proxy_m':float(z[-1,0]),'max_lateral_m':float(abs(z[:,1]).max()),'negative_forward_segments':int((np.diff(np.r_[0,z[:,0]])<0).sum()),'points':z.tolist(),'collision':None,'TTC':None,'route_progress':None})
 return {'summary':{k:float(np.mean([r[k] for r in records])) for k in ['first_residual_mps','kinematic_mps','max_accel_mps2','max_jerk_mps3','forward_proxy_m','max_lateral_m','negative_forward_segments']},'peak_first_abs_accel_mps2':max(abs(r['first_accel_mps2']) for r in records),'records':records}
def main():
 idle();configure_runtime();src=ROOT/'artifacts/masked-teacher-155m-continue2048';out=ROOT/'artifacts/teacher-boundary-short-training';out.mkdir(exist_ok=False);selection=json.loads((src/'selection.json').read_text());selection['train']=[selection['train'][i] for i in np.linspace(0,len(selection['train'])-1,128,dtype=int)];selection['validation']=selection['validation'][:16];(out/'selection.json').write_text(json.dumps(selection));ck=torch.load(src/'best.ckpt',map_location='cpu',weights_only=False);cfg=json.loads((src/'config.json').read_text());cfg.update(max_steps=96,learning_rate=1e-5,experiment='boundary-short-finetune-96steps');cfg['short_probe']={'initialization':'teacher2048 weights only','optimizer':'new same AdamW for all arms','physical_batch':8,'train_windows':128,'validation_windows':16,'seed':1801,'first_penalty_coefficient':.1,'accel_hinge_scale_mps2':6,'integrated':'5-channel FM targets retained for compatibility; xy heads still supervised but discarded at physical decode; not optimized three-channel architecture','safety':'unvalidated bounds, no safety acceptance'};(out/'config.json').write_text(json.dumps(cfg,indent=2));results=[]
 for mode in ['baseline','first_penalty','integrated']:
  pl.seed_everything(1801,workers=True);folder=out/mode;folder.mkdir();task=ProbeTask(cfg,ck['task_state']['manifest_sha'],ck['task_state']['settings']['selection_sha'],mode);task.load_state_dict(ck['state_dict'],strict=True);before=evaluate(task.model,selection['validation']);task.train();task.cpu();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  base=dict(dirpath=str(folder),auto_insert_metric_name=False,enable_version_counter=False,save_top_k=1)
  tr=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_steps=96,max_epochs=-1,max_time={'minutes':5},callbacks=[Progress(mode),ModelCheckpoint(**base,filename='latest',every_n_train_steps=48,save_on_train_epoch_end=False)],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='boundary-short',version=mode),num_sanity_val_steps=0,limit_val_batches=0,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=8)
  tr.fit(task,datamodule=Data(selection));wall=time.perf_counter()-start;peak=torch.cuda.max_memory_reserved()/2**30;after=evaluate(task.model,selection['validation']);result={'mode':mode,'steps':tr.global_step,'wall_seconds':wall,'peak_reserved_gib':peak,'before':before,'after':after};(folder/'report.json').write_text(json.dumps(result,indent=2));results.append({k:v for k,v in result.items() if k not in ['before','after']}|{'before':before['summary'],'after':after['summary'],'peak_first_abs_accel_before':before['peak_first_abs_accel_mps2'],'peak_first_abs_accel_after':after['peak_first_abs_accel_mps2']});print('RESULT '+json.dumps(results[-1]),flush=True);del task,tr;gc.collect();torch.cuda.empty_cache()
 (out/'summary.json').write_text(json.dumps({'results':results,'limits':'one seed,128 train/16 previously-used validation windows,96 updates,not driving acceptance. First-penalty adds a second forward so equal updates not equal FLOPs. Integrated changes decode, compare pre/post not just absolute.'},indent=2))
if __name__=='__main__':main()
