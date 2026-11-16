"""Bounded fusion-checkpoint continuation ablation: JEPA vs no JEPA, not from scratch."""
import copy,hashlib,json,time,gc
from collections import defaultdict
import torch,numpy as np,pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from masked_history_joint import MaskedHistoryJoint
from train_masked_teacher_bounded import Task,DiskSamples
from train_boundary_isolated_multiseed import SeedData
from train_boundary_short_comparison import Progress
from train_joint_occ import batch_to
from lightning_task import ROOT,configure_runtime,VerifiedCheckpointIO
from probe_teacher_history_matrix import idle
from stage1_objective import consistent_flow_loss
from audit_candidate_boundary import audit

class NoJEPA(MaskedHistoryJoint):
 def context(self,b):
  history=self.canonical_history(b);valid=b['history_valid'].bool();age=b['history_age_seconds'].float()
  if not torch.isfinite(age).all() or (age<0).any() or (age[:,1:]>age[:,:-1]).any():raise ValueError('Invalid ages')
  frames=self.encoder(history);bs,t,n,d=frames.shape;h=frames.new_zeros(bs,n,d);last_age=age[:,0];seen=torch.zeros(bs,device=age.device,dtype=torch.bool)
  for i in range(t):
   delta=torch.where(seen,last_age-age[:,i],torch.zeros_like(last_age)).clamp_min(0);timing=self.history_time(torch.stack([age[:,i]/5.,delta/5.],-1).to(frames.dtype))[:,None];x=frames[:,i]+timing;reset,update=self.temporal_gate(torch.cat([x,h],-1)).sigmoid().chunk(2,-1);candidate=self.temporal_candidate(torch.cat([x,reset*h],-1)).tanh();h=torch.where(valid[:,i,None,None],(1-update)*h+update*candidate,h);last_age=torch.where(valid[:,i],age[:,i],last_age);seen|=valid[:,i]
  return h+self.spatial+self.state(b['state'])[:,None]
class AblationTask(Task):
 def __init__(self,cfg,msha,ssha,seed,mode):
  super().__init__(cfg,msha,ssha);self.seed=seed;self.mode=mode;self.settings.update(ablation_mode=mode,seed=seed);self.trace=[];self.grads=[]
  if mode=='no_jepa':
   self.model=NoJEPA(768,15,12);self.model.predictor.requires_grad_(False);self.model.occ_head.requires_grad_(False);self.model.future_time.requires_grad_(False)
 def task_loss(self,b,split):
  with torch.random.fork_rng(devices=[self.device.index]):
   torch.manual_seed(self.seed*100000+self.global_step);before=torch.cuda.get_rng_state().clone();v,terms=consistent_flow_loss(self.model,b,self.settings['objective'])
   if self.mode=='jepa':
    for k,x in zip(['latent','variance','semantic'],self.model.auxiliary_loss(b)):v=v+self.settings['aux_weights'][k]*x;terms[k]=x
   after=torch.cuda.get_rng_state().clone()
  if self.global_step<3:self.trace.append({'step':self.global_step,'before':hashlib.sha256(before.cpu().numpy().tobytes()).hexdigest(),'after':hashlib.sha256(after.cpu().numpy().tobytes()).hexdigest()})
  self.log_dict({f'{split}/{k}':x for k,x in terms.items()},on_epoch=True,batch_size=len(b['state']));return v
 def configure_gradient_clipping(self,optimizer,gradient_clip_val=None,gradient_clip_algorithm=None):
  n=torch.nn.utils.clip_grad_norm_(self.parameters(),gradient_clip_val,error_if_nonfinite=True);self.grads.append({'step':self.global_step,'norm':float(n),'clipped':float(n)>gradient_clip_val})
@torch.no_grad()
def evaluate(m,names,reference,enabled):
 m.cuda().eval();ds=DiskSamples(names);rows=[];world=[];div=[]
 for start in range(0,len(ds),4):
  b=batch_to([ds[i] for i in range(start,min(start+4,len(ds)))],'cuda')
  with torch.random.fork_rng(devices=[0]),torch.autocast('cuda',dtype=torch.bfloat16):
   torch.manual_seed(1940+start);p=m.sample(b,k=4,steps=16).float();physical=(p*p.new_tensor([50.,10.,1,1,30.])).cpu().numpy();_,z=m.forecast_batch(b);future,valid,_=m.target_data(b);target=reference(future).float();persistent=reference(m.canonical_history(b)[:,-1:]).float().expand_as(target);own=m.teacher(future).float();current=m.teacher(m.canonical_history(b)[:,-1:]).float().expand_as(own)
   # Future poison must not change actor context, even in fusion arm.
   assert torch.equal(m.context(b),m.context({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)}))
  for j,q in enumerate(physical):
   div.append(float(np.sqrt(((q[:,:,:2]-q[:,:,:2].mean(0))**2).sum(-1).mean())))
   for k,trajectory in enumerate(q):
    a=audit(trajectory,float(b['state'][j,0])*30);derived=np.linalg.norm(np.diff(np.vstack(([0.,0.],trajectory[:,:2])),axis=0),axis=1)/.2;rows.append({'window':start+j,'candidate':k,'first_residual_mps':a['first_segment']['kinematic_residual_norm_mps'],'first_accel_mps2':a['first_segment']['signed_speed_accel_mps2'],'kinematic_mps':a['full']['mean_kinematic_residual_mps'],'max_accel_mps2':a['full']['max_abs_scalar_accel_mps2'],'max_jerk_mps3':a['full']['max_abs_scalar_jerk_mps3'],'forward_proxy_m':float(trajectory[-1,0]),'negative_forward_segments':int((np.diff(np.r_[0,trajectory[:,0]])<0).sum()),'nearzero':float((trajectory[:,4]<.5).mean()),'speed_disagreement':float(abs(derived-trajectory[:,4]).mean()),'points':trajectory.tolist(),'collision':None,'TTC':None,'route_progress':None})
   mask=valid[j];n=int(mask.sum());world.append({'window':start+j,'valid_tokens':n,'fixed_reference_mse':float((z[j].float()-target[j]).square().mean(-1)[mask].mean()) if n else None,'fixed_persistence_mse':float((persistent[j]-target[j]).square().mean(-1)[mask].mean()) if n else None,'own_ema_mse':float((z[j].float()-own[j]).square().mean(-1)[mask].mean()) if n else None,'own_persistence_mse':float((current[j]-own[j]).square().mean(-1)[mask].mean()) if n else None})
 keys=['first_residual_mps','kinematic_mps','max_accel_mps2','max_jerk_mps3','forward_proxy_m','negative_forward_segments','nearzero','speed_disagreement'];return {'summary':{**{k:float(np.mean([r[k] for r in rows])) for k in keys},'spread_rms_m':float(np.mean(div)),'global_first_accel':max(abs(r['first_accel_mps2']) for r in rows),'global_jerk':max(r['max_jerk_mps3'] for r in rows)},'world_summary':{k:float(np.mean([r[k] for r in world if r[k] is not None])) for k in ['fixed_reference_mse','fixed_persistence_mse','own_ema_mse','own_persistence_mse']},'world_head_active':enabled,'world_rows':world,'records':rows,'actor_future_poison_exact':True}
def main():
 idle();configure_runtime();src=ROOT/'artifacts/masked-teacher-155m-continue2048';out=ROOT/'artifacts/jepa-nojepa-continuation-paired';out.mkdir(exist_ok=False);train=[r['file'] for r in json.loads((ROOT/'artifacts/jepa-route-balanced-train-audit/selection.json').read_text())];groups=defaultdict(list)
 for p in sorted((ROOT/'data/occ_v3/cache/validation').rglob('*.npz')):groups[p.parent.name].append(str(p.relative_to(ROOT/'data/occ_v3')))
 val=[items[i] for _,items in sorted(groups.items()) for i in np.linspace(0,len(items)-1,3,dtype=int)];sel={'train':train,'validation':val};ss=json.dumps(sel,sort_keys=True);(out/'selection.json').write_text(ss);ssha=hashlib.sha256(ss.encode()).hexdigest();ck=torch.load(src/'best.ckpt',map_location='cpu',weights_only=False);cfg=json.loads((src/'config.json').read_text());cfg.update(max_steps=64,learning_rate=1e-5,experiment='fusion_checkpoint_JEPA_removal_continuation');cfg['ablation']={'seeds':[1930,1931],'steps':64,'batch':8,'train_windows':len(train),'dev_windows':len(val),'dev_routes':len(groups),'loss':'original FM+physical; JEPA arm adds original aux0.1each; NO rollout regularizer','limits':'Continuation from fusion-trained source, not from scratch; development only; no free truth or source OCC causality certificate','budget':'4arms each5min, total15min'};(out/'config.json').write_text(json.dumps(cfg,indent=2));reference=None;results=[];startall=time.perf_counter();print('CONFIG '+json.dumps(cfg['ablation']),flush=True)
 for seed in [1930,1931]:
  traces={}
  for mode in (['jepa','no_jepa'] if seed==1930 else ['no_jepa','jepa']):
   idle();assert time.perf_counter()-startall<900;pl.seed_everything(seed,workers=True);folder=out/f'{seed}-{mode}';folder.mkdir();task=AblationTask(cfg,ck['task_state']['manifest_sha'],ssha,seed,mode);task.load_state_dict(ck['state_dict'],strict=True);task.cuda().eval()
   if reference is None:reference=copy.deepcopy(task.model.teacher).eval().requires_grad_(False)
   if mode=='no_jepa':
    b=batch_to([DiskSamples(train)[0]],'cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):assert torch.equal(task.model.context(b),task.model.forecast_batch(b)[0])
   active=sum(p.numel() for p in task.model.parameters() if p.requires_grad);before=evaluate(task.model,val,reference,mode=='jepa');task.train();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();tr=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_steps=64,max_epochs=-1,max_time={'minutes':5},callbacks=[Progress(f'{seed}-{mode}'),ModelCheckpoint(dirpath=str(folder),filename='latest',auto_insert_metric_name=False,enable_version_counter=False,save_top_k=1,every_n_train_steps=32,save_on_train_epoch_end=False)],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='jepa-nojepa-continuation',version=f'{seed}-{mode}'),num_sanity_val_steps=0,limit_val_batches=0,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=8);tr.fit(task,datamodule=SeedData(sel,seed));wall=time.perf_counter()-start;allocated=torch.cuda.max_memory_allocated()/2**30;reserved=torch.cuda.max_memory_reserved()/2**30;ev=time.perf_counter();after=evaluate(task.model,val,reference,mode=='jepa');result={'seed':seed,'mode':mode,'updates':tr.global_step,'active_parameters':active,'train_wall_s':wall,'evaluation_after_s':time.perf_counter()-ev,'peak_allocated_gib':allocated,'peak_reserved_gib':reserved,'before':before,'after':after,'gradient_log':task.grads,'rng_trace':task.trace};traces[mode]=task.trace;(folder/'report.json').write_text(json.dumps(result,indent=2));brief={k:v for k,v in result.items() if k not in ['before','after','gradient_log','rng_trace']};brief.update(before=before['summary'],after=after['summary'],world=after['world_summary']);results.append(brief);print('RESULT '+json.dumps(brief),flush=True);del task,tr;gc.collect();torch.cuda.empty_cache()
  assert traces['jepa']==traces['no_jepa'],'paired RNG mismatch'
 (out/'summary.json').write_text(json.dumps({'results':results,'paired_first3_FM_rng_equal':True,'total_loop_wall_s':time.perf_counter()-startall,'limits':cfg['ablation']['limits']},indent=2));print('ALL_DONE',flush=True)
if __name__=='__main__':main()
