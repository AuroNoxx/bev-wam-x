"""Paired multi-seed short fine-tunes with isolated FM and boundary RNG streams."""
import json,time,gc,hashlib
import numpy as np,torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchdata.stateful_dataloader import StatefulDataLoader
from train_boundary_short_comparison import evaluate as base_evaluate,Progress
from train_joint_occ import batch_to
from rollout_dynamics_regularizer import rollout,dynamics_loss
from train_masked_teacher_bounded import Task,Data,DiskSamples
from lightning_task import ROOT,configure_runtime,VerifiedCheckpointIO
from probe_teacher_history_matrix import idle
class SeedData(Data):
 def __init__(self,selection,seed):super().__init__(selection);self.seed=seed
 def loader(self,split):return StatefulDataLoader(DiskSamples(self.selection[split]),batch_size=8,shuffle=split=='train',num_workers=0,generator=torch.Generator().manual_seed(self.seed if split=='train' else 1907))
class IsolatedTask(Task):
 def __init__(self,cfg,msha,ssha,seed,mode):
  super().__init__(cfg,msha,ssha);self.noise_seed=seed;self.arm=mode;self.settings.update(noise_seed=seed,boundary_arm=mode);self.rng_trace=[];self.rollout_log=[];self.gradient_log=[]
 def task_loss(self,b,split):
  step=self.global_step;devices=[self.device.index]
  with torch.random.fork_rng(devices=devices):
   torch.manual_seed(self.noise_seed*100000+step);before=torch.cuda.get_rng_state().clone();v=super().task_loss(b,split);after=torch.cuda.get_rng_state().clone()
  outer=torch.cuda.get_rng_state().clone()
  if self.arm=='rollout' and step%4==0:
   small={k:x[:2] for k,x in b.items()};buffers={k:x.clone() for k,x in self.model.named_buffers()}
   with torch.random.fork_rng(devices=devices):
    torch.manual_seed(self.noise_seed*100000+50000+step);p=rollout(self.model,small,4);reg,terms=dynamics_loss(p,small['state']);v=v+.01*reg
    row={'step':step,**{k:float(x.detach()) for k,x in terms.items()}}
    if step==0:
     params=[p for p in self.model.parameters() if p.requires_grad];norms={}
     for name,value in terms.items():
      grads=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True);norms[name]=sum(float(g.float().square().sum()) for g in grads if g is not None)**.5;del grads
     row['unweighted_component_gradient_norms']=norms
    self.rollout_log.append(row)
   assert torch.equal(outer,torch.cuda.get_rng_state()),'rollout RNG escaped isolation'
   assert all(torch.equal(buffers[k],x) for k,x in self.model.named_buffers()),'rollout mutated buffers'
  if step<3:self.rng_trace.append({'step':step,'fm_before':hashlib.sha256(before.cpu().numpy().tobytes()).hexdigest(),'fm_after':hashlib.sha256(after.cpu().numpy().tobytes()).hexdigest(),'extra_rng_isolated':True})
  return v
 def configure_gradient_clipping(self,optimizer,gradient_clip_val=None,gradient_clip_algorithm=None):
  norm=torch.nn.utils.clip_grad_norm_(self.parameters(),gradient_clip_val,error_if_nonfinite=True)
  self.gradient_log.append({'step':self.global_step,'preclip_norm':float(norm),'clipped':float(norm)>gradient_clip_val})
def evaluate(m,names):
 r=base_evaluate(m,names);rows=r['records'];ds=DiskSamples(names);derived=[];nearzero=[];diversity=[]
 for w in range(len(names)):
  group=[x for x in rows if x['window']==w];xy=[]
  for row in group:
   p=np.asarray(row['points']);speeds=np.linalg.norm(np.diff(np.vstack(([0.,0.],p[:,:2])),axis=0),axis=1)/.2;derived.append(float(np.abs(speeds-p[:,4]).mean()));nearzero.append(float((p[:,4]<.5).mean()));xy.append(p[:,:2])
  xy=np.asarray(xy);diversity.append(float(np.sqrt(((xy-xy.mean(0))**2).sum(-1).mean())))
 r['summary'].update(position_derived_vs_predicted_speed_mae_mps=float(np.mean(derived)),near_zero_speed_fraction=float(np.mean(nearzero)),candidate_spread_rms_m=float(np.mean(diversity)))
 b=batch_to([ds[i] for i in range(2)],'cuda')
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  torch.manual_seed(8811);p4=rollout(m,b,4);torch.manual_seed(8811);p16=rollout(m,b,16);difference=((p4[...,:2]-p16[...,:2])*p4.new_tensor([50,10])).norm(dim=-1).mean();_,z=m.forecast_batch(b);future,valid,_=m.target_data(b);target=m.teacher(future).float();persistent=m.teacher(m.canonical_history(b)[:,-1:]).float().expand_as(target);mask=valid.float();den=mask.sum().clamp_min(1)
  r['separate_2window_diagnostics']={'same_noise_4vs16_difference_m':float(difference),'world_latent_mse':float(((z.float()-target).square().mean(-1)*mask).sum()/den),'world_persistence_mse':float(((persistent-target).square().mean(-1)*mask).sum()/den)}
 return r

def main():
 idle();configure_runtime();src=ROOT/'artifacts/masked-teacher-155m-continue2048';prev=ROOT/'artifacts/teacher-boundary-short-training';out=ROOT/'artifacts/teacher-rollout-dynamics-paired';out.mkdir(exist_ok=False);selection=json.loads((prev/'selection.json').read_text());ss=json.dumps(selection,sort_keys=True);(out/'selection.json').write_text(ss);ssha=hashlib.sha256(ss.encode()).hexdigest();ck=torch.load(src/'best.ckpt',map_location='cpu',weights_only=False);basecfg=json.loads((prev/'config.json').read_text());basecfg['experiment']='rollout-dynamics-paired-2seed';basecfg['rng_contract']='FM noise/time:seed*100000+step; extra boundary:seed*100000+50000+step under fork_rng; paired data generator seed';basecfg['budget']='2seeds x2arms x64updates, each<=5min, overall<=15min';basecfg['max_steps']=64;basecfg['rollout_regularizer']={'coefficient':.01,'every_n_steps':4,'subbatch':2,'euler_steps':4,'accel_soft_threshold':4,'jerk_soft_threshold':10,'actor_input':'history/state/masks only','jerk_pre_t0_boundary':'masked unavailable','t0_velocity':'scalar-forward assumption','primary_evaluation_steps':16};basecfg['development_gates']={'require_both_seeds_full_and_first_residual_no_worse':True,'require_both_seeds_peak_accel_jerk_no_worse':True,'near_zero_fraction_increase_max':.05,'spread_ratio_min':.8,'note':'provisional development comparisons, not safety gates; forward/path proxies need context; no automatic promotion'};(out/'config.json').write_text(json.dumps(basecfg,indent=2));results=[];start_all=time.perf_counter()
 for seed in [1810,1811]:
  traces={}
  for mode in ['baseline','rollout']:
   idle();assert time.perf_counter()-start_all<900,'Total15min budget exceeded';pl.seed_everything(seed,workers=True);folder=out/f'{seed}-{mode}';folder.mkdir();cfg=json.loads(json.dumps(basecfg));cfg['short_probe'].update(seed=seed,mode=mode);task=IsolatedTask(cfg,ck['task_state']['manifest_sha'],ssha,seed,mode);task.load_state_dict(ck['state_dict'],strict=True);task.train();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
   tr=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_steps=64,max_epochs=-1,max_time={'minutes':5},callbacks=[Progress(f'{seed}-{mode}'),ModelCheckpoint(dirpath=str(folder),filename='latest',auto_insert_metric_name=False,enable_version_counter=False,save_top_k=1,every_n_train_steps=32,save_on_train_epoch_end=False)],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='rollout-dynamics-paired',version=f'{seed}-{mode}'),num_sanity_val_steps=0,limit_val_batches=0,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=8)
   tr.fit(task,datamodule=SeedData(selection,seed));wall=time.perf_counter()-start;peak=torch.cuda.max_memory_reserved()/2**30;after=evaluate(task.model,selection['validation']);trace=task.rng_trace;traces[mode]=trace;result={'seed':seed,'mode':mode,'steps':tr.global_step,'wall_seconds':wall,'peak_reserved_gib':peak,'rng_trace':trace,'gradient_log':task.gradient_log,'rollout_log':task.rollout_log,'after':after};(folder/'report.json').write_text(json.dumps(result,indent=2));brief={k:v for k,v in result.items() if k not in ['after','rng_trace','gradient_log','rollout_log']};brief.update(summary=after['summary'],peak_first_abs_accel=after['peak_first_abs_accel_mps2']);results.append(brief);print('RESULT '+json.dumps(brief),flush=True);del task,tr;gc.collect();torch.cuda.empty_cache()
  assert traces['baseline']==traces['rollout'],'Paired FM RNG traces differ'
 (out/'summary.json').write_text(json.dumps({'results':results,'paired_rng_trace_first3steps_equal':True,'limits':'Two seeds,128 train/16 repeatedly-used DEVELOPMENT windows,64updates; equal update/data not equal FLOPs; extra4step rollout on2samples every4updates; no collision/road/TTC/closed-loop acceptance'},indent=2));print('ALL_DONE',flush=True)
if __name__=='__main__':main()
