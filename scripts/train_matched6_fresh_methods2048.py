"""Pre-registered fresh matched6 end-to-end method comparison; no deployment."""
from train_matched6_regression512 import *
import math
OUT=ROOT/'artifacts/matched6-fresh-flow-regression2048'
class FreshTask(RegressionTask):
 def __init__(self,cfg,seed,scales,sha,method):
  super().__init__(cfg,seed,scales,sha);self.mode=method;self.model.method=method;self.settings.update(method=method,budget=2048,schedule='128warmup then cosine to1e-5 at2048',scope='fresh end-to-end matched6 development comparison')
 def configure_optimizers(self):
  opt=torch.optim.AdamW([p for p in self.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
  def rate(s):return (s+1)/128 if s<128 else .1+.9*.5*(1+math.cos(math.pi*min(1,(s-128)/1920)))
  return {'optimizer':opt,'lr_scheduler':{'scheduler':torch.optim.lr_scheduler.LambdaLR(opt,rate),'interval':'step'}}
 def training_step(self,b,i):
  loss=super().training_step(b,i);assert torch.isfinite(loss);return loss
@torch.no_grad()
def assess(m,names,strata):
 m.eval();ps=[];ys=[];vs=[];world=[]
 for st in range(0,len(names),12):
  b=batch_to([ReactiveData(names)[j] for j in range(st,min(st+12,len(names)))],'cuda')
  with torch.random.fork_rng(devices=[0]):
   torch.manual_seed(8300+st);p=m.sample(b,k=4,steps=16)
  assert torch.isfinite(p).all();ps.append(p.cpu().numpy());ys.append(b['trajectory'].cpu().numpy());vs.append((b['state'][:,0]*30).cpu().numpy())
  _,z=m.forecast_batch(b);f,valid,_=m.target_data(b);target=m.teacher(f).float();cur=m.teacher(m.canonical_history(b)[:,-1:]).float().expand_as(target)
  world.append([float(((z.float()-target).square().mean(-1)*valid).sum()),float(((cur-target).square().mean(-1)*valid).sum()),int(valid.sum())])
 p=np.concatenate(ps);y=np.concatenate(ys);v=np.concatenate(vs);result={'candidate0':grouped(p[:,0],y,v,names),'candidate_all':grouped(p.reshape(-1,25,5),np.repeat(y,4,0),np.repeat(v,4),[n for n in names for _ in range(4)])}
 const=np.zeros_like(y);const[...,3]=1;const[...,4]=v[:,None]/30;const[...,0]=v[:,None]*np.arange(1,26)*.2/50;zero=np.zeros_like(y);zero[...,3]=1
 result['constant']=grouped(const,y,v,names);result['stationary']=grouped(zero,y,v,names);result['labels']=grouped(y,y,v,names);result['strata']={}
 for name,files in strata.items():
  ids=[i for i,n in enumerate(names) if n in files]
  if not ids:result['strata'][name]={'n':0,'metrics':None};continue
  result['strata'][name]={'n':len(ids),'episodes':sorted(set(Path(names[i]).parent.name for i in ids)),'candidate0':grouped(p[ids,0],y[ids],v[ids],[names[i] for i in ids])['all'],'constant':grouped(const[ids],y[ids],v[ids],[names[i] for i in ids])['all'],'stationary':grouped(zero[ids],y[ids],v[ids],[names[i] for i in ids])['all']}
 xy=p[...,:2]*[50,10];result['candidate_spread_rms_m']=float(np.sqrt(((xy-xy.mean(1,keepdims=True))**2).sum(-1).mean()));a=np.array(world);result['world_same_ema']={'tokens':int(a[:,2].sum()),'prediction_mse':float(a[:,0].sum()/a[:,2].sum()),'persistence_mse':float(a[:,1].sum()/a[:,2].sum()),'free_truth':None};result['limits']='Repeated development episodes, oracle semantics/ego pose/nav; teacher lead truth excluded. Neither label fit nor candidate0 selection proves driving safety. No collision/TTC computed.'
 return result,p
class Progress(pl.Callback):
 def __init__(self,out,names,strata,globalstart):self.out=out;self.names=names;self.strata=strata;self.globalstart=globalstart;self.best=float('inf')
 def on_train_start(self,t,m):self.start=time.perf_counter()
 def on_train_batch_end(self,t,m,o,b,i):
  if time.perf_counter()-self.globalstart>3600:t.should_stop=True
  if t.global_step in [1,16,64] or t.global_step%128==0:
   torch.cuda.synchronize();e=time.perf_counter()-self.start;r={'step':t.global_step,'loss':float(o['loss']),'elapsed_s':e,'mean_s_step_with_eval':e/t.global_step,'remaining_estimate_s':e/t.global_step*(2048-t.global_step),'peak_gib':torch.cuda.max_memory_allocated()/2**30};(self.out/'status.json').write_text(json.dumps(r));print('STATUS',self.out.name,json.dumps(r),flush=True)
  if t.global_step not in [512,1024,2048]:return
  with sdpa_kernel(SDPBackend.MATH):ev,p=assess(m.model,self.names,self.strata)
  (self.out/f'eval{t.global_step}.json').write_text(json.dumps(ev));np.save(self.out/f'predictions{t.global_step}.npy',p);m.train();t.save_checkpoint(self.out/f'milestone{t.global_step}.ckpt')
  # Retained best is label-fit bookkeeping only; final2048 is pre-registered main comparison.
  score=ev['candidate0']['all']['position_diagnostic_m']
  if score<self.best:self.best=score;t.save_checkpoint(self.out/'best.ckpt')
  print('MILESTONE',self.out.name,t.global_step,json.dumps(ev['candidate0']['all']),flush=True)
def main():
 safe_idle();configure_runtime();OUT.mkdir(exist_ok=False);ReactiveData.acceleration=True;d=json.loads((ROOT/'data/carla_matched6_cache_v1/manifest.json').read_text());assert all(d['tests'].values());rows=json.loads((ROOT/'artifacts/matched6-start-frozen-audit/rows.json').read_text());strata={k:[] for k in ['future_start_not_yet_observable','start_early_motion_below_threshold','start_detectable_motion','hold1s','deceleration']}
 for r in rows:
  if r['future_start_label']:
   key='future_start_not_yet_observable' if r['lead_current_speed_eval_only']<.2 else ('start_detectable_motion' if r['noid_motion_above_OLD_train_threshold'] else 'start_early_motion_below_threshold');strata[key].append(r['file'])
  if r['hold_1s_label']:strata['hold1s'].append(r['file'])
  if r['decel_label']:strata['deceleration'].append(r['file'])
 gate={'automatic_closedloop':False,'required_before_further_guarded_pilot':'both seeds, each dev episode; no missing required group; primary final2048 candidate0 only','observability_coverage':'>=1 detectable-start window per held episode; absent is unknown, not pass','start':'detectable-start endpoint-speed MAE <=0.8*stationary baseline; forward endpoint positive; no backward segment excuses','decel':'endpoint-speed MAE <=0.8*constant baseline','hold':'hold1s predicted first1s displacement <=0.30m mean per episode','dynamics':'all-episode mean peak accel <=6m/s2; mean peak jerk <=15m/s3; negative vx fraction <=0.01','method_tradeoff':'FM must not worsen regression hold drift by >0.1m or mean peak jerk by >10%; improvement in fit cannot waive other gates','limits':'pilot admission gates only, not certified safety; no thresholds changed after results'}
 design={'seeds':[7301,7302],'methods':['fm','regression'],'steps':2048,'batch':12,'unique_train':2176,'dev':1088,'draws_per_arm':24576,'architecture':'existing width128 layers2 heads4 25x2 velocity; all trainable main trunk; no new geometry branch/topology','optimizer':'AdamW1e-4 wd.01 clip1 warmup128 cosine2048 floor1e-5','precision':'FP32 MATH','loss':'FM vectorfield or direct qMSE plus unchanged .2 gatedposition and .1eachJEPA; objectives differ, total losses not quality-comparable','sampling':'pure iid Gaussian,16Euler,K4; fixedcandidate0 primary; no best-of-K','budget':'15min perarm including milestone evaluation;60min overall; max2048 no autoextension','evaluation':'512/1024/2048 full repeateddev; final2048 primary; train-only setting; no officialtest','gates':gate,'limits':d['limits'],'teacher_information_gap':'teacher GT lead speed/gap;actor only causal OCC/state/nav; semantic/ego/map oracle remain; lead ID pose future schedule NEVER input','data_confound':'train actual cruise8.33-8.76 vsdev7.887-7.894 despite same source uniform;4train2dev overlapping windows not independentevents'}
 (OUT/'design.json').write_text(json.dumps(design,indent=2));(OUT/'strata_label_only.json').write_text(json.dumps(strata));cfg=json.loads((ROOT/'artifacts/structural-prototypes-light512/config.json').read_text());cfg.update(batch_size=12,max_steps=2048,warmup_steps=128,schedule_horizon=2048);scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m'];(OUT/'config.json').write_text(json.dumps(cfg));globalstart=time.perf_counter()
 for seed in design['seeds']:
  rng=np.random.default_rng(seed);seq=[]
  while len(seq)<24576:seq.extend(rng.permutation(d['train']).tolist())
  seq=seq[:24576];assert len(set(seq))==2176;sha=hashlib.sha256(json.dumps(seq).encode()).hexdigest();pl.seed_everything(seed);initial=FreshTask(cfg,seed,scales,sha,'fm');weights={k:v.cpu().clone() for k,v in initial.state_dict().items()};del initial;traces={}
  for method in (['fm','regression'] if seed==7301 else ['regression','fm']):
   safe_idle();assert time.perf_counter()-globalstart<3600;folder=OUT/f'{seed}-{method}';folder.mkdir();(folder/'selection.json').write_text(json.dumps(seq));(folder/'exposure.json').write_text(json.dumps({k:sum(n in set(files) for n in seq) for k,files in strata.items()}));pl.seed_everything(seed);task=FreshTask(cfg,seed,scales,sha,method);task.load_state_dict(weights);assert all(torch.equal(v,task.state_dict()[k].cpu()) for k,v in weights.items());task.cuda().eval();b=batch_to([ReactiveData(d['train'])[40]],'cuda');x=torch.randn(1,25,2,device='cuda');tt=torch.full((1,1),.3,device='cuda')
   with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
    a=task.model(b,x,tt);poison=task.model({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)},x,tt);assert torch.equal(a,poison);q=task.model.to_flow_space(b,b['trajectory']);rec=task.model.from_flow_space(b,q);assert float(abs(rec[...,4]-b['trajectory'][...,4]).max()*30)<1e-4
   (folder/'preflight.json').write_text(json.dumps({'paired_initialization_exact':True,'future_poison_exact':True,'roundtrip_velocity':True,'trainable_parameters':sum(p.numel() for p in task.parameters() if p.requires_grad),'unique_train':len(set(seq))}));del b,x,tt,a,poison,q,rec;torch.cuda.reset_peak_memory_stats();st=time.perf_counter();cb=Progress(folder,d['dev'],strata,globalstart)
   tr=pl.Trainer(accelerator='gpu',devices=1,precision='32-true',max_steps=2048,max_epochs=-1,max_time={'minutes':15},callbacks=[cb,ModelCheckpoint(dirpath=str(folder),filename='latest',enable_version_counter=False,save_top_k=1,every_n_train_steps=128,save_on_train_epoch_end=False),EpochLatest(folder/'latest.ckpt')],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='matched6-fresh-flow-regression2048',version=f'{seed}-{method}'),limit_val_batches=0,num_sanity_val_steps=0,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=32)
   with sdpa_kernel(SDPBackend.MATH):tr.fit(task,datamodule=CompleteDM(seq,[]))
   assert tr.global_step==2048,'budget ended, stop batch; no blind restart';traces[method]=task.trace;ck=torch.load(folder/'latest.ckpt',map_location='cpu',weights_only=False);assert ck['global_step']==2048 and ck['CompleteDM']['loader']['_sampler_iter_state']['samples_yielded']==24576
   (folder/'complete.json').write_text(json.dumps({'steps':2048,'wall_s':time.perf_counter()-st,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'rng_trace':task.trace,'grads':task.grads,'restorable_optimizer':bool(ck['optimizer_states']),'scheduler':ck['lr_schedulers'][0],'cursor':24576},default=str));print('ARM_DONE',folder.name,flush=True);del task,tr,ck;gc.collect();torch.cuda.empty_cache()
  assert traces['fm']==traces['regression']
 (OUT/'complete.json').write_text(json.dumps({'done':True,'paired_first3_rng_exact':True,'wall_s':time.perf_counter()-globalstart,'no_closedloop':True}));print('COMPLETE',flush=True)
if __name__=='__main__':main()
