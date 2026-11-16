"""Bounded deterministic capability diagnostic, not a new FM architecture."""
from train_fm_complete_topology256 import *
O=ROOT/'artifacts/matched6-regression512'
class RegressionTask(Task):
 def __init__(self,cfg,seed,scales,sha):
  super().__init__(cfg,seed,scales,sha);self.mode='regression';self.model.method='regression'
  with torch.no_grad():self.model.state[0].weight[:,5]/=50
  self.settings.update(method='regression',data='matched6',initialization='fresh',commands='current constant',precision='FP32+MATH',budget=512)
class Progress512(pl.Callback):
 def __init__(self,out):self.out=out
 def on_train_start(self,t,m):self.start=time.perf_counter()
 def on_train_batch_end(self,t,m,outputs,b,i):
  if t.global_step in [1,16] or t.global_step%128==0:
   torch.cuda.synchronize();r={'step':t.global_step,'budget':512,'loss':float(outputs['loss']),'elapsed_s':time.perf_counter()-self.start,'peak_gib':torch.cuda.max_memory_allocated()/2**30};(self.out/'status.json').write_text(json.dumps(r));print('STATUS',self.out.name,r,flush=True)
def grouped(p,y,v0,names):
 pv=p[...,4:5]*30*p[...,[3,2]];yv=y[...,4:5]*30*y[...,[3,2]];px=p[...,:2]*[50,10];yx=y[...,:2]*[50,10];ps=np.linalg.norm(pv,axis=-1);ys=y[...,4]*30
 initial=np.zeros((len(p),1,2));initial[:,0,0]=v0;acc=np.diff(np.concatenate([initial,pv],1),axis=1)/.2;jerk=np.diff(acc,axis=1)/.2
 groups={'all':np.ones(len(p),bool),'static_current':v0<.2,'hold_full5s':(v0<.2)&(ys.max(1)<.2),'start':(v0<.2)&(ys[:,-1]>2),'decelerating':(v0>=.2)&(ys[:,-1]<v0-1),'accelerating':(v0>=.2)&(ys[:,-1]>v0+1),'steady':(v0>=.2)&(abs(ys[:,-1]-v0)<=1)}
 for ep in sorted(set(Path(x).parent.name for x in names)):groups[ep]=np.array([Path(x).parent.name==ep for x in names])
 result={}
 for g,s in groups.items():
  n=int(s.sum())
  if not n:result[g]={'n':0,'metrics':None};continue
  result[g]={'n':n,'position_diagnostic_m':float(np.linalg.norm(px[s]-yx[s],axis=-1).mean()),'speed_MAE_02_06_1_5s':np.abs(ps[s]-ys[s]).mean(0)[[0,2,4,24]].tolist(),'endpoint_x_m':float(px[s,-1,0].mean()),'label_endpoint_x_m':float(yx[s,-1,0].mean()),'endpoint_speed_mps':float(ps[s,-1].mean()),'label_endpoint_speed_mps':float(ys[s,-1].mean()),'peak_vector_accel_with_t0_mean':float(np.linalg.norm(acc[s],axis=-1).max(1).mean()),'peak_vector_jerk_mean':float(np.linalg.norm(jerk[s],axis=-1).max(1).mean()),'negative_vx_fraction':float((pv[s,:,0]<-.05).mean()),'endpoint_abs_y_m':float(abs(px[s,-1,1]).mean()),'collision':None,'TTC':None,'closedloop_progress':None}
 return result
@torch.no_grad()
def run_eval(model,names):
 pred=[];truth=[];vel=[]
 for i in range(0,len(names),12):
  b=batch_to([ReactiveData(names)[j] for j in range(i,min(i+12,len(names)))],'cuda');p=model.sample(b,k=1,steps=16)[:,0];assert torch.isfinite(p).all();pred.append(p.cpu().numpy());truth.append(b['trajectory'].cpu().numpy());vel.append((b['state'][:,0]*30).cpu().numpy())
 p=np.concatenate(pred);y=np.concatenate(truth);v0=np.concatenate(vel);constant=np.zeros_like(p);constant[...,3]=1;constant[...,4]=v0[:,None]/30;constant[...,0]=v0[:,None]*(np.arange(1,26)*.2)/50;zero=np.zeros_like(p);zero[...,3]=1
 return {'model':grouped(p,y,v0,names),'constant_speed':grouped(constant,y,v0,names),'stationary':grouped(zero,y,v0,names),'label_dynamic_reference':grouped(y,y,v0,names),'limits':'offline diagnostics; no collision/TTC or closed-loop success; jerk excludes unknown pre-t0 acceleration; decoder direction is travel direction not independent body yaw'},p
def main():
 safe_idle();configure_runtime();O.mkdir(exist_ok=False);d=json.loads((ROOT/'data/carla_matched6_cache_v1/manifest.json').read_text());assert all(d['tests'].values());ReactiveData.acceleration=True
 cfg=json.loads((ROOT/'artifacts/structural-prototypes-light512/config.json').read_text());cfg.update(batch_size=12,max_steps=512);scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m']
 design={'seeds':[7201,7202],'steps':512,'batch':12,'data':d['counts'],'initialization':'fresh deterministic25; same existing condition/velocity representation; not new topology/physical head','loss':'direct q MSE plus unchanged .2 gated position and .1 eachJEPA; no FM training','optimizer':'AdamW1e-4 wd.01 clip1 warmup64 original512cosine','budget':'5minutes perseed; no automaticextension','selection':'uniform shuffled fulltrain cycles;6144draws,2176unique','precision':'FP32+MATH','decision':'offline capability vs constant/stationary; no automaticclosedloop; report bothseeds/episodes/allphases regardless outcome','limits':d['limits']};(O/'design.json').write_text(json.dumps(design,indent=2));allstart=time.perf_counter()
 for seed in design['seeds']:
  safe_idle();folder=O/str(seed);folder.mkdir();rng=np.random.default_rng(seed);seq=[]
  while len(seq)<6144:seq.extend(rng.permutation(d['train']).tolist())
  seq=seq[:6144];assert len(set(seq))==2176;(folder/'selection.json').write_text(json.dumps(seq));pl.seed_everything(seed);task=RegressionTask(cfg,seed,scales,hashlib.sha256(json.dumps(seq).encode()).hexdigest()).cuda().eval();b=batch_to([ReactiveData(d['train'])[40]],'cuda');z=torch.zeros(1,25,2,device='cuda');t=torch.zeros(1,1,device='cuda')
  with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
   p=task.model(b,z,t);q=task.model({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)},z,t);assert torch.equal(p,q)
   target=task.model.to_flow_space(b,b['trajectory']);reconstructed=task.model.from_flow_space(b,target);assert torch.isfinite(reconstructed).all();velerr=float(abs(reconstructed[...,4]-b['trajectory'][...,4]).max()*30);assert velerr<1e-4
  (folder/'preflight.json').write_text(json.dumps({'future_poison_exact':True,'velocity_roundtrip_max_mps':velerr,'parameters':sum(x.numel() for x in task.parameters() if x.requires_grad)}));del b,p,q,z,t;torch.cuda.reset_peak_memory_stats();st=time.perf_counter()
  trainer=pl.Trainer(accelerator='gpu',devices=1,precision='32-true',max_steps=512,max_epochs=-1,max_time={'minutes':5},callbacks=[Progress512(folder),ModelCheckpoint(dirpath=str(folder),filename='latest',enable_version_counter=False,save_top_k=1,every_n_train_steps=128,save_on_train_epoch_end=False),EpochLatest(folder/'latest.ckpt'),ModelCheckpoint(dirpath=str(folder),filename='best',enable_version_counter=False,monitor='validation_position_diagnostic_m',mode='min',save_top_k=1,save_on_train_epoch_end=False)],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='matched6-regression512',version=str(seed)),num_sanity_val_steps=0,val_check_interval=256,check_val_every_n_epoch=None,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=16)
  with sdpa_kernel(SDPBackend.MATH):trainer.fit(task,datamodule=CompleteDM(seq,d['dev'][::32]))
  assert trainer.global_step==512,'hard stop; no automatic restart';info={'steps':512,'wall_s':time.perf_counter()-st,'peak_gib':torch.cuda.max_memory_allocated()/2**30};(folder/'train_complete.json').write_text(json.dumps(info));print('TRAIN_COMPLETE',seed,info,flush=True);task.model.cuda().eval()
  with sdpa_kernel(SDPBackend.MATH):
   ev,pred=run_eval(task.model,d['dev']);fit,_=run_eval(task.model,d['train'][::8])
  (folder/'eval.json').write_text(json.dumps(ev));(folder/'fit_sample_eval.json').write_text(json.dumps(fit));np.save(folder/'dev_predictions.npy',pred);ck=torch.load(folder/'latest.ckpt',map_location='cpu',weights_only=False);assert ck['global_step']==512 and ck['CompleteDM']['loader']['_sampler_iter_state']['samples_yielded']==6144;(folder/'resume_verified.json').write_text(json.dumps({'step':512,'cursor':6144,'task_state':'task_state'in ck,'optimizer':bool(ck['optimizer_states']),'scheduler':bool(ck['lr_schedulers'])}));del ck,task,trainer;gc.collect();torch.cuda.empty_cache();print('ARM_COMPLETE',seed,ev['model']['all'],flush=True)
 (O/'complete.json').write_text(json.dumps({'done':True,'wall_s':time.perf_counter()-allstart}));print('COMPLETE',flush=True)
if __name__=='__main__':main()
