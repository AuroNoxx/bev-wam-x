from train_carla_reactive_history256 import *
from observed_motion_flow import ObservedMotionFlow
from functools import lru_cache
@lru_cache(None)
def features():return json.loads((ROOT/'artifacts/carla-observed-motion-paired512/features.json').read_text())
class ObservedData(ReactiveData):
 acceleration=True;observed=False
 def __getitem__(self,i):
  b=super().__getitem__(i);b['front_observation']=torch.tensor(features()[self.names[i]] if self.observed else [0.,0.,0.,0.,0.]);return b
class ObservedDM(ReactiveDM):
 def train_dataloader(self):return StatefulDataLoader(ObservedData(self.fit),batch_size=12,shuffle=False,num_workers=0)
 def val_dataloader(self):return StatefulDataLoader(ObservedData(self.val),batch_size=12,shuffle=False,num_workers=0)
class ObservedTask(Task):
 def __init__(self,cfg,seed,scales,sha):
  super().__init__(cfg,seed,scales,sha);self.model=ObservedMotionFlow();self.model.set_scales(scales);self.settings.update(observed_front_motion='action memory residual only, no teacher GT, no world loss change')
def main():
 safe_idle();configure_runtime();out=ROOT/'artifacts/carla-observed-motion-paired512';contract=json.loads((out/'feature_contract.json').read_text());assert contract['empty_unknown_test_pass'];assert not (out/'design.json').exists();d=json.loads((ROOT/'data/carla_reactive_hold_cache_v1/manifest.json').read_text());cfg=json.loads((ROOT/'artifacts/structural-prototypes-light512/config.json').read_text());cfg['batch_size']=12;scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m'];gates={'both_seeds_deceleration5s_MAE_relative_improvement_min':.2,'other_group_speed_MAE_worsening_max':.1,'other_groups':['command_hold','static_permitted','accelerating','steady'],'horizons_seconds':[.6,5],'candidate_peak_accel_mps2_max':3,'candidate_peak_jerk_mps3_max':5,'fraction_pass_both_min':.95,'safety_certification':False,'any_failure_no_closed_loop':True};design={'seeds':[2070,2071],'steps_per_arm':512,'batch':12,'initialization':'matching seed history256 weights, new AdamW reset and64warmup/512cosine, NOT strictresume','only_change':'same zero initialized additive action memory branch gets real OCC-only5features vszero; baselinebranchcanlearnconstantbias','world_loss':'unchanged','train_unique':514,'dev_unique':514,'gates':gates};(out/'design.json').write_text(json.dumps(design,indent=2));startall=time.perf_counter();allinfo={}
 for seed in [2070,2071]:
  source=ROOT/f'artifacts/carla-reactive-history-paired256/{seed}-history/latest.ckpt';sha=hashlib.sha256(source.read_bytes()).hexdigest();ck=torch.load(source,map_location='cpu',weights_only=False);oldweights={k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')};del ck;pl.seed_everything(seed);initial=ObservedMotionFlow();missing=initial.load_state_dict(oldweights,strict=False);assert set(missing.missing_keys)=={'front_condition.0.weight','front_condition.0.bias','front_condition.2.weight','front_condition.2.bias'} and not missing.unexpected_keys;weights={k:v.clone() for k,v in initial.state_dict().items()};del initial
  # Actual moving input. Both branch modes equal predecessor at step0; future labels excluded.
  m=ObservedMotionFlow().cuda().eval();m.load_state_dict(weights);old=NavigationScaledFlow(50).cuda().eval();old.load_state_dict(oldweights);ObservedData.observed=True;b=batch_to([ObservedData(d['train'])[130]],'cuda');a={**b,'front_observation':torch.zeros_like(b['front_observation'])};x=torch.randn(1,25,2,device='cuda');tt=torch.tensor([[.3]],device='cuda');tests={}
  with torch.no_grad():
   for precision in ['fp32','bf16']:
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
     y=m(b,x,tt);z=m(a,x,tt);p=m({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)},x,tt);original=old(b,x,tt)
    assert torch.equal(y,z) and torch.equal(y,p) and torch.equal(y,original);tests[precision+'_paired_predecessor_future_exact']=True
  (out/f'{seed}-preflight.json').write_text(json.dumps(tests));del m,old,b,a,x;gc.collect();torch.cuda.empty_cache();rng=np.random.default_rng(seed+100);seq=[]
  while len(seq)<6144:seq.extend(rng.permutation(d['train']).tolist())
  seq=seq[:6144];traces={}
  for arm,on in ([('base',False),('observed',True)] if seed==2070 else [('observed',True),('base',False)]):
   safe_idle();assert time.perf_counter()-startall<1200;ObservedData.observed=on;folder=out/f'{seed}-{arm}';folder.mkdir();(folder/'selection.json').write_text(json.dumps(seq));pl.seed_everything(seed);task=ObservedTask(cfg,seed,scales,hashlib.sha256(json.dumps(seq).encode()).hexdigest());task.model.load_state_dict(weights);torch.cuda.reset_peak_memory_stats();start=time.perf_counter();tr=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_steps=512,max_epochs=-1,max_time={'minutes':4},callbacks=[Status(folder),ModelCheckpoint(dirpath=str(folder),filename='latest',enable_version_counter=False,save_top_k=1,every_n_train_steps=128,save_on_train_epoch_end=False),EpochLatest(folder/'latest.ckpt'),ModelCheckpoint(dirpath=str(folder),filename='best',enable_version_counter=False,monitor='validation_position_diagnostic_m',mode='min',save_top_k=1,save_on_train_epoch_end=False)],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='carla-observed-motion-paired512',version=f'{seed}-{arm}'),num_sanity_val_steps=0,val_check_interval=256,check_val_every_n_epoch=None,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=16);tr.fit(task,datamodule=ObservedDM(seq,d['dev'][::32]));assert tr.global_step==512;info={'steps':512,'wall_s':time.perf_counter()-start,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'rng':task.trace,'parameters_trainable':sum(p.numel() for p in task.model.parameters() if p.requires_grad)};traces[arm]=task.trace;(folder/'train_complete.json').write_text(json.dumps(info));print('TRAIN_DONE',seed,arm,info['wall_s'],flush=True);task.model.cuda().eval();ref=copy.deepcopy(task.model.teacher).cuda().eval();evaluate.__wrapped__.__globals__['DiskSamples']=ObservedData;ev=evaluate(task.model,d['dev'],ref,True);(folder/'eval.json').write_text(json.dumps(ev));print('ARM_DONE',seed,arm,flush=True);del task,tr,ref;gc.collect();torch.cuda.empty_cache()
  assert traces['base']==traces['observed'];assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
 (out/'complete.json').write_text(json.dumps({'done':True,'rng_first3_equal':True,'old_checkpoints_unchanged':True,'wall_s':time.perf_counter()-startall}));print('COMPLETE',flush=True)
if __name__=='__main__':main()
