from train_carla_lateral_scale512 import *
from navigation_scaled_flow import NavigationScaledFlow
from train_masked_teacher_bounded import Status as BoundedStatus
class ReactiveData(torch.utils.data.Dataset):
 acceleration=False
 def __init__(self,names):self.names=names
 def __len__(self):return len(self.names)
 def __getitem__(self,i):
  with np.load(self.names[i]) as f:r={k:torch.from_numpy(f[k].copy()) for k in ['history','state','trajectory','future','history_valid','history_age_seconds']}
  r['geometry_valid']=r['history']!=255
  if not self.acceleration:r['state'][3]=0
  return r
class ReactiveDM(pl.LightningDataModule):
 def __init__(self,fit,val):super().__init__();self.fit=fit;self.val=val
 def train_dataloader(self):return StatefulDataLoader(ReactiveData(self.fit),batch_size=12,shuffle=False,num_workers=0)
 def val_dataloader(self):return StatefulDataLoader(ReactiveData(self.val),batch_size=12,shuffle=False,num_workers=0)
class Task(CompareTask):
 def __init__(self,cfg,seed,scales,sha):
  super().__init__(cfg,seed,'fm',scales,sha);self.model=NavigationScaledFlow(50);self.model.set_scales(scales);self.settings.update(reactive_commands='cruise/permission last issued .1s old',history_acceleration='slot3, difference .6s /5',state_valid_slot=6)
def main():
 safe_idle();configure_runtime();out=ROOT/'artifacts/carla-reactive-history-paired256';out.mkdir(exist_ok=False);d=json.loads((ROOT/'data/carla_reactive_hold_cache_v1/manifest.json').read_text());assert all(d['tests'].values());cfg=json.loads((ROOT/'artifacts/structural-prototypes-light512/config.json').read_text());cfg['batch_size']=12;scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m'];design={'seeds':[2070,2071],'steps':256,'batch':12,'data':d['counts'],'only_change':'state3 causal .6s acceleration versus zero, no gate','shared_commands':'cruise and permission last issued .1s old','optimizer':'AdamW1e-4 wd.01 clip1,64warmup original512cosine stopped256','limits':'one train road388/seed3101 and one dev road713/seed4101, cruise8 vs8.5 confounded; 99.25%unknown, teacher oracle lead excluded; FM offline diagnostic not driving','cells':[]};(out/'design.json').write_text(json.dumps(design,indent=2))
 # Equal initialization and future isolation on actual moving input, both precisions.
 pl.seed_everything(2070);m=NavigationScaledFlow(50).cuda().eval();m.set_scales(scales)
 with torch.no_grad():m.state[0].weight[:,5]/=50;m.state[0].weight[:,3]=0
 ReactiveData.acceleration=True;b=batch_to([ReactiveData(d['train'])[130]],'cuda');a={**b,'state':b['state'].clone()};a['state'][:,3]=0;x=torch.randn(1,25,2,device='cuda');tt=torch.tensor([[.3]],device='cuda');pre={}
 with torch.no_grad():
  for precision in ['fp32','bf16']:
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
    y=m(b,x,tt);z=m(a,x,tt);p=m({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)},x,tt)
   assert torch.equal(y,z) and torch.equal(y,p);pre[precision+'_init_and_future_exact']=True
 (out/'preflight.json').write_text(json.dumps(pre,indent=2));del m,b,a,x;torch.cuda.empty_cache();startall=time.perf_counter()
 for seed in [2070,2071]:
  rng=np.random.default_rng(seed);seq=[]
  while len(seq)<3072:seq.extend(rng.permutation(d['train']).tolist())
  seq=seq[:3072];assert len(set(seq))==514;pl.seed_everything(seed);initial=NavigationScaledFlow(50);initial.set_scales(scales)
  with torch.no_grad():initial.state[0].weight[:,5]/=50;initial.state[0].weight[:,3]=0
  weights={k:v.clone() for k,v in initial.state_dict().items()};del initial;traces={}
  for arm,on in ([('base',False),('history',True)] if seed==2070 else [('history',True),('base',False)]):
   safe_idle();assert time.perf_counter()-startall<1200;ReactiveData.acceleration=on;folder=out/f'{seed}-{arm}';folder.mkdir();pl.seed_everything(seed);task=Task(cfg,seed,scales,hashlib.sha256(json.dumps(seq).encode()).hexdigest());task.model.load_state_dict(weights);(folder/'selection.json').write_text(json.dumps(seq));torch.cuda.reset_peak_memory_stats();start=time.perf_counter();tr=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_steps=256,max_epochs=-1,max_time={'minutes':4},callbacks=[BoundedStatus(folder),ModelCheckpoint(dirpath=str(folder),filename='latest',enable_version_counter=False,save_top_k=1,every_n_train_steps=64,save_on_train_epoch_end=False),EpochLatest(folder/'latest.ckpt'),ModelCheckpoint(dirpath=str(folder),filename='best',enable_version_counter=False,monitor='validation_position_diagnostic_m',mode='min',save_top_k=1,save_on_train_epoch_end=False)],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='carla-reactive-history-paired256',version=f'{seed}-{arm}'),num_sanity_val_steps=0,val_check_interval=128,check_val_every_n_epoch=None,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=16);tr.fit(task,datamodule=ReactiveDM(seq,d['dev'][::32]));assert tr.global_step==256;traces[arm]=task.trace;info={'steps':256,'wall_s':time.perf_counter()-start,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'rng':task.trace,'parameters_trainable':sum(p.numel() for p in task.model.parameters() if p.requires_grad)};(folder/'train_complete.json').write_text(json.dumps(info));print('TRAIN_DONE',seed,arm,info['wall_s'],flush=True);task.model.cuda().eval();ref=copy.deepcopy(task.model.teacher).cuda().eval();evaluate.__wrapped__.__globals__['DiskSamples']=ReactiveData;ev=evaluate(task.model,d['dev'],ref,True);(folder/'eval.json').write_text(json.dumps(ev));print('ARM_DONE',seed,arm,flush=True);del task,tr,ref;gc.collect();torch.cuda.empty_cache()
  assert traces['base']==traces['history']
 (out/'complete.json').write_text(json.dumps({'done':True,'rng_first3_equal':True,'wall_s':time.perf_counter()-startall}));print('COMPLETE',flush=True)
if __name__=='__main__':main()
