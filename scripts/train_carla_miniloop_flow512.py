import os,psutil
from train_mixed_regression_flow512 import *
from carla_miniloop_adapter import actor_inputs
class CommonData(torch.utils.data.Dataset):
 def __init__(self,names):self.names=names
 def __len__(self):return len(self.names)
 def __getitem__(self,i):
  n=self.names[i]
  if Path(n).is_absolute():
   with np.load(n) as f:r={k:torch.from_numpy(f[k].copy()) for k in ['history','state','trajectory','future','history_valid','history_age_seconds']}
   r['geometry_valid']=r['history']!=255
  else:r=DiskSamples([n])[0]
  r['state'][1:]=0;return r
class DM(pl.LightningDataModule):
 def __init__(self,fit,val):super().__init__();self.fit=fit;self.val=val
 def train_dataloader(self):return StatefulDataLoader(CommonData(self.fit),batch_size=12,shuffle=False,num_workers=0)
 def val_dataloader(self):return StatefulDataLoader(CommonData(self.val),batch_size=12,shuffle=False,num_workers=0)
def safe_idle():
 parents={p.pid for p in psutil.Process().parents()}
 for p in psutil.process_iter(['name','cmdline']):
  if p.pid==os.getpid() or p.pid in parents:continue
  if (p.info['name'] or '').lower().startswith('python') and 'tensorboard' not in ' '.join(p.info['cmdline'] or []):raise RuntimeError('OtherPython:'+str(p.pid))
 # CARLA server allowed only idle by coordinator; no simultaneous simulator tick worker.
from functools import lru_cache
@lru_cache(maxsize=None)
def bucket(n):
 b=CommonData([n])[0];v=float(b['state'][0])*30;delta=float(b['trajectory'][-1,4])*30-v;return (int(np.digitize(v,[.5,3,7,12])),int(np.digitize(delta,[-2,2])))
def main():
 safe_idle();configure_runtime();out=ROOT/'artifacts/carla-miniloop-flow512';out.mkdir(exist_ok=True);d=json.loads((ROOT/'data/carla_miniloop_cache_v1/manifest.json').read_text());old=json.loads((ROOT/'artifacts/route-regression-flow1024/design.json').read_text());oldpool=sorted(set(old['streams']['2050']));by={}
 for n in oldpool:by.setdefault(bucket(n),[]).append(n)
 new=[n for n in d['train'] if bucket(n) in by];excluded=len(d['train'])-len(new);assert len(new)>100;held=d['dev'];oldheld=[x['file'] for x in old['held']];val=held[::56];scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m'];cfg=json.loads((ROOT/'artifacts/structural-prototypes-light512/config.json').read_text());cfg.update(experiment='speedonly-causal-source-package512',batch_size=12);design={'new_train':new,'new_dev':held,'old_dev':oldheld,'excluded_unmatched_new':excluded,'limits':'2train1dev episodes,semanticGT rayhits; sourcepackage comparison not isolatedcausal dataquality. Speed/5sfuture-speedchange buckets matched TRAIN labels only. Same speed-only state; no officialtest.'};(out/'design.json').write_text(json.dumps(design,indent=2));allstart=time.perf_counter();results=[]
 for seed in [2060,2061]:
  rng=np.random.default_rng(seed);newseq=rng.choice(new,6144).tolist();oldseq=[str(rng.choice(by[bucket(n)])) for n in newseq];pl.seed_everything(seed);init=CompareModel('fm');init.set_scales(scales);weights={k:v.clone() for k,v in init.state_dict().items()};del init
  for arm,seq in ([('old',oldseq),('new',newseq)] if seed==2060 else [('new',newseq),('old',oldseq)]):
   safe_idle();assert time.perf_counter()-allstart<1500;pl.seed_everything(seed);folder=out/f'{seed}-{arm}';
   if (folder/'report.json').exists():
    print('SKIP_COMPLETED',seed,arm,flush=True);continue
   folder.mkdir();(folder/'selection.json').write_text(json.dumps(seq));task=CompareTask(cfg,seed,'fm',scales,hashlib.sha256(json.dumps(seq).encode()).hexdigest());task.model.load_state_dict(weights);torch.cuda.reset_peak_memory_stats();start=time.perf_counter();trainer=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_steps=512,max_epochs=-1,max_time={'minutes':5},callbacks=[Status(folder),ModelCheckpoint(dirpath=str(folder),filename='latest',enable_version_counter=False,save_top_k=1,every_n_train_steps=128,save_on_train_epoch_end=False),EpochLatest(folder/'latest.ckpt'),ModelCheckpoint(dirpath=str(folder),filename='best',enable_version_counter=False,monitor='validation_position_diagnostic_m',mode='min',save_top_k=1,save_on_train_epoch_end=False)],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='carla-miniloop-flow512',version=f'{seed}-{arm}'),num_sanity_val_steps=0,val_check_interval=256,check_val_every_n_epoch=None,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False);trainer.fit(task,datamodule=DM(seq,val));assert trainer.global_step==512;wall=time.perf_counter()-start;task.model.cuda().eval();ref=copy.deepcopy(task.model.teacher).cuda().eval();evaluate.__wrapped__.__globals__['DiskSamples']=CommonData
   # CommonData must retain original legacy loader independent of evaluator global patch.
   report={'seed':seed,'arm':arm,'wall_s':wall,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'unique_train':len(set(seq)),'new_eval':evaluate(task.model,held,ref,True),'old_eval':evaluate(task.model,oldheld,ref,True)};(folder/'report.json').write_text(json.dumps(report));results.append({k:v for k,v in report.items() if k not in ['new_eval','old_eval']});print('DONE',seed,arm,wall,flush=True);del trainer,task,ref;gc.collect();torch.cuda.empty_cache()
 (out/'complete.json').write_text(json.dumps({'results':results,'wall_s':time.perf_counter()-allstart}));print('COMPLETE')
if __name__=='__main__':main()

