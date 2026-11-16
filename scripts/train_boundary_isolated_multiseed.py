"""Paired multi-seed short fine-tunes with isolated FM and boundary RNG streams."""
import json,time,gc,hashlib
import numpy as np,torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchdata.stateful_dataloader import StatefulDataLoader
from train_boundary_short_comparison import evaluate,extra_boundary,Progress
from train_masked_teacher_bounded import Task,Data,DiskSamples
from lightning_task import ROOT,configure_runtime,VerifiedCheckpointIO
from probe_teacher_history_matrix import idle
class SeedData(Data):
 def __init__(self,selection,seed):super().__init__(selection);self.seed=seed
 def loader(self,split):return StatefulDataLoader(DiskSamples(self.selection[split]),batch_size=8,shuffle=split=='train',num_workers=0,generator=torch.Generator().manual_seed(self.seed if split=='train' else 1907))
class IsolatedTask(Task):
 def __init__(self,cfg,msha,ssha,seed,mode):
  super().__init__(cfg,msha,ssha);self.noise_seed=seed;self.arm=mode;self.settings.update(noise_seed=seed,boundary_arm=mode);self.rng_trace=[]
 def task_loss(self,b,split):
  step=self.global_step;devices=[self.device.index] if self.device.type=='cuda' else []
  with torch.random.fork_rng(devices=devices):
   torch.manual_seed(self.noise_seed*100000+step);before=torch.cuda.get_rng_state().clone();v=super().task_loss(b,split);after=torch.cuda.get_rng_state().clone()
  outer=torch.cuda.get_rng_state().clone()
  if self.arm=='first_penalty':
   with torch.random.fork_rng(devices=devices):
    torch.manual_seed(self.noise_seed*100000+50000+step);boundary=extra_boundary(self.model,b);v=v+.1*boundary;self.log(split+'/boundary',boundary,on_epoch=True,batch_size=len(b['state']))
   assert torch.equal(outer,torch.cuda.get_rng_state()),'boundary RNG escaped isolation'
  if step<3:self.rng_trace.append({'step':step,'fm_before':hashlib.sha256(before.cpu().numpy().tobytes()).hexdigest(),'fm_after':hashlib.sha256(after.cpu().numpy().tobytes()).hexdigest(),'boundary_rng_isolated':True})
  return v

def main():
 idle();configure_runtime();src=ROOT/'artifacts/masked-teacher-155m-continue2048';prev=ROOT/'artifacts/teacher-boundary-short-training';out=ROOT/'artifacts/teacher-boundary-isolated-multiseed';out.mkdir(exist_ok=False);selection=json.loads((prev/'selection.json').read_text());ss=json.dumps(selection,sort_keys=True);(out/'selection.json').write_text(ss);ssha=hashlib.sha256(ss.encode()).hexdigest();ck=torch.load(src/'best.ckpt',map_location='cpu',weights_only=False);basecfg=json.loads((prev/'config.json').read_text());basecfg['experiment']='boundary-paired-rng-3seed';basecfg['rng_contract']='FM noise/time:seed*100000+step; extra boundary:seed*100000+50000+step under fork_rng; paired data generator seed';basecfg['budget']='3seeds x2arms x96updates, each<=5min, overall<=15min';(out/'config.json').write_text(json.dumps(basecfg,indent=2));results=[];start_all=time.perf_counter()
 for seed in [1802,1803,1804]:
  traces={}
  for mode in ['baseline','first_penalty']:
   idle();assert time.perf_counter()-start_all<900,'Total15min budget exceeded';pl.seed_everything(seed,workers=True);folder=out/f'{seed}-{mode}';folder.mkdir();cfg=json.loads(json.dumps(basecfg));cfg['short_probe'].update(seed=seed,mode=mode);task=IsolatedTask(cfg,ck['task_state']['manifest_sha'],ssha,seed,mode);task.load_state_dict(ck['state_dict'],strict=True);task.train();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
   tr=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_steps=96,max_epochs=-1,max_time={'minutes':5},callbacks=[Progress(f'{seed}-{mode}'),ModelCheckpoint(dirpath=str(folder),filename='latest',auto_insert_metric_name=False,enable_version_counter=False,save_top_k=1,every_n_train_steps=48,save_on_train_epoch_end=False)],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='boundary-isolated',version=f'{seed}-{mode}'),num_sanity_val_steps=0,limit_val_batches=0,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=8)
   tr.fit(task,datamodule=SeedData(selection,seed));wall=time.perf_counter()-start;peak=torch.cuda.max_memory_reserved()/2**30;after=evaluate(task.model,selection['validation']);trace=task.rng_trace;traces[mode]=trace;result={'seed':seed,'mode':mode,'steps':tr.global_step,'wall_seconds':wall,'peak_reserved_gib':peak,'rng_trace':trace,'after':after};(folder/'report.json').write_text(json.dumps(result,indent=2));brief={k:v for k,v in result.items() if k not in ['after','rng_trace']};brief.update(summary=after['summary'],peak_first_abs_accel=after['peak_first_abs_accel_mps2']);results.append(brief);print('RESULT '+json.dumps(brief),flush=True);del task,tr;gc.collect();torch.cuda.empty_cache()
  assert traces['baseline']==traces['first_penalty'],'Paired FM RNG traces differ'
 (out/'summary.json').write_text(json.dumps({'results':results,'paired_rng_trace_first3steps_equal':True,'limits':'Three seeds,128 train/16 reused validation,96updates; equal update/data budget not equal FLOPs, extra boundary has additional forward; no collision/road/TTC/closed-loop acceptance'},indent=2));print('ALL_DONE',flush=True)
if __name__=='__main__':main()
