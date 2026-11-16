"""Fresh paired experiment; hard stop 1024 per arm, no automatic continuation."""
from train_jepa_nojepa_paired import *
import math,shutil
class FreshTask(AblationTask):
 def configure_optimizers(self):
  o=torch.optim.AdamW([p for p in self.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
  def rate(s):return (s+1)/128 if s<128 else .1+.9*.5*(1+math.cos(math.pi*min(1,(s-128)/(4096-128))))
  return {'optimizer':o,'lr_scheduler':{'scheduler':torch.optim.lr_scheduler.LambdaLR(o,rate),'interval':'step'}}
class Status(pl.Callback):
 def __init__(self,folder):self.folder=folder;self.rows=[];self.hashes=[]
 def on_train_start(self,t,m):self.start=time.perf_counter()
 def on_train_batch_start(self,t,m,b,i):
  if t.global_step<3:self.hashes.append(hashlib.sha256(b['state'].cpu().numpy().tobytes()).hexdigest())
 def on_train_batch_end(self,t,m,o,b,i):
  loss=float(o['loss']);assert math.isfinite(loss)
  if t.global_step==1 or t.global_step%32==0:
   torch.cuda.synchronize();r={'seed':m.seed,'mode':m.mode,'step':t.global_step,'loss':loss,'lr_next':t.optimizers[0].param_groups[0]['lr'],'wall_s':time.perf_counter()-self.start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30};self.rows.append(r);(self.folder/'progress.json').write_text(json.dumps(self.rows,indent=2));print('PROGRESS '+json.dumps(r),flush=True)
def main():
 idle();configure_runtime();assert shutil.disk_usage(ROOT).free>45*2**30
 out=ROOT/'artifacts/fresh-jepa-paired-first1024';out.mkdir(exist_ok=True);src=ROOT/'artifacts/masked-teacher-155m-continue2048';sel=json.loads((src/'selection.json').read_text());dev=json.loads((ROOT/'artifacts/jepa-nojepa-continuation-paired/selection.json').read_text())['validation'];sel['validation']=dev;ss=json.dumps(sel,sort_keys=True);(out/'selection.json').write_text(ss);sha=hashlib.sha256(ss.encode()).hexdigest();cfg=json.loads((src/'config.json').read_text());cfg.update(max_steps=4096,learning_rate=1e-4,experiment='fresh-paired-hard-stop1024',warmup_steps=128);cfg['budget']={'seeds':[1960,1961],'hard_stop_per_arm':1024,'schedule_horizon_only':4096,'automatic_extension':False,'train_windows':len(sel['train']),'dev_windows':len(dev),'source_checkpoint_loaded':False};(out/'config.json').write_text(json.dumps(cfg,indent=2));results=[];allstart=time.perf_counter()
 for seed in [1960,1961]:
  pl.seed_everything(seed,workers=True);initial=MaskedHistoryJoint(768,15,12).cpu();state={k:v.detach().clone() for k,v in initial.state_dict().items()};assert all(torch.equal(a,b) for a,b in zip(initial.encoder.parameters(),initial.teacher.parameters()));del initial;traces={};hashes={}
  for mode in (['jepa','no_jepa'] if seed==1960 else ['no_jepa','jepa']):
   idle();assert time.perf_counter()-allstart<2400;assert shutil.disk_usage(ROOT).free>15*2**30;pl.seed_everything(seed,workers=True);folder=out/f'{seed}-{mode}';folder.mkdir(exist_ok=True);task=FreshTask(cfg,'fresh-no-checkpoint',sha,seed,mode);task.model.load_state_dict(state,strict=True);assert all(torch.equal(v,state[k]) for k,v in task.model.state_dict().items());status=Status(folder);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
   tr=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_steps=1024,max_epochs=-1,max_time={'minutes':10},callbacks=[status,ModelCheckpoint(dirpath=str(folder),filename='step{step}',auto_insert_metric_name=False,save_top_k=-1,every_n_train_steps=256,save_on_train_epoch_end=False)],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='fresh-jepa-paired-first1024',version=f'{seed}-{mode}'),num_sanity_val_steps=0,limit_val_batches=0,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=16);
   recovered=(folder/'step1024.ckpt').exists()
   if recovered:
    saved=torch.load(folder/'step1024.ckpt',map_location='cpu',weights_only=False);assert saved['global_step']==1024;task.load_state_dict(saved['state_dict'],strict=True);del saved;old=json.loads((folder/'progress.json').read_text());wall=old[-1]['wall_s'];peak=old[-1]['peak_allocated_gib'];reserved=old[-1]['peak_reserved_gib'];print('RECOVER_EVALUATION_ONLY '+str(folder),flush=True)
   else:
    tr.fit(task,datamodule=SeedData(sel,seed));wall=time.perf_counter()-start;assert tr.global_step==1024,'budget stopped arm early; do not continue';peak=torch.cuda.max_memory_allocated()/2**30;reserved=torch.cuda.max_memory_reserved()/2**30
   (folder/'training_metadata.json').write_text(json.dumps({'recovered_evaluation_only':recovered,'gradient_log':task.grads,'rng_trace':task.trace,'first3_data_hashes':status.hashes,'note':'recovered arm in-memory traces unavailable; all initial tensors were asserted exact in original run'},indent=2))
   # checkpoint is saved by callback inside fit, before loader teardown.
   ck=torch.load(folder/'step1024.ckpt',map_location='cpu',weights_only=False);assert ck['global_step']==1024 and ck['optimizer_states'] and ck['lr_schedulers'];del ck
   reference=copy.deepcopy(task.model.teacher).cuda().eval().requires_grad_(False);ev=evaluate(task.model,dev,reference,mode=='jepa');ev['world_summary']={k:v for k,v in ev['world_summary'].items() if k.startswith('own_')} if mode=='jepa' else None;ev['world_rows']=None if mode=='no_jepa' else [{k:v for k,v in r.items() if not k.startswith('fixed_')} for r in ev['world_rows']];traces[mode]=task.trace;hashes[mode]=status.hashes;r={'seed':seed,'mode':mode,'steps':1024,'train_wall_s':wall,'peak_allocated_gib':peak,'peak_reserved_gib':reserved,'active_parameters':sum(p.numel() for p in task.model.parameters() if p.requires_grad),'evaluation':ev,'gradient_log':task.grads,'rng_trace':task.trace,'first3_data_hashes':status.hashes,'limits':'reused development set; source OCC causality unverified; no free truth; collision/TTC/closed-loop unknown; world within own EMA space only; no test'};(folder/'report.json').write_text(json.dumps(r,indent=2));brief={k:v for k,v in r.items() if k not in ['evaluation','gradient_log','rng_trace','first3_data_hashes']};brief.update(metrics=ev['summary'],world=ev['world_summary']);results.append(brief);(out/'summary.partial.json').write_text(json.dumps(results,indent=2));print('RESULT '+json.dumps(brief),flush=True);del task,tr,reference;gc.collect();torch.cuda.empty_cache()
  assert (not traces['jepa'] or not traces['no_jepa']) or (traces['jepa']==traces['no_jepa'] and hashes['jepa']==hashes['no_jepa']),'pair mismatch';del state;gc.collect()
 (out/'summary.json').write_text(json.dumps({'results':results,'total_wall_s':time.perf_counter()-allstart,'fresh_shared_initialization':True,'paired_first3_data_FM_rng':'seed1961 asserted; seed1960 first-arm trace lost after completed training during evaluation error; deterministic shared seeds and initial tensors asserted','stopped_for_review':True},indent=2));print('ALL_DONE_REVIEW_REQUIRED',flush=True)
if __name__=='__main__':main()
