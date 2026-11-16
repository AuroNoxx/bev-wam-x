"""Single bounded low-LR tail continuation; preserve optimizer/RNG and append-only stream."""
from train_matched6_regression512 import *
O=ROOT/'artifacts/matched6-regression-resume1024-v3'
class ResumeTask(RegressionTask):
 def on_save_checkpoint(self,c):
  super().on_save_checkpoint(c);c['continuation_contract']=self.continuation
class ResumeGuard(pl.Callback):
 def __init__(self,folder,expected,sourceck):self.folder=folder;self.expected=expected;self.sourceck=sourceck;self.checked=False
 def on_train_start(self,t,m):
  assert t.global_step==512;assert t.optimizers[0].param_groups[0]['lr']==1e-5;assert t.lr_scheduler_configs[0].scheduler.last_epoch==512
  restored=t.optimizers[0].state_dict();saved=self.sourceck['optimizer_states'][0]
  for k,v in saved['state'].items():
   for f,x in v.items():assert torch.equal(restored['state'][k][f].cpu(),x.cpu()) if torch.is_tensor(x) else restored['state'][k][f]==x
  assert all(torch.equal(v.detach().cpu(),self.sourceck['state_dict'][k]) for k,v in m.state_dict().items());self.start=time.perf_counter();self.sourceck=None
 def on_train_batch_start(self,t,m,b,i):
  if not self.checked:
   assert t.global_step==512
   for k,v in self.expected.items():assert torch.equal(b[k].cpu(),v),f'resume batch mismatch {k}'
   self.checked=True;(self.folder/'first_batch_verified.json').write_text(json.dumps({'global_step_before_update':512,'next_epoch_stream_offset':0,'all_input_and_label_tensors_exact':True,'model_optimizer_scheduler_exact':True,'lr':t.optimizers[0].param_groups[0]['lr']}))
 def on_train_batch_end(self,t,m,outputs,b,i):
  if t.global_step in [513,528] or t.global_step%128==0:
   torch.cuda.synchronize();r={'step':t.global_step,'hard_stop':1024,'loss':float(outputs['loss']),'tail_elapsed_s':time.perf_counter()-self.start,'lr':t.optimizers[0].param_groups[0]['lr'],'peak_gib':torch.cuda.max_memory_allocated()/2**30};(self.folder/'status.json').write_text(json.dumps(r));print('STATUS',self.folder.name,r,flush=True)
def curves(pred,names):
 ys=[];v=[]
 for name in names:
  with np.load(name) as z:ys.append(z['trajectory'][...,4]*30);v.append(z['state'][0]*30)
 ys=np.stack(ys);v=np.array(v);ps=pred[...,4]*30;vx=ps*pred[...,3];g={'all':np.ones(len(names),bool),'start':(v<.2)&(ys[:,-1]>2),'decel':(v>=.2)&(ys[:,-1]<v-1),'hold1s':(v<.2)&(ys[:,:5].max(1)<.2),'steady':(v>=.2)&(abs(ys[:,-1]-v)<=1)};out={}
 for ep in sorted(set(Path(n).parent.name for n in names)):
  e=np.array([Path(n).parent.name==ep for n in names]);out[ep]={}
  for k,mask in g.items():
   s=e&mask;out[ep][k]={'n':int(s.sum()),'pred_speed25':ps[s].mean(0).tolist() if s.any() else None,'label_speed25':ys[s].mean(0).tolist() if s.any() else None,'pred_vx_within_std':float(vx[s].std(1).mean()) if s.any() else None,'label_speed_within_std':float(ys[s].std(1).mean()) if s.any() else None}
 return out
def main():
 safe_idle();configure_runtime();O.mkdir(exist_ok=False);d=json.loads((ROOT/'data/carla_matched6_cache_v1/manifest.json').read_text());ReactiveData.acceleration=True;cfg=json.loads((ROOT/'artifacts/structural-prototypes-light512/config.json').read_text());cfg.update(batch_size=12,max_steps=512);scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m'];design={'source_step':512,'target_step':1024,'additional_steps':512,'lr':'original scheduler has completed cosine; unchanged function clamps at1e-5 after512, no newwarmup/cosine/reset','budget':'5min perseed, hardmax1024,no automaticextension','data':'unchanged6144sequence; naturalnext epoch repeats original order','unchanged':'model/loss/precision/data/clip andoptimizerstate; originalconfig/settings retained, continuation metadata added separately','no_closed_loop':True};(O/'design.json').write_text(json.dumps(design,indent=2));startall=time.perf_counter()
 for seed in [7201,7202]:
  safe_idle();folder=O/str(seed);folder.mkdir();source=ROOT/f'artifacts/matched6-regression512/{seed}/latest.ckpt';sha=hashlib.sha256(source.read_bytes()).hexdigest();ck=torch.load(source,map_location='cpu',weights_only=False);old=json.loads((source.parent/'selection.json').read_text());rng=np.random.default_rng(seed);seq=[]
  while len(seq)<12288:seq.extend(rng.permutation(d['train']).tolist())
  seq=seq[:6144];assert seq==old and ck['global_step']==512 and ck['CompleteDM']['loader']['_sampler_iter_state']['samples_yielded']==6144;assert ck['optimizer_states'][0]['param_groups'][0]['lr']==1e-5
  continuation={**design,'source_sha256':sha,'extended_selection_sha256':hashlib.sha256(json.dumps(seq).encode()).hexdigest(),'legacy_settings_selection_sha256':ck['task_state']['settings']['selection_sha']};(folder/'continuation.json').write_text(json.dumps(continuation,indent=2));(folder/'selection.json').write_text(json.dumps(seq));pl.seed_everything(seed);task=ResumeTask(cfg,seed,scales,hashlib.sha256(json.dumps(old).encode()).hexdigest());task.continuation=continuation;assert task.settings==ck['task_state']['settings'];expected=batch_to([ReactiveData(seq)[j] for j in range(0,12)],'cpu');guard=ResumeGuard(folder,expected,ck);torch.cuda.reset_peak_memory_stats();st=time.perf_counter()
  trainer=pl.Trainer(accelerator='gpu',devices=1,precision='32-true',max_steps=1024,max_epochs=-1,max_time={'minutes':5},callbacks=[guard,ModelCheckpoint(dirpath=str(folder),filename='latest',enable_version_counter=False,save_top_k=1,every_n_train_steps=128,save_on_train_epoch_end=False),EpochLatest(folder/'latest.ckpt'),ModelCheckpoint(dirpath=str(folder),filename='best',enable_version_counter=False,monitor='validation_position_diagnostic_m',mode='min',save_top_k=1,save_on_train_epoch_end=False)],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='matched6-regression-resume1024',version=str(seed)),num_sanity_val_steps=0,val_check_interval=256,check_val_every_n_epoch=None,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=16)
  with sdpa_kernel(SDPBackend.MATH):trainer.fit(task,datamodule=CompleteDM(seq,d['dev'][::32]),ckpt_path=str(source),weights_only=False)
  assert trainer.global_step==1024;info={'step':1024,'additional512_wall_s':time.perf_counter()-st,'peak_gib':torch.cuda.max_memory_allocated()/2**30};(folder/'train_complete.json').write_text(json.dumps(info));task.model.cuda().eval()
  with sdpa_kernel(SDPBackend.MATH):ev,pred=run_eval(task.model,d['dev']);fit,_=run_eval(task.model,d['train'][::8])
  (folder/'eval.json').write_text(json.dumps(ev));(folder/'fit_sample_eval.json').write_text(json.dumps(fit));np.save(folder/'dev_predictions.npy',pred);oldpred=np.load(source.parent/'dev_predictions.npy');(folder/'episode_curves_512_1024.json').write_text(json.dumps({'512':curves(oldpred,d['dev']),'1024':curves(pred,d['dev'])},indent=2));saved=torch.load(folder/'latest.ckpt',map_location='cpu',weights_only=False);assert saved['global_step']==1024 and saved['CompleteDM']['loader']['_sampler_iter_state']['samples_yielded']==6144;assert hashlib.sha256(source.read_bytes()).hexdigest()==sha;print('ARM_COMPLETE',seed,info,ev['model']['all'],flush=True);del task,trainer,guard,ck,saved;gc.collect();torch.cuda.empty_cache()
 (O/'complete.json').write_text(json.dumps({'done':True,'wall_s':time.perf_counter()-startall,'source_checkpoints_unchanged':True}));print('COMPLETE',flush=True)
if __name__=='__main__':main()

