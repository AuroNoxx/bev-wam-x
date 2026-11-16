from train_jepa_nojepa_paired import *
from train_joint_occ import JointTask
from occ_structural_prototypes import PrototypeJoint,prototype_loss,future_positive_targets
from train_fresh_jepa_first1024_v2 import Status
import math
class ProtoTask(JointTask):
 def __init__(self,cfg,seed,arm,scales,selection_sha):
  super().__init__('fusion',cfg,'prototype-v2-data-audit',selection_sha,seed);self.seed=seed;self.mode=arm;self.trace=[];self.grads=[];self.model=PrototypeJoint(128,2,4,'velocity2' if arm=='velocity_only' else 'xy5','future_positive' if arm=='world_only' else 'current_support');self.model.set_scales(scales);self.settings.update(action=self.model.action_kind,world=self.model.world_kind,residual_scale=scales)
 def task_loss(self,b,split):
  with torch.random.fork_rng(devices=[self.device.index]):
   torch.manual_seed(self.seed*100000+self.global_step);before=torch.cuda.get_rng_state().clone();v,terms=prototype_loss(self.model,b,self.settings['objective'])
   for k,x in zip(['latent','variance','semantic'],self.model.auxiliary_loss(b)):v=v+.1*x;terms[k]=x
   after=torch.cuda.get_rng_state().clone()
  if split=='train' and self.global_step<3:self.trace.append({'step':self.global_step,'before':hashlib.sha256(before.numpy().tobytes()).hexdigest(),'after':hashlib.sha256(after.numpy().tobytes()).hexdigest()})
  self.log_dict({split+'/'+k:x for k,x in terms.items()},on_epoch=True,batch_size=len(b['state']));return v
 def configure_gradient_clipping(self,o,gradient_clip_val=None,gradient_clip_algorithm=None):
  n=torch.nn.utils.clip_grad_norm_(self.parameters(),gradient_clip_val,error_if_nonfinite=True);self.grads.append({'step':self.global_step,'norm':float(n)})
 def configure_optimizers(self):
  opt=torch.optim.AdamW([p for p in self.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
  def rate(s):return (s+1)/64 if s<64 else .1+.9*.5*(1+math.cos(math.pi*min(1,(s-64)/192)))
  return {'optimizer':opt,'lr_scheduler':{'scheduler':torch.optim.lr_scheduler.LambdaLR(opt,rate),'interval':'step'}}
 def validation_step(self,b,i):
  with torch.random.fork_rng(devices=[self.device.index]):
   torch.manual_seed(2019+i);p=self.model.sample(b,k=1,steps=8);error=((p[:,0,:,:2]-b['trajectory'][...,:2])*p.new_tensor([50,10])).norm(dim=-1).mean();self.log('validation_position_diagnostic_m',error,on_epoch=True,batch_size=len(b['state']))
class EpochLatest(pl.Callback):
 def __init__(self,path):self.path=path
 def on_train_epoch_end(self,t,m):t.save_checkpoint(self.path)
@torch.no_grad()
def world_common(m,names):
 ds=DiskSamples(names);rows=[]
 for st in range(0,len(ds),4):
  b=batch_to([ds[i] for i in range(st,min(st+4,len(ds)))],'cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):
   _,p=m.forecast_batch(b);f,valid,frac=future_positive_targets(b);z=m.teacher(f).float();cur=m.teacher(m.canonical_history(b)[:,-1:]).float().expand_as(z);dyn=valid&(frac[...,3:5].sum(-1)>.05)
   for mask,name in [(valid,'all_future_positive'),(dyn,'dynamic_class_proxy')]:
    n=int(mask.sum());rows.append({'kind':name,'n':n,'pred_sum':float(((p.float()-z).square().mean(-1)*mask).sum()),'persistence_sum':float(((cur-z).square().mean(-1)*mask).sum())})
 return {k:{'tokens':sum(r['n'] for r in rows if r['kind']==k),'prediction_mse':sum(r['pred_sum'] for r in rows if r['kind']==k)/max(1,sum(r['n'] for r in rows if r['kind']==k)),'persistence_mse':sum(r['persistence_sum'] for r in rows if r['kind']==k)/max(1,sum(r['n'] for r in rows if r['kind']==k))} for k in ['all_future_positive','dynamic_class_proxy']}
def main():
 idle();configure_runtime();out=ROOT/'artifacts/structural-prototypes-light256';out.mkdir(exist_ok=False);base=ROOT/'artifacts/residual-representation-paired128';sel=json.loads((base/'selection.json').read_text());scales=json.loads((base/'scale_calibration.json').read_text())['scales_m'];cfg=json.loads((ROOT/'configs/joint-occ-v1.json').read_text());cfg.update(width=128,max_steps=256,learning_rate=1e-4,experiment='structural-isolated-prototypes',warmup_steps=64,best_monitor='validation_position_diagnostic_m',best_mode='min',best_limit='label-fit diagnostic only,notbest driving; no test');sha=hashlib.sha256(json.dumps(sel,sort_keys=True).encode()).hexdigest();(out/'config.json').write_text(json.dumps(cfg,indent=2));(out/'selection.json').write_text(json.dumps(sel,indent=2));results=[];allstart=time.perf_counter()
 for seed in [2010,2011]:
  pl.seed_everything(seed);initial=PrototypeJoint();initial.set_scales(scales);state={k:v.detach().clone() for k,v in initial.state_dict().items()};del initial;traces={}
  for arm in (['baseline','velocity_only','world_only'] if seed==2010 else ['world_only','velocity_only','baseline']):
   idle();assert time.perf_counter()-allstart<900;pl.seed_everything(seed);folder=out/f'{seed}-{arm}';folder.mkdir();task=ProtoTask(cfg,seed,arm,scales,sha);weights=dict(state)
   if arm=='velocity_only':weights['trajectory.weight']=weights['trajectory.weight'][:,:2].clone();weights['output.weight']=weights['output.weight'][:2].clone();weights['output.bias']=weights['output.bias'][:2].clone()
   task.model.load_state_dict(weights,strict=True);status=Status(folder);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();callbacks=[status,ModelCheckpoint(dirpath=str(folder),filename='latest',enable_version_counter=False,save_top_k=1,every_n_train_steps=64,save_on_train_epoch_end=False),EpochLatest(folder/'latest.ckpt'),ModelCheckpoint(dirpath=str(folder),filename='best',enable_version_counter=False,monitor='validation_position_diagnostic_m',mode='min',save_top_k=1,save_on_train_epoch_end=False)];tr=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_steps=256,max_epochs=-1,max_time={'minutes':3},callbacks=callbacks,plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='structural-prototypes-light256',version=f'{seed}-{arm}'),num_sanity_val_steps=0,val_check_interval=128,check_val_every_n_epoch=None,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=16);tr.fit(task,datamodule=SeedData(sel,seed));assert tr.global_step==256;wall=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/2**30;traces[arm]=task.trace;latest=torch.load(folder/'latest.ckpt',map_location='cpu',weights_only=False);assert latest['global_step']==256;del latest;reference=copy.deepcopy(task.model.teacher).cuda().eval();ev=evaluate(task.model,sel['validation'],reference,True);ev['world_summary_common_future_positive']=world_common(task.model,sel['validation']);r={'seed':seed,'arm':arm,'steps':256,'active_parameters':sum(p.numel() for p in task.model.parameters() if p.requires_grad),'wall_s':wall,'peak_allocated_gib':peak,'rng_trace':task.trace,'gradient_log':task.grads,'checkpoint_files':[p.name for p in folder.glob('*.ckpt')],'best_metric':'validation_position_diagnostic_m—not driving score','eval':ev};(folder/'report.json').write_text(json.dumps(r,indent=2));brief={k:v for k,v in r.items() if k not in ['eval','gradient_log','rng_trace']};brief.update(metrics=ev['summary'],world_common=ev['world_summary_common_future_positive']);results.append(brief);(out/'summary.partial.json').write_text(json.dumps(results,indent=2));print('RESULT '+json.dumps(brief),flush=True);del task,tr,reference,weights;gc.collect();torch.cuda.empty_cache()
  assert traces['baseline']==traces['velocity_only']==traces['world_only'];del state;gc.collect()
 (out/'summary.json').write_text(json.dumps({'results':results,'wall_s':time.perf_counter()-allstart,'paired_base_rng_first3':True,'limits':'smallmodel128/2/4,84train42same-routeinternaldiagnostic; velocityhead2d commonbackboneinit/prefixheadinit not identicalfullarchitecture; physicalconsistencybyconstructionnot learnedquality; worldcomparison commonfuturepositivemask withinownEMA; no test/closedloop/sourcecausalitycertificate'},indent=2));print('DONE')
if __name__=='__main__':main()
