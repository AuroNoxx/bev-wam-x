from test_fixed_context_decoder_fit import *
import copy
OLD=ROOT/'artifacts/matched6-fixed-context-decoder256'
OUT=ROOT/'artifacts/matched6-fixed-context-decoder2048'
BaseDM=ContextDM
BaseTask=DecoderTask
class ContextDM(BaseDM):
 def load_state_dict(self,s):self.pending=s
 def train_dataloader(self):
  loader=super().train_dataloader()
  if hasattr(self,'pending'):loader.load_state_dict(self.pending['loader'])
  return loader
class DecoderTask(BaseTask):
 def on_load_checkpoint(self,c):self.pending_rng=c['task_state'];self.saved_opt=c['optimizer_states'][0]
 def training_step(self,b,i):
  if hasattr(self,'pending_rng'):
   assert self.global_step==256
   assert all(torch.equal(b[k].cpu(),v) for k,v in self.expected.items())
   opt=self.optimizers().optimizer.state_dict();assert opt['param_groups']==self.saved_opt['param_groups']
   for k,v in opt['state'].items():
    for n,x in v.items():
     y=self.saved_opt['state'][k][n];assert torch.equal(x.cpu(),y.cpu()) if torch.is_tensor(x) else x==y
   torch.set_rng_state(self.pending_rng['rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in self.pending_rng['cuda_rng']]);del self.pending_rng
   print('RESTORE_GATE_PASS',self.metadata['seed'],flush=True)
  return super().training_step(b,i)
 def configure_gradient_clipping(self,opt,gradient_clip_val=None,gradient_clip_algorithm=None):
  norm=torch.nn.utils.clip_grad_norm_(self.parameters(),1.,error_if_nonfinite=True)
  if self.global_step%256==0:self.grads.append({'step':self.global_step,'preclip_norm':float(norm)})
class Monitor(pl.Callback):
 def __init__(self,out,data,initial):self.out=out;self.data=data;self.rows=[{'step':256,**initial}];self.stale=0;self.reason='hard2048';self.best=initial['action_mse']
 def on_train_start(self,t,m):self.start=time.perf_counter()
 def on_train_batch_end(self,t,m,o,b,i):
  if t.global_step==257:print('FIRST_UPDATE',m.metadata['seed'],float(o['loss']),flush=True)
  if t.global_step%256:return
  m.eval()
  with sdpa_kernel(SDPBackend.MATH):r=evaluate_unit(m,self.data)
  m.train();r.update(step=t.global_step,elapsed_s=time.perf_counter()-self.start)
  prev=self.rows[-1];worst=max(x['vector_component_rmse_mps'] for x in r['items']);oldworst=max(x['vector_component_rmse_mps'] for x in prev['items']);sign=sum(x['pred_delta']*x['target_delta']>0 for x in r['items']);oldsign=sum(x['pred_delta']*x['target_delta']>0 for x in prev['items'])
  stagnant=(prev['action_mse']-r['action_mse'])/max(prev['action_mse'],1e-12)<.01 and worst>=oldworst and sign<=oldsign
  self.stale=self.stale+1 if stagnant else 0;self.rows.append(r)
  if r['all10_fit_pass']:self.reason='all10_fit_pass';t.should_stop=True
  elif self.stale>=3:self.reason='three_stagnant_checkpoints';t.should_stop=True
  t.save_checkpoint(self.out/'latest.ckpt')
  if r['action_mse']<self.best:self.best=r['action_mse'];t.save_checkpoint(self.out/'best.ckpt')
  (self.out/'trajectory.json').write_text(json.dumps(self.rows,indent=2));print('EVAL',m.metadata['seed'],r['step'],r['action_mse'],worst,sign,self.reason,flush=True)
def main():
 safe_idle();configure_runtime();OUT.mkdir(exist_ok=True);ReactiveData.acceleration=True
 design=json.loads((OLD/'design.json').read_text());names=design['names'];groups=json.loads((ROOT/'artifacts/matched6-ambiguity-shape-v2/report.json').read_text())['exact_duplicate_actor_groups']
 cfg=json.loads((ROOT/'artifacts/structural-prototypes-light512/config.json').read_text());cfg.update(batch_size=12,max_steps=512);scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m']
 (OUT/'plan.json').write_text(json.dumps({'total_step_limit':2048,'new_updates':1792,'per_seed_time_limit_s':120,'lr':.0001,'schedule':'unchanged constant','data_stream':'original 256-batch epoch repeats naturally','context_limit':'reconstructed, not original context hash certification','stop':'all10 RMSE<=.1 and correct trend; or 3 checkpoints <1% MSE gain with worst error and direction not improving; nonfinite immediate','scope':'train-only unit diagnostic not driving model'},indent=2))
 for seed in [7202]:
  safe_idle();out=OUT/str(seed);out.mkdir();path=OLD/str(seed)/'latest.ckpt';sha=hashlib.sha256(path.read_bytes()).hexdigest();ck=torch.load(path,map_location='cpu',weights_only=False)
  source=RegressionTask(cfg,seed,scales,'unit');original=torch.load(ROOT/f'artifacts/matched6-regression-resume1024-v3/{seed}/latest.ckpt',map_location='cpu',weights_only=False);source.load_state_dict(original['state_dict']);m=source.model.cuda().eval();b=batch_to([ReactiveData(names)[i] for i in range(10)],'cuda')
  with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
   context=m.context(b);target=m.to_flow_space(b,b['trajectory'])
   for i,g in enumerate(groups):
    gb=batch_to([ReactiveData([r['file'] for r in g])[j] for j in range(len(g))],'cuda');target[i]=m.to_flow_space(gb,gb['trajectory']).mean(0)
   data={'context':context.cpu(),'target':target.cpu(),'v0':(b['state'][:,0]*30).cpu()}
  task=DecoderTask(m,ck['task_state']['metadata']);fixed={k:p.detach().cpu().clone() for k,p in m.named_parameters() if not p.requires_grad};task.load_state_dict(ck['state_dict']);assert all(torch.equal(v,dict(m.named_parameters())[k].detach().cpu()) for k,v in fixed.items());task.cuda().eval()
  with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
   assert torch.equal(context,m.context(b));assert torch.equal(context,m.context({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)}));before=evaluate_unit(task,data)
  old=json.loads((OLD/str(seed)/'report.json').read_text())['after'];diff=max(abs(a-b) for x,y in zip(before['items'],old['items']) for a,b in zip(x['pred_vx25'],y['pred_vx25']));assert diff<=1e-6
  (out/'preflight.json').write_text(json.dumps({'reconstructed_prediction_max_abs':diff,'frozen_encoder_exact':True,'context_future_poison_exact':True,'old_context_hash_available':False,'source_sha':sha}));task.expected=data;mon=Monitor(out,data,before)
  torch.cuda.reset_peak_memory_stats();st=time.perf_counter();trainer=pl.Trainer(accelerator='gpu',devices=1,precision='32-true',max_steps=2048,max_epochs=-1,max_time={'minutes':2},callbacks=[mon],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='matched6-fixed-context-decoder2048',version=str(seed)),enable_checkpointing=False,num_sanity_val_steps=0,enable_progress_bar=False,enable_model_summary=False,gradient_clip_val=1,log_every_n_steps=32)
  with sdpa_kernel(SDPBackend.MATH):trainer.fit(task,datamodule=ContextDM(data),ckpt_path=str(path),weights_only=False)
  trainer.save_checkpoint(out/'latest.ckpt');assert hashlib.sha256(path.read_bytes()).hexdigest()==sha;assert all(torch.equal(v,dict(m.named_parameters())[k].detach().cpu()) for k,v in fixed.items());task.cuda().eval()
  with sdpa_kernel(SDPBackend.MATH):after=evaluate_unit(task,data)
  (out/'report.json').write_text(json.dumps({'step':trainer.global_step,'reason':mon.reason,'before':before,'after':after,'grads':task.grads,'wall_s':time.perf_counter()-st,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'frozen_unchanged':True,'source_unchanged':True},indent=2));print('DONE',seed,trainer.global_step,flush=True)
  del source,m,task,trainer,ck,original,b,context,target,gb;gc.collect();torch.cuda.empty_cache()
 (OUT/'complete.json').write_text(json.dumps({'complete':True}))
if __name__=='__main__':main()

