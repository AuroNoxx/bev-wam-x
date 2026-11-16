"""Bounded learnability unit test, not a driving checkpoint or main training."""
from train_matched6_regression512 import *
O=ROOT/'artifacts/matched6-fixed-context-decoder256'
class ContextData(torch.utils.data.Dataset):
 def __init__(self,data,repeats):self.data=data;self.repeats=repeats
 def __len__(self):return 10*self.repeats
 def __getitem__(self,i):return {k:v[i%10] for k,v in self.data.items()}
class ContextDM(pl.LightningDataModule):
 def __init__(self,data):super().__init__();self.data=data
 def train_dataloader(self):self.loader=StatefulDataLoader(ContextData(self.data,256),batch_size=10,shuffle=False,num_workers=0);return self.loader
 def state_dict(self):return {'loader':self.loader.state_dict()} if hasattr(self,'loader') else {}
class DecoderTask(pl.LightningModule):
 def __init__(self,model,metadata):
  super().__init__();self.model=model;self.metadata=metadata;self.grads=[]
  self.model.requires_grad_(False)
  for name in ['trajectory','time','transformer','output']:getattr(self.model,name).requires_grad_(True)
  self.model.position.requires_grad_(True)
 def forward(self,b):return self.model.velocity(b['context'],torch.zeros(len(b['context']),25,2,device=self.device),torch.zeros(len(b['context']),1,device=self.device))
 def training_step(self,b,i):
  pred=self(b);loss=(pred-b['target']).square().mean();assert torch.isfinite(loss);self.log('unit_action_mse',loss);return loss
 def configure_optimizers(self):
  opt=torch.optim.AdamW([p for p in self.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01);scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:1.)
  return {'optimizer':opt,'lr_scheduler':{'scheduler':scheduler,'interval':'step'}}
 def configure_gradient_clipping(self,opt,gradient_clip_val=None,gradient_clip_algorithm=None):
  pos=self.model.position.grad;norm=torch.nn.utils.clip_grad_norm_(self.parameters(),1.,error_if_nonfinite=True)
  if self.global_step in [0,1,15,63,127,255]:self.grads.append({'step_before_update':self.global_step,'global_preclip_norm':float(norm),'position_postclip_norm':float(pos.norm()),'position_temporal_contrast_postclip_norm':float((pos-pos.mean(1,keepdim=True)).norm()),'output_postclip_norm':float(self.model.output.weight.grad.norm())})
 def on_save_checkpoint(self,c):c['task_state']={'metadata':self.metadata,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'scope':'unit test only, do not deploy'}
class UnitStatus(pl.Callback):
 def __init__(self,out):self.out=out
 def on_train_start(self,t,m):self.start=time.perf_counter()
 def on_train_batch_end(self,t,m,o,b,i):
  if t.global_step in [1,16,64,128,256]:r={'step':t.global_step,'loss':float(o['loss']),'elapsed_s':time.perf_counter()-self.start};(self.out/'status.json').write_text(json.dumps(r));print('STATUS',self.out.name,r,flush=True)
@torch.no_grad()
def evaluate_unit(task,data):
 b={k:v.cuda() for k,v in data.items()};q=task(b);target=b['target'];physical=(q-target)*q.new_tensor([10,.5]);v=q*q.new_tensor([10,.5]);v[...,0]+=b['v0'][:,None];truth=target*q.new_tensor([10,.5]);truth[...,0]+=b['v0'][:,None];rows=[]
 for i in range(10):
  err=float(physical[i].square().mean().sqrt());delta=float(v[i,-1,0]-v[i,0,0]);expected=float(truth[i,-1,0]-truth[i,0,0]);rows.append({'item':i,'vector_component_rmse_mps':err,'longitudinal_rmse_mps':float(physical[i,:,0].square().mean().sqrt()),'lateral_rmse_mps':float(physical[i,:,1].square().mean().sqrt()),'pred_vx25':v[i,:,0].tolist(),'target_vx25':truth[i,:,0].tolist(),'pred_delta':delta,'target_delta':expected,'diagnostic_fit_pass':err<=.1 and delta*expected>0})
 constant=torch.zeros_like(target);stationary=constant.clone();stationary[:,:,0]=-b['v0'][:,None]/10
 return {'action_mse':float((q-target).square().mean()),'constant_speed_action_mse':float((constant-target).square().mean()),'stationary_action_mse':float((stationary-target).square().mean()),'all10_fit_pass':all(r['diagnostic_fit_pass'] for r in rows),'items':rows}
def main():
 safe_idle();configure_runtime();O.mkdir(exist_ok=False);ReactiveData.acceleration=True;groups=json.loads((ROOT/'artifacts/matched6-ambiguity-shape-v2/report.json').read_text())['exact_duplicate_actor_groups'];rr=json.loads((ROOT/'artifacts/matched6-start-frozen-audit/rows.json').read_text());decel=[r['file'] for r in rr if r['split']=='train' and r['decel_label'] and r['noid_motion_above_OLD_train_threshold']];selected=[decel[i] for i in np.linspace(0,len(decel)-1,5,dtype=int)];names=[g[0]['file'] for g in groups]+selected;assert len(names)==10;design={'seeds':[7201,7202],'steps':256,'batch':10,'budget':'2minutes each, no extension','input':'fixed actual train-only context from original1024 encoder; no future input','targets':'5empirical exact-actor group means and5fixedtrain decel labels','objective':'actionqMSE only, diagnosticisolation not mainobjective comparison','optimizer':'AdamW1e-4 wd.01 constantlr clip1 FP32+MATH','criterion':'each vectorcomponent RMSE<=.1m/s and correct opposite endpoint trend; unitfit only not safety','names':names};(O/'design.json').write_text(json.dumps(design,indent=2));cfg=json.loads((ROOT/'artifacts/structural-prototypes-light512/config.json').read_text());cfg.update(batch_size=12,max_steps=512);scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m'];allstart=time.perf_counter()
 for seed in [7201,7202]:
  safe_idle();folder=O/str(seed);folder.mkdir();path=ROOT/f'artifacts/matched6-regression-resume1024-v3/{seed}/latest.ckpt';sha=hashlib.sha256(path.read_bytes()).hexdigest();ck=torch.load(path,map_location='cpu',weights_only=False);source=RegressionTask(cfg,seed,scales,'unit');source.load_state_dict(ck['state_dict']);m=source.model.cuda().eval();b=batch_to([ReactiveData(names)[i] for i in range(10)],'cuda')
  with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
   context=m.context(b);poison=m.context({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)});assert torch.equal(context,poison);target=m.to_flow_space(b,b['trajectory'])
   for i,g in enumerate(groups):
    gb=batch_to([ReactiveData([r['file'] for r in g])[j] for j in range(len(g))],'cuda');target[i]=m.to_flow_space(gb,gb['trajectory']).mean(0)
   data={'context':context.cpu(),'target':target.cpu(),'v0':(b['state'][:,0]*30).cpu()}
  task=DecoderTask(m,{**design,'seed':seed,'source_sha':sha});fixed={k:v.detach().cpu().clone() for k,v in task.model.named_parameters() if not v.requires_grad};task.cuda().eval()
  with sdpa_kernel(SDPBackend.MATH):before=evaluate_unit(task,data)
  (folder/'before.json').write_text(json.dumps(before));(folder/'preflight.json').write_text(json.dumps({'context_future_poison_exact':True,'trainable_parameters':sum(p.numel() for p in task.parameters() if p.requires_grad),'frozen_context_shape':list(context.shape)}));torch.cuda.reset_peak_memory_stats();st=time.perf_counter();trainer=pl.Trainer(accelerator='gpu',devices=1,precision='32-true',max_steps=256,max_epochs=-1,max_time={'minutes':2},callbacks=[UnitStatus(folder),ModelCheckpoint(dirpath=str(folder),filename='latest',enable_version_counter=False,every_n_train_steps=64,save_top_k=1,save_on_train_epoch_end=False),EpochLatest(folder/'latest.ckpt')],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='matched6-fixed-context-decoder256',version=str(seed)),num_sanity_val_steps=0,enable_progress_bar=False,enable_model_summary=False,gradient_clip_val=1,log_every_n_steps=16)
  with sdpa_kernel(SDPBackend.MATH):trainer.fit(task,datamodule=ContextDM(data))
  assert trainer.global_step==256;task.cuda().eval()
  with sdpa_kernel(SDPBackend.MATH):after=evaluate_unit(task,data)
  assert all(torch.equal(v,dict(task.model.named_parameters())[k].detach().cpu()) for k,v in fixed.items());assert hashlib.sha256(path.read_bytes()).hexdigest()==sha;result={'before':before,'after':after,'gradients':task.grads,'wall_s':time.perf_counter()-st,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'frozen_parameters_unchanged':True,'source_unchanged':True};(folder/'report.json').write_text(json.dumps(result,indent=2));print('RESULT',seed,before['action_mse'],after['action_mse'],after['all10_fit_pass'],result['wall_s'],flush=True);del source,m,task,trainer,ck,b,context,target;gc.collect();torch.cuda.empty_cache()
 (O/'complete.json').write_text(json.dumps({'done':True,'wall_s':time.perf_counter()-allstart}));print('COMPLETE',flush=True)
if __name__=='__main__':main()
