"""Matched-data regression vs Flow learnability diagnostic, not driving deployment."""
from train_scene_balance_light512 import *
from pathlib import Path
from torch.nn import functional as F
class CompareModel(PrototypeJoint):
 def __init__(self,method):super().__init__(128,2,4,'velocity2','current_support');self.method=method
 @torch.no_grad()
 def sample(self,b,k=4,steps=16):
  if self.method=='fm':return super().sample(b,k,steps)
  q=self.velocity(self.context(b),torch.zeros(len(b['state']),25,2,device=b['state'].device),torch.zeros(len(b['state']),1,device=b['state'].device));return self.from_flow_space(b,q)[:,None].expand(-1,k,-1,-1).clone()
class CompareTask(ProtoTask):
 def __init__(self,cfg,seed,method,scales,sha):
  super().__init__(cfg,seed,'velocity_only',scales,sha);self.mode=method;self.model=CompareModel(method);self.model.set_scales(scales);self.settings.update(method=method,comparison='action FM vs deterministic q MSE; shared physical position and JEPA losses')
 def task_loss(self,b,split):
  with torch.random.fork_rng(devices=[self.device.index]):
   torch.manual_seed(self.seed*100000+self.global_step);before=torch.cuda.get_rng_state().clone();target=self.model.to_flow_space(b,b['trajectory']);noise=torch.randn(len(target),25,5,device=self.device)[...,:2];t=torch.rand(len(target),1,device=self.device)
   if self.mode=='fm':
    x=(1-t[:,:,None])*noise+t[:,:,None]*target;pred=self.model(b,x,t).float();loss=F.mse_loss(pred,target-noise);q=x+(1-t[:,:,None])*pred
   else:
    q=self.model(b,torch.zeros_like(noise),torch.zeros_like(t)).float();loss=F.mse_loss(q,target)
   physical=self.model.from_flow_space(b,q);cfg=self.settings['objective'];gate=(t[:,0]>=cfg['auxiliary_min_t']).float()/(1-cfg['auxiliary_min_t']);err=(physical[...,:2]-b['trajectory'][...,:2])*q.new_tensor([50,10]);position=(F.smooth_l1_loss(err/cfg['position_scale_m'],torch.zeros_like(err),reduction='none').flatten(1).mean(1)*gate).mean();total=loss+cfg['weights']['position']*position;terms={'action_objective':loss,'position':position}
   for key,val in zip(['latent','variance','semantic'],self.model.auxiliary_loss(b)):total=total+.1*val;terms[key]=val
   after=torch.cuda.get_rng_state().clone()
  if split=='train' and self.global_step<3:self.trace.append({'step':self.global_step,'before':hashlib.sha256(before.cpu().numpy().tobytes()).hexdigest(),'after':hashlib.sha256(after.cpu().numpy().tobytes()).hexdigest()})
  self.log_dict({split+'/'+k:v for k,v in terms.items()},on_epoch=True,batch_size=len(target));return total
class Data12(pl.LightningDataModule):
 def __init__(self,fit):super().__init__();self.fit=fit
 def train_dataloader(self):return StatefulDataLoader(DiskSamples(self.fit*512),batch_size=12,shuffle=False,num_workers=0)
 def val_dataloader(self):return StatefulDataLoader(DiskSamples(self.fit),batch_size=12,shuffle=False,num_workers=0)
def main():
 idle();configure_runtime();out=ROOT/'artifacts/mixed-regression-flow512';design=json.loads((out/'design.json').read_text());fit=[x['file'] for x in design['fit']];held=design['held'];scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m'];cfg=json.loads((ROOT/'artifacts/structural-prototypes-light512/config.json').read_text());cfg.update(experiment='mixed_regression_flow_small_fit',batch_size=12,initialization='fresh paired',budget='2seeds x2arms x512,5min perarm15min total',comparison='Same velocity2 decoder and conditional architecture; FM and direct q regression are different objectives; no total-loss quality comparison');(out/'config.json').write_text(json.dumps(cfg,indent=2));sha=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest();allstart=time.perf_counter();results=[]
 for seed in [2040,2041]:
  pl.seed_everything(seed);initial=CompareModel('fm');initial.set_scales(scales);weights={k:v.clone() for k,v in initial.state_dict().items()};del initial;traces={}
  for method in (['fm','regression'] if seed==2040 else ['regression','fm']):
   idle();assert time.perf_counter()-allstart<900;pl.seed_everything(seed);folder=out/f'{seed}-{method}';folder.mkdir(exist_ok=False);task=CompareTask(cfg,seed,method,scales,sha);task.model.load_state_dict(weights,strict=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();callbacks=[Status(folder),ModelCheckpoint(dirpath=str(folder),filename='latest',enable_version_counter=False,save_top_k=1,every_n_train_steps=128,save_on_train_epoch_end=False),EpochLatest(folder/'latest.ckpt'),ModelCheckpoint(dirpath=str(folder),filename='best',enable_version_counter=False,monitor='validation_position_diagnostic_m',mode='min',save_top_k=1,save_on_train_epoch_end=False)];trainer=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_steps=512,max_epochs=-1,max_time={'minutes':5},callbacks=callbacks,plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='mixed-regression-flow512',version=f'{seed}-{method}'),num_sanity_val_steps=0,val_check_interval=128,check_val_every_n_epoch=None,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=16);trainer.fit(task,datamodule=Data12(fit));assert trainer.global_step==512;wall=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/2**30;traces[method]=task.trace;ck=torch.load(folder/'latest.ckpt',map_location='cpu',weights_only=False);assert ck['global_step']==512;del ck;task.model.cuda().eval();ref=copy.deepcopy(task.model.teacher).cuda().eval();fit_eval=evaluate(task.model,fit,ref,True);held_eval=evaluate(task.model,held,ref,True);r={'seed':seed,'method':method,'steps':512,'parameters':sum(p.numel() for p in task.model.parameters() if p.requires_grad),'wall_s':wall,'peak_allocated_gib':peak,'rng':task.trace,'gradients':task.grads,'fit':fit_eval,'held':held_eval};(folder/'report.json').write_text(json.dumps(r,indent=2));brief={k:v for k,v in r.items() if k not in ['rng','gradients','fit','held']};brief.update(fit=fit_eval['summary'],held=held_eval['summary']);results.append(brief);(out/'summary.partial.json').write_text(json.dumps(results,indent=2));print('RESULT',json.dumps(brief),flush=True);del task,trainer,ref;gc.collect();torch.cuda.empty_cache()
  assert traces['fm']==traces['regression'];del weights
 (out/'summary.json').write_text(json.dumps({'results':results,'wall_s':time.perf_counter()-allstart,'limits':design['limits']+' Regression deterministic repeatedK predictions are not independent candidates. Same update/sample budget, not equal sampling FLOPs. No physical action/safety acceptance. Current support-only world unchanged.','rng_first3_matched':True},indent=2));print('DONE')
if __name__=='__main__':main()
