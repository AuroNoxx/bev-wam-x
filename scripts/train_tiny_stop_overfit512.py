"""Bounded eight-window generation overfit gate, never a driving acceptance test."""
from train_scene_balance_light512 import *
from collections import defaultdict
from pathlib import Path
@torch.no_grad()
def gate_eval(m,names):
 ds=DiskSamples(names);b=batch_to([ds[i] for i in range(len(ds))],'cuda');metrics=[]
 for noise_seed in [2100,2101,2102]:
  with torch.random.fork_rng(devices=[0]),torch.autocast('cuda',dtype=torch.bfloat16):
   torch.manual_seed(noise_seed);p=m.sample(b,k=4,steps=16).float();torch.manual_seed(noise_seed);poison={**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)};pp=m.sample(poison,k=4,steps=16).float();assert torch.equal(p,pp)
  v=p[...,4]*30;truth=b['trajectory'][:,None];xy=(p[...,:2]-truth[...,:2])*p.new_tensor([50,10]);vel=p[...,4:5]*30*p[...,[3,2]];targetv=truth[...,4:5]*30*truth[...,[3,2]];acc=(vel-torch.cat([torch.stack([b['state'][:,0]*30,torch.zeros_like(b['state'][:,0])],-1)[:,None,None,:].expand(-1,4,-1,-1),vel[:,:,:-1]],2))/.2;jerk=torch.diff(acc,dim=2)/.2
  metrics.append({'noise_seed':noise_seed,'stop_fraction':float((v[:5,:,-1]<.5).float().mean()),'stop_end_speed':float(v[:5,:,-1].mean()),'position_fit_m':float(xy.norm(dim=-1).mean()),'velocity_rmse_mps':float((vel-targetv).square().mean().sqrt()),'normal_end_speed_mae':float(abs(v[5:,:,-1]-truth[5:,:, -1,4]*30).mean()),'normal_end_stopped_fraction':float((v[5:,:,-1]<.5).float().mean()),'normal_forward_proxy_m':float((p[5:,:,-1,0]*50).mean()),'mean_peak_vector_accel':float(acc.norm(dim=-1).max(-1).values.mean()),'mean_peak_vector_jerk':float(jerk.norm(dim=-1).max(-1).values.mean())})
 summary={k:float(np.mean([r[k] for r in metrics])) for k in metrics[0] if k!='noise_seed'};passed=all(r['stop_fraction']>=.9 and r['position_fit_m']<1 and r['velocity_rmse_mps']<.5 and r['normal_end_speed_mae']<1 and r['normal_end_stopped_fraction']==0 for r in metrics);return {'summary':summary,'by_noise':metrics,'gate_pass':passed,'limits':'Fit gate only, not safety; 3 fixed eval noise seeds,4 correlated candidates; historical labels not unique safe behavior.'}
class OverfitTask(ProtoTask):
 def __init__(self,*a,fit_names=None,**kw):super().__init__(*a,**kw);self.fit_names=fit_names;self.gates=[];self.consecutive=0
 def validation_step(self,b,i):
  if i:raise RuntimeError('Expected one fixed fit batch')
  result=gate_eval(self.model,self.fit_names);result['step']=self.global_step;self.gates.append(result);self.consecutive=self.consecutive+1 if result['gate_pass'] else 0;self.log('validation_position_diagnostic_m',result['summary']['position_fit_m'],batch_size=8);print('GATE',self.seed,json.dumps(result),flush=True)
  if self.consecutive>=2:self.trainer.should_stop=True
class SaveGate(pl.Callback):
 def __init__(self,folder):self.folder=folder
 def on_validation_end(self,t,m):(self.folder/'gates.json').write_text(json.dumps(m.gates,indent=2))
def main():
 idle();configure_runtime();out=ROOT/'artifacts/tiny-stop-overfit512-v2';out.mkdir(exist_ok=True);src=ROOT/'artifacts/scene-balance-light512';d=json.loads((src/'design.json').read_text());stops=json.loads((ROOT/'artifacts/scene-balance-frozen-stop-noise/selection.json').read_text())['train_stop'];by=defaultdict(list)
 for n in stops:by[Path(n).parent.name].append(n)
 chosen=[v[len(v)//2] for _,v in sorted(by.items())];assert len(chosen)==5;stoproutes=set(by);normal=defaultdict(list)
 for n in sorted(set(sum([v for arms in d['streams'].values() for v in arms.values()],[]))):
  if Path(n).parent.name in stoproutes:continue
  with np.load(ROOT/'data/occ_v3'/n) as f:v0=float(f['state'][0])*30;v1=float(f['trajectory'][-1,4])*30
  if min(v0,v1)>2 and abs(v1-v0)<1:normal[Path(n).parent.name].append(n)
 keys=sorted(normal);chosen += [normal[keys[i]][len(normal[keys[i]])//2] for i in np.linspace(0,len(keys)-1,3,dtype=int)];held=[r['file'] for r in d['validation']];assert not set(chosen)&set(held);(out/'selection.json').write_text(json.dumps({'fit':chosen,'held':held,'stop_first5':True},indent=2));cfg=json.loads((src/'config.json').read_text());cfg.update(experiment='strict_eight_window_overfit',initialization='weights only balanced512; optimizer reset explicitly',gate='all3noise:stop>=.9,position<1m,velocityRMSE<.5m/s,normalendMAE<1,normalstop0 twice; max512');(out/'config.json').write_text(json.dumps(cfg,indent=2));scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m'];allstart=time.perf_counter();results=[]
 for seed in [2020,2021]:
  if (out/str(seed)/'report.json').exists():
   completed=json.loads((out/str(seed)/'report.json').read_text());assert completed['steps']==512;results.append({k:v for k,v in completed.items() if k not in ['gates','held']});continue
  idle();assert time.perf_counter()-allstart<600;pl.seed_everything(seed);folder=out/str(seed);folder.mkdir();ck=torch.load(src/f'{seed}-balanced/latest.ckpt',map_location='cpu',weights_only=False);task=OverfitTask(cfg,seed,'velocity_only',scales,'tiny-fixed8',fit_names=chosen);task.load_state_dict(ck['state_dict'],strict=True);del ck;task.model.cuda().eval();before=gate_eval(task.model,chosen);(folder/'before.json').write_text(json.dumps(before,indent=2));task.train();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();callbacks=[Status(folder),SaveGate(folder),ModelCheckpoint(dirpath=str(folder),filename='latest',enable_version_counter=False,save_top_k=1,every_n_train_steps=128,save_on_train_epoch_end=False),EpochLatest(folder/'latest.ckpt'),ModelCheckpoint(dirpath=str(folder),filename='best',enable_version_counter=False,monitor='validation_position_diagnostic_m',mode='min',save_top_k=1,save_on_train_epoch_end=False)];trainer=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_steps=512,max_epochs=-1,max_time={'minutes':4},callbacks=callbacks,plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='tiny-stop-overfit512',version=str(seed)),num_sanity_val_steps=0,val_check_interval=128,check_val_every_n_epoch=None,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=16);trainer.fit(task,datamodule=StreamData({'train':chosen*512,'validation':chosen}));wall=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/2**30;task.model.cuda().eval();after=gate_eval(task.model,chosen);ref=copy.deepcopy(task.model.teacher).cuda().eval();held_result=evaluate(task.model,held,ref,True);r={'seed':seed,'steps':trainer.global_step,'wall_s':wall,'peak_gib':peak,'before':before,'after':after,'gates':task.gates,'held':held_result};(folder/'report.json').write_text(json.dumps(r,indent=2));results.append({k:v for k,v in r.items() if k not in ['gates','held']});print('RESULT',json.dumps(results[-1]),flush=True);del task,trainer,ref;gc.collect();torch.cuda.empty_cache()
 (out/'summary.json').write_text(json.dumps({'results':results,'wall_s':time.perf_counter()-allstart,'limits':'Eight-window fit only; source causal input uncertified; pure noise eval no future inputs; no test/closedloop; normal/stop label-fit diagnostic not driving acceptance'},indent=2));print('DONE')
if __name__=='__main__':main()


