"""Bounded deterministic head diagnostic; frozen encoder, no FM replacement."""
from train_carla_reactive_history256 import *
from rear_trajectory_contract import RearTrajectoryDecoder,supervision
from audit_signed_curvature_expressivity import rows
from torch import nn
O=ROOT/'artifacts/rear-frozen-diagnostic512'
class Packed(torch.utils.data.Dataset):
 def __init__(self,data,indices):self.data=data;self.indices=indices
 def __len__(self):return len(self.indices)
 def __getitem__(self,i):return {k:v[self.indices[i]] for k,v in self.data.items()}
class DM(pl.LightningDataModule):
 def __init__(self,data,seq,n):super().__init__();self.data=data;self.seq=seq;self.n=n
 def train_dataloader(self):
  self.loader=StatefulDataLoader(Packed(self.data,self.seq),batch_size=12,shuffle=False,num_workers=0)
  if hasattr(self,"pending"):self.loader.load_state_dict(self.pending["loader"])
  return self.loader
 def val_dataloader(self):return StatefulDataLoader(Packed(self.data,list(range(self.n,len(self.data['features'])))),batch_size=32,shuffle=False,num_workers=0)
 def state_dict(self):return {'loader':self.loader.state_dict()} if hasattr(self,'loader') else {}
 def load_state_dict(self,state):self.pending=state
class HeadTask(pl.LightningModule):
 def __init__(self,width,source,source_sha):
  super().__init__();self.save_hyperparameters();self.head=nn.Linear(width,12);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias);self.decoder=RearTrajectoryDecoder().float()
 def on_save_checkpoint(self,checkpoint):
  checkpoint["task_state"]={"source":self.hparams.source,"source_sha":self.hparams.source_sha,"representation":"rear-continuous-v1","rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"steps":self.global_step}
 def on_load_checkpoint(self,checkpoint):
  assert checkpoint["task_state"]["source_sha"]==self.hparams.source_sha
  self.resume_metadata=checkpoint["task_state"]
 def forward(self,b):
  c=self.head(b['features']).reshape(-1,6,2)*b['features'].new_tensor([1.,.01]);p=self.decoder(b['v0'],c);p['coefficients']=c;return p
 def loss(self,b):
  p=self(b);terms=supervision(p,b);loss=terms['position_m2'][0]/25+terms['signed_speed_m2ps2'][0]/25+terms['body_yaw_rad2'][0]/.01+terms['interval_yaw_rate_rad2ps2'][0]/.01
  if not torch.isfinite(loss):raise RuntimeError('nonfinite diagnostic loss')
  return loss,p,terms
 def training_step(self,b,i):
  loss,_,terms=self.loss(b);self.log('train_loss',loss);return loss
 def validation_step(self,b,i):
  loss,p,_=self.loss(b);self.log('validation_position_diagnostic_m',torch.linalg.vector_norm(p['actor_xy']-b['actor_xy'],dim=-1).mean(),batch_size=len(b['v0']));self.log('validation_loss',loss,batch_size=len(b['v0']))
 def configure_optimizers(self):
  opt=torch.optim.AdamW(self.head.parameters(),lr=1e-4,weight_decay=.01)
  sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/64,1.)*.5*(1+math.cos(math.pi*max(0,s-64)/(512-64))))
  return {'optimizer':opt,'lr_scheduler':{'scheduler':sch,'interval':'step'}}
class Progress(pl.Callback):
 def __init__(self,out):self.out=out
 def on_train_start(self,t,m):self.start=time.perf_counter()
 def on_train_batch_end(self,t,m,outputs,b,i):
  if t.global_step in [1,16] or t.global_step%128==0:
   torch.cuda.synchronize();s={'step':t.global_step,'loss':float(outputs['loss']),'elapsed_s':time.perf_counter()-self.start,'peak_gib':torch.cuda.max_memory_allocated()/2**30};(self.out/'status.json').write_text(json.dumps(s));print('STATUS',self.out.name,json.dumps(s),flush=True)

def make_labels(names):
 records=[]
 for name in names:
  p=Path(name);rr=rows('carla_causal_reactive_20261116_hold_v3',p.parent.name);i=int(p.stem);window=rr[i:i+51];base=np.array(window[0]['ego_world']);T=np.linalg.inv(base);rot=base[:3,:3].T;ys=np.unwrap([-np.arctan2((rot@np.array(x['ego_world'])[:3,:3])[1,0],(rot@np.array(x['ego_world'])[:3,:3])[0,0]) for x in window]);v=np.array([(np.array(x['ego_world'])[:3,:3].T@np.array(x['velocity_world']))[0] for x in window]);xy=np.array([(T@np.array(x['ego_world']))[:2,3]*[1,-1] for x in window]);targetv=v[2::2];targety=ys[2::2];rate=np.diff(ys[::2])/.2;vm=(v[::2][1:]+v[::2][:-1])/2;valid=np.ones(25,bool)
  records.append({'v0':np.float32(v[0]),'signed_speed':targetv.astype('f'),'actor_xy':xy[2::2].astype('f'),'body_yaw':targety.astype('f'),'interval_yaw_rate':rate.astype('f'),'valid':valid,'yaw_observable':valid,'interval_valid':valid,'curvature_identifiable':abs(vm)>=.5})
 return {k:torch.from_numpy(np.stack([r[k] for r in records])) for k in records[0]}

def metrics(pred,b):
 result={};v0=b['v0'].numpy();truth=b['signed_speed'].numpy();groups=np.where(v0<.2,'static_current',np.where(truth[:,-1]<v0-1,'decelerating',np.where(truth[:,-1]>v0+1,'accelerating','steady')));groups[(v0<.2)&(truth[:,-1]>2)]='start'
 xy=pred['actor_xy'].numpy();sp=pred['signed_speed'].numpy();txy=b['actor_xy'].numpy();zero=np.zeros((len(xy),1,2));vel=np.diff(np.concatenate([zero,xy],1),axis=1)/.2;acc=np.diff(vel,axis=1)/.2;jerk=np.diff(acc,axis=1)/.2
 for g in ['all']+sorted(set(groups)):
  sel=np.ones(len(xy),bool) if g=='all' else groups==g
  result[g]={'n':int(sel.sum()),'position_error_mean_m':float(np.linalg.norm(xy[sel]-txy[sel],axis=-1).mean()),'speed_MAE_0.2_0.6_1_5s':np.abs(sp[sel]-truth[sel]).mean(0)[[0,2,4,24]].tolist(),'endpoint_x_mean_m':float(xy[sel,-1,0].mean()),'label_endpoint_x_mean_m':float(txy[sel,-1,0].mean()),'position_derived_peak_accel_mean':float(np.linalg.norm(acc[sel],axis=-1).max(1).mean()),'position_derived_peak_jerk_mean':float(np.linalg.norm(jerk[sel],axis=-1).max(1).mean()),'negative_speed_fraction':float((sp[sel]<-.05).mean()),'collision':None,'TTC':None}
 return result
@torch.no_grad()
def evaluate_head(task,data,n):
 allp={'actor_xy':[],'signed_speed':[],'body_yaw':[]}
 for start in range(n,len(data['features']),32):
  b={k:v[start:start+32].cuda() for k,v in data.items()};p=task(b)
  for k in allp:allp[k].append(p[k].cpu())
 return {k:torch.cat(v) for k,v in allp.items()}

def main():
 safe_idle();configure_runtime();O.mkdir(exist_ok=True);startall=time.perf_counter();manifest=json.loads((ROOT/'data/carla_reactive_hold_cache_v1/manifest.json').read_text());names=manifest['train']+manifest['dev'];n=len(manifest['train']);labels=make_labels(names);ReactiveData.acceleration=True;ds=ReactiveData(names)
 design={'seeds':[2070,2071],'steps':512,'batch':12,'train_unique':n,'dev_unique':len(names)-n,'encoder':'matching history256 frozen eval; full context flatten480x128 no pooling','head':'zero linear61440->12, accel1/curvature.01 units','loss_scales':{'position_m':5,'speed_mps':5,'body_yaw_rad':.1,'interval_yaw_rate_radps':.1},'optimizer':'AdamW1e-4 wd.01 clip1 warmup64 cosine512','physical_geometry':'historical corroborated, not collection-instance certified','no_coeff_target':True,'no_closed_loop':True,'not_single_variable_vs_FM':True,'world_frozen':True};(O/'design.json').write_text(json.dumps(design,indent=2));summary={}
 for seed in [2070,2071]:
  safe_idle();assert time.perf_counter()-startall<600;source=ROOT/f'artifacts/carla-reactive-history-paired256/{seed}-history/latest.ckpt';sha=hashlib.sha256(source.read_bytes()).hexdigest();ck=torch.load(source,map_location='cpu',weights_only=False);m=NavigationScaledFlow(50);m.load_state_dict({k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')});m.cuda().eval();m.requires_grad_(False);before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};feats=[]
  with torch.no_grad():
   probe=batch_to([ds[130]],'cuda');p=m.context(probe);q=m.context({**probe,'future':torch.full_like(probe['future'],255),'trajectory':torch.full_like(probe['trajectory'],999)});assert torch.equal(p,q)
   for start in range(0,len(names),12):feats.append(m.context(batch_to([ds[j] for j in range(start,min(start+12,len(names)))],'cuda')).flatten(1).cpu())
  assert all(torch.equal(before[k],v.cpu()) for k,v in m.state_dict().items());data={**labels,'features':torch.cat(feats)};del m,ck,feats,probe,p,q;torch.cuda.empty_cache();folder=O/str(seed);folder.mkdir(exist_ok=True);torch.save(data,folder/'frozen_features_and_labels.pt');rng=np.random.default_rng(seed+100);seq=[]
  while len(seq)<6144:seq.extend(rng.permutation(n).tolist())
  seq=seq[:6144];(folder/'selection.json').write_text(json.dumps(seq));pl.seed_everything(seed);task=HeadTask(data['features'].shape[1],str(source),sha).cuda();b={k:v[:12].cuda() for k,v in data.items()}
  with torch.no_grad():
   p=task(b);straight=b['v0'][:,None]*torch.arange(1,26,device='cuda')[None]*.2;assert torch.allclose(p['actor_xy'][:,:,0],straight,atol=1e-4) and torch.count_nonzero(p['actor_xy'][:,:,1])==0
  (folder/'preflight.json').write_text(json.dumps({'future_poison_context_exact':True,'frozen_buffers_exact':True,'zero_head_straight_initialization':True,'trainable_parameters':sum(x.numel() for x in task.parameters() if x.requires_grad)}));torch.cuda.reset_peak_memory_stats();start=time.perf_counter();tr=pl.Trainer(accelerator='gpu',devices=1,precision='32-true',max_steps=512,max_epochs=-1,max_time={'minutes':4},callbacks=[Progress(folder),ModelCheckpoint(dirpath=str(folder),filename='latest',enable_version_counter=False,save_top_k=1,every_n_train_steps=128,save_on_train_epoch_end=False),ModelCheckpoint(dirpath=str(folder),filename='best',enable_version_counter=False,monitor='validation_position_diagnostic_m',mode='min',save_top_k=1,save_on_train_epoch_end=False)],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='rear-frozen-diagnostic512',version=str(seed)),num_sanity_val_steps=0,val_check_interval=256,check_val_every_n_epoch=None,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=16);resume=folder/"recovered128.ckpt";tr.fit(task,datamodule=DM(data,seq,n),ckpt_path=str(resume) if resume.exists() else None);assert tr.global_step==512;task.cuda().eval();pred=evaluate_head(task,data,n);dev={k:v[n:] for k,v in data.items()};zeros=torch.zeros(len(dev['v0']),6,2);baseline=RearTrajectoryDecoder().float()(dev['v0'],zeros);stationary=RearTrajectoryDecoder().float()(torch.zeros_like(dev['v0']),zeros);r={'trained':metrics(pred,dev),'zero_coefficients':metrics(baseline,dev),'stationary':metrics(stationary,dev),'labels':metrics({'actor_xy':dev['actor_xy'],'signed_speed':dev['signed_speed']},dev),'steps':512,'train_wall_s':time.perf_counter()-start,'peak_gib':torch.cuda.max_memory_allocated()/2**30};(folder/'result.json').write_text(json.dumps(r,indent=2));torch.save(pred,folder/'predictions.pt');assert hashlib.sha256(source.read_bytes()).hexdigest()==sha;summary[str(seed)]=r;(O/'summary.partial.json').write_text(json.dumps(summary,indent=2));print('SEED_COMPLETE',seed,r['train_wall_s'],flush=True);del task,tr,data;gc.collect();torch.cuda.empty_cache()
 (O/'complete.json').write_text(json.dumps({'done':True,'wall_s':time.perf_counter()-startall,'encoder_sources_unchanged':True}));print('COMPLETE',flush=True)
if __name__=='__main__':main()
