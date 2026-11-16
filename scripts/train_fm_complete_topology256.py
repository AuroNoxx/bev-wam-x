"""Bounded complete-window attention topology comparison, FP32 MATH only."""
from train_carla_reactive_history256 import *
from fixed_causal_flow import FixedCausalFlow
from torch.nn.attention import sdpa_kernel,SDPBackend
import evaluate_jepa_fp32 as evaluation
O=ROOT/'artifacts/fm-complete-topology-paired256'
class CompleteDM(ReactiveDM):
 def train_dataloader(self):
  self.loader=StatefulDataLoader(ReactiveData(self.fit),batch_size=12,shuffle=False,num_workers=0)
  if hasattr(self,'pending'):self.loader.load_state_dict(self.pending['loader'])
  return self.loader
 def state_dict(self):return {'loader':self.loader.state_dict()} if hasattr(self,'loader') else {}
 def load_state_dict(self,state):self.pending=state
class TopologyTask(Task):
 def __init__(self,cfg,seed,scales,sha,causal):
  super().__init__(cfg,seed,scales,sha)
  if causal:self.model=FixedCausalFlow(50);self.model.set_scales(scales)
  self.settings.update(attention='fixed_causal' if causal else 'bidirectional',precision='FP32',attention_backend='MATH',train_scope='480complete reactive windows; no partial labels')
class Progress(pl.Callback):
 def __init__(self,out,globalstart):self.out=out;self.globalstart=globalstart
 def on_train_start(self,t,m):self.start=time.perf_counter()
 def on_train_batch_end(self,t,m,outputs,b,i):
  if time.perf_counter()-self.globalstart>720:t.should_stop=True
  if t.global_step in [1,16] or t.global_step%64==0:
   torch.cuda.synchronize();e=time.perf_counter()-self.start;r={'step':t.global_step,'budget':256,'loss':float(outputs['loss']),'elapsed_s':e,'mean_s_step':e/t.global_step,'remaining_train_estimate_s':e/t.global_step*(256-t.global_step),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30};(self.out/'status.json').write_text(json.dumps(r));print('STATUS',self.out.name,json.dumps(r),flush=True)
def main():
 safe_idle();configure_runtime();O.mkdir(exist_ok=False);startall=time.perf_counter();data=json.loads((ROOT/'data/carla_reactive_hold_cache_v1/manifest.json').read_text());auditroot=ROOT/'artifacts/permission-prefix-audit-v1';complete={}
 for split in ['train','dev']:
  labels=json.loads((auditroot/f'{split}_label_only_manifest.json').read_text());assert len(labels)==len(data[split]);complete[split]=[name for name,row in zip(data[split],labels) if row['change_post_frame_index'] is None];assert len(complete[split])==480
 cfg=json.loads((ROOT/'artifacts/structural-prototypes-light512/config.json').read_text());cfg.update(batch_size=12,max_steps=256,precision='32-true');scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m'];design={'seeds':[2070,2071],'steps_per_arm':256,'batch':12,'initialization':'matching history256 weights, reset optimizer; pretrained already saw514 including34partial','unique_train':480,'primary_dev':480,'secondary_command_change_dev':34,'only_change':'trajectory self attention topology; same output and FM position JEPA objectives','loss':'FM MSE +0.2 gated position SmoothL1/5m at flow_t>=.5;0.1 eachJEPA latent variance semantic; no newkinematicloss','schedule':'AdamW1e-4 wd.01 clip1 warmup64/original512cosine stopped256','precision':'FP32+MATH botharms training/eval','budget':'4minutes perarm;12minutes total; no automaticrestart/extension','evaluation':'fixed initial same-seed EMA reference+ownEMA persistence, K4Euler16, full514 stratified','no_closed_loop':True};(O/'design.json').write_text(json.dumps(design,indent=2));(O/'complete_selection.json').write_text(json.dumps(complete));ReactiveData.acceleration=True;evaluation.DiskSamples=ReactiveData;brief=[]
 for seed in [2070,2071]:
  source=ROOT/f'artifacts/carla-reactive-history-paired256/{seed}-history/latest.ckpt';source_sha=hashlib.sha256(source.read_bytes()).hexdigest();ck=torch.load(source,map_location='cpu',weights_only=False);weights={k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')};del ck;reference=NavigationScaledFlow(50);reference.load_state_dict(weights);reference=copy.deepcopy(reference.teacher).cuda().eval();rng=np.random.default_rng(seed+100);seq=[]
  while len(seq)<3072:seq.extend(rng.permutation(complete['train']).tolist())
  seq=seq[:3072];assert len(set(seq))==480;sha=hashlib.sha256(json.dumps(seq).encode()).hexdigest();traces={}
  for arm,causal in ([('bidirectional',False),('causal',True)] if seed==2070 else [('causal',True),('bidirectional',False)]):
   safe_idle();assert time.perf_counter()-startall<720;folder=O/f'{seed}-{arm}';folder.mkdir();(folder/'selection.json').write_text(json.dumps(seq));pl.seed_everything(seed);task=TopologyTask(cfg,seed,scales,sha,causal);task.model.load_state_dict(weights);assert all(torch.equal(weights[k],v.cpu()) for k,v in task.model.state_dict().items());task.cuda().eval();b=batch_to([ReactiveData(complete['train'])[40]],'cuda');x=torch.randn(1,25,2,device='cuda');tt=torch.full((1,1),.3,device='cuda')
   with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
    p=task.model(b,x,tt);q=task.model({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)},x,tt);assert torch.equal(p,q)
   (folder/'preflight.json').write_text(json.dumps({'source_weights_exact':True,'actor_future_poison_exact':True,'parameters_trainable':sum(p.numel() for p in task.model.parameters() if p.requires_grad),'source_sha256':source_sha}));del b,x,p,q;torch.cuda.reset_peak_memory_stats();st=time.perf_counter();tr=pl.Trainer(accelerator='gpu',devices=1,precision='32-true',max_steps=256,max_epochs=-1,max_time={'minutes':4},callbacks=[Progress(folder,startall),ModelCheckpoint(dirpath=str(folder),filename='latest',enable_version_counter=False,save_top_k=1,every_n_train_steps=64,save_on_train_epoch_end=False),EpochLatest(folder/'latest.ckpt'),ModelCheckpoint(dirpath=str(folder),filename='best',enable_version_counter=False,monitor='validation_position_diagnostic_m',mode='min',save_top_k=1,save_on_train_epoch_end=False)],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(ROOT/'logs/tensorboard/v1'),name='fm-complete-topology-paired256',version=f'{seed}-{arm}'),num_sanity_val_steps=0,val_check_interval=128,check_val_every_n_epoch=None,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=16)
   with sdpa_kernel(SDPBackend.MATH):tr.fit(task,datamodule=CompleteDM(seq,complete['dev'][::24]))
   if tr.global_step!=256:raise RuntimeError('Hard budget stop before256; no automaticrestart')
   info={'steps':tr.global_step,'train_wall_s':time.perf_counter()-st,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'rng':task.trace,'grad_norms':task.grads};traces[arm]=task.trace;(folder/'train_complete.json').write_text(json.dumps(info));print('TRAIN_COMPLETE',folder.name,info['train_wall_s'],flush=True)
   with sdpa_kernel(SDPBackend.MATH):ev=evaluation.evaluate_fp32(task.model,data['dev'],reference,True)
   (folder/'eval.json').write_text(json.dumps(ev));latest=torch.load(folder/'latest.ckpt',map_location='cpu',weights_only=False);assert latest['global_step']==256 and latest['CompleteDM']['loader']['_sampler_iter_state']['samples_yielded']==3072;brief.append({'seed':seed,'arm':arm,'train_wall_s':info['train_wall_s'],'peak_gib':info['peak_gib'],'summary':ev['summary'],'world':ev['world_summary']});(O/'summary.partial.json').write_text(json.dumps(brief,indent=2));assert hashlib.sha256(source.read_bytes()).hexdigest()==source_sha;del task,tr,latest;gc.collect();torch.cuda.empty_cache();print('ARM_COMPLETE',folder.name,flush=True)
  assert traces['causal']==traces['bidirectional'];del reference
 (O/'complete.json').write_text(json.dumps({'done':True,'paired_first3_rng_exact':True,'source_weights_unchanged':True,'wall_s':time.perf_counter()-startall}));print('COMPLETE',flush=True)
if __name__=='__main__':main()
