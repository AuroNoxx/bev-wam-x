"""Read-only final evaluation of native v1; writes metrics only, no optimizer."""
import os,json,time,argparse,hashlib,psutil
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from b2d_native_occ import ROOT
from native_occ_cached_data import NativeCachedDataset
from occ_qwen_native_flow import NativeOccQwenJointFlow,load_text_backbone
from occ_qwen_performance import enable_channels_last

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='artifacts/occ-qwen-native-training/best.ckpt');a=p.parse_args()
 for proc in psutil.process_iter(['name','cmdline']):
  if proc.info['name'] and proc.info['name'].lower()=='python.exe' and any('train_occ_qwen_native' in s for s in (proc.info['cmdline'] or [])):raise RuntimeError('Training process still alive')
 status=json.loads((ROOT/'artifacts/occ-qwen-native-training/status.json').read_text());assert status['phase']=='completed' and status['global_step']==13240
 torch.set_num_threads(4);folder=ROOT/'artifacts/occ-qwen-native-final-audit';folder.mkdir(exist_ok=True)
 cp=torch.load(ROOT/a.checkpoint,map_location='cpu',weights_only=False);step=cp['global_step'];epoch=cp['epoch']+1
 model=NativeOccQwenJointFlow(load_text_backbone(ROOT/'models/Qwen3-VL-2B-Instruct'))
 model.load_state_dict({k.removeprefix('model.'):v for k,v in cp['state_dict'].items()},strict=True)
 perf=cp['task_state']['performance'];assert perf['channels_last'];enable_channels_last(model);model.cuda().eval();del cp
 ds=NativeCachedDataset(perf['native_aligned_cache'],'validation');assert len(ds)==1124
 loader=DataLoader(ds,batch_size=8,shuffle=False,num_workers=0);iterator=iter(loader)
 torch.manual_seed(1907);started=time.monotonic();n=0;total={};world={};jerk=[];scalarjerk=[];posjerk=[]
 def add(k,x):total[k]=total.get(k,0.)+float(x.double().sum())
 with torch.inference_mode():
  for i,b in enumerate(iterator):
   b={k:v.cuda() for k,v in b.items()};bs=len(b['state'])
   with torch.autocast('cuda',dtype=torch.bfloat16):
    loss,_=model.loss(**b) # preserve production validation RNG consumption before sample
    out=model.sample(b['history'],b['state'],steps=8)
    target_world=model.world_target(b['future'])
    now=model.teacher.frames(b['history'][:,-1:]).float().expand(-1,2,-1,-1).flatten(1,2)
   pred=out['trajectory_xy'].float();target=b['trajectory'][...,:2]*pred.new_tensor([50.,10.]);dist=(pred-target).norm(dim=-1)
   add('ade_sum',dist.mean(1));add('loss_sum',loss*bs)
   for sec,ix in [(1,4),(2,9),(3,14),(5,24)]:add('endpoint_'+str(sec)+'s_sum',dist[:,ix])
   velocity=out['residual_velocity'].float()*10+torch.stack([b['state'][:,0]*30,torch.zeros(bs,device='cuda')],-1)[:,None]
   speed=velocity.norm(dim=-1);add('speed_mae_sum',(speed-b['trajectory'][...,4]*30).abs().mean(1))
   vpos=torch.diff(torch.cat([torch.zeros_like(pred[:,:1]),pred],1),dim=1)/.2
   jpos=torch.diff(vpos,n=2,dim=1)/(.2**2);jvel=torch.diff(velocity,n=2,dim=1)/(.2**2);jsp=torch.diff(speed,n=2,dim=1)/(.2**2)
   posjerk.append(jpos.norm(dim=-1).cpu());jerk.append(jvel.norm(dim=-1).cpu());scalarjerk.append(jsp.abs().cpu())
   known=((b['future']>0)&(b['future']<8)).float().mean(2);mask=F.adaptive_avg_pool2d(known.flatten(0,1)[:,None],(4,4)).reshape(bs,2,16)>0
   for name,latent in [('prediction',out['world_latent']),('persistence',now)]:
    err=(latent.float()-target_world).square().mean(-1).reshape(bs,2,16)
    for h in range(2):world[name+str(h)]=world.get(name+str(h),0.)+float((err[:,h]*mask[:,h]).sum())
   for h in range(2):world['count'+str(h)]=world.get('count'+str(h),0)+int(mask[:,h].sum())
   n+=bs
   if i%20==0:print(json.dumps({'evaluated':n,'total':1124,'seconds':time.monotonic()-started}),flush=True)
 def jm(parts):
  x=torch.cat(parts);return {'mean_mps3':float(x.mean()),'p95_mps3':float(torch.quantile(x.flatten(),.95)),'max_mps3':float(x.max()),'point_fraction_over_10':float((x>10).float().mean()),'trajectory_fraction_any_over_10':float((x>10).any(1).float().mean())}
 result={'checkpoint':a.checkpoint,'epoch':epoch,'step':step,'samples':n,'split':'validation only; Town13; test untouched','candidate_count':1,'euler_steps':8,'seed':1907,'batch_size':8,'trajectory_dt_s':.2,'same_rng_consumption_as_training_validation':True,'ade_m':total['ade_sum']/n,'endpoint_error_m':{str(s):total['endpoint_'+str(s)+'s_sum']/n for s in [1,2,3,5]},'speed_mae_mps':total['speed_mae_sum']/n,'validation_loss':total['loss_sum']/n,'jerk_from_positions':jm(posjerk),'jerk_from_velocity_vectors':jm(jerk),'jerk_from_scalar_speed':jm(scalarjerk),'world_mse':{name:{str(h+1)+'s':world[name+str(h)]/max(1,world['count'+str(h)]) for h in range(2)} for name in ['prediction','persistence']},'world_contract':'same checkpoint EMA encoder, current ego frame, original latent scale; occupied-support mask pooled4x4, mean64 features; persistence repeats current-frame latent; no future actor input','seconds':time.monotonic()-started,'peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9,'weights_or_source_data_modified':False}
 (folder/'best-evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
