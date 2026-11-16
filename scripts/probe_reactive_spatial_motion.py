"""Frozen model representation readout, not driving/model training."""
import sys,json,time,hashlib,gc
from pathlib import Path
import numpy as np,torch
from train_carla_miniloop_flow512 import safe_idle,configure_runtime,batch_to
from navigation_scaled_flow import NavigationScaledFlow
from carla_reactive_hold_adapter import actor
R=Path(__file__).resolve().parents[1];OUT=R/'artifacts/reactive-spatial-motion-probe';DATA=R/'data/carla_reactive_hold_cache_v1';AR=R/'data/carla_causal_instance_audit_20261116_v1'
PROJECTION=None
class Data:
 def __init__(self,split,on):
  self.split=split;self.on=on;self.names=json.loads((DATA/'manifest.json').read_text()).get(split,[])
  if split=='audit':self.rows=json.loads((AR/'episode0/frames.json').read_text())
 def __len__(self):return 414 if self.split=='audit' else len(self.names)
 def __getitem__(self,i):
  if self.split=='audit':
   idx=i+36;raw=[]
   for j in [idx-6,idx-4,idx-2,idx]:
    with np.load(AR/f'episode0/{j:05d}.npz') as z:raw.append({k:z[k].copy() for k in ['lidar_xyz','lidar_tag','lidar_world','ego_world']})
   r=actor(raw,self.rows[idx-6:idx+1])
  else:
   with np.load(self.names[i]) as z:r={k:z[k].copy() for k in ['history','state','history_valid','history_age_seconds']}
  if not self.on:r['state'][3]=0
  r['geometry_valid']=r['history']!=255
  return {k:torch.from_numpy(v) for k,v in r.items()}
@torch.no_grad()
def extract(m,split,on):
 ds=Data(split,on);states=[];contexts=[];spatial=[];poison=True
 for start in range(0,len(ds),8):
  b=batch_to([ds[i] for i in range(start,min(start+8,len(ds)))],'cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):
   c=m.context(b).float()
   if start==0:
    # Forbidden extra fields, including labels, are ignored by actual context path.
    q=m.context({**b,'future':torch.full_like(b['history'][:,:2],255),'trajectory':torch.full((len(b['state']),25,5),999.,device='cuda'),'teacher_lead_id':torch.full((len(b['state']),),999,device='cuda')}).float();assert torch.equal(c,q)
  assert c.shape[1:]==(480,128);spatial.extend((c.flatten(1)@PROJECTION).cpu().numpy());contexts.extend(c.reshape(len(b['state']),3,160,128).mean(2).flatten(1).cpu().numpy());states.extend(b['state'][:,[0,1,2,3,4,5,6]].cpu().numpy())
 return {'context':np.array(contexts,dtype=np.float64),'spatial384':np.array(spatial,dtype=np.float64),'state':np.array(states,dtype=np.float64)}
def main():
 global PROJECTION
 safe_idle();configure_runtime();OUT.mkdir(exist_ok=False);start=time.perf_counter();gen=torch.Generator(device='cpu').manual_seed(2160);PROJECTION=(torch.randn(480*128,384,generator=gen)/np.sqrt(480*128)).cuda();labels=json.loads((R/'artifacts/occ-noid-motion-diagnostic/rows.json').read_text());spec={'alpha':1.,'spatial':'flatten fixed token order480x128 then label-independent Gaussian projection61440to384 seed2160; same384 dimensions as pool','fit':'train-only standardized ridge meanMSE+alphaL2, intercept unpenalized','context':'mean160spatialtokens percurrent/pred1/pred2block, concatenate384','readouts':['state','context','context_plus_state'],'target':'past .6s average lead forward motion m/s, same earlier references','limits':'meanpool can lose space; failure is not proof no information; no mainmodel updates; train1episode,held dev and audit share route/eventseed; GT label/eval only','seeds':[2070,2071]};(OUT/'design.json').write_text(json.dumps(spec,indent=2));results={}
 for seed in [2070,2071]:
  for arm in ['base','history']:
   safe_idle();path=R/f'artifacts/carla-reactive-history-paired256/{seed}-{arm}/latest.ckpt';sha=hashlib.sha256(path.read_bytes()).hexdigest();ck=torch.load(path,map_location='cpu',weights_only=False);m=NavigationScaledFlow(50);m.load_state_dict({k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')});del ck;m.cuda().eval().requires_grad_(False);fit=extract(m,'train',arm=='history');fit['context_plus_state']=np.c_[fit['context'],fit['state']];fit['spatial_plus_state']=np.c_[fit['spatial384'],fit['state']];y=np.array([r['gt_past_mean_speed'] for r in labels['train']]);models={}
   for name,x in fit.items():
    mu=x.mean(0);sd=x.std(0).clip(1e-5);z=(x-mu)/sd;ym=y.mean();w=np.linalg.solve(z.T@z/len(z)+np.eye(z.shape[1]),z.T@(y-ym)/len(z));models[name]=(mu,sd,w,ym)
   result={}
   for split in ['train','dev','audit']:
    data=fit if split=='train' else extract(m,split,arm=='history');data['context_plus_state']=np.c_[data['context'],data['state']];data['spatial_plus_state']=np.c_[data['spatial384'],data['state']];truth=np.array([r['gt_past_mean_speed'] for r in labels[split]]);valid=np.array([r['valid'] for r in labels[split]]);preds={'zero':np.zeros(len(truth)),'geometry':np.array([r['displacement_s']/.6 if r['valid'] else np.nan for r in labels[split]])}
    for name,(mu,sd,w,ym) in models.items():preds[name]=(data[name]-mu)/sd@w+ym
    result[split]={}
    for group,lo,hi in [('all',-1e9,1e9),('static',-.05,.05),('slow',.05,.5),('medium',.5,2),('moving',2,1e9)]:
     mask=(truth>=lo)&(truth<hi);result[split][group]={'n':int(mask.sum()),'geometry_common_n':int((mask&valid).sum()),'readouts':{}}
     for name,p in preds.items():
      use=mask&np.isfinite(p);common=use&valid;e=p[use]-truth[use];result[split][group]['readouts'][name]={'n':int(use.sum()),'MAE':float(abs(e).mean()) if len(e) else None,'RMSE':float(np.sqrt((e**2).mean())) if len(e) else None,'common_geometry_MAE':float(abs(p[common]-truth[common]).mean()) if common.any() else None,'pred_mean':float(p[use].mean()) if use.any() else None}
   assert hashlib.sha256(path.read_bytes()).hexdigest()==sha;result['checkpoint_unchanged']=True;results[f'{seed}-{arm}']=result;(OUT/'report.partial.json').write_text(json.dumps(results,indent=2));print('ARM_DONE',seed,arm,flush=True);del m,fit;gc.collect();torch.cuda.empty_cache()
 (OUT/'report.json').write_text(json.dumps({'results':results,'wall_s':time.perf_counter()-start,'future_extra_fields_invariant':True},indent=2));print('DONE',time.perf_counter()-start)
if __name__=='__main__':main()
