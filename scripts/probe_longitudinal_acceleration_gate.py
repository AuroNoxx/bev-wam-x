"""CPU-only longitudinal observability probe; no policy/checkpoint updates."""
import json,time,hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
from carla_nav_contract import cache_near
R=Path(__file__).resolve().parents[1]
OUT=R/'artifacts/carla-longitudinal-acceleration-gate'
MAPPING={'carla_miniloop_cache_v1':'carla_causal_miniloop_20261116_town12_r2','carla_startstop_cache_v1':'carla_causal_miniloop_20261116_startstop_v1','carla_recovery_cache_v1':'carla_causal_miniloop_20261116_recovery_v1','carla_recovery_dev_cache_v2':'carla_causal_miniloop_20261116_recovery_dev_v2'}
@lru_cache(None)
def rows(root,ep): return json.loads((R/'data'/root/ep/'frames.json').read_text())
def observed(history,nav):
 assert len(history)==7
 t=np.array([x['timestamp'] for x in history]);v=np.array([np.linalg.norm(x['velocity_world'][:2]) for x in history]);assert np.allclose(np.diff(t),.1,atol=1e-4)
 a=(v[-1]-v[0])/(t[-1]-t[0]);return np.r_[v[-1],nav],a

def extract(names):
 X=[];A=[];Y=[];groups=[];roads=[]
 for name in names:
  p=Path(name);rr=rows(MAPPING[p.parent.parent.name],p.parent.name);i=int(p.stem);nav=cache_near(p)*50;x,a=observed(rr[i-6:i+1],nav)
  with np.load(p) as z: assert abs(x[0]-float(z['state'][0])*30)<1e-5
  y=np.array([np.linalg.norm(rr[i+j]['velocity_world'][:2])-x[0] for j in [2,6,10]])
  group='static' if x[0]<.2 else 'decelerating' if y[1]<-.2 else 'steady' if abs(y[1])<=.2 else 'accelerating'
  X.append(x);A.append(a);Y.append(y);groups.append(group);roads.append(int(rr[i]['road']))
 return dict(x=np.array(X),a=np.array(A),y=np.array(Y),g=np.array(groups),roads=sorted(set(roads)))
def main():
 OUT.mkdir(exist_ok=False);t=time.perf_counter();d=json.loads((R/'artifacts/carla-directstate-paired512/design.json').read_text());names=sorted(set(d['selection']));spec={'alpha':.01,'gate':'feature only: clip((current horizontal speed-.2)/.3,0,1); preset, not dev tuned; does not gate predictor output','regularization':'mean squared error + alpha sum standardized coefficient squared; unpenalized intercept','horizons_seconds':[.2,.6,1.],'train_unique':len(names),'predict':'future horizontal speed minus current horizontal speed, m/s','base':'current horizontal speed, near navigation x/y metres','only_added':'(current speed - speed at t-0.6)/actual elapsed 0.6 seconds','groups':'static current speed<.2; else decelerating delta_v_0.6<-.2; steady abs(delta_v_0.6)<=.2; else accelerating; labels for reporting only','limits':'closed-form diagnostic only, repeated development routes; no expert optimality/safety claim; no future phase/desired speed/time index in features','selection_sha256':hashlib.sha256(json.dumps(names).encode()).hexdigest()};(OUT/'design.json').write_text(json.dumps(spec,indent=2))
 synthetic=[{'timestamp':k*.1,'velocity_world':[2+3*k*.1,0,0]} for k in range(7)];assert abs(observed(synthetic,[1,2])[1]-3)<1e-12
 fit=extract(names);models={}
 for name,x in [('base',fit['x']),('plus_acceleration',np.column_stack([fit['x'],fit['a']])),('gated_acceleration',np.column_stack([fit['x'],fit['a']*np.clip((fit['x'][:,0]-.2)/.3,0,1)]))]:
  mu=x.mean(0);sd=x.std(0).clip(1e-5);z=(x-mu)/sd;ym=fit['y'].mean(0);w=np.linalg.solve(z.T@z/len(z)+.01*np.eye(z.shape[1]),z.T@(fit['y']-ym)/len(z));models[name]=(mu,sd,w,ym)
 result={};dev_roads=set()
 for split,ns in [('fit',names)]+list(d['devs'].items()):
  data=fit if split=='fit' else extract(ns)
  if split!='fit':dev_roads.update(data['roads']);assert not set(data['roads'])&set(fit['roads'])
  preds={'constant_speed':np.zeros_like(data['y']),'constant_acceleration':data['a'][:,None]*np.array([.2,.6,1.])}
  for name,(mu,sd,w,ym) in models.items():
   x=data['x'] if name=='base' else np.column_stack([data['x'],data['a']*(np.clip((data['x'][:,0]-.2)/.3,0,1) if name=='gated_acceleration' else 1)]);preds[name]=(x-mu)/sd@w+ym
  result[split]={'roads':data['roads'],'groups':{}}
  for group in ['all','static','steady','decelerating','accelerating','transition_speed']:
   mask=np.ones(len(ns),bool) if group=='all' else ((data['x'][:,0]>=.2)&(data['x'][:,0]<.5)) if group=='transition_speed' else data['g']==group;y=data['y'][mask]
   result[split]['groups'][group]={'n':int(mask.sum()),'label_mean_delta_v':y.mean(0).tolist() if len(y) else None,'models':{key:{'MAE_mps':np.abs(pred[mask]-y).mean(0).tolist(),'RMSE_mps':np.sqrt(((pred[mask]-y)**2).mean(0)).tolist()} for key,pred in preds.items()} if len(y) else {}}
 # Actor extraction accepts only seven past/current rows. Mutating future cannot affect it.
 p=Path(names[0]);rr=rows(MAPPING[p.parent.parent.name],p.parent.name);i=int(p.stem);nav=cache_near(p)*50;before=observed(rr[i-6:i+1],nav);copy=json.loads(json.dumps(rr));
 for row in copy[i+1:]:row['velocity_world']=[1e8,-1e8,1e8];row['target_speed']=-999
 after=observed(copy[i-6:i+1],nav);assert np.array_equal(before[0],after[0]) and before[1]==after[1]
 (OUT/'report.json').write_text(json.dumps({'metrics':result,'train_roads':fit['roads'],'dev_roads':sorted(dev_roads),'future_poison_exact':True,'synthetic_acceleration_pass':True,'wall_s':time.perf_counter()-t},indent=2));print('DONE',time.perf_counter()-t)
if __name__=='__main__':main()
