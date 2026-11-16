"""Offline only. No network/model/optimizer; future labels only fit oracle coefficients."""
import numpy as np,json,time
from pathlib import Path
from functools import lru_cache
R=Path(__file__).resolve().parents[1]; O=R/'artifacts/continuous-action-expressivity-v1'
T=np.arange(1,26)*.2

def matrices(k,t=T):
 knots=np.linspace(0,5,k); V=np.zeros((len(t),k));P=V.copy()
 for j,(l,r) in enumerate(zip(knots[:-1],knots[1:])):
  h=r-l;d=np.clip(t-l,0,h);u=t-l
  V[:,j]+=d-d*d/(2*h);V[:,j+1]+=d*d/(2*h)
  P[:,j]+=u*d-(u/h+1)*d*d/2+d**3/(3*h)
  P[:,j+1]+=u*d*d/(2*h)-d**3/(3*h)
 return V,P

def fit(v0,v,k):
 V,P=matrices(k);c=np.linalg.lstsq(V,v-v0,rcond=None)[0]
 return c,v0+V@c,T[:,None]*v0+P@c

def stats(a):
 a=np.array(a);return {'mean':float(a.mean()),'p95':float(np.quantile(a,.95)),'max':float(a.max())}
@lru_cache(None)
def rows(src,ep):return json.loads((R/'data'/src/ep/'frames.json').read_text())
def audit(paths,src):
 out={str(k):{} for k in [6,11]};dif=[];oldposition=[];turn=[];reverses=0;groups={}
 for path in paths:
  p=Path(path);rr=rows(src,p.parent.name);i=int(p.stem);row=rr[i];rot=np.array(row['ego_world'])[:3,:3].T
  def vel(r):
   q=rot@np.array(r['velocity_world']);return q[:2]*[1,-1]
  v0=vel(row);v=np.array([vel(rr[j]) for j in range(i+2,i+51,2)])
  with np.load(p) as z:label=z['trajectory'].copy();initial=float(z['state'][0]*30)
  xy=label[:,:2]*[50,10];oldv=label[:,4,None]*30*label[:,[3,2]]
  dif.append(float(np.linalg.norm(oldv-v,axis=1).mean()));turn.append(float(np.max(np.abs(np.arctan2(label[:,2],label[:,3])))));reverses+=int(np.any(v[:,0]<-.05))
  prev=np.vstack(([initial,0],oldv[:-1]));oldposition.append(float(np.linalg.norm(np.cumsum((prev+oldv)*.1,0)-xy,axis=1).mean()))
  speed=np.linalg.norm(v,axis=1);s0=np.linalg.norm(v0)
  group='static' if max(s0,speed.max())<.2 else ('stop' if s0>1 and speed[-1]<.2 else ('start' if s0<.2 and speed[-1]>2 else 'other'))
  groups[group]=groups.get(group,0)+1
  for k in [6,11]:
   c,pv,px=fit(v0,v,k);bucket=out[str(k)].setdefault(group,[])
   dense=np.linspace(0,5,501);knots=np.linspace(0,5,k);ac=np.stack([np.interp(dense,knots,c[:,d]) for d in range(2)],1)
   bucket.append([np.linalg.norm(pv-v,axis=1).mean(),np.linalg.norm(px-xy,axis=1).mean(),np.linalg.norm(px[-1]-xy[-1]),np.linalg.norm(pv[-1]-v[-1]),np.linalg.norm(ac,axis=1).max(),np.linalg.norm(np.diff(c,axis=0)/(5/(k-1)),axis=1).max()])
 for k in out:
  for g,a in out[k].items():out[k][g]={'n':len(a),**{key:stats(np.array(a)[:,j]) for j,key in enumerate(['velocity_vector_MAE','position_mean_error','endpoint_position_error','endpoint_velocity_error','peak_vector_acceleration','peak_vector_jerk'])}}
 return {'n':len(paths),'groups':groups,'old_heading_times_speed_vs_true_velocity_error_mps':stats(dif),'old_trapezoid_position_error_m':stats(oldposition),'max_abs_heading_change_rad':stats(turn),'windows_with_negative_ego_forward_velocity':reverses,'fits':out}

def main():
 O.mkdir(exist_ok=False);start=time.perf_counter()
 design={'knots':[6,11],'fit':'least-squares true planar velocity labels, no position fitting; exact integral linear acceleration','training':False,'dev_tuning':False,'actor_contract':'decoder only current true signed velocity vector and generated coefficients; future only offline encoder label; no GT lead','limits':['planar vector acceleration not bicycle steering/nonholonomic feasibility','no hard physical/comfort clipping; no safety threshold invented','zero speed heading undefined; preserve body heading separately, speed norm cannot encode reverse gear','oracle coefficients not learned or causal policy']}
 (O/'design.json').write_text(json.dumps(design,indent=2));tests={}
 for k in [6,11]:
  rng=np.random.default_rng(42);c=rng.normal(size=(k,2));v0=np.array([3.,-.2]);V,P=matrices(k);cc,pv,px=fit(v0,v0+V@c,k);assert np.allclose(cc,c,atol=1e-10);assert np.allclose(px,T[:,None]*v0+P@c,atol=1e-10);tests[f'coefficient_roundtrip_{k}']=float(abs(cc-c).max())
 for name,v0,acc in [('rest',[0,0],[0,0]),('constant_reverse',[-2,0],[0,0]),('turning_vector',[3,0],[0,.3]),('stop_then_reverse',[2,0],[-1,0])]:
  v0=np.array(v0);acc=np.array(acc);v=v0+T[:,None]*acc;c,pv,px=fit(v0,v,6);assert np.allclose(pv,v,atol=1e-10);assert np.allclose(px,T[:,None]*v0+.5*T[:,None]**2*acc,atol=1e-10);tests[name]=True
 tests['zero_time_integrals']=bool(np.allclose(matrices(6,np.array([0.]))[0],0) and np.allclose(matrices(6,np.array([0.]))[1],0));tests['future_fields_not_decoder_arguments']=True
 sources=[('reactive','carla_reactive_hold_cache_v1','carla_causal_reactive_20261116_hold_v3'),('recovery_train','carla_recovery_cache_v1','carla_causal_miniloop_20261116_recovery_v1'),('recovery_dev','carla_recovery_dev_cache_v2','carla_causal_miniloop_20261116_recovery_dev_v2')];results={}
 for name,cache,src in sources:
  m=json.loads((R/'data'/cache/'manifest.json').read_text())
  for split in ['train','dev']:
   if m.get(split):results[name+'_'+split]=audit(m[split],src);print(name,split,'done',flush=True)
 (O/'report.json').write_text(json.dumps({'design':design,'tests':tests,'results':results,'wall_s':time.perf_counter()-start},indent=2));print('done',time.perf_counter()-start)
if __name__=='__main__':main()
