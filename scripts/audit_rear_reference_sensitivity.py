"""Historical-geometry sensitivity ONLY. No training/simulator; future data are labels."""
import json,time,numpy as np
from pathlib import Path
from audit_signed_curvature_expressivity import R,rows,matrices,basis,decode,D,stats
O=R/'artifacts/rear-reference-sensitivity-v1';G=R/'logs/occ-closed-loop/20261115-220619-smoke/report.json'
L=2.874995708465576;d=1.4375089406967163;STEER=.6108652381980153

def correction(y,offset):return offset*np.stack([np.cos(y)-1,np.sin(y)],1)
def run(paths,src):
 out={};lat0=[];latrear=[];mask_total=0;rankbad=0;zero=0;maxsteer=0
 for path in paths:
  p=Path(path);rr=rows(src,p.parent.name);i=int(p.stem);r=rr[i:i+51];base=np.array(r[0]['ego_world']);rot=base[:3,:3].T;trans=np.linalg.inv(base)
  yaw=np.unwrap(np.array([-np.arctan2((rot@np.array(x['ego_world'])[:3,:3])[1,0],(rot@np.array(x['ego_world'])[:3,:3])[0,0]) for x in r]));bv=np.array([np.array(x['ego_world'])[:3,:3].T@np.array(x['velocity_world']) for x in r]);vel=bv[:,0];lat=-bv[:,1];wcenter=np.gradient(yaw,.1);lat0.extend(abs(lat));latrear.extend(abs(lat-d*wcenter))
  times=np.arange(51)*.1;vm=(vel[:-1]+vel[1:])/2;w=np.diff(yaw)/.1;mask=abs(vm)>=.5;A=basis(times[:-1]+.05)[mask]*vm[mask,None];rank=np.linalg.matrix_rank(A) if len(A) else 0;rankbad+=rank<6;zero+=not mask.any();mask_total+=int(mask.sum())
  c=np.linalg.lstsq(A,w[mask],rcond=None)[0] if mask.any() else np.zeros(6);a=np.linalg.lstsq(matrices(6,times)[0],vel-vel[0],rcond=None)[0];sv,sy,px=decode(vel[0],a,c);idx=np.arange(20,501,20);actual=np.array([(trans@np.array(x['ego_world']))[:2,3]*[1,-1] for x in r])[2::2];sy=sy[idx];px=px[idx]
  group='static' if max(abs(vel))<.2 else ('stop' if vel[0]>1 and abs(vel[-1])<.2 else ('start' if abs(vel[0])<.2 and vel[-1]>2 else 'other'));v=[]
  for off in [0,d-.1,d,d+.1]:
   err=np.linalg.norm(px+correction(sy,off)-actual,axis=1);v.extend([float(err.mean()),float(err[-1])])
  maxsteer=max(maxsteer,float(np.max(abs(np.arctan(L*c)))));out.setdefault(group,[]).append(v)
 return {'n':len(paths),'groups':{g:{'n':len(a),'errors':{str(off):{'position_mean_error':stats(np.array(a)[:,2*j]),'endpoint_error':stats(np.array(a)[:,2*j+1])} for j,off in enumerate([0,d-.1,d,d+.1])}} for g,a in out.items()},'actor_lateral_abs':stats(lat0),'inferred_rear_lateral_abs':stats(latrear),'curvature_rank_deficient':int(rankbad),'all_low_speed_masked':int(zero),'observed_intervals':mask_total,'interval_total':50*len(paths),'max_inferred_knot_steer_rad':maxsteer,'wheelbase_plusminus5pct_same_curvature_steer_rad':[float(np.arctan((1+q)*np.tan(maxsteer))) for q in [-.05,.05]]}

def main():
 O.mkdir(exist_ok=False);start=time.perf_counter();assert G.exists()
 design={'historical_source':str(G),'episode_geometry_certified':False,'wheelbase_m':L,'actor_to_rear_distance_m':d,'historical_maxsteer_rad':STEER,'offset_sensitivity_m':[-.1,0,.1],'wheelbase_relative_sensitivity':[-.05,.05],'sensitivity_not_confidence_interval':True,'curvature_mask_min_abs_speed_mps':.5,'no_dev_tuning':True,'forward_model':'rear dx=v cos(yaw), dy=v sin(yaw), dyaw=v*k; actor displacement=rear displacement+d*(heading-current heading)','lateral_reference_diagnostic':'v_rear_y=v_actor_y-d*yaw_rate; yaw rate central pose differences is OFFLINE only','low_speed':'unobserved curvature not assigned supervision zero; current masks causal from observed state, label fitting masks offline','training':False}
 (O/'design.json').write_text(json.dumps(design,indent=2));tests={}
 for name,v,k in [('forward',5,.1),('reverse',-2,.1),('static',0,.1)]:
  y=v*k*D;rear=np.stack([np.sin(y)/k,(1-np.cos(y))/k],1);actor=rear+correction(y,d);back=actor-correction(y,d);assert np.allclose(back,rear);_,py,px=decode(v,np.zeros(6),np.ones(6)*k);err=float(abs(px+correction(py,d)-actor).max());assert err<1e-4;tests[name]={'roundtrip_max':float(abs(back-rear).max()),'integrated_max_error':err}
 results={}
 for name,cache,src in [('reactive','carla_reactive_hold_cache_v1','carla_causal_reactive_20261116_hold_v3'),('recovery_train','carla_recovery_cache_v1','carla_causal_miniloop_20261116_recovery_v1'),('recovery_dev','carla_recovery_dev_cache_v2','carla_causal_miniloop_20261116_recovery_dev_v2')]:
  m=json.loads((R/'data'/cache/'manifest.json').read_text())
  for split in ['train','dev']:
   if m.get(split):results[name+'_'+split]=run(m[split],src);print(name,split,flush=True)
 (O/'report.json').write_text(json.dumps({'design':design,'tests':tests,'results':results,'wall_s':time.perf_counter()-start},indent=2));print('DONE',time.perf_counter()-start)
if __name__=='__main__':main()
