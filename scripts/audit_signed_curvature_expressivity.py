"""Offline signed speed / curvature oracle-label expressivity. No training or simulator."""
import numpy as np,json,time
from pathlib import Path
from audit_continuous_action_expressivity import R,rows,matrices,stats
O=R/'artifacts/signed-curvature-expressivity-v1';D=np.linspace(0,5,501);T=np.arange(1,26)*.2

def basis(t,k=6):
 knots=np.linspace(0,5,k);return np.stack([np.interp(t,knots,np.eye(k)[j]) for j in range(k)],1)
def decode(v0,acc,curv):
 V,P=matrices(len(acc),D);v=v0+V@acc;k=basis(D,len(curv))@curv;dy=v*k; yaw=np.r_[0,np.cumsum((dy[1:]+dy[:-1])*.005)];vv=v[:,None]*np.stack([np.cos(yaw),np.sin(yaw)],1);xy=np.vstack((np.zeros(2),np.cumsum((vv[1:]+vv[:-1])*.005,axis=0)));return v,yaw,xy

def run(paths,src):
 records=[];maskcounts=[];lateral=[];reverse=0
 for path in paths:
  p=Path(path);rr=rows(src,p.parent.name);i=int(p.stem);r=rr[i:i+51];base=np.array(r[0]['ego_world']);rot=base[:3,:3].T
  yaw=np.unwrap(np.array([-np.arctan2((rot@np.array(x['ego_world'])[:3,:3])[1,0],(rot@np.array(x['ego_world'])[:3,:3])[0,0]) for x in r]));vel=[];lat=[]
  for x in r:
   q=np.array(x['ego_world'])[:3,:3].T@np.array(x['velocity_world']);vel.append(q[0]);lat.append(q[1])
  vel=np.array(vel);lateral.extend(abs(np.array(lat)));reverse+=int(vel.min()<-.05);times=np.arange(51)*.1
  acc=np.linalg.lstsq(matrices(6,times)[0],vel-vel[0],rcond=None)[0]
  # Interval yaw rate and midpoint speed; near-zero undefined, never treated as curvature zero target.
  vm=(vel[:-1]+vel[1:])/2;w=np.diff(yaw)/.1;mask=abs(vm)>=.5;B=basis(times[:-1]+.05)
  # Fit observable yaw rate v*k, rather than dividing by noisy small speed.
  A=B[mask]*vm[mask,None];c=np.linalg.lstsq(A,w[mask],rcond=None)[0] if mask.any() else np.zeros(6)
  sv,sy,sxy=decode(vel[0],acc,c);ix=np.arange(20,501,20);actual=np.array([(np.linalg.inv(base)@np.array(x['ego_world']))[:2,3]*[1,-1] for x in r])[::2][1:]
  group='static' if max(abs(vel))<.2 else ('stop' if vel[0]>1 and abs(vel[-1])<.2 else ('start' if abs(vel[0])<.2 and vel[-1]>2 else 'other'))
  records.append({'group':group,'speed_mae':float(abs(sv[ix]-vel[::2][1:]).mean()),'position_mae':float(np.linalg.norm(sxy[ix]-actual,axis=1).mean()),'endpoint_error':float(np.linalg.norm(sxy[-1]-actual[-1])),'yaw_mae_rad':float(abs(sy[ix]-yaw[::2][1:]).mean()),'curvature_abs_peak':float(max(abs(c))),'curvature_fit_rank':int(np.linalg.matrix_rank(A)) if len(A) else 0});maskcounts.append(int(mask.sum()))
 groups={}
 for g in sorted(set(x['group'] for x in records)):
  a=[x for x in records if x['group']==g];groups[g]={'n':len(a),**{key:stats([x[key] for x in a]) for key in ['speed_mae','position_mae','endpoint_error','yaw_mae_rad','curvature_abs_peak']}}
 return {'n':len(paths),'groups':groups,'lateral_body_velocity_mps':stats(lateral),'windows_negative_signed_speed':reverse,'curvature_observed_intervals':stats(maskcounts),'rank_deficient_windows':sum(x['curvature_fit_rank']<6 for x in records)}

def main():
 O.mkdir(exist_ok=False);start=time.perf_counter();design={'knots':6,'dt_integration_s':.01,'curvature_observable_min_abs_speed_mps':.5,'preset_not_dev_tuned':True,'fit':'true current signed velocity; future speed/yaw labels only offline coefficient encoder','origin':'recorded actor transform, not verified rear axle','wheelbase':'not recorded/verified; cannot map curvature to physical steer','coordinate':'CARLA x forward y right -> FLU x forward y left, yaw negated','unknown_curvature':'zero default if all masked is arbitrary, not observed truth','training':False};(O/'design.json').write_text(json.dumps(design,indent=2))
 tests={}
 for name,v,k in [('rest',0,.2),('forward',5,.1),('reverse',-2,.1)]:
  sv,y,xy=decode(v,np.zeros(6),np.ones(6)*k);truth=np.stack([np.sin(v*k*D)/k,(1-np.cos(v*k*D))/k],1);err=float(abs(xy-truth).max());assert err<1e-4;assert np.allclose(y,v*k*D);tests[name]={'max_position_error':err,'final_yaw':float(y[-1]),'no_sideways_or_yaw_at_rest':bool(v!=0 or (np.all(xy==0) and np.all(y==0)))}
 results={}
 for name,cache,src in [('reactive','carla_reactive_hold_cache_v1','carla_causal_reactive_20261116_hold_v3'),('recovery_train','carla_recovery_cache_v1','carla_causal_miniloop_20261116_recovery_v1'),('recovery_dev','carla_recovery_dev_cache_v2','carla_causal_miniloop_20261116_recovery_dev_v2')]:
  m=json.loads((R/'data'/cache/'manifest.json').read_text())
  for split in ['train','dev']:
   if m.get(split):results[name+'_'+split]=run(m[split],src);print(name,split,flush=True)
 (O/'report.json').write_text(json.dumps({'design':design,'tests':tests,'results':results,'wall_s':time.perf_counter()-start},indent=2));print('DONE',time.perf_counter()-start)
if __name__=='__main__':main()
