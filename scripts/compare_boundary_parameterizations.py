"""Isolated boundary-aware output parameterization feasibility; no weight updates."""
import json,unittest
from pathlib import Path
import numpy as np
from audit_candidate_boundary import audit
ROOT=Path(__file__).resolve().parents[1]
CONFIGS={'integrate_only':{},'accel_yaw':{'amin':-4.,'amax':2.,'yawrate':.3},'accel_yaw_jerk':{'amin':-4.,'amax':2.,'yawrate':.3,'jerk':5.}}
def project(points,v0,cfg):
 p=np.asarray(points,float);desired_v=p[:,4];desired_h=np.unwrap(np.arctan2(p[:,2],p[:,3]));v=max(0.,v0);h=0.;a=0.;xy=np.zeros(2);out=[];infeasible=0
 for dv,dh in zip(desired_v,desired_h):
  previous=np.array([np.cos(h),np.sin(h)])*v
  if 'amin' in cfg:
   lo=max(cfg['amin'],-v/.2);hi=cfg['amax']
   if 'jerk' in cfg:lo=max(lo,a-cfg['jerk']*.2);hi=min(hi,a+cfg['jerk']*.2)
   if lo>hi:
    infeasible+=1;raise ValueError('Constraints infeasible: do not silently clip/safety claim')
   a=float(np.clip((dv-v)/.2,lo,hi));v=v+a*.2
   delta=(dh-h+np.pi)%(2*np.pi)-np.pi;h+=float(np.clip(delta,-cfg['yawrate']*.2,cfg['yawrate']*.2))
  else:v=max(0.,dv);h=dh
  now=np.array([np.cos(h),np.sin(h)])*v;xy=xy+(previous+now)*.1;out.append([*xy,np.sin(h),np.cos(h),v])
 return np.asarray(out)
class Tests(unittest.TestCase):
 def test_uniform(self):
  t=np.arange(1,26)*.2;p=np.column_stack([10*t,t*0,t*0,t*0+1,t*0+10]);np.testing.assert_allclose(project(p,10,CONFIGS['accel_yaw_jerk']),p,atol=1e-12)
 def test_jump(self):
  p=np.tile([0,0,0,1,20.],(25,1));z=project(p,1,CONFIGS['accel_yaw']);r=audit(z,1);self.assertLessEqual(r['full']['max_abs_scalar_accel_mps2'],2+1e-9);self.assertLess(r['full']['max_kinematic_residual_mps'],1e-10)
 def test_jerk(self):
  p=np.tile([0,0,0,1,20.],(25,1));z=project(p,1,CONFIGS['accel_yaw_jerk']);r=audit(z,1);self.assertLessEqual(r['full']['max_abs_scalar_jerk_mps3'],5+1e-9)
 def test_stop(self):
  p=np.tile([0,0,0,1,0.],(25,1));z=project(p,20,CONFIGS['accel_yaw']);self.assertAlmostEqual(z[-1,0],50,places=8);self.assertLess(abs(z[-1,4]),1e-8)
 def test_boundary_gap(self):
  p=np.tile([0,0,0,1,20.],(25,1));z=project(p,1,CONFIGS['accel_yaw_jerk']);self.assertAlmostEqual((z[0,4]-1)/.2,1)
def main():
 result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));assert result.wasSuccessful()
 src=ROOT/'artifacts/masked-teacher-155m-continue2048/postfit-evidence/report.json';source=json.loads(src.read_text());out=ROOT/'artifacts/teacher-boundary-parameterization-comparison';out.mkdir(exist_ok=False);records=[]
 for si,s in enumerate(source['samples']):
  v0=s['ego_state'][0]*30
  for ci,p in enumerate(s['candidate_xy_m_sin_cos_speed_mps']):
   p=np.asarray(p);variants={'original':p,**{name:project(p,v0,c) for name,c in CONFIGS.items()}}
   for name,z in variants.items():
    q=audit(z,v0);records.append({'input':si,'candidate':ci,'variant':name,'first':q['first_segment'],'metrics':q['full'],'position_change_vs_original_mean_m':float(np.linalg.norm(z[:,:2]-p[:,:2],axis=1).mean()),'position_change_vs_original_final_m':float(np.linalg.norm(z[-1,:2]-p[-1,:2])),'forward_displacement_proxy_m':float(z[-1,0]),'max_abs_lateral_position_m':float(abs(z[:,1]).max()),'collision':None,'route_progress':None,'points':z.tolist()})
 stop=np.tile([0,0,0,1,0.],(25,1));counter={}
 for name,c in CONFIGS.items():
  z=project(stop,20,c);counter[name]={'forward_m':float(z[-1,0]),'end_speed_mps':float(z[-1,4]),'note':'unobserved synthetic obstacle at40m would be beyond reach of a collision-free stop for some bounded profiles; no simulator contact claim'}
 report={'scope':'Eight same predicted candidates, deterministic postprocessing only; not new trained model, no labels/test/checkpoint reads','configs':CONFIGS,'initial_accel_for_jerk':0,'initial_accel_note':'explicit assumed0 for constraint demonstration; observed acceleration not used and can differ','synthetic_tests':result.testsRun,'records':records,'braking_counterexample_20mps_target0':counter,'limits':['Bounds are illustrative, not vehicle-calibrated safety/comfort limits','Postprocessing can displace path into obstacles and change intent','Forward displacement is not route progress or efficiency certification','Exact kinematic consistency is by construction, not learned competence','Yaw-rate bound alone does not bound lateral acceleration/curvature','No future truth used for online input; no training updates']}
 (out/'report.json').write_text(json.dumps(report,indent=2));summary={}
 for v in ['original',*CONFIGS]:
  rows=[r for r in records if r['variant']==v];summary[v]={'mean_residual_mps':float(np.mean([r['metrics']['mean_kinematic_residual_mps'] for r in rows])),'largest_abs_scalar_accel_mps2':max(r['metrics']['max_abs_scalar_accel_mps2'] for r in rows),'largest_abs_scalar_jerk_mps3':max(r['metrics']['max_abs_scalar_jerk_mps3'] for r in rows),'mean_position_change_m':float(np.mean([r['position_change_vs_original_mean_m'] for r in rows])),'max_final_position_change_m':max(r['position_change_vs_original_final_m'] for r in rows),'mean_forward_proxy_m':float(np.mean([r['forward_displacement_proxy_m'] for r in rows]))}
 (out/'summary.json').write_text(json.dumps({'summary':summary,'counterexample':counter},indent=2));print(json.dumps({'summary':summary,'counterexample':counter},indent=2))
if __name__=='__main__':main()
