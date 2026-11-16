"""Independent CPU boundary-inclusive candidate consistency audit, no model execution."""
import json,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def decode(z):return np.asarray(z,dtype=float)*np.array([50.,10.,1.,1.,30.])
def encode(z):return np.asarray(z,dtype=float)/np.array([50.,10.,1.,1.,30.])
def audit(points,v0,dt=.2):
 p=np.asarray(points,dtype=float);assert p.shape==(25,5) and np.isfinite(p).all();xy=np.vstack(([0.,0.],p[:,:2]));speed=np.r_[v0,p[:,4]];d=p[:,[3,2]];d=d/np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-4);velocity=np.vstack(([v0,0.],d*p[:,4,None]));chord=np.diff(xy,axis=0)/dt;avg=(velocity[:-1]+velocity[1:])/2;res=chord-avg;sa=np.diff(speed)/dt;va=np.diff(velocity,axis=0)/dt;sj=np.diff(sa)/dt;vj=np.diff(va,axis=0)/dt;gs=np.linalg.norm(chord,axis=1);upper=np.maximum(abs(speed[:-1]),abs(speed[1:]));rn=np.linalg.norm(res,axis=1)
 return {'assumption':'t0 position=(0,0),heading=0,velocity=(observed scalar speed,0),as training contract; true lateral velocity unavailable','first_segment':{'signed_speed_accel_mps2':float(sa[0]),'vector_accel_xy_mps2':va[0].tolist(),'position_chord_velocity_xy_mps':chord[0].tolist(),'trapezoid_velocity_xy_mps':avg[0].tolist(),'kinematic_residual_xy_mps':res[0].tolist(),'kinematic_residual_norm_mps':float(rn[0])},'full':{'mean_abs_scalar_accel_mps2':float(abs(sa).mean()),'max_abs_scalar_accel_mps2':float(abs(sa).max()),'max_abs_scalar_accel_segment':int(abs(sa).argmax()),'mean_abs_scalar_jerk_mps3':float(abs(sj).mean()),'max_abs_scalar_jerk_mps3':float(abs(sj).max()),'mean_kinematic_residual_mps':float(rn.mean()),'max_kinematic_residual_mps':float(rn.max()),'max_vector_accel_mps2':float(np.linalg.norm(va,axis=1).max()),'max_vector_jerk_mps3':float(np.linalg.norm(vj,axis=1).max()),'chord_exceeds_endpoint_max_speed_segments':np.where(gs>upper+1e-5)[0].tolist()},'segments':[{'t0_s':round(i*dt,3),'t1_s':round((i+1)*dt,3),'scalar_accel_mps2':float(sa[i]),'kinematic_residual_mps':float(rn[i]),'chord_speed_mps':float(gs[i]),'endpoint_max_speed_mps':float(upper[i])} for i in range(25)]}
class Tests(unittest.TestCase):
 def test_constant(self):
  t=np.arange(1,26)*.2;p=np.column_stack([10*t,t*0,t*0,t*0+1,t*0+10]);r=audit(p,10);self.assertLess(r['full']['max_kinematic_residual_mps'],1e-12);self.assertEqual(r['full']['max_abs_scalar_accel_mps2'],0)
 def test_acceleration(self):
  t=np.arange(1,26)*.2;p=np.column_stack([3*t+t*t,t*0,t*0,t*0+1,3+2*t]);r=audit(p,3);self.assertLess(r['full']['max_kinematic_residual_mps'],1e-12);self.assertAlmostEqual(r['full']['mean_abs_scalar_accel_mps2'],2);self.assertLess(r['full']['max_abs_scalar_jerk_mps3'],1e-12)
 def test_roundtrip(self):
  p=np.random.default_rng(1).normal(size=(25,5));np.testing.assert_allclose(decode(encode(p)),p,atol=1e-14)
 def test_first_jump(self):
  p=np.tile([0,0,0,1,10.],(25,1));r=audit(p,0);self.assertEqual(r['first_segment']['signed_speed_accel_mps2'],50);self.assertEqual(r['full']['max_abs_scalar_accel_segment'],0)
 def test_left_axis(self):
  t=np.arange(1,26)*.2;p=np.column_stack([t*0,5*t,t*0+1,t*0,t*0+5]);r=audit(p,0);self.assertGreater(r['first_segment']['position_chord_velocity_xy_mps'][1],0);self.assertLess(max(x['kinematic_residual_mps'] for x in r['segments'][1:]),1e-12)
def main():
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);result=unittest.TextTestRunner().run(suite);assert result.wasSuccessful()
 source=ROOT/'artifacts/masked-teacher-155m-continue2048/postfit-evidence/report.json';r=json.loads(source.read_text());out={'source':str(source.relative_to(ROOT)),'kind':'prediction-only, no training/checkpoint/test access','unit_contract':'current ego FLU; normalized x50,y10,speed30;sin/cos heading normalized; dt=.2s','tests_passed':result.testsRun,'inputs':[],'limits':['True measured t0 lateral velocity missing: use existing training scalar-forward assumption','Jerk starts between acceleration-interval midpoints .1 and .3s; no claim about prehistory jerk','Scalar derivatives do not equal complete vector comfort','Chord speed greater than endpoint speeds diagnoses incompatibility with endpoint-interpolated velocity, not impossibility under unbounded hidden within-interval acceleration','No collision/TTC/free-space certification; trajectories not executed']}
 for sample in r['samples']:
  out['inputs'].append({'file':sample['validation_file'],'v0_mps':sample['ego_state'][0]*30,'candidates':[{'id':i,**audit(p,sample['ego_state'][0]*30)} for i,p in enumerate(sample['candidate_xy_m_sin_cos_speed_mps'])]})
 path=source.parent/'boundary-inclusive-audit.json';path.write_text(json.dumps(out,indent=2));print(json.dumps({'tests':result.testsRun,'inputs':[{'file':s['file'],'v0':s['v0_mps'],'candidates':[{'id':x['id'],'first':x['first_segment'],'full':x['full']} for x in s['candidates']]} for s in out['inputs']]},indent=2))
if __name__=='__main__':main()
