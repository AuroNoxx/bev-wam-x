import unittest,numpy as np
from learned_policy_bridge_contract import *
class Tests(unittest.TestCase):
 def payload(self):return {'history':np.zeros(SHAPE,dtype=np.uint8),'state':np.zeros(20),'history_valid':np.ones(4,dtype=bool),'history_age_seconds':np.array([.6,.4,.2,0]),'geometry_valid':np.ones(SHAPE,dtype=bool)}
 def test_unknown_is_not_free(self):
  b=self.payload();b['history'][0,0,0,0]=4;r=validate_actor_payload(b);self.assertFalse(r['free_space_certified']);self.assertEqual(r['canonical_history'][0,0,0,0],4);self.assertEqual(r['canonical_history'][-1,0,0,0],255)
 def test_future_field_rejected(self):
  b=self.payload();b['future']=np.zeros(1)
  with self.assertRaises(ValueError):validate_actor_payload(b)
 def test_xyz_rejected(self):
  b=self.payload();b['history']=np.zeros((4,160,64,20),dtype=np.uint8)
  with self.assertRaises(ValueError):validate_actor_payload(b)
 def test_missing_current_rejected(self):
  b=self.payload();b['history_valid'][-1]=False
  with self.assertRaises(ValueError):validate_actor_payload(b)
 def test_invalid_history_is_unknown(self):
  b=self.payload();b['history'][:]=4;b['history_valid'][0]=False;self.assertTrue((validate_actor_payload(b)['canonical_history'][0]==255).all())
 def test_timing(self):np.testing.assert_allclose(validate_timing([1,3,5,7],[1.,1.2,1.4,1.6],1.6),[.6,.4,.2,0])
 def test_future_time_rejected(self):
  with self.assertRaises(ValueError):validate_timing([1,3,5,7],[1.,1.2,1.4,1.6],1.5)
 def test_duplicate_frame_rejected(self):
  with self.assertRaises(ValueError):validate_timing([1,3,3,7],[1.,1.2,1.4,1.6],1.6)
 def test_decode_units_and_unknowns(self):
  p=np.tile([1.,1.,0.,1.,.5],(25,1));r=candidate_to_physical(p);np.testing.assert_allclose(r['trajectory_flu_m_sin_cos_mps'][0],[50,10,0,1,15]);self.assertIsNone(r['collision']);self.assertFalse(r['execution_authorized'])
 def test_negative_speed_rejected(self):
  p=np.tile([0,0,0,1,-.1],(25,1))
  with self.assertRaises(ValueError):candidate_to_physical(p)
 def test_nan_rejected(self):
  p=np.tile([0.,0.,0.,1.,1.],(25,1));p[0,0]=np.nan
  with self.assertRaises(ValueError):candidate_to_physical(p)
if __name__=='__main__':unittest.main()
