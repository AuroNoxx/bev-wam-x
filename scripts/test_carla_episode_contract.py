import unittest
from carla_episode_contract import validate_ticks,event_coverage
class Tests(unittest.TestCase):
 def test_ticks(self):self.assertTrue(validate_ticks([{'frame':1,'timestamp_s':2.},{'frame':2,'timestamp_s':2.05}]))
 def test_gap(self):
  with self.assertRaises(ValueError):validate_ticks([{'frame':1,'timestamp_s':2.},{'frame':3,'timestamp_s':2.05}])
 def test_wrong_delta(self):
  with self.assertRaises(ValueError):validate_ticks([{'frame':1,'timestamp_s':2.},{'frame':2,'timestamp_s':2.1}])
 def test_empty_not_coverage(self):self.assertFalse(event_coverage([],[],True,True,[])['recording_complete'])
 def test_missing_frame(self):self.assertFalse(event_coverage([1,2],[1],True,True,[])['recording_complete'])
 def test_transport_error(self):self.assertFalse(event_coverage([1],[1],True,True,['timeout'])['recording_complete'])
 def test_complete_not_safety(self):
  r=event_coverage([1,2],[1,2],True,True,[]);self.assertTrue(r['recording_complete']);self.assertFalse(r['sensor_semantics_certified']);self.assertIsNone(r['driving_pass'])
if __name__=='__main__':unittest.main()
