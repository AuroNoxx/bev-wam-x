import unittest
from driving_metric_contract import evaluate_episode,paired_comparison,scenario_fingerprint
class ContractTests(unittest.TestCase):
 def ep(self,rows=None):return {'schema':'driving-metrics-v1','samples':rows or [],'coverage':{}}
 def test_unknown_not_zero(self):
  r=evaluate_episode(self.ep());self.assertIsNone(r['safety']['collision_events']['value']);self.assertIsNone(r['feasibility_comfort']['mean_abs_speed_jerk']['value']);self.assertIsNone(r['overall_driving_pass'])
 def test_instrumented_empty_events(self):
  e=self.ep();e['coverage']={'collision':True};self.assertEqual(evaluate_episode(e)['safety']['collision_events']['value'],0)
 def test_event_dedup(self):
  e=self.ep();e['coverage']={'collision':True};v={'id':'sensor1/frame9/contact2','kind':'collision'};e['events']=[v,v.copy()];self.assertEqual(evaluate_episode(e)['safety']['collision_events']['value'],1)
 def test_conflict(self):
  e=self.ep();e['events']=[{'id':'x','kind':'collision'},{'id':'x','kind':'offroad'}]
  with self.assertRaises(ValueError):evaluate_episode(e)
 def test_no_event_identity_unknown(self):
  e=self.ep();e['coverage']={'collision':True};e['events']=[{'kind':'collision'}];self.assertIsNone(evaluate_episode(e)['safety']['collision_events']['value'])
 def test_derivative_units(self):
  r=evaluate_episode(self.ep([{'t_s':i*.1,'speed_mps':2*i*.1} for i in range(4)]));self.assertAlmostEqual(r['feasibility_comfort']['mean_abs_speed_accel']['value'],2);self.assertAlmostEqual(r['feasibility_comfort']['mean_abs_speed_jerk']['value'],0)
 def test_gap_not_bridged(self):
  r=evaluate_episode(self.ep([{'t_s':0,'speed_mps':0},{'t_s':1,'speed_mps':20}]));self.assertIsNone(r['feasibility_comfort']['mean_abs_speed_accel']['value'])
 def test_duplicate_timestamp_rejected(self):
  with self.assertRaises(ValueError):evaluate_episode(self.ep([{'t_s':0},{'t_s':0}]))
 def test_ttc_conditions(self):
  rows=[{'t_s':0,'speed_mps':10,'lead_valid':True,'lead_in_path':True,'bumper_gap_m':20,'closing_speed_mps':5}];r=evaluate_episode(self.ep(rows));self.assertEqual(r['safety']['min_time_headway']['value'],2);self.assertEqual(r['safety']['min_ttc']['value'],4)
 def test_nonclosing_and_stationary_not_zero_ttc(self):
  r=evaluate_episode(self.ep([{'t_s':0,'speed_mps':0,'lead_valid':True,'lead_in_path':True,'bumper_gap_m':20,'closing_speed_mps':-1}]));self.assertIsNone(r['safety']['min_ttc']['value']);self.assertIsNone(r['safety']['min_time_headway']['value'])
 def test_invalid_lead(self):
  r=evaluate_episode(self.ep([{'t_s':0,'speed_mps':20,'bumper_gap_m':2,'closing_speed_mps':10}]));self.assertIsNone(r['safety']['min_ttc']['value'])
 def test_latency_dedup(self):
  e=self.ep();e['planning_events']=[{'id':'1','wall_ms':10}]*4;r=evaluate_episode(e);self.assertEqual(r['latency']['planning_p95']['n'],1)
 def test_pairing_requires_metadata(self):
  with self.assertRaises(ValueError):paired_comparison({}, {})
  with self.assertRaises(ValueError):paired_comparison({'scenario_fingerprint':'a'},{'scenario_fingerprint':'b'})
  self.assertTrue(paired_comparison({'scenario_fingerprint':'a'},{'scenario_fingerprint':'a'})['comparable_scenario'])
 def test_scenario_requires_fields(self):
  with self.assertRaises(ValueError):scenario_fingerprint({'map':'Town12'})
 def test_scenario_seed_changes_signature(self):
  a={k:'fixture' for k in ['simulator_version','map','weather','route','initial_actors','sensor_config','controller_version']};a.update(traffic_seed=1,fixed_delta_seconds=.05)
  self.assertNotEqual(scenario_fingerprint(a),scenario_fingerprint({**a,'traffic_seed':2}))
if __name__=='__main__':unittest.main()
