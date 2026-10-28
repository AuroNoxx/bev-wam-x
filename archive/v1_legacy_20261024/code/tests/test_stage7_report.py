"""Acceptance must distinguish trajectory diversity from physical consistency."""
import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
from scripts.report_stage7 import kinematics,compare_repeated_predictions,resources
from scripts.recover_stage7_motion import observed_motion


class GateTests(unittest.TestCase):
    def test_missing_early_pss_is_counted_and_never_replaced_by_rss(self):
        rows=[dict(timestamp=1,available_ram_bytes=50,gpu=[],processes=[dict(rss_bytes=999)]),
              dict(timestamp=2,available_ram_bytes=40,gpu=[],processes=[dict(rss_bytes=11,pss_bytes=9)]),
              dict(timestamp=3,available_ram_bytes=60,gpu=[],processes=[])]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'resources.jsonl';p.write_text('\n'.join(json.dumps(r) for r in rows))
            actual=resources(p)
        self.assertEqual(actual['maximum_sum_worker_pss_bytes'],9)
        self.assertEqual(actual['missing_pss_process_records'],1)
        self.assertEqual(actual['complete_pss_samples'],2)
        self.assertEqual(actual['minimum_available_ram_bytes'],40)

    def test_repeated_predictions_are_strict_but_elapsed_measurements_can_change(self):
        old=dict(candidates=np.ones((8,10,4)),fm_latencies_ms=np.array([6.,9.]))
        new=dict(candidates=old['candidates'].copy(),fm_latencies_ms=np.array([7.,6.]))
        prediction,timing=compare_repeated_predictions(new,old)
        self.assertEqual(prediction['candidates'],0.)
        self.assertEqual(timing['fm_latencies_ms'],3.)
        new['candidates'][0,0,0]+=.1
        with self.assertRaises(AssertionError):compare_repeated_predictions(new,old)

    def test_recovered_motion_uses_observed_speed_and_wrapped_backward_heading(self):
        states=np.zeros((5,6),np.float32);states[:,3]=np.arange(5)+4
        states[:,2]=[3.10,3.12,-3.14,-3.12,-3.10]
        motion=observed_motion(states)
        np.testing.assert_allclose(motion[:,0],[.5,.6,.7,.8])
        self.assertTrue(np.all(np.abs(motion[:,1])<.3))
        self.assertTrue(np.all(motion[:,1]>0))
        with self.assertRaises(ValueError):observed_motion(states[-4:])

    def test_heading_mismatch_cannot_pass_with_plausible_speed(self):
        a=np.zeros((1,8,10,4));a[...,0]=np.arange(1,11)*.2;a[...,2]=1
        self.assertEqual(kinematics(a)['pointwise_valid_fraction'],1)
        a[...,3]=np.pi/2
        self.assertAlmostEqual(kinematics(a)['pointwise_valid_fraction'],.1)
        self.assertEqual(kinematics(a)['all_points_valid_candidate_fraction'],0)

    def test_acceleration_speed_and_nonfinite_checks_are_joint(self):
        a=np.zeros((1,1,10,4));a[...,0]=np.arange(1,11)*.2;a[...,2]=1
        a[0,0,4,2]=4  # two adjacent accelerations fail
        a[0,0,7,2]=-1  # invalid speed and two adjacent accelerations
        a[0,0,0,0]=np.nan
        self.assertLess(kinematics(a)['pointwise_valid_fraction'],.8)


if __name__=='__main__':unittest.main()
