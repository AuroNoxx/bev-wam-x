import tempfile
import unittest
from pathlib import Path
import numpy as np
from planning.candidates import (score_candidates,static_cost,actor_tracks,dynamic_cost,
    kinematic_reasons,emergency_trajectory)
from configs.planner7 import Planner7Config


def road():
    bev=np.zeros((4,128,128),np.uint8);bev[3]=255;bev[0,:,56:72]=255
    return bev


def straight():
    return np.column_stack([np.arange(1,11)*1.4,np.zeros(10),np.ones(10)*7,np.zeros(10)]).astype(np.float32)


class PlanningTests(unittest.TestCase):
    def test_footprint_detects_boundary_and_raster_exit(self):
        bev=road();path=straight();self.assertEqual(static_cost(bev,path,4.5,1.9),0.)
        path[:,1]=5.;self.assertGreater(static_cost(bev,path,4.5,1.9),.5)
        path[:,1]=50.;self.assertEqual(static_cost(bev,path,4.5,1.9),1.)

    def test_ego_motion_does_not_appear_as_actor_motion(self):
        a=road();b=road();a[2,42:46,62:66]=255;b[2,44:48,62:66]=255
        tracks=actor_tracks([a,b],[np.zeros(6),np.array([1.,0,0,0,0,0])])
        np.testing.assert_allclose(tracks[1],0,atol=1e-6);self.assertTrue(tracks[3][0])
        self.assertGreater(dynamic_cost(straight(),tracks,4.5,1.9,Planner7Config()),0.)

    def test_fixed_tie_break_and_explicit_all_rejected_fallback(self):
        bev=road();state=np.zeros(6);path=straight()
        result=score_candidates(np.stack([path,path]),bev,[bev],[state],4.5,1.9)
        self.assertEqual(result['selected_index'],0)
        invalid=path.copy();invalid[:,2]=20
        result=score_candidates(np.stack([invalid]),bev,[bev],[state],4.5,1.9)
        self.assertEqual(result['selected_index'],-1)
        stop=emergency_trajectory(7.)
        self.assertTrue(np.all(np.diff(stop[:,2])<=0));self.assertGreater(stop[-1,0],0)
        self.assertFalse(kinematic_reasons(stop))

    def test_risk_disabled_is_missing_and_enabled_threshold_rejects(self):
        bev=road();state=np.zeros(6);bank=np.stack([straight(),straight()])
        result=score_candidates(bank,bev,[bev],[state],4.5,1.9,
            np.array([[.9,.01],[.1,.01]]),[.5,.5])
        self.assertEqual(result['selected_index'],1)
        self.assertIn('crash_threshold',result['rejection_reasons'][0])
        # Frozen protocol prunes ABOVE validation thresholds, not at equality.
        equal=score_candidates(bank[:1],bev,[bev],[state],4.5,1.9,np.array([[.5,.5]]),[.5,.5])
        self.assertEqual(equal['selected_index'],0)

    def test_public_logger_records_unobserved_counterfactual_without_fake_target(self):
        from mini_wa_jepa.visualization import RerunLogger
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'planning.rrd'
            with RerunLogger(spawn=False,save_path=path) as log:
                log.set_step(0,0.);log.log_bev(road())
                log.log_planning(candidates=np.stack([straight()]*8),selected_id='STOP',
                    selected_trajectory=emergency_trajectory(7.),selection_source='rule_emergency_stop_all_rejected',
                    predicted=np.zeros((8,4,64,256)),scores=np.ones(8),terms=np.ones((8,8)),
                    rejection_reasons=[['speed']]*8,executed_action=[0,-1])
            self.assertGreater(path.stat().st_size,1000)


if __name__=='__main__':unittest.main()
