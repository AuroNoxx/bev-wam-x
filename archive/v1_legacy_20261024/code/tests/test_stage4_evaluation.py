"""Acceptance-critical delayed target alignment and held-out seed checks."""
import copy
import unittest

import numpy as np

from scripts.evaluate_joint_closed_loop import delayed_latent_targets
from scripts.report_stage4 import validate_closed_loop


class EvaluationContractTests(unittest.TestCase):
    def test_delayed_actual_targets_use_exact_steps_and_mask_missing_tail(self):
        records=[dict(step=step,encoded=np.full((2,3),step,np.float32),
            latent=np.full((4,2,3),-step,np.float32)) for step in (0,5,10,20)]
        predicted,target,valid=delayed_latent_targets(records,.02)
        np.testing.assert_array_equal(valid,[[1,1,0,1],[1,0,1,0],[0,1,0,0],[0,0,0,0]])
        np.testing.assert_array_equal(target[0,0],records[1]['encoded'])
        np.testing.assert_array_equal(target[0,3],records[-1]['encoded'])
        self.assertTrue(np.all(target[~valid]==0) and np.all(predicted[1]==-5))
        with self.assertRaises(ValueError):delayed_latent_targets(records[::-1],.02)


    def test_acceptance_rejects_smoke_and_duplicate_or_wrong_seeds(self):
        report=dict(completed=True,split='test',max_steps=None,control_dt_s=.02,
            episodes=[dict(seed=base+i) for base in (309000,319000) for i in range(10)])
        validate_closed_loop(report)
        for update in (dict(max_steps=100),dict(completed=False),dict(control_dt_s=.1),dict(split='val')):
            with self.assertRaises(ValueError):validate_closed_loop(dict(report,**update))
        broken=copy.deepcopy(report);broken['episodes'][-1]['seed']=309000
        with self.assertRaises(ValueError):validate_closed_loop(broken)


if __name__=='__main__':unittest.main()
