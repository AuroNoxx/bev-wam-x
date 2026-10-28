"""Regression coverage for absent labels, cohort denominators and honest reports."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from evaluation.bev_wa_metrics import TrajectoryMetrics, open_loop_gate, render_stage4_report


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.pred = np.zeros((3, 10, 4))
        self.target = np.zeros_like(self.pred)
        self.valid = np.ones((3, 10), bool)
        self.eligible = np.array([True, False, False])

    def metrics(self):
        meter = TrajectoryMetrics()
        meter.update(self.pred, self.target, self.valid, self.eligible)
        return meter.result()

    def test_missing_label_is_not_zero_target_and_endpoint_is_not_last_valid_point(self):
        self.pred[0, :, 0] = .2
        self.pred[1, :, 0] = 2
        self.pred[2, :, 0] = 100
        self.valid[1, 5:] = False
        self.valid[2] = False
        self.target[~self.valid] = np.nan
        result = self.metrics()
        self.assertAlmostEqual(result['val_ade_2s'], .2)
        self.assertAlmostEqual(result['val_fde_2s'], .2)
        self.assertTrue(result['open_loop_gate']['passed'])
        all_valid = result['cohorts']['all_valid_labels']
        self.assertEqual(all_valid['valid_points'], 15)
        self.assertEqual(all_valid['missing_points'], 15)
        self.assertEqual(all_valid['endpoints'], 1)
        self.assertAlmostEqual(all_valid['val_ade_2s'], (2 + 10) / 15)
        self.assertAlmostEqual(all_valid['val_fde_2s'], .2)
        self.assertIsNone(result['cohorts']['ineligible']['val_fde_2s'])
        json.dumps(result, allow_nan=False)

    def test_all_missing_and_no_eligible_stay_missing(self):
        self.valid[:] = False
        self.eligible[:] = False
        result = self.metrics()
        for field in ('val_ade_2s', 'val_fde_2s', 'neg_speed_pct'):
            self.assertIsNone(result[field])
        self.assertFalse(result['open_loop_gate']['passed'])
        json.dumps(result, allow_nan=False)

    def test_nonfinite_valid_prediction_does_not_silently_disappear(self):
        self.pred[0, 0, 0] = np.nan
        result = self.metrics()
        self.assertEqual(result['cohorts']['eligible']['valid_points'], 10)
        self.assertEqual(result['cohorts']['eligible']['nonfinite_points'], 1)
        self.assertIsNone(result['val_ade_2s'])
        self.assertFalse(result['open_loop_gate']['passed'])
        json.dumps(result, allow_nan=False)

    def test_negative_speed_uses_explicit_valid_denominator(self):
        self.pred[0, 0, 2] = -1
        self.pred[1:, :, 2] = -1
        result = self.metrics()
        self.assertEqual(result['neg_speed_pct'], 10)
        self.assertFalse(result['open_loop_gate']['passed'])

    def test_exact_original_thresholds_and_no_rounding(self):
        row = dict(anchors=1, valid_points=10, endpoints=1, val_ade_2s=.25,
                   val_fde_2s=.60, neg_speed_pct=.01)
        self.assertTrue(open_loop_gate(row)['passed'])
        for field in ('val_ade_2s', 'val_fde_2s', 'neg_speed_pct'):
            altered = row | {field: row[field] + 0.00000001}
            self.assertFalse(open_loop_gate(altered)['passed'], field)
        self.assertFalse(open_loop_gate(row | {'valid_points': 9})['passed'])

    def test_streaming_partition_does_not_change_weighting(self):
        self.pred[:, :, 0] = np.arange(3)[:, None]
        self.valid[2, 3:] = False
        expected = self.metrics()
        meter = TrajectoryMetrics()
        for i in range(3):
            meter.update(self.pred[i:i+1], self.target[i:i+1], self.valid[i:i+1], self.eligible[i:i+1])
        self.assertEqual(expected, meter.result())

    def test_report_does_not_claim_driving_or_physical_data_success(self):
        result = {'fair_arms': {'D': {'seed_20261022': self.metrics()}},
                  'comparative_analysis': {'significant': True}, 'gates': {'all': True}}
        report = render_stage4_report(result)
        self.assertIn('0.25', report)
        self.assertIn('0.60', report)
        self.assertIn('0.01%', report)
        self.assertIn('阶段4整体尚未验收', report)
        for false_claim in ('驾驶能力已具备', '100% 行为物理达标', '全部通过', '显著优于', '85.0'):
            self.assertNotIn(false_claim, report)
        absent = render_stage4_report({'fair_arms': {'D': {'seed_20261022': {}}}})
        self.assertIn('缺失', absent)
        self.assertIn('未通过或缺少证据', absent)

    def test_legacy_main_cannot_start_training_without_explicit_mode(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/run_bev_wa_stage4_remediated.py'
        process = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('required', process.stderr)

    def test_evaluate_open_loop_passes_dataset_masks_on_cpu(self):
        import torch
        from scripts.run_bev_wa_stage4_remediated import evaluate_open_loop
        class FakeModel:
            def eval(self):
                return self
            def solve_euler(self, *args, **kwargs):
                return torch.from_numpy(self_pred).float()
        self_pred = self.pred.copy()
        self_pred[0, :, 0] = .125
        self_pred[1:, :, 0] = 100
        self.valid[1:] = False
        batch = {key: torch.zeros(3, 1) for key in ('history_bev','ego_motion','route_points','route_valid','vehicle','maneuver_allowed')}
        batch.update(trajectory=torch.from_numpy(self.target).float(),
                     future_traj_mask=torch.from_numpy(self.valid), action_mask=torch.from_numpy(self.eligible))
        result = evaluate_open_loop(FakeModel(), [batch], torch.device('cpu'))
        self.assertAlmostEqual(result['val_ade_2s'], .125)
        self.assertEqual(result['cohorts']['eligible']['anchors'], 1)
        self.assertTrue(result['open_loop_gate']['passed'])
        self.assertEqual(len(result['sample_targets']), 1)


if __name__ == '__main__':
    unittest.main()
