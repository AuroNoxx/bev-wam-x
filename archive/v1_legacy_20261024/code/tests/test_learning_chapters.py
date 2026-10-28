"""教学实验的数学边界与数据语义；浏览器联动另有实际事件检查。"""
import unittest

import numpy as np

from chapters.chapter0_hardware import ui as hardware
from chapters.chapter1_bev import ui as bev
from chapters.chapter1_bev.src.semantic_bev import ego_to_pixels, pixels_to_ego
from chapters.chapter2_baseline import ui as baseline
from chapters.chapter3_latent_vicreg import ui as vicreg
from chapters.chapter4_joint_world_action import ui as joint
from chapters.chapter5_stress_testing import ui as stress


class LearningExperimentsTests(unittest.TestCase):
    def test_attention_quadratic_scaling_and_accumulation_independence(self):
        values = hardware.make_controls().value
        first = hardware.compute(values)
        larger = hardware.compute(dict(values, tokens=values['tokens'] * 4, accum=4))
        self.assertAlmostEqual(larger['score_mib'], first['score_mib'] * 16)
        self.assertEqual(larger['effective_batch'], first['effective_batch'] * 4)

    def test_calibrated_pixel_center_and_forward_left_signs(self):
        np.testing.assert_allclose(ego_to_pixels([[0, 0], [10, 5]]), [[63.5, 63.5], [53.5, 43.5]])
        points = np.array([[12.25, -3.75], [-20, 10]])
        np.testing.assert_allclose(pixels_to_ego(ego_to_pixels(points)), points)

    def test_fixed_horizon_fde_stays_missing_when_label_is_invalid(self):
        values = baseline.make_controls().value
        constant_bias = baseline.compute(dict(values, bias=1., noise=0., valid=10))
        self.assertAlmostEqual(constant_bias['ade'], 1.)
        self.assertAlmostEqual(constant_bias['fde'], 1.)
        incomplete = baseline.compute(dict(values, valid=7))
        self.assertIsNone(incomplete['fde'])

    def test_zero_feature_collapse_keeps_variance_penalty(self):
        values = vicreg.make_controls().value
        collapsed = vicreg.compute(dict(values, scale=0))
        self.assertEqual(collapsed['prediction'], 0)
        self.assertGreater(collapsed['variance'], .98)
        self.assertGreater(collapsed['total'], 24)

    def test_constant_turn_straight_line_and_mirror_symmetry(self):
        values = joint.make_controls().value
        straight = joint.compute(dict(values, speed=10., yaw=0.))
        np.testing.assert_allclose(straight['prior'][-1], [20, 0, 10, 0])
        left = joint.compute(dict(values, yaw=.25))['prior']
        right = joint.compute(dict(values, yaw=-.25))['prior']
        np.testing.assert_allclose(left[:, 0], right[:, 0])
        np.testing.assert_allclose(left[:, 1], -right[:, 1])

    def test_clean_identity_and_causal_three_frame_hold(self):
        values = stress.make_controls().value
        clean = stress.compute(dict(values, depth=0., boundary=0., dropped=0))
        np.testing.assert_array_equal(clean['clean'], clean['degraded'])
        self.assertFalse(clean['changed'].any())
        frozen = stress.compute(dict(values, depth=0., boundary=0., dropped=3))
        for frame in range(4):
            np.testing.assert_array_equal(frozen['frozen'][frame], frozen['clean'][0])
        self.assertFalse(np.array_equal(frozen['clean'][0], frozen['clean'][3]))

    def test_boundary_removal_preserves_other_semantic_channels(self):
        values = stress.make_controls().value
        data = stress.compute(dict(values, depth=0., boundary=1., dropped=0))
        self.assertFalse(data['degraded'][:, 1].any())
        for channel in [0, 2, 3]:
            np.testing.assert_array_equal(data['clean'][:, channel], data['degraded'][:, channel])


if __name__ == '__main__':
    unittest.main()
