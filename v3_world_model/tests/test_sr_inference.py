import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from v3_world_model.async_lidar import full_revolution
from v3_world_model.geometry import carla_pose
from v3_world_model.sr_viewer import perception_summary, synthetic_scene, build_scene, SceneMesh
from v3_world_model.evaluate_sr import match_centers


def packet(t, x=0.):
    scan = np.array([[i, j, 0, 1] for j in (1, 2) for i in range(4)], dtype='<f4')
    return SimpleNamespace(timestamp=t, raw_data=scan.tobytes(),
                           transform=SimpleNamespace(get_matrix=lambda:carla_pose(x=x)),
                           get_point_count=lambda channel:4)


class SRInferenceTests(unittest.TestCase):
    def test_center_matching_maximizes_in_radius_matches(self):
        # Plain minimum-distance assignment chooses 0 + 3.1 (only one valid).
        # Radius-aware matching chooses 2.9 + 2.9 (two valid).
        truth = np.array([[0., 0.], [2.9, 0.]])
        x = (2.9**2 + 2.9**2 - 3.1**2) / (2*2.9)
        predicted = np.array([[0., 0.], [x, np.sqrt(2.9**2-x*x)]])
        self.assertEqual(match_centers(predicted, truth), 2)
        self.assertEqual(match_centers(np.empty((0, 2)), truth), 0)

    def test_full_sweep_trims_each_channel_and_compensates_motion(self):
        scan, info = full_revolution([(packet(.06, 2), .06), (packet(.12, 4), .06)],
                                    carla_pose(x=4), .12, channels=2)
        self.assertEqual(scan.shape, (14, 4))
        np.testing.assert_allclose(scan[:3, 0], [-1, 0, 1])
        np.testing.assert_allclose(scan[:6, 1], [-1, -1, -1, -2, -2, -2])
        self.assertEqual(info['span_seconds'], .1)

    def test_partial_future_or_missing_packets_are_not_a_full_sweep(self):
        self.assertIsNone(full_revolution([(packet(.05), .05)], np.eye(4), .05, channels=2))
        self.assertIsNone(full_revolution([(packet(.05), .05), (packet(.15), .05)], np.eye(4), .15, channels=2))
        self.assertIsNone(full_revolution([(packet(.10), .10)], np.eye(4), .05, channels=2))

    def test_empty_detections_never_claim_clear_road(self):
        scene = synthetic_scene()
        scene['boxes'][:] = 0
        scene['object_scores'] = np.full(32, .12)
        summary = perception_summary(scene)
        self.assertEqual(summary[2], 0)
        self.assertEqual(summary[4], '未检出前车')
        self.assertIn('12%', summary[5])
        scene['boxes'][0] = [20, 0, 1, 4, 2, 2, 0, 0, 0, 0, 1]
        self.assertNotEqual(summary, perception_summary(scene))

    def test_renderer_does_not_cap_number_of_vehicles(self):
        scene = synthetic_scene()
        scene['boxes'][:] = 0
        for i in range(4):
            scene['boxes'][i] = [20 + i, -3.5*i, 1, 4, 2, 2, 0, 0, 0, 0, 1]
        with patch.object(SceneMesh, 'vehicle') as vehicle:
            build_scene(scene)
        self.assertEqual(vehicle.call_count, 5)


if __name__ == '__main__':
    unittest.main()
