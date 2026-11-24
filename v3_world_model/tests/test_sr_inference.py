import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from v3_world_model.async_lidar import full_revolution
from v3_world_model.geometry import H, W, carla_pose, ego_delta, grid, pixels
from v3_world_model.sr_geometry import clean_drivable_mask, dashed_segments, road_curb_paths
from v3_world_model.sr_postprocess import SRPostProcessor
from v3_world_model.sr_viewer import perception_summary, prediction_scene, synthetic_scene, build_scene, SceneMesh
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

    def test_lane_fitting_preserves_overlapping_vehicle_predictions(self):
        scene = synthetic_scene()
        scene['boxes'][:] = 0
        scene['fit_geometry'] = True
        for i in range(4):
            scene['boxes'][i] = [20 + i, -.3 * i, 1, 4, 2, 2, 0, 0, 0, 0, 1]
        expected = scene['boxes'][:4].copy()
        with patch.object(SceneMesh, 'vehicle') as vehicle:
            build_scene(scene)
        self.assertEqual(vehicle.call_count, 5)
        np.testing.assert_array_equal([call.args[0] for call in vehicle.call_args_list[:4]], expected)

    def test_clean_drivable_mask_removes_islands_and_keeps_large_hole(self):
        xy = grid()
        road = ((xy[..., 0] >= -30.0) & (xy[..., 0] <= 60.0) & (np.abs(xy[..., 1]) <= 6.0)).astype(np.float32)
        tiny_pin = (xy[..., 0] >= 10.0) & (xy[..., 0] <= 11.2) & (np.abs(xy[..., 1]) <= 0.6)
        large_hole = (xy[..., 0] >= 25.0) & (xy[..., 0] <= 35.0) & (xy[..., 1] >= -2.5) & (xy[..., 1] <= 2.5)
        island = (xy[..., 0] >= -25.0) & (xy[..., 0] <= -22.5) & (xy[..., 1] >= 20.0) & (xy[..., 1] <= 22.0)
        road[tiny_pin] = 0.0
        road[large_hole] = 0.0
        road[island] = 1.0

        cleaned = clean_drivable_mask(road)
        ego_uv = np.rint(pixels(np.array([[0.0, 0.0]]))[0]).astype(int)
        self.assertEqual(cleaned[ego_uv[1], ego_uv[0]], 1.0)
        self.assertTrue((cleaned[tiny_pin] >= 0.5).all())
        center_hole = np.rint(pixels(np.array([[30.0, 0.0]]))[0]).astype(int)
        self.assertEqual(cleaned[center_hole[1], center_hole[0]], 0.0)
        self.assertEqual(cleaned[island].max(), 0.0)

    def test_dashed_segments_emits_uniform_dashes_from_noisy_snake(self):
        xy = grid()
        road = (np.abs(xy[..., 1]) <= 6.0).astype(np.float32)
        wave_y = 0.25 * np.sin(xy[..., 0] * 0.35)
        snake = ((xy[..., 0] >= -20.0) & (xy[..., 0] <= 60.0) & (np.abs(xy[..., 1] - wave_y) <= 0.65)).astype(np.float32)

        sweep = dashed_segments(snake, road, dash_len=3.0, gap_len=5.5, fit=True)
        self.assertIsNotNone(sweep)
        points, _, seg_starts = sweep
        breaks = set(seg_starts.tolist())
        dashes, cur = [], []
        for idx in range(len(points) - 1):
            if idx in breaks:
                if not cur:
                    cur = [points[idx], points[idx + 1]]
                else:
                    cur.append(points[idx + 1])
            elif cur:
                dashes.append(np.asarray(cur))
                cur = []
        if cur:
            dashes.append(np.asarray(cur))

        self.assertGreaterEqual(len(dashes), 6)
        lengths = [float(np.linalg.norm(np.diff(d, axis=0), axis=1).sum()) for d in dashes]
        self.assertTrue(all(0.75 <= l <= 3.15 for l in lengths), lengths)
        centers = sorted(float(d[:, 0].mean()) for d in dashes)
        spacings = np.diff(centers)
        self.assertTrue((spacings >= 7.5).all(), spacings)

    def test_road_curb_paths_stay_open_across_roi_borders(self):
        xy = grid()
        road = (np.abs(xy[..., 1]) <= 7.5).astype(np.float32)
        sweep = road_curb_paths(road, fit=True)
        self.assertIsNotNone(sweep)
        points, _, seg_starts = sweep
        p0, p1 = points[seg_starts], points[seg_starts + 1]
        near_roi_end = (np.maximum(p0[:, 0], p1[:, 0]) >= 77.0) | (np.minimum(p0[:, 0], p1[:, 0]) <= -37.0)
        cross_gap = np.abs(p1[:, 1] - p0[:, 1]) > 2.0
        self.assertFalse((near_roi_end & cross_gap).any())
        self.assertTrue((p0[:, 1] > 5.0).any() and (p0[:, 1] < -5.0).any())

    def test_sr_postprocess_filters_aligns_and_tracks(self):
        xy = grid()
        road = (np.abs(xy[..., 1]) <= 6.0).astype(np.float32)
        lane_paths = [np.array([[-30.0, 0.0], [60.0, 0.0]], dtype=np.float32)]
        raw_boxes = np.array([
            [0.4, 0.2, 0.0, 4.5, 1.9, 1.5, 0.0, 0.0, 0.0, 0.0, 1.0],          # ego overlap -> drop
            [20.0, 1.8, 0.0, 4.5, 1.9, 1.5, 0.35, 5.0, 0.0, 0.0, 1.0],         # noisy heading -> align
            [20.4, 2.0, 0.0, 4.5, 1.9, 1.5, 0.10, 5.0, 0.0, 0.0, 1.0],         # duplicate overlap -> NMS
            [25.0, -20.0, 0.0, 4.5, 1.9, 1.5, 0.0, 0.0, 0.0, 0.0, 1.0],        # off-road void -> drop
            [32.0, 0.0, 0.0, 4.5, 1.9, 1.5, np.pi / 2, 0.0, 4.0, 0.0, 1.0],    # cross-traffic -> keep yaw
        ], dtype=np.float32)
        scores = np.array([0.95, 0.85, 0.65, 0.80, 0.90], dtype=np.float32)

        kept, kept_scores = SRPostProcessor.clean_frame(raw_boxes, scores, road, lane_paths)
        self.assertEqual(len(kept), 2)
        self.assertLess(abs(float(kept[1, 6])), 0.12)
        self.assertLess(abs(float(kept[0, 6]) - np.pi / 2), 0.05)

        pp = SRPostProcessor()
        low_conf = np.array([[40.0, 1.8, 0.0, 4.5, 1.9, 1.5, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        b0, _, ids0, _ = pp.update(low_conf, np.array([0.55], np.float32), road, lane_paths)
        self.assertEqual(int((b0[:, -1] > 0.5).sum()), 0)  # hysteresis blocks single unconfirmed frame
        delta = ego_delta(carla_pose(x=0.0, y=0.0, yaw=0.0), carla_pose(x=1.0, y=0.0, yaw=0.0))
        low_conf_moved = low_conf.copy()
        low_conf_moved[0, 0] -= 1.0
        b1, _, ids1, _ = pp.update(low_conf_moved, np.array([0.55], np.float32), road, lane_paths, ego_delta=delta)
        self.assertEqual(int((b1[:, -1] > 0.5).sum()), 1)
        self.assertGreater(int(ids1[0]), 0)

    def test_prediction_scene_live_cleans_while_diagnostic_stays_raw(self):
        xy = grid()
        road_logits = np.where(np.abs(xy[..., 1]) <= 5.0, 4.0, -4.0).astype(np.float32)
        lane_logits = np.full((1, 4, H, W), -4.0, dtype=np.float32)
        lane_logits[0, 0] = road_logits
        boxes = np.zeros((1, 32, 10), dtype=np.float32)
        boxes[0, 0, :8] = [20.0, -22.0, 0.0, 4.5, 1.9, 1.5, 0.0, 1.0]  # off-road ghost
        cls_logits = np.zeros((1, 32, 5), dtype=np.float32)
        obj_logits = np.full((1, 32), -4.0, dtype=np.float32)
        obj_logits[0, 0] = 2.0  # sigmoid ~ 0.88
        pred = {
            "boxes": torch.from_numpy(boxes),
            "cls_logits": torch.from_numpy(cls_logits),
            "obj_logits": torch.from_numpy(obj_logits),
            "lane_logits": torch.from_numpy(lane_logits),
            "future_occupancy_logits": torch.zeros((1, 3, 2, H, W), dtype=torch.float32),
            "trajectory": torch.zeros((1, 3, 2), dtype=torch.float32),
        }
        diag = prediction_scene(pred, mode="DIAGNOSTIC")
        live = prediction_scene(pred, mode="LIVE")
        self.assertFalse(diag["fit_geometry"])
        self.assertEqual(int((diag["boxes"][:, -1] > 0.5).sum()), 1)
        self.assertTrue(live["fit_geometry"])
        self.assertEqual(int((live["boxes"][:, -1] > 0.5).sum()), 0)

    def test_footprint_nms_suppresses_split_and_phantom_two_wheeler_keeps_neighbor_lane(self):
        xy = grid()
        road = (np.abs(xy[..., 1]) <= 8.0).astype(np.float32)
        lane_paths = [np.array([[-30.0, 0.0], [60.0, 0.0]], dtype=np.float32)]
        raw_boxes = np.array([
            [20.0, 0.0, 0.5, 4.6, 1.9, 1.5, 0.0, 8.0, 0.0, 0.0, 1.0],   # primary car
            [22.8, 0.0, 0.5, 4.6, 1.9, 1.5, 0.02, 8.0, 0.0, 0.0, 1.0],  # along-track split duplicate -> drop
            [21.0, 0.4, 0.5, 2.0, 0.8, 1.4, 0.05, 8.0, 0.0, 3.0, 1.0],  # phantom two_wheeler on same car -> drop
            [20.0, 3.5, 0.5, 4.6, 1.9, 1.5, 0.0, 8.0, 0.0, 0.0, 1.0],   # adjacent lane car -> keep
        ], dtype=np.float32)
        scores = np.array([0.88, 0.72, 0.68, 0.85], dtype=np.float32)

        kept, _ = SRPostProcessor.clean_frame(raw_boxes, scores, road, lane_paths)
        self.assertEqual(len(kept), 2)
        ys = sorted(float(y) for y in kept[:, 1])
        self.assertAlmostEqual(ys[0], 0.0, delta=0.2)
        self.assertAlmostEqual(ys[1], 3.5, delta=0.2)

    def test_tracker_suppresses_stale_overlapping_track(self):
        xy = grid()
        road = (np.abs(xy[..., 1]) <= 8.0).astype(np.float32)
        lane_paths = [np.array([[-30.0, 0.0], [60.0, 0.0]], dtype=np.float32)]
        pp = SRPostProcessor()

        frame0 = np.array([[20.0, 0.0, 0.5, 4.6, 1.9, 1.5, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        b0, _, _, _ = pp.update(frame0, np.array([0.82], np.float32), road, lane_paths)
        self.assertEqual(int((b0[:, -1] > 0.5).sum()), 1)

        # Jumped high-confidence front-half detection outside Hungarian gate (4.2m > 3.8m)
        # must suppress the stale unmatched frame0 track rather than cloning the car.
        frame1 = np.array([[24.2, 0.0, 0.5, 4.6, 1.9, 1.5, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        b1, _, _, _ = pp.update(frame1, np.array([0.85], np.float32), road, lane_paths)
        active1 = b1[b1[:, -1] > 0.5]
        self.assertEqual(len(active1), 1)
        self.assertAlmostEqual(float(active1[0, 0]), 24.2, delta=0.2)


if __name__ == '__main__':
    unittest.main()
