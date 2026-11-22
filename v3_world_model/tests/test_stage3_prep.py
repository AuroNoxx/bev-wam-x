"""Offline acceptance tests. Run with python -m unittest discover -s v3_world_model/tests -v."""
import argparse
import json
import tarfile
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

from v1_dual_system.checkpoint_io import VerifiedCheckpointIO
from v3_world_model import collect_v3
from v3_world_model.calibration_probe import coverage, ego_silhouette, lane_reprojection, metrics_pass, mosaic, RULES
from v3_world_model.data import Clips
from v3_world_model.geometry import (CONFIG_SHA256, DT, H, W, Z, NAMES, SIZES, camera_mounts,
                                     carla_pose, delta_error, ego_delta, grid, metric, pixels,
                                     project, projection_tables, world_at)
from v3_world_model.labels import decode_boxes, encode_boxes, exclusive_markings, object_slots
from v3_world_model.objectives import LANE_POS_WEIGHT, OCC_POS_WEIGHT, segmentation, stage31_losses
from v3_world_model.observations import AUDIT_SENSORS, OBS_FIELDS, command_vector, validate_observation
from v3_world_model.pack_colab_v3 import assert_split, main as pack_cli, read_manifest, unpack_verified
from v3_world_model.perception_model import BEVWAMXPerception, DECODE_SHAPE, sdpa_context
from v3_world_model.qc_v3 import LABEL_SHAPES, event_count, load_npz
from v3_world_model.sim import connect
from v3_world_model.sr_viewer import SRViewer, geometry, ribbon, synthetic_scene
from v3_world_model.train_stage31_colab import Stage31, lock, lr_factor, train


def labels():
    targets = {name: np.zeros(shape, np.float32) for name, shape in LABEL_SHAPES.items()}
    for k in ("lanes", "occupancy", "future_occupancy"):
        targets[k] = targets[k].astype(np.uint8)
    for k in ("vector_exists", "vector_valid", "flags"):
        targets[k] = targets[k].astype(bool)
    targets["lanes"][0, :, 51:68] = 1
    targets["lanes"][1, ::5, 52] = 1
    targets["lanes"][2, :, 68] = 1
    targets["lanes"][3, :, 69] = 1
    targets["vector_valid"][0] = True
    targets["vector_exists"][0] = True
    targets["vectors"][0, :, 0] = np.linspace(-20, 55, 24)
    targets["boxes"][0] = [12., 0., .9, 4.2, 1.9, 1.6, .1, 6., 0., 0., 1.]
    targets["box_ids"] = np.full(32, -1, np.int64)
    targets["box_ids"][0] = 301
    targets["ego_state"][0] = 8.
    targets["trajectory"][:, 0] = np.arange(1, 26) * .8
    targets["trajectory"][:, 2] = 8.
    targets["nav_state"] = np.array([0., .2, 1.], dtype=np.float32)
    targets["lane_state"] = np.array([1, 1, 0], np.int64)
    targets["thw"] = np.float32(1.)
    targets["thw_actor_id"] = np.int64(301)
    targets["occupancy"][2, 88:93, 57:62] = 1
    targets["future_occupancy"][:, 78:82, 57:62] = 1
    return targets


def create_capture(root):
    rng = np.random.default_rng(641)
    town = "Town12"
    for index, (route, start) in enumerate((("a", "y"), ("a", "z"), ("b", "z"), ("c", "w"))):
        folder = root / f"episode-{index:05d}"
        folder.mkdir(parents=True)
        frame_ids = list(range(index * 20 + 1, index * 20 + 5))
        info = dict(config_sha256=CONFIG_SHA256, dt=DT, map=town, server="1.0-test",
                    frame_ids=frame_ids, future_tail_ticks=25, route_id=f"{town}/{route}",
                    initial_state_id=f"{town}/{start}", scenario="left_change" if index == 0 else "straight",
                    lane_change=dict(side="follow", requested_frame=None, ended_frame=None, status="unrequested"))
        (folder / "episode.json").write_text(json.dumps(info), encoding="utf-8")
        images = [cv2.imencode(".png", np.full((h, w, 3), (42, 69, 100), np.uint8))[1].tobytes()
                  for w, h in SIZES]
        previous = None
        for i, fid in enumerate(frame_ids):
            frame = folder / str(fid)
            frame.mkdir()
            for n, image in zip(NAMES, images):
                (frame / f"{n}.png").write_bytes(image)
            pose = carla_pose(x=i * .8, y=.01 * i, yaw=i * .03)
            scan = np.column_stack((rng.uniform(10.5, 13.5, 120), rng.uniform(-.6, .6, 120),
                                    rng.uniform(.5, 1.5, 120), rng.random(120))).astype(np.float32)
            np.savez_compressed(frame / "obs.npz", lidar_bev=np.zeros((6, H, W), np.float16),
                                ego_delta=ego_delta(previous, pose) if previous is not None else np.zeros(3, np.float32),
                                nav_anchors=np.array([[12, 0, 0], [24, i, 0], [36, 2*i, 0]], np.float32),
                                ego_speed=np.array([8., 0.], np.float32), cmd=command_vector("follow"))
            cameras = pose @ camera_mounts()
            sensor_poses = np.concatenate((cameras, pose[None]))
            np.savez_compressed(frame / "audit.npz", pose=pose, lidar_points=scan,
                                sensor_poses=sensor_poses, sensor_ids=np.full(len(AUDIT_SENSORS), fid, np.int64),
                                timestamps=np.full(len(AUDIT_SENSORS), fid * DT, np.float64), lidar_angle=np.float64(0))
            np.savez_compressed(frame / "labels.npz", **labels())
            previous = pose


class Stage3PrepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cv2.setNumThreads(1)

    def test_geometry_and_capture_redline(self):
        self.assertEqual(sum(w*h for w, h in SIZES), 518400)
        uv, visible = projection_tables()
        self.assertEqual(uv.shape, (5, 6, H, W, 2))
        self.assertTrue((uv[~visible] == -1).all())
        self.assertEqual(coverage(camera_mounts()), (1., 0))
        for n, mount, (w, h) in zip(NAMES, camera_mounts(), SIZES):
            p = (mount[:3, 3] + mount[:3, 0] * 17) * [1, -1, 1]
            xy, valid, depth = project(p, n, mount)
            self.assertTrue(valid)
            np.testing.assert_allclose(xy, [w / 2, h / 2], atol=1e-8)
            self.assertAlmostEqual(depth, 17.)
        np.testing.assert_array_equal(pixels([[80., 30.], [-40., -30.]]),
                                      [[-.5, -.5], [W - .5, H - .5]])
        np.testing.assert_allclose(metric(pixels(grid())), grid(), rtol=0, atol=1e-12)
        a, b = carla_pose(4, -3, .3, 179, -3), carla_pose(3.1, -2, .45, -179, -2)
        delta = ego_delta(a, b)
        self.assertLess(delta_error(delta, a, b), 1e-5)
        self.assertAlmostEqual(float(delta[2]), np.deg2rad(-2), places=6)
        with self.assertRaisesRegex(ValueError, "prohibited"):
            connect("127.0.0.1", 1, "/Game/Maps/Town04_Opt")
        with patch("sys.argv", ["collect_v3", "--map", "Town04_Opt", "--calibration", "unused",
                                "--out", "unused", "--pilot-frames", "500"]):
            with patch.object(collect_v3, "connect") as called:
                with self.assertRaisesRegex(ValueError, "Town04"):
                    collect_v3.main()
                called.assert_not_called()

    def test_labels_and_events(self):
        masks = np.ones((4, H, W), np.uint8)
        clean = exclusive_markings(masks)
        self.assertEqual(np.count_nonzero(clean[1] & clean[2]), 0)
        sample = np.tile(labels()["boxes"][0], (40, 1))
        sample[:, 0] = np.arange(40, 0, -1)
        slots = object_slots(sample)
        self.assertEqual(slots.shape, (32, 11))
        np.testing.assert_array_equal(slots[:, 0], np.arange(1, 33))
        encoded = encode_boxes(slots)
        self.assertEqual(np.max(abs(decode_boxes(encoded) - slots)), 0.0)
        self.assertLess(np.max(abs(np.hypot(encoded["sin_yaw"], encoded["cos_yaw"]) - 1)), 1e-4)
        self.assertEqual(event_count(np.r_[np.ones(14), np.zeros(7), np.ones(11)], 10), 2)

    def test_back_calibration_visibility_and_display_gamma(self):
        mounts = camera_mounts()
        rear = mounts[NAMES.index("back")]
        old = carla_pose(-2.0, 0, 1.45, 180, -12)
        shape = (180, 320)
        lo, hi = np.array([-2.55, -1.0, -.2]), np.array([2.55, 1.0, 1.53])
        self.assertTrue(ego_silhouette("back", old, shape, lo, hi)[-30:].any())
        self.assertFalse(ego_silhouette("back", rear, shape, lo, hi).any())

        points = np.array([[-4., 0, -.2], [-10., 1, -.2], [-19., -2., -.2], [-31., 0, -.2]])
        uv, valid, _ = project(points, "back", rear)
        self.assertTrue(valid[1:3].all())
        u, v = np.rint(uv[1:3]).astype(int).T
        road_tag = 6
        semantic = np.zeros(shape, np.uint8)
        semantic[v, u] = road_tag
        mask = np.zeros(shape, bool)
        mask[v[1], u[1]] = True
        distances, blocked, subpixel = lane_reprojection(points, "back", rear, semantic, mask, road_tag)
        self.assertEqual((len(distances), blocked, subpixel), (1, 1, 0))
        self.assertLess(distances[0], .72)
        with patch.dict(RULES, min_marking_width_px=1.5):
            _, blocked, subpixel = lane_reprojection(points, "back", rear, semantic, mask, road_tag)
            self.assertEqual((blocked, subpixel), (0, 1))

        cams = {name: dict(matches=25, lane_p95_px=4., max_ego_occlusion_fraction=0.) for name in NAMES}
        report = dict(frames=20, coverage_fraction=1., largest_angular_gap_deg=0,
                      worst_mount_abs_error=0., cameras=cams)
        self.assertTrue(metrics_pass(report))
        cams["back"]["lane_p95_px"] = 4.01
        self.assertFalse(metrics_pass(report))
        cams["back"].update(lane_p95_px=4., max_ego_occlusion_fraction=.001)
        self.assertFalse(metrics_pass(report))
        camera_images = [np.full((h, w, 3), 16, np.uint8) for w, h in SIZES]
        masks = [np.zeros(img.shape[:2], bool) for img in camera_images]
        self.assertGreater(mosaic(camera_images, mounts, masks).max(), 50)

    def test_lane_change_command_tracks_request_and_recentering(self):
        def lane(lane_id, y):
            return SimpleNamespace(road_id=8, lane_id=lane_id,
                                   transform=SimpleNamespace(get_matrix=lambda: carla_pose(y=y)))

        source, target = lane(1, 0.), lane(2, -3.6)
        tracker = collect_v3.LaneChangeCommand()
        self.assertEqual(tracker.advance(100, source, carla_pose()), "follow")
        tracker.start(112, "left", target)
        self.assertEqual(tracker.advance(112, source, carla_pose()), "left")
        # The next lane can be reported before the vehicle has finished crossing and straightened.
        self.assertEqual(tracker.advance(113, target, carla_pose(y=-2.0)), "left")
        self.assertEqual(tracker.advance(114, target, carla_pose(y=-3.6, yaw=12)), "left")
        for frame in (115, 116):
            self.assertEqual(tracker.advance(frame, target, carla_pose(y=-3.6)), "left")
        self.assertEqual(tracker.advance(117, target, carla_pose(y=-3.6)), "follow")
        self.assertEqual(tracker.advance(118, target, carla_pose(y=-3.6)), "follow")
        self.assertEqual((tracker.status, tracker.ended_frame), ("completed", 117))

        missed = collect_v3.LaneChangeCommand()
        missed.start(112, "right", target)
        self.assertEqual(missed.advance(112 + collect_v3.LANE_CHANGE_TIMEOUT, source, carla_pose()), "follow")
        self.assertEqual(missed.status, "timed_out")

    def test_blocked_spawns_retry_without_committing_a_partial_episode(self):
        transform = SimpleNamespace(location=SimpleNamespace(x=0., y=0., z=.2), rotation=object())
        carla = SimpleNamespace(Location=lambda x, y, z: SimpleNamespace(x=x, y=y, z=z),
                                Transform=lambda loc, rot: SimpleNamespace(location=loc, rotation=rot))
        world = Mock()
        world.try_spawn_actor.return_value = None
        blocked_route = (transform, None, None, 0., False, (False, False), "route-a")
        other_route = (object(), None, None, 0., False, (False, False), "route-b")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pilot"
            staging = root / ".partial-00000"
            with self.assertRaisesRegex(collect_v3.SpawnUnavailable, "Ego spawn"):
                collect_v3.episode(carla, world, None, None, blocked_route, "straight",
                                   4, None, staging, "Town12", "1.0", 714, 0)
            self.assertAlmostEqual(world.try_spawn_actor.call_args.args[1].location.z, .55)
            self.assertAlmostEqual(transform.location.z, .2)
            self.assertFalse(staging.exists())

            tried = []

            def run_episode(*args):
                route = args[4]
                self.assertFalse(args[8].exists())
                tried.append(route[-1])
                if route is blocked_route:
                    raise collect_v3.SpawnUnavailable("Ego spawn is blocked")
                args[8].mkdir()

            argv = ["collect_v3", "--map", "Town12", "--scenarios", "straight", "--pilot-frames", "4",
                    "--calibration", "unused", "--out", str(root)]
            with (patch("sys.argv", argv), patch.object(collect_v3, "connect", return_value=(carla, Mock(), world, "1.0")),
                  patch.object(collect_v3, "require_pass"), patch.object(collect_v3, "synchronous", return_value=nullcontext()),
                  patch.object(collect_v3, "RoadSurfaces"),
                  patch.object(collect_v3, "catalog", return_value=[blocked_route, other_route]),
                  patch.object(collect_v3, "select_route", side_effect=lambda routes, *_: routes[0]),
                  patch.object(collect_v3, "episode", side_effect=run_episode)):
                collect_v3.main()
            self.assertEqual(tried, ["route-a", "route-b"])
            self.assertEqual(world.tick.call_count, 1)
            self.assertTrue((root / "episode-00000").is_dir())
            self.assertFalse(staging.exists())

    def test_sparse_positive_weight_and_focal_dice(self):
        logits = torch.full((1, 4, 16, 16), -8., requires_grad=True)
        target = torch.zeros_like(logits)
        target[:, :, 4, 5] = 1.
        lane = segmentation(logits, target, LANE_POS_WEIGHT)
        lane_grad = torch.autograd.grad(lane, logits, retain_graph=True)[0]
        self.assertLess(lane_grad[0, 1, 4, 5], lane_grad[0, 0, 4, 5] * 6)
        focal_dice = segmentation(logits, target, OCC_POS_WEIGHT, focal=True)
        weighted_dice = segmentation(logits, target, OCC_POS_WEIGHT)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=torch.tensor(OCC_POS_WEIGHT))
        prob = logits.sigmoid()
        expected_dice_loss = 1. - ((2 * (prob * target).sum((-1, -2)) + 1) /
                                  ((prob + target).sum((-1, -2)) + 1)).mean()
        torch.testing.assert_close(weighted_dice, bce + expected_dice_loss)
        self.assertGreater(focal_dice.item(), weighted_dice.item())
        focal_dice.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(logits.grad[0, 0, 4, 5], 0.)

    def test_model_forward_gradient_and_intent_isolation(self):
        torch.manual_seed(317)
        model = BEVWAMXPerception(lidar_dropout=0).eval()
        params = sum(p.numel() for p in model.parameters())
        self.assertTrue(20_000_000 <= params <= 30_000_000)
        rgb = tuple(torch.randint(0, 256, (1, 3, h, w), dtype=torch.uint8) for w, h in SIZES)
        lidar = torch.randn(1, 6, H, W)
        prev = torch.randn(1, 64, 256)
        motion = torch.tensor([[-.93, .04, -.02]])
        query = dict(cmd=torch.tensor([[0., 1., 0.]]),
                     nav_anchors=torch.tensor([[[12., 0., 0.], [24., 0., 0.], [36., 0., 0.]]]),
                     ego_speed=torch.tensor([[8., .04]]))
        changed = dict(cmd=torch.tensor([[0., 0., 1.]]),
                       nav_anchors=query["nav_anchors"] + torch.tensor([0., 6., 0.]),
                       ego_speed=torch.tensor([[22., -.25]]))
        with sdpa_context("math"):
            left = model(rgb, lidar, prev, motion, **query)
            with torch.no_grad():
                for field, value in changed.items():
                    with self.subTest(planning_field=field):
                        result = model(rgb, lidar, prev, motion, **dict(query, **{field: value}))
                        for key in left.keys() - {"trajectory"}:
                            self.assertEqual((left[key] - result[key]).abs().max().item(), 0.0, key)
                        self.assertGreater((left["trajectory"] - result["trajectory"]).abs().max().item(), 0)
                decoded = model.decode_perception(left["z_t"])
        self.assertEqual(DECODE_SHAPE, (40, 30))
        self.assertEqual(model.dense_queries.shape[1], 40 * 30)
        self.assertEqual(tuple(left["z_t"].shape), (1, 64, 256))
        self.assertEqual(tuple(left["lane_logits"].shape), (1, 4, H, W))
        self.assertEqual(tuple(left["occupancy_logits"].shape), (1, Z, H, W))
        self.assertEqual(tuple(left["future_occupancy_logits"].shape), (1, 5, H, W))
        for key in left:
            self.assertTrue(torch.isfinite(left[key]).all(), key)
        for key, tensor in decoded.items():
            self.assertTrue(torch.equal(tensor, left[key]), key)
        targets = {k: torch.as_tensor(v)[None] for k, v in labels().items()}
        loss = sum(stage31_losses(left, targets).values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))

    def test_pack_verify_split_and_lightning(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            create_capture(source)
            packed = base / "packed"
            with patch("sys.argv", ["pack_colab_v3", "--source", str(source), "--out", str(packed),
                                    "--max-shard-bytes", "65000", "--verify-pack"]):
                pack_cli()
            result = read_manifest(packed / "manifest.json")
            self.assertGreaterEqual(len(result["shards"]), 2)
            for shard in result["shards"]:
                with tarfile.open(packed / shard["name"]) as archive:
                    self.assertTrue(all(Path(n).name in {"obs.npz", "targets.npz"} or n.endswith(".webp")
                                        for n in archive.getnames()))
            self.assertEqual(len(list(source.glob("*/*/audit.npz"))), 16)
            self.assertFalse(list(source.glob("*/*/depth.npz")))
            check = json.loads((packed / "verification.json").read_text(encoding="utf-8"))
            self.assertEqual(check, dict(sampled_frames=16, max_abs_diff=0.0,
                                         route_overlap=0, initial_state_overlap=0))
            matching = next(v for v in result["splits"].values() if "ep-000000" in v)
            self.assertTrue({"ep-000000", "ep-000001", "ep-000002"} <= set(matching))
            with self.assertRaisesRegex(ValueError, "leakage"):
                assert_split(result["episodes"], dict(train=["ep-000000", "ep-000003"],
                                                     val=["ep-000001", "ep-000002"]))
            dataset = base / "dataset"
            unpack_verified(packed / "manifest.json", dataset)
            val = Clips(dataset, "val")
            tr = Clips(dataset, "train")
            fixed_lut = np.broadcast_to(np.arange(255, -1, -1, dtype=np.uint8)[None, :, None],
                                        (1, 256, 3)).copy()
            with patch("v3_world_model.data.color_jitter_table", return_value=fixed_lut) as jitter:
                raw = val[0]["inputs"]
                jitter.assert_not_called()
                augmented = tr[0]["inputs"]
                jitter.assert_called_once()
            self.assertEqual(raw.keys(), OBS_FIELDS.keys() | {"rgb"})
            self.assertTrue(torch.equal(augmented["rgb"][0], 255 - raw["rgb"][0]))
            raw_file = next(dataset.glob("ep-000000/*/obs.npz"))
            stored_obs = load_npz(raw_file)
            validate_observation(stored_obs)
            with self.assertRaisesRegex(ValueError, "keys"):
                validate_observation(dict(stored_obs, trajectory=np.zeros(3)))
            torch.testing.assert_close(raw["cmd"], torch.from_numpy(np.tile(command_vector("follow"), (4, 1))))
            self.assertEqual(tuple(raw["nav_anchors"].shape), (4, 3, 3))
            self.assertEqual(tuple(raw["ego_speed"].shape), (4, 2))
            run = base / "run"
            args = argparse.Namespace(dataset=dataset, out=run, batch=1, accumulate=1, max_updates=1,
                                      checkpoint_every=1, lr=2e-4, lidar_dropout=0., seed=317, accelerator="cpu",
                                      precision="32-true", sdpa="math", workers=0, resume=False,
                                      warmup_updates=3, schedule_updates=5)
            with lock():
                train(args)
            info = json.loads((run / "training-result.json").read_text(encoding="utf-8"))
            self.assertEqual(info["actual_updates"], 1)
            self.assertAlmostEqual(info["learning_rate_after_update"], args.lr * 2 / 3)
            self.assertFalse(list(run.glob("*.pending")))
            io = VerifiedCheckpointIO()
            ckpt = io.load_checkpoint(run / "latest.ckpt", map_location="cpu", weights_only=False)
            best = io.load_checkpoint(run / "best.ckpt", map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["global_step"], 1)
            self.assertEqual(best["global_step"], 1)
            self.assertTrue(ckpt["optimizer_states"])
            self.assertTrue(ckpt["optimizer_states"][0]["param_groups"][0]["foreach"])
            self.assertEqual(ckpt["lr_schedulers"][0]["last_epoch"], 1)
            self.assertTrue((run / "validation.jsonl").read_text(encoding="utf-8").strip())
            restored = Stage31(ckpt["task_state"])
            restored.load_state_dict(ckpt["state_dict"])
            for name, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, ckpt["state_dict"][name]), name)
                self.assertTrue(torch.equal(value, best["state_dict"][name]), name)
            del ckpt, best, restored
            args.resume, args.max_updates, args.schedule_updates, args.warmup_updates = True, 2, None, None
            with lock():
                train(args)
            result = json.loads((run / "training-result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["actual_updates"], 2)
            self.assertEqual(result["schedule_updates"], 5)
            self.assertAlmostEqual(result["learning_rate_after_update"], args.lr)
            self.assertLess(lr_factor(4, 3, 5), lr_factor(3, 3, 5))
            self.assertAlmostEqual(lr_factor(5, 3, 5), .05)

    def test_sr_offscreen(self):
        scene = synthetic_scene()
        band = ribbon(scene["trajectory"])
        np.testing.assert_allclose(np.linalg.norm(band[1::2, :2] - band[::2, :2], axis=1), 1.9, atol=2e-6)
        self.assertGreater(band[0, 6], band[-1, 6])
        opaque, blend = geometry(scene, True)
        self.assertGreater(len(opaque), 200)
        self.assertGreater(len(blend), 200)
        viewer = SRViewer(size=(640, 420), backend="software")
        try:
            img = viewer.render(scene)
        finally:
            viewer.close()
        b, g, r = img.astype(np.int16).transpose(2, 0, 1)
        self.assertGreater(np.count_nonzero((b > 110) & (b > r + 45) & (b > g + 15)), 200)
        self.assertGreater(np.count_nonzero((r > 200) & (g > 150) & (b < 135)), 140)
        self.assertGreater(np.count_nonzero((r > 130) & (r > g + 70)), 30)
        self.assertGreater(np.count_nonzero((r > 195) & (g > 195) & (b > 195)), 200)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sr_preview.png"
            path.write_bytes(cv2.imencode(".png", img)[1].tobytes())
            self.assertGreater(path.stat().st_size, 2_000)


if __name__ == "__main__":
    unittest.main()
