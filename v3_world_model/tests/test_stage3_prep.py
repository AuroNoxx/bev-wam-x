"""Offline acceptance tests. Run with python -m unittest discover -s v3_world_model/tests -v."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from v1_dual_system.checkpoint_io import VerifiedCheckpointIO
from v3_world_model import collect_v3
from v3_world_model.calibration_probe import coverage
from v3_world_model.geometry import (CONFIG_SHA256, DT, H, W, Z, NAMES, SIZES, camera_mounts,
                                     carla_pose, delta_error, ego_delta, grid, metric, pixels,
                                     project, projection_tables, world_at)
from v3_world_model.labels import decode_boxes, encode_boxes, exclusive_markings, object_slots
from v3_world_model.objectives import stage31_losses
from v3_world_model.pack_colab_v3 import assert_split, main as pack_cli, read_manifest, unpack_verified
from v3_world_model.perception_model import BEVWAMXPerception, sdpa_context
from v3_world_model.qc_v3 import LABEL_SHAPES, event_count
from v3_world_model.sim import connect
from v3_world_model.sr_viewer import SRViewer, geometry, ribbon, synthetic_scene
from v3_world_model.train_stage31_colab import Stage31, lock, train


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
                    initial_state_id=f"{town}/{start}", scenario="straight")
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
                                ego_delta=ego_delta(previous, pose) if previous is not None else np.zeros(3, np.float32))
            cameras = pose @ camera_mounts()
            sensor_poses = np.concatenate((cameras, pose[None], cameras))
            np.savez_compressed(frame / "audit.npz", pose=pose, lidar_points=scan,
                                sensor_poses=sensor_poses, sensor_ids=np.full(13, fid, np.int64),
                                timestamps=np.full(13, fid * DT, np.float64), lidar_angle=np.float64(0))
            np.savez_compressed(frame / "labels.npz", **labels())
            np.savez_compressed(frame / "depth.npz",
                                **{n: np.full((h, w), 1200, np.uint16) for n, (w, h) in zip(NAMES, SIZES)})
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

    def test_model_forward_gradient_and_intent_isolation(self):
        torch.manual_seed(317)
        model = BEVWAMXPerception(lidar_dropout=0).eval()
        params = sum(p.numel() for p in model.parameters())
        self.assertTrue(20_000_000 <= params <= 30_000_000)
        rgb = tuple(torch.randint(0, 256, (1, 3, h, w), dtype=torch.uint8) for w, h in SIZES)
        lidar = torch.randn(1, 6, H, W)
        prev = torch.randn(1, 64, 256)
        motion = torch.tensor([[-.93, .04, -.02]])
        with sdpa_context("math"):
            left = model(rgb, lidar, prev, motion, torch.tensor([[0., 1., 0.]]))
            with torch.no_grad():
                right = model(rgb, lidar, prev, motion, torch.tensor([[0., 0., 1.]]))
        self.assertEqual(tuple(left["z_t"].shape), (1, 64, 256))
        for key in left:
            self.assertTrue(torch.isfinite(left[key]).all(), key)
            if key != "trajectory":
                self.assertEqual((left[key] - right[key]).abs().max().item(), 0.0, key)
        self.assertGreater((left["trajectory"] - right["trajectory"]).abs().max().item(), 0)
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
                                    "--max-shard-bytes", "180000", "--verify-pack"]):
                pack_cli()
            result = read_manifest(packed / "manifest.json")
            self.assertGreaterEqual(len(result["shards"]), 2)
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
            # Training decodes only obs.npz, the six WebPs, and training targets.
            for path in dataset.glob("*/*/*.npz"):
                if path.name in ("audit.npz", "depth.npz"):
                    path.unlink()
            run = base / "run"
            args = argparse.Namespace(dataset=dataset, out=run, batch=1, accumulate=1, max_updates=1,
                                      checkpoint_every=1, lr=2e-4, lidar_dropout=0., seed=317, accelerator="cpu",
                                      precision="32-true", sdpa="math", workers=0, resume=False)
            with lock():
                train(args)
            info = json.loads((run / "training-result.json").read_text(encoding="utf-8"))
            self.assertEqual(info["actual_updates"], 1)
            self.assertFalse(list(run.glob("*.pending")))
            io = VerifiedCheckpointIO()
            ckpt = io.load_checkpoint(run / "latest.ckpt", map_location="cpu", weights_only=False)
            best = io.load_checkpoint(run / "best.ckpt", map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["global_step"], 1)
            self.assertEqual(best["global_step"], 1)
            self.assertTrue(ckpt["optimizer_states"])
            self.assertTrue(ckpt["optimizer_states"][0]["param_groups"][0]["foreach"])
            self.assertTrue((run / "validation.jsonl").read_text(encoding="utf-8").strip())
            restored = Stage31(ckpt["task_state"])
            restored.load_state_dict(ckpt["state_dict"])
            for name, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, ckpt["state_dict"][name]), name)
                self.assertTrue(torch.equal(value, best["state_dict"][name]), name)
            del ckpt, best, restored
            args.resume, args.max_updates = True, 2
            with lock():
                train(args)
            result = json.loads((run / "training-result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["actual_updates"], 2)

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
