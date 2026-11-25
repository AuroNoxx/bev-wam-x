"""Tests for Stage 3.1c ResNet34-FPN + CenterPoint/DN-DETR + MapTR perception pipeline."""
import argparse
import gc
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from v1_dual_system.checkpoint_io import VerifiedCheckpointIO
from v3_world_model.build_maptr_targets import extract_frame_maptr, pack_overlay, unpack_overlay
from v3_world_model.geometry import H, NAMES, SIZES, W, grid, metric
from v3_world_model.labels import MAP_POINTS, MAP_SLOTS, OBJECT_CLASSES, OBJECT_SLOTS, VECTOR_X
from v3_world_model.objectives import maptr_loss, stage31c_losses
from v3_world_model.pack_colab_v3 import bit_pack, pack, read_manifest, unpack_verified
from v3_world_model.perception_model import BEVWAMXPerception, CLIP_LEN, _sample_bev_grid
from v3_world_model.sr_viewer import SRViewer, prediction_scene
from v3_world_model.train_stage31_colab import Stage31, lock, train
from v3_world_model.tests.test_stage3_prep import create_capture


class Stage31cPerceptionTests(unittest.TestCase):
    def _dummy_inputs(self, batch=2):
        rgb = tuple(torch.randint(0, 255, (batch, 3, h, w), dtype=torch.uint8) for w, h in SIZES)
        lidar = torch.randn(batch, 6, H, W, dtype=torch.float32) * 0.2
        cmd = torch.tensor([[1.0, 0.0, 0.0]] * batch, dtype=torch.float32)
        nav = torch.tensor([[[12.0, 0.0, 0.0], [24.0, 0.2, 0.0], [36.0, 0.5, 0.0]]] * batch, dtype=torch.float32)
        spd = torch.tensor([[15.0, 0.02]] * batch, dtype=torch.float32)
        return rgb, lidar, cmd, nav, spd

    def _dummy_targets(self, batch=2):
        boxes = torch.zeros(batch, OBJECT_SLOTS, 11, dtype=torch.float32)
        boxes[:, 0] = torch.tensor([18.0, 0.0, 0.75, 4.5, 1.9, 1.5, 0.0, 12.0, 0.0, 0.0, 1.0])
        boxes[:, 1] = torch.tensor([26.0, -3.5, 0.70, 2.0, 0.8, 1.4, 0.05, 10.0, 0.0, 3.0, 1.0])
        lanes = torch.zeros(batch, 4, H, W, dtype=torch.float32)
        lanes[:, 0, :, 40:80] = 1.0
        lanes[:, 1, :, 55] = 1.0
        map_polys = torch.zeros(batch, MAP_SLOTS, MAP_POINTS, 2, dtype=torch.float32)
        xs = torch.linspace(-10.0, 50.0, MAP_POINTS)
        map_polys[:, 0] = torch.stack((xs, torch.full_like(xs, -1.8)), dim=-1)
        map_polys[:, 1] = torch.stack((xs, torch.full_like(xs, 1.8)), dim=-1)
        map_cls = torch.zeros(batch, MAP_SLOTS, dtype=torch.long)
        map_cls[:, 1] = 1
        map_valid = torch.zeros(batch, MAP_SLOTS, dtype=torch.bool)
        map_valid[:, :2] = True
        return dict(
            lanes=lanes,
            lanes_cont=lanes,
            map_polylines=map_polys,
            map_classes=map_cls,
            map_valid=map_valid,
            boxes=boxes,
            vectors=torch.zeros(batch, 7, len(VECTOR_X), 2, dtype=torch.float32),
            vector_valid=torch.ones(batch, 7, len(VECTOR_X), dtype=torch.bool),
            vector_exists=torch.ones(batch, 7, dtype=torch.bool),
            occupancy=torch.zeros(batch, 8, H, W, dtype=torch.float32),
            future_occupancy=torch.zeros(batch, 5, H, W, dtype=torch.float32),
            trajectory=torch.zeros(batch, 25, 3, dtype=torch.float32),
        )

    def test_sample_bev_grid_exact_pixel_alignment(self):
        xy_grid = torch.from_numpy(grid((H, W)).astype(np.float32))  # (H, W, 2)
        bev = xy_grid.permute(2, 0, 1).unsqueeze(0)  # (1, 2, H, W)
        pts = xy_grid[17, 43].reshape(1, 1, 2)
        sampled = _sample_bev_grid(bev, pts)
        torch.testing.assert_close(sampled[0, 0], xy_grid[17, 43], atol=1e-4, rtol=1e-4)

    def test_stage31c_forward_backward_dn_isolation_and_latent_decode(self):
        torch.manual_seed(11)
        model = BEVWAMXPerception(lidar_dropout=0.0, arch="stage31c", pretrained=False).train()
        rgb, lidar, cmd, nav, spd = self._dummy_inputs(batch=2)
        targets = self._dummy_targets(batch=2)

        pred = model(rgb, lidar_bev=lidar, cmd=cmd, nav_anchors=nav, ego_speed=spd, gt_boxes=targets["boxes"])
        self.assertIn("dn_outputs", pred)
        self.assertEqual(pred["center_heatmap_logits"].shape, (2, len(OBJECT_CLASSES), H, W))
        self.assertEqual(pred["map_polylines"].shape, (2, MAP_SLOTS, MAP_POINTS, 2))
        self.assertEqual(pred["boxes"].shape, (2, OBJECT_SLOTS, 10))
        self.assertEqual(pred["z_t"].shape, (2, 128, 256))

        losses = stage31c_losses(pred, targets)
        total = sum(losses.values())
        self.assertTrue(torch.isfinite(total))
        total.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), name)

        # Verify eval + intent isolation + bev=None Stage 3.2 rollout compatibility.
        model.eval()
        with torch.no_grad():
            p_a = model(rgb, lidar_bev=lidar, cmd=cmd, nav_anchors=nav, ego_speed=spd)
            p_b = model(rgb, lidar_bev=lidar,
                        cmd=torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                        nav_anchors=nav + 5.0,
                        ego_speed=spd + 10.0)
            for key in ("z_t", "boxes", "obj_logits", "cls_logits", "map_polylines", "map_obj_logits", "lane_logits"):
                self.assertEqual(float((p_a[key] - p_b[key]).abs().max()), 0.0, key)
            p_dec = model.decode_perception(p_a["z_t"], bev=None)
            self.assertEqual(p_dec["boxes"].shape, (2, OBJECT_SLOTS, 10))
            self.assertEqual(p_dec["map_polylines"].shape, (2, MAP_SLOTS, MAP_POINTS, 2))
            self.assertTrue(torch.isfinite(p_dec["lane_logits"]).all())

    def test_final_map_coordinates_backpropagate_through_all_refinements(self):
        model = BEVWAMXPerception(lidar_dropout=0., arch="stage31c", pretrained=False).eval()
        # Constant residuals isolate the coordinate path from gradients through query features.
        with torch.no_grad():
            model.map_init_pts[-1].weight.zero_()
            model.map_init_pts[-1].bias.zero_()
            for head in model.map_pt_heads:
                head[-1].weight.zero_()
                head[-1].bias.fill_(0.5)
        points = model.decode_perception(torch.zeros(1, 128, 256))["map_polylines"]
        expected = torch.tensor([60., 30.]) * torch.tensor(1.5).tanh() + torch.tensor([20., 0.])
        torch.testing.assert_close(points, expected.expand_as(points))
        points.abs().mean().backward()
        for head in (model.map_init_pts, *model.map_pt_heads):
            grad = head[-1].bias.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all() and (grad != 0).all())

    def test_maptr_loss_is_direction_permutation_invariant(self):
        torch.manual_seed(7)
        xs = torch.linspace(-10.0, 50.0, MAP_POINTS)
        poly_fwd = torch.stack((xs, 0.01 * xs * xs - 2.0), dim=-1)
        poly_rev = poly_fwd.flip(0)
        gt_f = torch.zeros(1, MAP_SLOTS, MAP_POINTS, 2)
        gt_r = torch.zeros(1, MAP_SLOTS, MAP_POINTS, 2)
        gt_f[0, 0] = poly_fwd
        gt_r[0, 0] = poly_rev
        cls_t = torch.zeros(1, MAP_SLOTS, dtype=torch.long)
        valid_t = torch.zeros(1, MAP_SLOTS, dtype=torch.bool)
        valid_t[0, 0] = True

        p_pts = torch.randn(1, MAP_SLOTS, MAP_POINTS, 2) * 0.2
        p_pts[0, 3] = poly_fwd + 0.15
        p_cls = torch.zeros(1, MAP_SLOTS, 3)
        p_obj = torch.full((1, MAP_SLOTS), -3.0)
        p_obj[0, 3] = 3.0
        pred = dict(map_polylines=p_pts, map_cls_logits=p_cls, map_obj_logits=p_obj, aux_map=())

        loss_f = maptr_loss(pred, dict(map_polylines=gt_f, map_classes=cls_t, map_valid=valid_t))
        loss_r = maptr_loss(pred, dict(map_polylines=gt_r, map_classes=cls_t, map_valid=valid_t))
        torch.testing.assert_close(loss_f, loss_r, atol=1e-5, rtol=1e-5)

    def test_extract_frame_maptr_and_sr_viewer_vector_render(self):
        masks = np.zeros((4, H, W), dtype=np.uint8)
        y = metric(np.column_stack((np.arange(W), np.zeros(W))))[:, 1]
        masks[0, :, np.abs(y) < 6.0] = 1
        col_c = int(W // 2)
        for row in range(0, H, 1):
            if row % 12 < 4:
                masks[1, row, col_c - 6] = 1
            masks[2, row, col_c + 6] = 1
        polys, cls_ids, valid, cont = extract_frame_maptr(masks)
        self.assertTrue(valid.any())
        self.assertIn(0, cls_ids[valid].tolist())
        self.assertIn(1, cls_ids[valid].tolist())
        self.assertIn(2, cls_ids[valid].tolist())
        self.assertEqual(cont.shape, (4, H, W))
        # Bit-pack exactness check.
        packed = bit_pack(cont)
        unpacked = np.unpackbits(packed, count=4 * H * W, bitorder="little").reshape(4, H, W)
        self.assertTrue(np.array_equal(cont, unpacked))

        # Verify SRViewer renders Stage 3.1c predictions (including map_polylines_vis) cleanly on software backend.
        model = BEVWAMXPerception(lidar_dropout=0.0, arch="stage31c", pretrained=False).eval()
        rgb, lidar, cmd, nav, spd = self._dummy_inputs(batch=1)
        with torch.no_grad():
            pred = model(rgb, lidar_bev=lidar, cmd=cmd, nav_anchors=nav, ego_speed=spd)
            pred["map_obj_logits"].fill_(3.0)
        scene = prediction_scene(pred, speed=15.0, cruise=20.0, clean=True, mode="LIVE")
        self.assertEqual(scene["road_source"], "dense")
        self.assertIsNotNone(scene["map_polylines_vis"])
        viewer = SRViewer(size=(480, 270), backend="software")
        img = viewer.render(scene)
        self.assertEqual(img.shape, (270, 480, 3))
        self.assertGreater(int(img.std()), 5)
        scene_maptr = prediction_scene(pred, speed=15.0, cruise=20.0, clean=True, mode="LIVE", road_source="maptr")
        self.assertEqual(scene_maptr["road_source"], "maptr")
        img_maptr = viewer.render(scene_maptr)
        self.assertEqual(img_maptr.shape, (270, 480, 3))
        self.assertGreater(int(img_maptr.std()), 5)

    def test_stage31c_lightning_train_resume_and_overlay_roundtrip(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source, packed, dataset = root / "source", root / "packed", root / "dataset"
            create_capture(source)
            pack([source], packed, validation_fraction=0.25)
            unpack_verified(packed / "manifest.json", dataset)
            manifest = read_manifest(dataset / "manifest.json", check_shards=False)
            for ep in manifest["episodes"]:
                ep_dir = dataset / ep["id"]
                with np.load(ep_dir / "targets.npz") as t:
                    lanes_arr = t["lanes"].copy()
                nf = int(lanes_arr.shape[0])
                polys = np.zeros((nf, MAP_SLOTS, MAP_POINTS, 2), dtype=np.float16)
                xs = np.linspace(-10.0, 40.0, MAP_POINTS, dtype=np.float32)
                polys[:, 0] = np.stack((xs, np.full_like(xs, -1.8)), axis=-1).astype(np.float16)
                cls_ids = np.zeros((nf, MAP_SLOTS), dtype=np.uint8)
                valid = np.zeros((nf, MAP_SLOTS), dtype=bool)
                valid[:, 0] = True
                np.savez_compressed(ep_dir / "maptr_targets.npz", map_polylines=polys,
                                    map_classes=cls_ids, map_valid=valid, lanes_cont=lanes_arr)

            overlay_dir = root / "overlay"
            pack_overlay(dataset, manifest, overlay_dir)
            for ep in manifest["episodes"]:
                (dataset / ep["id"] / "maptr_targets.npz").unlink()
            unpack_overlay(overlay_dir, dataset)
            for ep in manifest["episodes"]:
                self.assertTrue((dataset / ep["id"] / "maptr_targets.npz").is_file())

            run = root / "run31c"
            args = argparse.Namespace(
                dataset=dataset, out=run, arch="stage31c", balanced_maps=True, no_pretrained=True,
                batch=1, accumulate=1, max_updates=1, checkpoint_every=1, lr=2e-4,
                lidar_dropout=0.0, seed=317, accelerator="cpu", precision="32-true",
                sdpa="math", workers=0, resume=False, warmup_updates=2, schedule_updates=4,
            )
            with lock():
                train(args)
            res1 = json.loads((run / "training-result.json").read_text(encoding="utf-8"))
            self.assertEqual(res1["architecture"], "stage31c")
            self.assertEqual(res1["actual_updates"], 1)
            val_rows = [json.loads(line) for line in (run / "validation.jsonl").read_text(encoding="utf-8").splitlines() if line]
            self.assertTrue(val_rows)
            for k in ("val_det_p3m", "val_det_r3m", "val_det_f1", "val_map_l1_m", "val_lane_dashed_iou"):
                self.assertIn(k, val_rows[-1])

            ckpt = VerifiedCheckpointIO().load_checkpoint(run / "latest.ckpt", map_location="cpu", weights_only=False)
            self.assertTrue((run / "best_det.ckpt").is_file())
            self.assertEqual(len(ckpt["optimizer_states"][0]["param_groups"]), 2)
            head_lr = ckpt["optimizer_states"][0]["param_groups"][0]["lr"]
            bb_lr = ckpt["optimizer_states"][0]["param_groups"][1]["lr"]
            self.assertAlmostEqual(bb_lr, head_lr * 0.3, places=9)
            del ckpt

            args.resume, args.max_updates = True, 2
            with lock():
                train(args)
            res2 = json.loads((run / "training-result.json").read_text(encoding="utf-8"))
            self.assertEqual(res2["actual_updates"], 2)

            run_distill = root / "run31c_zdistill"
            args_z = argparse.Namespace(
                dataset=dataset, out=run_distill, arch="stage31c", balanced_maps=True, no_pretrained=True,
                init_from=run / "latest.ckpt", z_aux_weight=0.6, z_distill_weight=0.5, z_all_frames=True,
                batch=1, accumulate=1, max_updates=1, checkpoint_every=1, lr=5e-5,
                lidar_dropout=0.0, seed=317, accelerator="cpu", precision="32-true",
                sdpa="math", workers=0, resume=False, warmup_updates=1, schedule_updates=2,
            )
            with lock():
                train(args_z)
            z_rows = [json.loads(line) for line in (run_distill / "validation.jsonl").read_text(encoding="utf-8").splitlines() if line]
            self.assertTrue(z_rows)
            for k in ("val_z_det_p3m", "val_z_det_r3m", "val_z_det_f1", "val_z_drivable_iou"):
                self.assertIn(k, z_rows[-1])
            self.assertTrue((run_distill / "best_z_det.ckpt").is_file())
            gc.collect()

    def test_nominal_action_hermite_avoids_early_curb_cut_and_slows_in_90deg_turn(self):
        from v3_world_model.live_carla_demo import _nominal_action

        traj = np.column_stack((np.linspace(0.6, 15.0, 25), np.zeros(25), np.full(25, 6.0))).astype(np.float32)
        nav_turn = np.array([[9.0, -6.0, 0.0], [9.0, -18.0, 0.0], [9.0, -30.0, 0.0]], dtype=np.float32)
        steer_turn, _, brake_turn, v_turn = _nominal_action(traj, nav_turn, speed_mps=6.5, zone_limit_mps=7.0, last_steer=0.0)
        self.assertGreater(steer_turn, 0.0)
        self.assertLess(steer_turn, 0.30)
        self.assertLess(v_turn, 5.0)
        self.assertGreater(brake_turn, 0.0)

        nav_straight = np.array([[12.0, 0.0, 0.0], [24.0, 0.0, 0.0], [36.0, 0.0, 0.0]], dtype=np.float32)
        steer_st, _, brake_st, v_st = _nominal_action(traj, nav_straight, speed_mps=14.0, zone_limit_mps=15.3, last_steer=0.0)
        self.assertAlmostEqual(steer_st, 0.0, places=3)
        self.assertEqual(brake_st, 0.0)
        self.assertAlmostEqual(v_st, 15.3, places=1)

        # Approaching a 90deg right turn still 12m ahead must NEVER swerve left (zero Runge overshoot).
        nav_before_turn = np.array([[12.0, 0.0, 0.0], [12.0, -12.0, 0.0], [12.0, -24.0, 0.0]], dtype=np.float32)
        steer_pre, _, _, v_pre = _nominal_action(traj, nav_before_turn, speed_mps=6.5, zone_limit_mps=7.0, last_steer=0.0)
        self.assertGreaterEqual(steer_pre, -1e-4)
        self.assertLess(v_pre, 5.0)

    def test_stage32_supports_stage31c_perception_and_nested_finite_check(self):
        from v3_world_model.eval_stage32 import _finite_tensor_dict
        from v3_world_model.train_stage32 import Stage32WorldModel

        cfg = dict(wm_layers=2, wm_heads=4, perception_architecture="stage31c", perception_ckpt=None)
        mod = Stage32WorldModel(cfg).eval()
        self.assertEqual(mod.perception.arch, "stage31c")
        with torch.no_grad():
            dec = mod.perception.decode_perception(torch.zeros(1, 128, 256), bev=None)
        self.assertTrue(_finite_tensor_dict(dec))


if __name__ == "__main__":
    unittest.main()
