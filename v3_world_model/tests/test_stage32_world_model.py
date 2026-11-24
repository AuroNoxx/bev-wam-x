import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from v1_dual_system.checkpoint_io import VerifiedCheckpointIO
from v3_world_model.geometry import CONFIG_SHA256
from v3_world_model.perception_model import D, LATENT_TOKENS
from v3_world_model.train_stage32 import LatentRolloutDataset, Stage32WorldModel, train
from v3_world_model.world_model import ActionConditionedWorldModel


def _write_synthetic_latents(root: Path, n_train=3, n_val=2, seq_len=8):
    root.mkdir(parents=True, exist_ok=True)
    episodes = []
    splits = {"train": [], "val": []}
    g = torch.Generator().manual_seed(32)
    for split, count, prefix in (("train", n_train, "ep-tr"), ("val", n_val, "ep-val")):
        for idx in range(count):
            ep_id = f"{prefix}-{idx:03d}"
            payload = {
                "episode_id": ep_id,
                "map": "Town12",
                "dt": 0.1,
                "z": torch.randn(seq_len, LATENT_TOKENS, D, generator=g, dtype=torch.bfloat16),
                "ego_delta": torch.randn(seq_len, 3, generator=g, dtype=torch.float32) * 0.2,
                "cmd": torch.tensor([[1., 0., 0.]] * seq_len, dtype=torch.float32),
                "nav_anchors": torch.zeros(seq_len, 3, 3, dtype=torch.float32),
                "ego_speed": torch.ones(seq_len, 2, dtype=torch.float32) * 5.0,
                "ego_state": torch.randn(seq_len, 7, generator=g, dtype=torch.float32) * 0.5,
                "flags": torch.zeros(seq_len, 3, dtype=torch.bool),
                "thw": torch.full((seq_len,), -1.0, dtype=torch.float32),
                "lane_state": torch.zeros(seq_len, 3, dtype=torch.int64),
                "nav_state": torch.ones(seq_len, 3, dtype=torch.float32),
                "trajectory": torch.zeros(seq_len, 25, 3, dtype=torch.float32),
            }
            payload["ego_state"][:, 0] = 12.0
            if idx == 0:
                payload["flags"][2, 1] = True
                payload["thw"][2] = 1.5
            torch.save(payload, root / f"{ep_id}.pt")
            splits[split].append(ep_id)
            episodes.append({"id": ep_id, "map": "Town12", "dt": 0.1, "num_frames": seq_len, "file": f"{ep_id}.pt"})
    manifest = {
        "architecture": "stage32-latents-v1",
        "config_sha256": CONFIG_SHA256,
        "pack_sha256": "synthetic-pack-sha256",
        "splits": splits,
        "episodes": episodes,
        "total_frames": (n_train + n_val) * seq_len,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


class Stage32WorldModelTests(unittest.TestCase):
    def test_rollout_shapes_and_temporal_causality(self):
        torch.manual_seed(0)
        wm = ActionConditionedWorldModel(layers=2, heads=8).eval()
        z0 = torch.randn(2, LATENT_TOKENS, D)
        kin0 = torch.tensor([[0.4, 0.0], [0.5, 0.05]], dtype=torch.float32)
        actions_a = torch.zeros(2, 4, 3, dtype=torch.float32)
        actions_a[:, :, 1] = 0.3
        actions_b = actions_a.clone()
        actions_b[:, 2, 0] = 0.85

        with torch.no_grad():
            out_a = wm.rollout(z0, kin0, actions_a)
            out_b = wm.rollout(z0, kin0, actions_b)

        self.assertEqual(out_a["z_next"].shape, (2, 4, LATENT_TOKENS, D))
        self.assertEqual(out_a["next_kinematics"].shape, (2, 4, 4))
        self.assertEqual(out_a["next_delta"].shape, (2, 4, 3))
        self.assertEqual(out_a["risk_logits"].shape, (2, 4, 3))
        self.assertEqual(out_a["inv_thw"].shape, (2, 4, 1))
        self.assertEqual(out_a["nav_progress"].shape, (2, 4, 3))
        torch.testing.assert_close(out_a["z_next"][:, :2], out_b["z_next"][:, :2], atol=0.0, rtol=0.0)
        torch.testing.assert_close(out_a["next_delta"][:, :2], out_b["next_delta"][:, :2], atol=0.0, rtol=0.0)
        self.assertGreater((out_a["z_next"][:, 2] - out_b["z_next"][:, 2]).abs().max().item(), 1e-4)

    def test_counterfactual_action_sensitivity(self):
        torch.manual_seed(1)
        wm = ActionConditionedWorldModel(layers=2, heads=8).eval()
        z0 = torch.randn(1, LATENT_TOKENS, D)
        kin0 = torch.tensor([[0.4, 0.0]], dtype=torch.float32)
        left = torch.tensor([[[0.7, 0.2, 0.0]]], dtype=torch.float32)
        right = torch.tensor([[[-0.7, 0.2, 0.0]]], dtype=torch.float32)
        with torch.no_grad():
            out_l = wm.rollout(z0, kin0, left)
            out_r = wm.rollout(z0, kin0, right)
        self.assertGreater((out_l["z_next"] - out_r["z_next"]).abs().max().item(), 1e-4)
        self.assertGreater((out_l["next_delta"] - out_r["next_delta"]).abs().max().item(), 1e-4)

    def test_decode_perception_consistency_backpropagates_only_to_world_model(self):
        cfg = dict(
            architecture="stage32-wm-v1",
            config_sha256=CONFIG_SHA256,
            pack_sha256="p",
            latents_manifest_sha256="m",
            perception_ckpt=None,
            horizon=3,
            wm_layers=2,
            wm_heads=8,
            physical_batch=2,
            gradient_accumulation=1,
            max_steps=1,
            checkpoint_every=1,
            learning_rate=3e-4,
            warmup_updates=0,
            schedule_updates=1,
            teacher_forcing=0.0,
            seed=32,
            accelerator="cpu",
            precision="32-true",
            sdpa="math",
            num_workers=0,
        )
        module = Stage32WorldModel(cfg)
        batch = dict(
            z=torch.randn(2, 4, LATENT_TOKENS, D),
            kinematics=torch.randn(2, 4, 4),
            actions=torch.randn(2, 3, 3).clamp(-1., 1.),
            delta=torch.randn(2, 3, 3) * 0.2,
            flags=torch.zeros(2, 3, 3),
            inv_thw=torch.zeros(2, 3, 1),
            nav=torch.ones(2, 3, 3) * 0.5,
        )
        loss = module.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        wm_grads = [p.grad for p in module.world_model.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in wm_grads))
        self.assertTrue(any(g.abs().max().item() > 0 for g in wm_grads))
        self.assertTrue(all((not p.requires_grad) and (p.grad is None) for p in module.perception.parameters()))

    def test_stage32_lightning_train_resume_and_verified_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latents = root / "latents"
            out = root / "run"
            _write_synthetic_latents(latents, n_train=3, n_val=2, seq_len=8)

            train_ds = LatentRolloutDataset(latents, "train", horizon=3, stride=2)
            val_ds = LatentRolloutDataset(latents, "val", horizon=3, stride=3)
            self.assertEqual(set(train_ds.episodes) & set(val_ds.episodes), set())

            base_args = SimpleNamespace(
                latents=latents, perception_ckpt=None, out=out, resume=False,
                max_steps=1, physical_batch=2, grad_accum=1, horizon=3,
                train_stride=2, val_stride=3, layers=2, heads=8, lr=3e-4,
                teacher_forcing=0.0, warmup_updates=0, schedule_updates=4,
                val_check_interval=1.0, checkpoint_every=1, workers=0, seed=320,
                accelerator="cpu", precision="32-true", sdpa="math",
            )
            train(base_args)
            res1 = json.loads((out / "training-result.json").read_text(encoding="utf-8"))
            self.assertEqual(res1["actual_updates"], 1)
            self.assertTrue((out / "best.ckpt").is_file() and (out / "latest.ckpt").is_file())
            self.assertTrue((out / "validation.jsonl").is_file())

            io = VerifiedCheckpointIO()
            ck_best = io.load_checkpoint(out / "best.ckpt", map_location="cpu")
            self.assertEqual(ck_best["task_state"]["config_sha256"], CONFIG_SHA256)

            resume_args = SimpleNamespace(**vars(base_args))
            resume_args.resume = True
            resume_args.max_steps = 2
            train(resume_args)
            res2 = json.loads((out / "training-result.json").read_text(encoding="utf-8"))
            self.assertEqual(res2["actual_updates"], 2)


if __name__ == "__main__":
    unittest.main()
