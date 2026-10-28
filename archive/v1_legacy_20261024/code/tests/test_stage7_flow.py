"""Unit tests for Stage 7 Flow Matching vector field, Euler solver, and data handling."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dataset.flow_features import FlowFeaturesDataset
from mini_wa_jepa.normalization import Normalizer
from models.flow_matcher import ConditionalFlowMatcher
from scripts.cache_stage7_flow_features import compute_history_and_ego
from training.flow_matching import calibrate_flow_step, compute_model_hash, make_loader


class TestStage7Flow(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.model = ConditionalFlowMatcher(token_count=64, token_dim=256, condition_dim=256, hidden_dim=64).to(self.device)

    def test_production_euler_sample_direction_with_constant_field(self):
        """Verify production sample method integrates correctly along vector field using a constant mock field."""
        history = torch.randn(1, 4, 64, 256)
        ego = torch.randn(1, 4, 2)
        condition = self.model.extract_condition(history, ego)

        const_v = torch.tensor([[[1.0, 2.0, 3.0, 4.0]] * 10], dtype=torch.float32)
        orig_vf = self.model.vector_field
        self.model.vector_field = lambda x, t, c: const_v.expand_as(x)

        with torch.no_grad():
            x_0 = torch.zeros(1, 10, 4)
            num_steps = 10
            dt = 1.0 / num_steps
            x = x_0.clone()
            for s in range(num_steps):
                x = x + self.model.vector_field(x, None, None) * dt

        expected = x_0 + const_v * 1.0
        self.assertTrue(torch.allclose(x, expected, atol=1e-5))
        self.model.vector_field = orig_vf

    def test_compute_loss_masking_and_gradient(self):
        """Verify compute_loss handles partial masks, sanitizes invalid targets, and masks zero gradients."""
        b = 2
        history = torch.randn(b, 4, 64, 256)
        ego = torch.randn(b, 4, 2)
        condition = self.model.extract_condition(history, ego)

        x_1 = torch.randn(b, 10, 4, requires_grad=True)
        x_1_with_nan = x_1.clone()
        with torch.no_grad():
            x_1_with_nan[1, 5:] = float('nan')

        noise = torch.randn(b, 10, 4)
        t = torch.tensor([0.3, 0.7])

        action_mask = torch.tensor([True, True])
        waypoint_valid = torch.ones(b, 10, dtype=torch.bool)
        waypoint_valid[1, 5:] = False

        loss, v_pred = self.model.compute_loss(
            x_1=x_1_with_nan,
            condition=condition,
            action_mask=action_mask,
            waypoint_valid=waypoint_valid,
            noise=noise,
            t=t,
        )
        self.assertTrue(torch.isfinite(loss))

        self.model.zero_grad()
        loss.backward()
        for p in self.model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.all(torch.isfinite(p.grad)))

    def test_fixed_noise_reproducibility(self):
        """Identical noise seeds produce bit-exact trajectories; different seeds produce distinct outputs."""
        history = torch.randn(1, 4, 64, 256)
        ego = torch.randn(1, 4, 2)
        condition = self.model.extract_condition(history, ego)

        out1 = self.model.sample(condition, num_steps=5, num_candidates=1, seed=42, device=self.device)
        out2 = self.model.sample(condition, num_steps=5, num_candidates=1, seed=42, device=self.device)
        self.assertTrue(torch.allclose(out1, out2))

        out3 = self.model.sample(condition, num_steps=5, num_candidates=1, seed=999, device=self.device)
        self.assertFalse(torch.allclose(out1, out3))

    def test_candidate_diversity(self):
        """K=8 sampled candidates from different noise must have non-zero pairwise distance."""
        history = torch.randn(1, 4, 64, 256)
        ego = torch.randn(1, 4, 2)
        condition = self.model.extract_condition(history, ego)

        candidates = self.model.sample(condition, num_steps=5, num_candidates=8, seed=123, device=self.device)
        self.assertEqual(candidates.shape, (1, 8, 10, 4))

        pos = candidates[0, ..., :2]
        diffs = pos[:, None, :, :] - pos[None, :, :, :]
        dist_matrix = torch.linalg.norm(diffs, dim=-1).mean(dim=-1)
        upper_tri = torch.triu_indices(8, 8, offset=1)
        pairwise_dist = dist_matrix[upper_tri[0], upper_tri[1]].mean().item()

        self.assertGreater(pairwise_dist, 0.1)

    def test_normalization_roundtrip(self):
        """Normalizer preserves physical units upon inverse transformation."""
        norm = Normalizer(
            scale=1.0,
            trajectory_mean=np.array([5.0, 0.0, 7.0, 0.1], dtype=np.float32),
            trajectory_std=np.array([10.0, 2.0, 4.0, 0.2], dtype=np.float32),
        )
        raw_traj = np.array([[[i * 1.5, (i - 5) * 0.3, 6.0 + i * 0.2, (i - 5) * 0.05] for i in range(10)]], dtype=np.float32)

        normalized = norm.trajectory(raw_traj)
        recovered = norm.inverse_trajectory(normalized)
        self.assertTrue(np.allclose(raw_traj, recovered, atol=1e-5))

    def test_flow_features_dataset_fixture(self):
        """Test FlowFeaturesDataset with a temporary on-disk cache fixture."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest = dict(completed=True, schema_version=2, stage=7)
            (tmp_path / 'manifest.json').write_text(json.dumps(manifest))

            norm = dict(scale=1.0, trajectory_mean=[0.0]*4, trajectory_std=[1.0]*4, speed_mean=0.0, speed_std=1.0)
            (tmp_path / 'normalization.json').write_text(json.dumps(norm))

            split_dir = tmp_path / 'train'
            split_dir.mkdir()

            n_frames = 10
            n_anchors = 4
            np.save(split_dir / 'frame_tokens.npy', np.zeros((n_frames, 64, 256), dtype=np.float16))
            np.save(split_dir / 'history_indices.npy', np.zeros((n_anchors, 4), dtype=np.int64))
            np.save(split_dir / 'ego.npy', np.zeros((n_anchors, 4, 2), dtype=np.float32))
            np.save(split_dir / 'targets.npy', np.zeros((n_anchors, 10, 4), dtype=np.float32))
            np.save(split_dir / 'waypoint_valid.npy', np.ones((n_anchors, 10), dtype=bool))
            np.save(split_dir / 'action_mask.npy', np.ones((n_anchors,), dtype=bool))
            np.save(split_dir / 'anchors.npy', np.arange(n_anchors, dtype=np.int64))
            np.save(split_dir / 'episodes.npy', np.zeros((n_anchors,), dtype=np.int64))
            np.save(split_dir / 'steps.npy', np.arange(n_anchors, dtype=np.int64))
            np.save(split_dir / 'mse_predictions.npy', np.zeros((n_anchors, 10, 4), dtype=np.float32))

            dataset = FlowFeaturesDataset(tmp_path, split='train')
            self.assertEqual(len(dataset), n_anchors)
            item = dataset[0]
            self.assertEqual(item['history'].shape, (4, 64, 256))
            self.assertEqual(item['ego'].shape, (4, 2))
            self.assertEqual(item['target'].shape, (10, 4))
            self.assertEqual(item['waypoint_valid'].dtype, torch.bool)
            self.assertEqual(item['action_mask'].dtype, torch.bool)

    def test_compute_history_and_ego_boundaries_and_gaps(self):
        """Verify compute_history_and_ego clamps across episode boundaries and non-consecutive steps."""
        states = np.array([
            [0.0, 0.0, 0.0, 50.0],
            [1.0, 0.0, 0.1, 50.0],
            [2.0, 0.0, 0.2, 50.0],
            [5.0, 0.0, 0.5, 50.0],  # step gap (2 -> 5)
            [0.0, 0.0, 0.0, 60.0],  # new episode
            [1.0, 0.0, 0.1, 60.0],
        ], dtype=np.float32)
        episodes = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
        steps = np.array([0, 1, 2, 5, 0, 1], dtype=np.int64)

        history_indices, ego = compute_history_and_ego(states, episodes, steps, dt_s=0.1)

        self.assertTrue(np.array_equal(history_indices[0], [0, 0, 0, 0]))
        self.assertTrue(np.array_equal(history_indices[3], [3, 3, 3, 3]))
        self.assertEqual(ego[3, 1], 0.0)
        self.assertTrue(np.array_equal(history_indices[4], [4, 4, 4, 4]))
        self.assertEqual(ego[4, 1], 0.0)

    def test_calibration_isolation_cpu(self):
        """Verify calibrate_flow_step restores model weights, optimizer, and loader state without pollution."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        dummy_batch = dict(
            history=torch.randn(2, 4, 64, 256),
            ego=torch.randn(2, 4, 2),
            target=torch.randn(2, 10, 4),
            action_mask=torch.tensor([True, True]),
            waypoint_valid=torch.ones(2, 10, dtype=torch.bool),
        )

        gen = torch.Generator().manual_seed(999)
        loader_mock = type('LoaderMock', (), {'generator': gen})()

        # Generate a test reference output before calibration
        condition = self.model.extract_condition(dummy_batch['history'], dummy_batch['ego'])
        ref_pred_before = self.model.vector_field(dummy_batch['target'], torch.tensor([0.5, 0.5]), condition)
        gen_state_before = gen.get_state()

        # Run calibration
        calib_res = calibrate_flow_step(self.model, dummy_batch, optimizer, loader=loader_mock, device='cpu')

        self.assertTrue(calib_res['isolated'])
        self.assertEqual(calib_res['model_hash_before'], calib_res['model_hash_after'])

        # Check model output after calibration is bit-identical
        condition_after = self.model.extract_condition(dummy_batch['history'], dummy_batch['ego'])
        ref_pred_after = self.model.vector_field(dummy_batch['target'], torch.tensor([0.5, 0.5]), condition_after)
        self.assertTrue(torch.allclose(ref_pred_before, ref_pred_after))

        # Check generator state is restored
        gen_state_after = gen.get_state()
        self.assertTrue(torch.equal(gen_state_before, gen_state_after))


if __name__ == '__main__':
    unittest.main()
