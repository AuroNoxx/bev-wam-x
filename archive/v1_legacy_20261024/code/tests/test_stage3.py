import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from configs.stage3 import Stage3Config
from dataset.latent import LatentPairs
from dataset.storage import ShardWriter
from evaluation.latent import LatentMoments, paired_episode_bootstrap, spectrum_metrics
from models.latent import LatentModel
from training.vicreg import content_covariance, latent_loss


class PairTests(unittest.TestCase):
    def test_pairs_cross_shards_but_never_resets_gaps_or_action_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'manifest.json').write_text(json.dumps(dict(schema_version=2, completed=True, config=dict(dt_s=.1))))
            state = np.zeros((6, 6), np.float32); state[:, 2] = np.arange(6)*.1
            writer = ShardWriter(root/'train', 2)
            writer.add(dict(bev=np.zeros((6, 4, 128, 128), np.uint8), state=state,
                episode_id=np.array([0, 0, 0, 1, 1, 1]), step_id=np.array([0, 1, 2, 0, 1, 3]),
                future_trajectory=np.zeros((6, 10, 4), np.float32), future_valid=np.zeros((6, 10), bool),
                action_mask=np.zeros(6, bool), noise_active=np.ones(6, bool), recovery_active=np.ones(6, bool)))
            writer.close()
            data = LatentPairs(root)
            np.testing.assert_array_equal(data.anchors, [0, 1, 3])
            self.assertAlmostEqual(float(data[1]['ego'][1]), 1., places=5)
            self.assertEqual(float(data[2]['ego'][1]), 0.)
            self.assertTrue(data[1]['noisy']); self.assertTrue(data[1]['recovery'])


class CovarianceTests(unittest.TestCase):
    def test_position_only_rank_cannot_pass_as_content(self):
        torch.manual_seed(3)
        position = torch.randn(1, 16, 8)
        z = position.repeat(32, 1, 1)
        self.assertLess(float(content_covariance(z).abs().max()), 1e-10)
        moments = LatentMoments(16, 8, 'cpu'); moments.update(z)
        health, _ = moments.finish()
        self.assertLess(health['effective_rank_ratio'], .01)
        self.assertGreater(health['raw_rank_ratio'], .4)

    def test_streamed_health_matches_whole_set_covariance(self):
        torch.manual_seed(4)
        z = torch.randn(31, 4, 7).double() + torch.randn(1, 4, 7).double()*10
        moments = LatentMoments(4, 7, 'cpu')
        for block in z.split(6): moments.update(block)
        _, arrays = moments.finish()
        centered = (z - z.mean(0)).flatten(0, 1)
        expected = centered.T @ centered / (4*30)
        np.testing.assert_allclose(arrays['covariance'], expected.numpy(), atol=1e-12)
        pooled = z.mean(1).numpy()
        np.testing.assert_allclose(arrays['pooled_covariance'], np.cov(pooled, rowvar=False), atol=1e-12)

    def test_entropy_uses_covariance_eigenvalues_without_whitening(self):
        metrics, _ = spectrum_metrics(np.diag([1., 1., 0., 0.]))
        self.assertAlmostEqual(metrics['effective_rank_ratio'], .5)
        self.assertEqual(spectrum_metrics(np.zeros((4, 4)))[0]['effective_rank'], 0.)

    def test_confidence_interval_resamples_paired_episodes(self):
        copy = np.arange(1, 13, dtype=float)
        episode = np.repeat(np.arange(4), 3)
        result = paired_episode_bootstrap(copy*.5, copy, episode, 200, 9)
        self.assertEqual(result['episode_count'], 4)
        self.assertLess(result['delta_ci95'][1], 0)
        np.testing.assert_allclose(result['ratio_ci95'], [.5, .5])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_parameter_budgets(self):
        model = LatentModel(Stage3Config())
        self.assertTrue(6e6 <= sum(p.numel() for p in model.encoder.parameters()) <= 8e6)
        self.assertTrue(15e6 <= sum(p.numel() for p in model.parameters()) <= 30e6)

    def test_no_future_leak_and_both_encoder_branches_have_gradients(self):
        cfg = replace(Stage3Config(), encoder_layers=1, predictor_layers=1)
        model = LatentModel(cfg)
        bev = torch.randint(0, 256, (3, 4, 128, 128), dtype=torch.uint8)
        future = torch.randint(0, 256, bev.shape, dtype=torch.uint8)
        ego = torch.randn(3, 2)
        with sdpa_kernel(SDPBackend.MATH):
            prediction, target, current = model(bev, ego, future)
            self.assertTrue(torch.equal(prediction, current))
            torch.nn.init.normal_(model.predictor.output.weight, std=.01)
            prediction, target, current = model(bev, ego, future)
            alternate = model(bev, ego, future.flip(0))[0]
            torch.testing.assert_close(prediction, alternate)
            target.retain_grad(); current.retain_grad()
            loss, _ = latent_loss(prediction, target, current, cfg); loss.backward()
        self.assertGreater(float(target.grad.norm()), 0)
        self.assertGreater(float(current.grad.norm()), 0)
        self.assertTrue(torch.isfinite(model.encoder.stem[0].weight.grad).all())
        with self.assertRaises(ValueError): model.encoder(bev.float())


if __name__ == '__main__':
    unittest.main()
