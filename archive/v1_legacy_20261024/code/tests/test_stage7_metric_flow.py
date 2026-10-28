"""Scientific loss contracts: legacy parity, units, masks and actual sampling."""
import unittest
import numpy as np
import torch

from mini_wa_jepa.normalization import Normalizer
from models.flow_matcher_metric import MetricFlowMatcher, physical_endpoint_losses
from models.flow_matcher_transformer import TransformerKinematicFlowMatcher


class MetricLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.norm = Normalizer(scale=2., trajectory_mean=np.array([2., 1., 4., .1]),
                               trajectory_std=np.array([3., 4., 5., .2]))
        self.model = MetricFlowMatcher(token_count=4, token_dim=8, d_model=16,
                                      n_heads=2, n_layers=1, dim_feedforward=32,
                                      residual_std=torch.full((10, 4), .4), normalizer=self.norm).eval()
        self.h = torch.randn(2, 4, 4, 8)
        self.e = torch.randn(2, 4, 2) * .1
        self.y = torch.randn(2, 10, 4)
        self.mask = torch.ones(2, 10, dtype=torch.bool)
        self.action = torch.ones(2, dtype=torch.bool)
        self.noise = torch.randn_like(self.y)
        self.t = torch.tensor([.2, .8])

    def test_zero_weights_match_v2_loss_and_gradients(self):
        original = TransformerKinematicFlowMatcher(token_count=4, token_dim=8, d_model=16,
                    n_heads=2, n_layers=1, dim_feedforward=32, normalizer=self.norm).eval()
        original.load_state_dict(self.model.state_dict())
        a, va = original.compute_loss(self.h, self.e, self.y, self.action, self.mask, self.noise, self.t)
        b, vb = self.model.compute_loss(self.h, self.e, self.y, self.action, self.mask, self.noise, self.t)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(va, vb, rtol=0, atol=0)
        a.backward(); b.backward()
        for pa, pb in zip(original.parameters(), self.model.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)

    def test_reconstruction_has_physical_metres_and_endpoint_gradient(self):
        for p in self.model.parameters():
            p.data.zero_()
        loss, _, parts = self.model.compute_loss(self.h, self.e, self.y, self.action, self.mask,
                    self.noise, self.t, ade_weight=.25, fde_weight=.5, return_parts=True)
        prior = self.model.compute_norm_prior(self.e)
        residual = (self.y - prior) / self.model.residual_std
        z_t = (1 - self.t[:, None, None]) * self.noise + self.t[:, None, None] * residual
        raw = self.norm.inverse_trajectory((prior + z_t * self.model.residual_std).detach().numpy())
        gt = self.norm.inverse_trajectory(self.y.numpy())
        distances = np.linalg.norm(raw[..., :2] - gt[..., :2], axis=-1)
        self.assertAlmostEqual(parts['reconstructed_ade_m'].item(), distances.mean(), places=5)
        self.assertAlmostEqual(parts['reconstructed_fde_m'].item(), distances[:, -1].mean(), places=5)
        loss.backward()
        self.assertGreater(self.model.out_proj.bias.grad.abs().sum().item(), 0)

    def test_invalid_nan_labels_and_all_invalid_batch_are_safe(self):
        self.mask[0, 5:] = False
        self.action[1] = False
        self.y[0, 5:] = float('nan'); self.y[1] = float('nan')
        for valid in [self.mask, torch.zeros_like(self.mask)]:
            self.model.zero_grad()
            loss, _, p = self.model.compute_loss(self.h, self.e, self.y, self.action, valid,
                        self.noise, self.t, ade_weight=.25, fde_weight=.5, return_parts=True)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(p['reconstructed_fde_m'].item(), 0.)
            loss.backward()
            self.assertTrue(all(torch.isfinite(x.grad).all() for x in self.model.parameters() if x.grad is not None))
            if not valid.any(): self.assertEqual(loss.item(), 0.)

    def test_terminal_loss_does_not_substitute_earlier_valid_waypoint(self):
        predicted = torch.ones(2, 10, 4, requires_grad=True)
        target = torch.zeros_like(predicted)
        valid = torch.ones(2, 10, dtype=torch.bool); valid[:, -1] = False
        ade, fde = physical_endpoint_losses(predicted, target, valid)
        self.assertAlmostEqual(ade.item(), 2 ** .5, places=5)
        self.assertEqual(fde.item(), 0.)

    def test_batched_evaluation_keeps_original_per_anchor_noise(self):
        from scripts.run_stage7_metric_ablation import sample_anchors
        with torch.no_grad():
            batched = sample_anchors(self.model, self.h, self.e, [7, 91], .8)
            for i, anchor in enumerate([7, 91]):
                single = self.model.sample(self.h[i:i+1], self.e[i:i+1],
                    num_candidates=8, temperature=.8, seed=42+anchor)
                torch.testing.assert_close(batched[i], single[0], rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
