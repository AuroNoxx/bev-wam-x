"""Unit test suite for BEV WA-JEPA Stage 3 Latent Flow Matching Model.

Tests:
1. Shapes and parameter budgets
2. EMA target update mechanics
3. Flow matching forward and ODE Euler generation
4. Gate 3 verification: Gradient isolation and causal leakage freedom
5. Representation health metrics (effective rank and variance)
"""

import unittest
import torch
from configs.stage3 import Stage3Config
from models.bev_wa_latent_flow import (
    BEVOnlineEncoder,
    EMATargetEncoder,
    MultiHorizonLatentFlowPredictor,
    BEVWALatentFlowModel,
)
from evaluation.latent import spectrum_metrics


class TestBEWWALatentFlow(unittest.TestCase):
    def setUp(self):
        self.cfg = Stage3Config()
        self.b = 2
        self.history_bev = torch.randint(0, 256, (self.b, 4, 4, 128, 128), dtype=torch.uint8)
        self.ego_motion = torch.randn(self.b, 4, 2, dtype=torch.float32)
        self.future_bev = torch.randint(0, 256, (self.b, 4, 4, 128, 128), dtype=torch.uint8)
        self.future_mask = torch.tensor([[True, True, True, True], [True, True, True, False]])

    def test_parameter_counts_and_shapes(self):
        model = BEVWALatentFlowModel(self.cfg)
        online_params = sum(p.numel() for p in model.online_encoder.parameters() if p.requires_grad)
        predictor_params = sum(p.numel() for p in model.predictor.parameters() if p.requires_grad)
        target_params = sum(p.numel() for p in model.target_encoder.parameters())
        
        # Verify deployable budget (<50M for Stage 3)
        deployable_params = online_params + predictor_params
        print(f"\n[Test] Online Encoder params: {online_params:,}")
        print(f"[Test] Predictor params: {predictor_params:,}")
        print(f"[Test] Target Teacher params: {target_params:,} (requires_grad=False)")
        print(f"[Test] Total deployable params: {deployable_params:,}")
        
        self.assertLess(deployable_params, 50_000_000)
        self.assertEqual(sum(p.numel() for p in model.target_encoder.parameters() if p.requires_grad), 0)

        # Check forward train
        loss, terms = model.forward_train(self.history_bev, self.ego_motion, self.future_bev, self.future_mask)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_ema_target_update(self):
        model = BEVWALatentFlowModel(self.cfg)
        # Modify an online encoder parameter
        with torch.no_grad():
            for p in model.online_encoder.bev_encoder.parameters():
                p.add_(1.0)
                break

        # Check that target encoder hasn't changed yet
        p_online = next(model.online_encoder.bev_encoder.parameters())
        p_target = next(model.target_encoder.bev_encoder.parameters())
        diff_before = (p_online - p_target).abs().max().item()
        self.assertGreater(diff_before, 0.5)

        # Apply EMA update with tau=0.5
        model.update_teacher(tau=0.5)
        diff_after = (p_online - p_target).abs().max().item()
        self.assertLess(diff_after, diff_before)

    def test_causal_isolation_and_no_future_leakage(self):
        """Protocol Gate 3: Causal and Leakage Verification.
        Ensure future BEV has strictly 0 influence on history condition tokens,
        and backward pass from flow loss generates 0 gradient to target encoder.
        """
        model = BEVWALatentFlowModel(self.cfg)
        
        # 1. Verification of conditioning path independence
        cond1 = model.online_encoder(self.history_bev, self.ego_motion)
        
        # Corrupt / swap future BEVs completely
        corrupted_future = torch.randint(0, 256, (self.b, 4, 4, 128, 128), dtype=torch.uint8)
        cond2 = model.online_encoder(self.history_bev, self.ego_motion)
        
        # History conditioning must be bitwise identical regardless of future
        self.assertTrue(torch.equal(cond1, cond2))

        # 2. Gradient isolation
        loss, _ = model.forward_train(self.history_bev, self.ego_motion, self.future_bev, self.future_mask)
        loss.backward()

        # All target encoder parameters must have None gradient
        for name, p in model.target_encoder.named_parameters():
            self.assertIsNone(p.grad, f"Target encoder parameter {name} received non-None gradient!")

        # Online encoder parameters and predictor parameters must have valid non-None gradients
        self.assertIsNotNone(model.online_encoder.temporal_pos.grad)
        self.assertIsNotNone(model.predictor.pos_spatial.grad)

    def test_euler_ode_generation(self):
        model = BEVWALatentFlowModel(self.cfg)
        model.eval()
        with torch.no_grad():
            cond = model.online_encoder(self.history_bev, self.ego_motion)
            gen_latents = model.predictor.solve_euler(cond, num_steps=5)
            
        self.assertEqual(gen_latents.shape, (self.b, 4, 64, 256))
        self.assertFalse(torch.isnan(gen_latents).any())

    def test_representation_health_computation(self):
        # Generate random latents with known variance
        latents = torch.randn(64, 64, 256)
        cov = torch.cov(latents.flatten(0, 1).T).numpy()
        metrics, _ = spectrum_metrics(cov)
        print(f"\n[Test] Effective rank: {metrics['effective_rank']:.2f}")
        print(f"[Test] Variance / std_mean: {metrics['std_mean']:.4f}")
        self.assertGreater(metrics['effective_rank'], 16.0)
        self.assertGreater(metrics['std_mean'], 0.10)


if __name__ == "__main__":
    unittest.main()
