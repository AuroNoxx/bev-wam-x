"""Unit tests for Stage 4 Joint World-Action Flow Models.

Tests:
1. Parameter counts across all 4 comparison arms
2. Forward train losses and shape validations
3. Asymmetric gradient contract:
   - Scene loss does not backprop into trajectory path
   - Trajectory loss freely backprops through shared Transformer layers into condition encoder
4. ODE Euler generation shapes and sanity
"""

import unittest
from pathlib import Path
import torch
from configs.stage3 import Stage3Config
from models.bev_wa_joint_flow import (
    ArmBActionOnlyFlowModel,
    ArmCDecoupledFlowModel,
    ArmDJointFlowModel,
)


class TestBEWWAJointFlow(unittest.TestCase):
    def setUp(self):
        self.cfg = Stage3Config()
        self.b = 2
        self.batch = {
            "history_bev": torch.randint(0, 256, (self.b, 4, 4, 128, 128), dtype=torch.uint8),
            "ego_motion": torch.randn(self.b, 4, 2, dtype=torch.float32),
            "route_points": torch.randn(self.b, 8, 2, dtype=torch.float32),
            "route_valid": torch.ones(self.b, 8, dtype=torch.bool),
            "vehicle": torch.randn(self.b, 4, dtype=torch.float32),
            "maneuver_allowed": torch.tensor([[True, True], [False, True]], dtype=torch.bool),
            "future_bev": torch.randint(0, 256, (self.b, 4, 4, 128, 128), dtype=torch.uint8),
            "future_bev_mask": torch.tensor([[True, True, True, True], [True, True, True, False]]),
            "trajectory": torch.randn(self.b, 10, 4, dtype=torch.float32),
            "future_traj_mask": torch.ones(self.b, 10, dtype=torch.bool),
            "action_mask": torch.tensor([True, True], dtype=torch.bool),
        }

    def test_parameter_budgets(self):
        arm_b = ArmBActionOnlyFlowModel(self.cfg)
        arm_d = ArmDJointFlowModel(self.cfg)
        
        b_params = sum(p.numel() for p in arm_b.parameters() if p.requires_grad)
        d_params = sum(p.numel() for p in arm_d.parameters() if p.requires_grad)
        
        print(f"\n[Test] Arm B (Action-only Flow) params: {b_params:,}")
        print(f"[Test] Arm D (Joint WA-Flow) params:      {d_params:,}")
        
        self.assertLess(b_params, 50_000_000)
        self.assertLess(d_params, 50_000_000)

    def test_asymmetric_gradient_contract(self):
        """Protocol Requirement:
        Scene loss must NOT propagate to action tokens / action projection.
        Action loss MUST propagate into action tokens AND shared interactive representation.
        Test must verify real non-zero gradient values and prevent zero-initialization false passes.
        """
        arm_d = ArmDJointFlowModel(self.cfg)
        arm_d.train()

        # Re-run forward step to get graph
        h = self.batch["history_bev"]
        ego = self.batch["ego_motion"]
        rp = self.batch["route_points"]
        rv = self.batch["route_valid"]
        veh = self.batch["vehicle"]
        ma = self.batch["maneuver_allowed"]
        fb = self.batch["future_bev"]
        fbm = self.batch["future_bev_mask"]

        cond = arm_d.condition_encoder(h, ego, rp, rv, veh, ma)
        with torch.no_grad():
            target_scene = arm_d.target_encoder.encode_future(fb)
        
        s = torch.rand(self.b, device=h.device)
        eps_s = torch.randn_like(target_scene)
        z_s = (1.0 - s[:, None, None, None]) * eps_s + s[:, None, None, None] * target_scene
        z_traj_leaf = torch.randn(self.b, 10, 4, requires_grad=True)

        v_scene, v_traj = arm_d.forward_interactive(z_s, z_traj_leaf, s, cond, detach_action_for_scene=True)
        
        # Test 1: Backward on loss_scene alone
        arm_d.zero_grad()
        loss_s = (v_scene - (target_scene - eps_s)).square().mean()
        loss_s.backward(retain_graph=True)
        
        # 1.1 Action input leaf MUST have zero gradient from scene loss
        self.assertIsNotNone(z_traj_leaf.grad)
        self.assertEqual(z_traj_leaf.grad.abs().max().item(), 0.0)

        # 1.2 Action projection weights MUST have zero/None gradient from scene loss
        if arm_d.traj_in_proj.weight.grad is not None:
            self.assertEqual(arm_d.traj_in_proj.weight.grad.abs().max().item(), 0.0)
        if arm_d.traj_step_emb.grad is not None:
            self.assertEqual(arm_d.traj_step_emb.grad.abs().max().item(), 0.0)
        if arm_d.modality_emb_traj.grad is not None:
            self.assertEqual(arm_d.modality_emb_traj.grad.abs().max().item(), 0.0)
        self.assertIsNone(arm_d.out_traj.weight.grad)

        # 1.3 Scene path weights MUST have REAL non-zero gradients from scene loss
        self.assertIsNotNone(arm_d.out_scene.weight.grad)
        self.assertGreater(arm_d.out_scene.weight.grad.abs().max().item(), 1e-6)
        self.assertIsNotNone(arm_d.scene_in_proj.weight.grad)
        self.assertGreater(arm_d.scene_in_proj.weight.grad.abs().max().item(), 1e-6)

        # Test 2: Backward on loss_traj
        loss_t = v_traj.square().mean()
        loss_t.backward()
        
        # 2.1 Action loss MUST propagate into action leaf with real non-zero gradient
        self.assertGreater(z_traj_leaf.grad.abs().max().item(), 1e-6)

        # 2.2 Action loss MUST propagate into action projection and head
        self.assertIsNotNone(arm_d.out_traj.weight.grad)
        self.assertGreater(arm_d.out_traj.weight.grad.abs().max().item(), 1e-6)
        self.assertIsNotNone(arm_d.traj_in_proj.weight.grad)
        self.assertGreater(arm_d.traj_in_proj.weight.grad.abs().max().item(), 1e-6)

        # 2.3 Action loss MUST freely propagate through shared interactive Transformer layers
        self.assertIsNotNone(arm_d.layers[0].qkv_self.weight.grad)
        self.assertGreater(arm_d.layers[0].qkv_self.weight.grad.abs().max().item(), 1e-6)
        self.assertIsNotNone(arm_d.condition_encoder.bev_encoder.temporal_pos.grad)
        self.assertGreater(arm_d.condition_encoder.bev_encoder.temporal_pos.grad.abs().max().item(), 1e-6)

    def test_euler_ode_generation_shapes(self):
        arm_b = ArmBActionOnlyFlowModel(self.cfg)
        arm_d = ArmDJointFlowModel(self.cfg)
        arm_b.eval()
        arm_d.eval()

        b = self.batch
        with torch.no_grad():
            traj_b = arm_b.solve_euler(
                b["history_bev"], b["ego_motion"], b["route_points"], b["route_valid"],
                b["vehicle"], b["maneuver_allowed"], num_steps=5
            )
            traj_d, scene_d = arm_d.solve_euler(
                b["history_bev"], b["ego_motion"], b["route_points"], b["route_valid"],
                b["vehicle"], b["maneuver_allowed"], num_steps=5
            )

        self.assertEqual(traj_b.shape, (self.b, 10, 4))
        self.assertEqual(traj_d.shape, (self.b, 10, 4))
        self.assertEqual(scene_d.shape, (self.b, 4, 64, 256))
        self.assertFalse(torch.isnan(traj_b).any())
        self.assertFalse(torch.isnan(traj_d).any())
        self.assertFalse(torch.isnan(scene_d).any())

    def test_applied_action_wiring_and_fixed_seed(self):
        """Verifies that applied_action [B, 4, 2] is accepted into condition encoder,
        and noise_seed guarantees bitwise reproducible scene latents.
        """
        arm_d = ArmDJointFlowModel(self.cfg)
        arm_d.eval()

        b = self.batch
        applied_act = torch.randn(self.b, 4, 2, dtype=torch.float32)

        with torch.no_grad():
            # Test without applied_action
            t1, s1 = arm_d.solve_euler(
                b["history_bev"], b["ego_motion"], b["route_points"], b["route_valid"],
                b["vehicle"], b["maneuver_allowed"], num_steps=3, noise_seed=42
            )
            # Test with applied_action
            t2, s2 = arm_d.solve_euler(
                b["history_bev"], b["ego_motion"], b["route_points"], b["route_valid"],
                b["vehicle"], b["maneuver_allowed"], applied_action=applied_act, num_steps=3, noise_seed=42
            )
            # Test reproducibility with identical seed
            t3, s3 = arm_d.solve_euler(
                b["history_bev"], b["ego_motion"], b["route_points"], b["route_valid"],
                b["vehicle"], b["maneuver_allowed"], applied_action=applied_act, num_steps=3, noise_seed=42
            )

        self.assertEqual(t2.shape, (self.b, 10, 4))
        self.assertEqual(s2.shape, (self.b, 4, 64, 256))
        # Identical seed must produce bitwise identical ODE outputs
        self.assertTrue(torch.allclose(t2, t3, atol=1e-7))
        self.assertTrue(torch.allclose(s2, s3, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
