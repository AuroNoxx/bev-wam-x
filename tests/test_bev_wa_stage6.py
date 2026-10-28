"""Unit tests for Stage 6: Consequence Consistency Diagnostics."""

import unittest
import sys
from pathlib import Path
WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import numpy as np
import torch

from configs.stage3 import Stage3Config
from models.bev_wa_joint_flow import ArmDJointFlowModel
from mini_wa_jepa.geometry import world_to_ego, wrap_angle


class TestStage6Diagnostics(unittest.TestCase):
    def setUp(self):
        self.cfg = Stage3Config()
        self.b = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ArmDJointFlowModel(self.cfg).to(self.device).eval()

        self.batch = {
            "history_bev": torch.randint(0, 256, (self.b, 4, 4, 128, 128), dtype=torch.uint8).to(self.device),
            "ego_motion": torch.randn(self.b, 4, 2, dtype=torch.float32).to(self.device),
            "route_points": torch.randn(self.b, 8, 2, dtype=torch.float32).to(self.device),
            "route_valid": torch.ones(self.b, 8, dtype=torch.bool).to(self.device),
            "vehicle": torch.randn(self.b, 4, dtype=torch.float32).to(self.device),
            "maneuver_allowed": torch.tensor([[True, True], [False, True]], dtype=torch.bool).to(self.device),
            "future_bev": torch.randint(0, 256, (self.b, 4, 4, 128, 128), dtype=torch.uint8).to(self.device),
            "future_bev_mask": torch.ones(self.b, 4, dtype=torch.bool).to(self.device),
            "trajectory": torch.randn(self.b, 10, 4, dtype=torch.float32).to(self.device),
            "future_traj_mask": torch.ones(self.b, 10, dtype=torch.bool).to(self.device),
            "action_mask": torch.tensor([True, True], dtype=torch.bool).to(self.device),
        }

    def test_coupled_euler_output_shapes(self):
        """Verifies that Arm D outputs both trajectory [B, 10, 4] and future scene latents [B, 4, 64, 256]."""
        with torch.no_grad():
            z_traj, z_scene = self.model.solve_euler(
                self.batch["history_bev"],
                self.batch["ego_motion"],
                self.batch["route_points"],
                self.batch["route_valid"],
                self.batch["vehicle"],
                self.batch["maneuver_allowed"],
                num_steps=3,
            )

        self.assertEqual(z_traj.shape, (self.b, 10, 4))
        self.assertEqual(z_scene.shape, (self.b, 4, 64, 256))

    def test_world_to_ego_geometry(self):
        """Verifies coordinate transformation for open-loop branch tracking."""
        pos0 = np.array([10.0, 20.0], dtype=np.float32)
        heading0 = 0.0  # Facing east (world +x)

        # A point 5m in front of vehicle in world frame
        p_world = np.array([[15.0, 20.0]], dtype=np.float32)
        p_ego = world_to_ego(p_world, pos0, heading0)

        # In ego frame, x should be +5.0, y should be 0.0
        self.assertAlmostEqual(float(p_ego[0, 0]), 5.0, places=4)
        self.assertAlmostEqual(float(p_ego[0, 1]), 0.0, places=4)

        # Vehicle facing north (heading = pi/2)
        heading_north = np.pi / 2.0
        p_north = np.array([[10.0, 25.0]], dtype=np.float32)  # 5m north
        p_ego_north = world_to_ego(p_north, pos0, heading_north)

        # In ego frame, forward is x, so x should be +5.0
        self.assertAlmostEqual(float(p_ego_north[0, 0]), 5.0, places=4)
        self.assertAlmostEqual(float(p_ego_north[0, 1]), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
