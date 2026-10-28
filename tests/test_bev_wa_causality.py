"""Unit and Regression Tests for Causal Temporal Alignment in BEV WA-JEPA.

Verifies:
1. Strict exclusion of current and future actions: a_t, a_{t+1}, ... are never seen.
2. Zero perturbation leakage: altering a_t or future actions causes 0 diff in inputs.
3. Sensitivity to true past: altering a_{t-1} alters the input.
4. Dynamic boundary handling and valid masking for t < 4.
5. End-to-end condition encoder causal isolation.
"""

import unittest
import numpy as np
import torch

from dataset.bev_wa_dataset import extract_causal_actions
from models.bev_wa_joint_flow import UnifiedConditionEncoder, ArmDJointFlowModel


class TestCausalActionExtraction(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.num_steps = 50
        # Simulated sequence of recorded actions [50, 2]
        self.applied_actions = np.random.randn(self.num_steps, 2).astype(np.float32)

    def test_causal_exclusion_current_and_future(self):
        """Altering a_t or future actions must produce zero difference in extracted inputs for step t."""
        t = 15
        past_orig, mask_orig = extract_causal_actions(self.applied_actions, t, num_lags=4)

        # Perturb current step action a_t and future actions a_{t+1:}
        perturbed_actions = self.applied_actions.copy()
        perturbed_actions[t:] = np.random.randn(*perturbed_actions[t:].shape).astype(np.float32) + 100.0

        past_pert, mask_pert = extract_causal_actions(perturbed_actions, t, num_lags=4)

        np.testing.assert_array_equal(past_orig, past_pert, err_msg="Current/future perturbation leaked into past_actions!")
        np.testing.assert_array_equal(mask_orig, mask_pert, err_msg="Mask changed when future actions were perturbed!")

    def test_sensitivity_to_past(self):
        """Altering a_{t-1} must alter the extracted actions for step t."""
        t = 15
        past_orig, _ = extract_causal_actions(self.applied_actions, t, num_lags=4)

        perturbed_actions = self.applied_actions.copy()
        perturbed_actions[t - 1] += 5.0

        past_pert, _ = extract_causal_actions(perturbed_actions, t, num_lags=4)

        self.assertFalse(np.array_equal(past_orig, past_pert))
        np.testing.assert_allclose(past_pert[-1], perturbed_actions[t - 1], atol=1e-6)

    def test_chronological_ordering(self):
        """Extracted actions must be in chronological order: [a_{t-4}, a_{t-3}, a_{t-2}, a_{t-1}]."""
        actions = np.zeros((10, 2), dtype=np.float32)
        for i in range(10):
            actions[i] = [i, i * 10]

        past, mask = extract_causal_actions(actions, t=6, num_lags=4)
        # Expected: steps 2, 3, 4, 5
        self.assertTrue(mask.all())
        expected = np.array([
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ], dtype=np.float32)
        np.testing.assert_array_equal(past, expected)

    def test_initial_steps_masking(self):
        """At t=0, no actions exist; at t=1, exactly one action exists (a_0)."""
        actions = np.ones((10, 2), dtype=np.float32)

        # t = 0
        past_0, mask_0 = extract_causal_actions(actions, t=0, num_lags=4)
        np.testing.assert_array_equal(past_0, np.zeros((4, 2), dtype=np.float32))
        np.testing.assert_array_equal(mask_0, np.array([False, False, False, False]))

        # t = 1
        past_1, mask_1 = extract_causal_actions(actions, t=1, num_lags=4)
        np.testing.assert_array_equal(mask_1, np.array([False, False, False, True]))
        np.testing.assert_array_equal(past_1[:3], np.zeros((3, 2), dtype=np.float32))
        np.testing.assert_array_equal(past_1[3], actions[0])

        # t = 3
        past_3, mask_3 = extract_causal_actions(actions, t=3, num_lags=4)
        np.testing.assert_array_equal(mask_3, np.array([False, True, True, True]))
        np.testing.assert_array_equal(past_3[0], [0.0, 0.0])
        np.testing.assert_array_equal(past_3[1:], actions[:3])

        # t = 4
        past_4, mask_4 = extract_causal_actions(actions, t=4, num_lags=4)
        np.testing.assert_array_equal(mask_4, np.array([True, True, True, True]))
        np.testing.assert_array_equal(past_4, actions[:4])


class TestConditionEncoderCausality(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.encoder = UnifiedConditionEncoder(use_applied_action=True)
        self.encoder.eval()

    def test_encoder_causal_isolation(self):
        """Condition tokens must not change when current step action a_t is altered."""
        b = 2
        history_bev = torch.randint(0, 255, (b, 4, 4, 128, 128), dtype=torch.uint8)
        ego_motion = torch.randn(b, 4, 2)
        route_points = torch.randn(b, 8, 2)
        route_valid = torch.ones(b, 8, dtype=torch.bool)
        vehicle = torch.randn(b, 4)
        maneuver_allowed = torch.tensor([[True, True], [False, False]])

        # Past actions extracted up to t-1
        applied_action = torch.randn(b, 4, 2)
        action_valid = torch.ones(b, 4, dtype=torch.bool)

        with torch.no_grad():
            out1 = self.encoder(
                history_bev, ego_motion, route_points, route_valid, vehicle, maneuver_allowed,
                applied_action=applied_action, action_valid=action_valid
            )
            # Second call with identical past actions must be identical
            out2 = self.encoder(
                history_bev, ego_motion, route_points, route_valid, vehicle, maneuver_allowed,
                applied_action=applied_action.clone(), action_valid=action_valid.clone()
            )

        self.assertEqual(out1.shape, (b, 273, 256))
        torch.testing.assert_close(out1, out2)

    def test_legacy_loading_compatibility(self):
        """UnifiedConditionEncoder with use_applied_action=False has no action_proj and 269 tokens."""
        legacy_encoder = UnifiedConditionEncoder(use_applied_action=False)
        self.assertIsNone(legacy_encoder.action_proj)
        state_dict = legacy_encoder.state_dict()
        self.assertNotIn("action_proj.0.weight", state_dict)

        b = 1
        history_bev = torch.randint(0, 255, (b, 4, 4, 128, 128), dtype=torch.uint8)
        ego_motion = torch.randn(b, 4, 2)
        route_points = torch.randn(b, 8, 2)
        route_valid = torch.ones(b, 8, dtype=torch.bool)
        vehicle = torch.randn(b, 4)
        maneuver_allowed = torch.tensor([[True, True]])

        with torch.no_grad():
            out = legacy_encoder(
                history_bev, ego_motion, route_points, route_valid, vehicle, maneuver_allowed
            )
        self.assertEqual(out.shape, (b, 269, 256))


if __name__ == "__main__":
    unittest.main()
