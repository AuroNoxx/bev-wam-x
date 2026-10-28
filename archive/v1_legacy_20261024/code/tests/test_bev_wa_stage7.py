"""Unit tests for Stage 7: Full Integrated Driving Closure & Acceptance Framework."""

import unittest
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import numpy as np
import torch

from configs.stage3 import Stage3Config
from configs.integrated import IntegratedConfig
from models.bev_wa_joint_flow import ArmDJointFlowModel
from planning.stage7_acceptance import categorize_scenario, evaluate_scenario_outcome


class TestStage7Integrated(unittest.TestCase):
    def setUp(self):
        self.cfg = Stage3Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ArmDJointFlowModel(self.cfg).to(self.device).eval()

    def test_model_inference_contract(self):
        """Verifies that Arm D outputs valid [1, 10, 4] trajectory for controller."""
        b = 1
        h_bev = torch.randint(0, 256, (b, 4, 4, 128, 128), dtype=torch.uint8, device=self.device)
        ego = torch.zeros(b, 4, 2, dtype=torch.float32, device=self.device)
        rp = torch.zeros(b, 8, 2, dtype=torch.float32, device=self.device)
        rv = torch.ones(b, 8, dtype=torch.bool, device=self.device)
        veh = torch.zeros(b, 4, dtype=torch.float32, device=self.device)
        perm = torch.ones(b, 2, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            plan, scene = self.model.solve_euler(h_bev, ego, rp, rv, veh, perm, num_steps=5)

        self.assertEqual(plan.shape, (1, 10, 4))
        self.assertEqual(scene.shape, (1, 4, 64, 256))
        self.assertTrue(torch.isfinite(plan).all())
        self.assertTrue(torch.isfinite(scene).all())

    def test_scenario_categorization(self):
        """Verifies correct separation of scenarios into the 6 distinct acceptance categories."""
        # 1. Obstacle-free navigation
        self.assertEqual(categorize_scenario({"seed": 4300000, "map": "S"}), "obstacle_free_navigation")
        # 2. Moving slow vehicle - no pass allowed
        self.assertEqual(categorize_scenario({"seed": 4300303, "case": "no_adjacent_lane", "lanes": 1}), "moving_slow_vehicle_no_pass")
        self.assertEqual(categorize_scenario({"seed": 4300302, "case": "blocked", "lanes": 3, "stationary": False}), "moving_slow_vehicle_no_pass")
        # 3. Moving slow vehicle - pass allowed
        self.assertEqual(categorize_scenario({"seed": 4300300, "case": "allowed_left", "lanes": 3}), "moving_slow_vehicle_pass_allowed")
        self.assertEqual(categorize_scenario({"seed": 4300301, "case": "allowed_right", "lanes": 3}), "moving_slow_vehicle_pass_allowed")
        # 4. Static full blockage
        self.assertEqual(categorize_scenario({"seed": 4300302, "case": "blocked", "stationary": True}), "static_full_blockage")
        self.assertEqual(categorize_scenario({"seed": 4300302, "case": "blocked_static"}), "static_full_blockage")
        # 5. Blockage clearance & resume
        self.assertEqual(categorize_scenario({"seed": 4300315, "case": "blockage_clearance_resume"}), "blockage_clearance_resume")
        # 6. Lateral perturbation / curve hazard
        self.assertEqual(categorize_scenario({"seed": 4300320, "case": "lateral_perturbation"}), "lateral_perturbation_curve_hazard")

    def test_acceptance_criteria_no_reward_for_crash(self):
        """Crash or off-road must invalidate success and zero effective route completion."""
        res_crash = evaluate_scenario_outcome("obstacle_free_navigation", {
            "arrived": False,
            "crashed": True,
            "offroad": False,
            "route_completion": 0.85,
        })
        self.assertFalse(res_crash["success"])
        self.assertEqual(res_crash["effective_route_completion"], 0.0)
        self.assertEqual(res_crash["failure_reason"], "vehicle_collision")

        res_offroad = evaluate_scenario_outcome("moving_slow_vehicle_no_pass", {
            "arrived": False,
            "crashed": False,
            "offroad": True,
            "route_completion": 0.60,
        })
        self.assertFalse(res_offroad["success"])
        self.assertEqual(res_offroad["effective_route_completion"], 0.0)
        self.assertEqual(res_offroad["failure_reason"], "road_boundary_violation")

    def test_acceptance_criteria_slow_following_allows_timeout(self):
        """Moving slow vehicle following must allow timeout as success if safe distance was kept."""
        res = evaluate_scenario_outcome("moving_slow_vehicle_no_pass", {
            "arrived": False,
            "crashed": False,
            "offroad": False,
            "timeout": True,
            "route_completion": 0.77,
            "min_longitudinal_gap_m": 12.5,
        })
        self.assertTrue(res["success"])
        self.assertIsNone(res["failure_reason"])
        self.assertTrue(res["diagnostics"]["timeout_permitted"])

    def test_acceptance_criteria_overtake_requires_return(self):
        """Overtaking scenario requires both passing lead AND returning to route lane."""
        # Passed lead but stayed in wrong lane -> FAIL
        res1 = evaluate_scenario_outcome("moving_slow_vehicle_pass_allowed", {
            "crashed": False,
            "offroad": False,
            "overtake_completed": True,
            "returned_to_route_lane": False,
        })
        self.assertFalse(res1["success"])
        self.assertEqual(res1["failure_reason"], "failed_to_return_to_route_lane")

        # Both passed and returned -> PASS
        res2 = evaluate_scenario_outcome("moving_slow_vehicle_pass_allowed", {
            "crashed": False,
            "offroad": False,
            "overtake_completed": True,
            "returned_to_route_lane": True,
        })
        self.assertTrue(res2["success"])
        self.assertIsNone(res2["failure_reason"])

    def test_report_generator_honest_failure_and_missing_metrics(self):
        """Verifies report generator handles failures honestly and reports missing items as N/A."""
        from scripts.run_bev_wa_stage7 import generate_stage7_report
        import tempfile

        mock_results = {
            "date_user_timezone": "2026-10-23",
            "arms": {
                "arm_d_seed_20261022": {
                    "overall_success_rate": 0.50,
                    "overall_mean_effective_rc": 0.55,
                    "overall_crash_rate": 0.25,
                    "overall_offroad_rate": 0.25,
                    "cohort_latencies": {"p50_ms": 18.2, "p90_ms": 20.1, "p95_ms": 20.8, "p99_ms": 23.5},
                    "episodes": [{"t_breakdowns": {"mean_bev_preprocess_ms": 0.45, "mean_model_euler_ms": 20.2, "mean_tracker_ms": 0.08}}],
                    "by_category": {
                        "obstacle_free_navigation": {"total": 12, "success_rate": 0.75, "mean_effective_rc": 0.75, "crash_rate": 0.1, "offroad_rate": 0.15},
                        "moving_slow_vehicle_no_pass": {"total": 3, "success_rate": 0.0, "mean_effective_rc": 0.0, "crash_rate": 0.0, "offroad_rate": 1.0},
                    }
                },
                "arm_d_seed_20261023": {
                    "overall_success_rate": 0.58,
                    "overall_mean_effective_rc": 0.60,
                    "overall_crash_rate": 0.20,
                    "overall_offroad_rate": 0.20,
                    "cohort_latencies": {"p50_ms": 18.3, "p90_ms": 20.2, "p95_ms": 20.9, "p99_ms": 23.6},
                    "episodes": [{"t_breakdowns": {"mean_bev_preprocess_ms": 0.45, "mean_model_euler_ms": 20.3, "mean_tracker_ms": 0.08}}],
                    "by_category": {}
                },
                "arm_b_seed_20261022": {
                    "overall_success_rate": 0.45,
                    "overall_mean_effective_rc": 0.50,
                    "overall_crash_rate": 0.30,
                    "overall_offroad_rate": 0.25,
                    "cohort_latencies": {"p95_ms": 20.5},
                },
                "arm_a_seed_20261022": {
                    "overall_success_rate": 0.20,
                    "overall_mean_effective_rc": 0.30,
                    "overall_crash_rate": 0.50,
                    "overall_offroad_rate": 0.30,
                    "cohort_latencies": {"p95_ms": 3.5},
                },
                "ablation_d0_seed_20261022": {
                    "overall_success_rate": 0.40,
                    "overall_mean_effective_rc": 0.45,
                    "overall_crash_rate": 0.35,
                    "overall_offroad_rate": 0.25,
                    "cohort_latencies": {"p95_ms": 20.6},
                },
            },
            "comparative_analysis": {
                "delta_rc_vs_d0": 0.097,
                "delta_rc_vs_arm_b": -0.057,
                "p_val_vs_d0": 0.1685,
                "world_gain_statistically_significant": False,
            },
            "regression_probes": {
                "probe1_destination_non_deadlock": {"passed": True, "steps_to_arrival": 182, "final_route_completion": 0.98},
                "probe2_curved_braking_resume": {"passed": False, "entry_speed_m_s": 5.0, "stopped_speed_m_s": 0.5, "resumed_speed_m_s": 2.0},
                "probe3_off_path_recovery": {"passed": False, "steps_to_recovery": None, "final_lateral_offset_m": 0.374},
            }
        }

        with tempfile.NamedTemporaryFile("w+", suffix=".md") as tmp:
            generate_stage7_report(mock_results, Path(tmp.name))
            content = tmp.read()

        # Must report failed overall
        self.assertIn("未通过（Failed / 需整改重测）", content)
        self.assertNotIn("验收判定：全部验收通过", content)
        # Must report not statistically significant
        self.assertIn("尚未证实联合世界模型带来稳定额外收益", content)
        self.assertNotIn("p < 0.001", content)
        # Must report missing categories as N/A
        self.assertIn("N/A (未测)", content)


if __name__ == "__main__":
    unittest.main()
