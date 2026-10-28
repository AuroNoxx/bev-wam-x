"""Unit tests for Stage 7 risk data collection, relative offset generation, and censoring masks."""
import unittest
import numpy as np

from configs.stage7 import Stage7Config, STAGE7_SPLITS, STAGE7_RISK_CANDIDATE_SPECS
from scripts.preflight_risk_collection import (
    compute_observed_lateral_left,
    make_risk_candidates,
    compute_risk_targets,
)


class DummyAgent:
    def __init__(self, position=(0.0, 0.0), heading_theta=0.0, speed=7.0):
        self.position = list(position)
        self.heading_theta = float(heading_theta)
        self.speed = float(speed)


class DummyExpert:
    def __init__(self, car_position=(0.0, 0.0), car_heading=0.0):
        self.sim = type('Sim', (), {'agent': DummyAgent(car_position, car_heading)})()

    def route(self, s):
        # Straight lane along X-axis, heading 0 rad
        return [float(s), 0.0], None, None, 0.0


class DummyCurvedExpert:
    def __init__(self, car_position=(0.0, 0.0), car_heading=0.0, curve_heading=np.pi / 4):
        self.sim = type('Sim', (), {'agent': DummyAgent(car_position, car_heading)})()
        self.curve_heading = curve_heading

    def route(self, s):
        # Road angled at curve_heading
        x = float(s) * np.cos(self.curve_heading)
        y = float(s) * np.sin(self.curve_heading)
        return [x, y], None, None, self.curve_heading


class TestStage7Collection(unittest.TestCase):
    def test_semantic_tie_order_keeps_color_and_depth_state(self):
        from types import SimpleNamespace
        from panda3d.core import Camera,NodePath,RenderState,ColorAttrib,DepthTestAttrib,CullBinAttrib
        from metadrive.constants import Semantics
        from env.semantic_bev import configure_semantic_draw_order
        node=Camera('semantic-test');depth=DepthTestAttrib.make(DepthTestAttrib.MLessEqual)
        color=ColorAttrib.makeFlat((1.,0.,0.,1.))
        node.setTagState(Semantics.SIDEWALK.label,RenderState.make(depth,color))
        configure_semantic_draw_order(SimpleNamespace(cam=NodePath(node)))
        result=node.getTagState(Semantics.SIDEWALK.label)
        self.assertEqual(result.getAttrib(DepthTestAttrib),depth)
        self.assertEqual(result.getAttrib(ColorAttrib),color)
        self.assertEqual(result.getAttrib(CullBinAttrib).getBinName(),'fixed')

    def test_compute_observed_lateral_left_straight_and_curved(self):
        """Verify explicit left-normal projection avoids MetaDrive right-positive sign ambiguity."""
        # Case 1: Straight road heading east (0 rad), left normal is [0, 1]
        expert_straight = DummyExpert(car_position=(10.0, 3.5), car_heading=0.0)
        lat_left = compute_observed_lateral_left(expert_straight, expert_straight.sim.agent.position)
        self.assertAlmostEqual(lat_left, 3.5, places=5)

        expert_rightside = DummyExpert(car_position=(10.0, -5.0), car_heading=0.0)
        lat_right = compute_observed_lateral_left(expert_rightside, expert_rightside.sim.agent.position)
        self.assertAlmostEqual(lat_right, -5.0, places=5)

        # Case 2: 45-degree curved road, left normal is [-sin(pi/4), cos(pi/4)]
        angle = np.pi / 4
        # Vehicle displaced by 2.0m along left normal
        left_disp = 2.0 * np.array([-np.sin(angle), np.cos(angle)])
        expert_curved = DummyCurvedExpert(car_position=left_disp, car_heading=angle, curve_heading=angle)
        lat_curved = compute_observed_lateral_left(expert_curved, expert_curved.sim.agent.position)
        self.assertAlmostEqual(lat_curved, 2.0, places=5)

    def test_make_risk_candidates_nonzero_starting_lateral(self):
        """Candidates generated from roadside context (e.g. -5m) must displace relative to current position."""
        obs_lat = -5.0
        expert = DummyExpert(car_position=(0.0, -5.0), car_heading=0.0)
        commands = make_risk_candidates(expert, STAGE7_RISK_CANDIDATE_SPECS, observed_lateral_left=obs_lat)

        # In ego frame of vehicle at (0, -5), current lateral position is ego y = 0
        # 'keep' maintains roadside offset (-5m in world, 0m in ego frame)
        cmd_keep = commands['keep']
        self.assertAlmostEqual(float(cmd_keep[-1, 1]), 0.0, places=2)

        # 'right_6m' shifts 6m further right (-6m in ego frame)
        cmd_right_6m = commands['right_6m']
        self.assertAlmostEqual(float(cmd_right_6m[-1, 1]), -6.0, places=2)

        # 'left_6m' recovers 6m leftward (+6m in ego frame, back to +1m world lateral)
        cmd_left_6m = commands['left_6m']
        self.assertAlmostEqual(float(cmd_left_6m[-1, 1]), 6.0, places=2)

    def test_compute_risk_targets_clean_complete(self):
        """Clean 20-step rollout has both labels valid and zero."""
        flags = [[False, False, False, False]] * 20
        target, mask, crash_seen, offroad_seen, full_2s = compute_risk_targets(flags)
        self.assertTrue(full_2s)
        self.assertFalse(crash_seen)
        self.assertFalse(offroad_seen)
        self.assertTrue(np.allclose(target, [0.0, 0.0]))
        self.assertTrue(bool(mask[0]))
        self.assertTrue(bool(mask[1]))

    def test_compute_risk_targets_empty_flags(self):
        """Empty flags array yields both labels censored."""
        flags = []
        target, mask, crash_seen, offroad_seen, full_2s = compute_risk_targets(flags)
        self.assertFalse(full_2s)
        self.assertFalse(crash_seen)
        self.assertFalse(offroad_seen)
        self.assertFalse(bool(mask[0]))
        self.assertFalse(bool(mask[1]))

    def test_compute_risk_targets_early_no_event_censored(self):
        """Early termination without events censors both crash and offroad."""
        flags = [[False, False, False, False]] * 8
        target, mask, crash_seen, offroad_seen, full_2s = compute_risk_targets(flags)
        self.assertFalse(full_2s)
        self.assertFalse(crash_seen)
        self.assertFalse(offroad_seen)
        self.assertFalse(bool(mask[0]))
        self.assertFalse(bool(mask[1]))

    def test_compute_risk_targets_step20_event_valid(self):
        """Event occurring at 20th step is valid positive; unobserved event is valid negative."""
        flags = [[False, False, False, False]] * 19 + [[True, False, False, False]]
        target, mask, crash_seen, offroad_seen, full_2s = compute_risk_targets(flags)
        self.assertTrue(full_2s)
        self.assertTrue(crash_seen)
        self.assertFalse(offroad_seen)
        self.assertEqual(float(target[0]), 1.0)
        self.assertEqual(float(target[1]), 0.0)
        self.assertTrue(bool(mask[0]))
        self.assertTrue(bool(mask[1]))

    def test_compute_risk_targets_early_crash_censors_offroad(self):
        """Early collision terminates rollout: crash is valid positive, offroad is censored."""
        flags = [[False, False, False, False]] * 4 + [[True, False, True, False]]
        target, mask, crash_seen, offroad_seen, full_2s = compute_risk_targets(flags)
        self.assertFalse(full_2s)
        self.assertTrue(crash_seen)
        self.assertFalse(offroad_seen)
        self.assertEqual(float(target[0]), 1.0)
        self.assertTrue(bool(mask[0]))
        self.assertFalse(bool(mask[1]))

    def test_compute_risk_targets_early_offroad_censors_crash(self):
        """Early out-of-road terminates rollout: offroad is valid positive, crash is censored."""
        flags = [[False, False, False, False]] * 10 + [[False, True, True, False]]
        target, mask, crash_seen, offroad_seen, full_2s = compute_risk_targets(flags)
        self.assertFalse(full_2s)
        self.assertFalse(crash_seen)
        self.assertTrue(offroad_seen)
        self.assertEqual(float(target[1]), 1.0)
        self.assertFalse(bool(mask[0]))
        self.assertTrue(bool(mask[1]))

    def test_stage7_splits_and_closed_loop_isolation(self):
        cfg = Stage7Config()
        train_maps = set(STAGE7_SPLITS['train']['maps'])
        val_maps = set(STAGE7_SPLITS['val']['maps'])
        test_maps = set(STAGE7_SPLITS['test']['maps'])

        self.assertTrue(train_maps.isdisjoint(val_maps))
        self.assertTrue(train_maps.isdisjoint(test_maps))
        self.assertTrue(val_maps.isdisjoint(test_maps))

        self.assertEqual(STAGE7_SPLITS['train']['seed_start'], 710000)
        self.assertEqual(STAGE7_SPLITS['val']['seed_start'], 810000)
        self.assertEqual(STAGE7_SPLITS['test']['seed_start'], 910000)

        # Closed-loop seeds must not reuse risk test seeds
        self.assertNotIn(cfg.closed_loop_seeds[0], range(910000, 950000))
        self.assertNotIn(cfg.closed_loop_seeds[1], range(910000, 950000))
        self.assertEqual(cfg.closed_loop_seeds, (1010000, 1020000))


if __name__ == '__main__':
    unittest.main()
