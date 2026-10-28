"""Failure-driven tests for physical, chronology-safe stop/resume collection."""
import unittest
import numpy as np
from dataset.behavior_audit import StandstillGate, audit_episode
from dataset.recording import future_labels_v2

EVENTS = ('crash', 'crash_vehicle', 'crash_object', 'crash_sidewalk', 'out_of_road', 'arrive_dest', 'max_step')


def fixture():
    n, release = 80, 45
    states = np.zeros((n+1, 6), np.float32)
    states[:20, 3] = 2.
    states[46:, 3] = 4.
    states[1:, 0] = np.cumsum(states[:-1, 3]*.1)
    npc = np.zeros((n, 1, 6), np.float32)
    npc[:, 0, 0] = 14.
    before = np.ones((n, 1), bool); before[release+1:] = False
    after = np.ones((n, 1), bool); after[release:] = False
    gap = npc[:, :, 0] - states[:-1, None, 0] - 4.
    npc[~before] = 0.
    gap[~before] = np.nan
    lat = np.zeros((n, 1), np.float32); lat[~before] = np.nan
    actions = np.zeros((n, 2), np.float32); actions[20:release, 1] = -1.
    hold = np.zeros(n, bool); hold[20:release] = True
    release_event = np.zeros(n, bool); release_event[release] = True
    traj, valid, _ = future_labels_v2(states, np.zeros(n+1, bool))
    return dict(states=states, events=np.zeros((n, len(EVENTS)), bool),
                action=actions.copy(), applied_action=actions.copy(), valid=valid[:n], trajectory=traj[:n],
                diagnostic_npc_states=npc, diagnostic_npc_present_before=before,
                diagnostic_npc_present_after=after, diagnostic_npc_clearance_m=gap,
                diagnostic_npc_lateral_m=lat, diagnostic_release_event=release_event,
                diagnostic_teacher_hold_brake=hold, diagnostic_npc_length_m=np.full((n, 1), 4.),
                diagnostic_ego_length_m=np.full(n, 4.), diagnostic_road_heading_rad=np.zeros(n),
                step_id=np.arange(n), time_s=np.arange(n)*.1, transition_valid=np.ones(n, bool),
                action_mask=np.ones(n, bool))


class StandstillGateTest(unittest.TestCase):
    def test_initial_static_never_qualifies(self):
        gate = StandstillGate()
        for _ in range(100):
            self.assertFalse(gate.observe(0., 0., True, 0., 6.))
        self.assertFalse(gate.holding)

    def test_completed_intervals_and_reset(self):
        gate = StandstillGate()
        gate.observe(2., 3., True, 0., 6.)
        for _ in range(25):
            self.assertFalse(gate.observe(0., 3., True, 0., 6.))
        self.assertTrue(gate.observe(0., 3., True, 0., 6.))
        self.assertFalse(gate.observe(.06, 3., True, 0., 6.))
        self.assertEqual(gate.consecutive_intervals, 0)
        self.assertTrue(gate.holding)  # full-brake latch survives numerical creep

    def test_missing_far_or_moving_blocker(self):
        for present, npc_speed, gap in [(False, 0., 6.), (True, 1., 6.), (True, 0., 30.)]:
            gate = StandstillGate()
            gate.observe(2., 3., present, npc_speed, gap)
            for _ in range(40):
                self.assertFalse(gate.observe(0., 3., present, npc_speed, gap))


class RawAuditTest(unittest.TestCase):
    def audit(self, arrays):
        return audit_episode(arrays, 'static_blockage_and_resume', EVENTS)

    def test_valid_stop_clear_resume(self):
        result = self.audit(fixture())
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['metrics']['stationary_intervals_before_release'], 25)

    def test_accumulated_creep_cannot_pass(self):
        arrays = fixture()
        arrays['states'][35, 3] = .06
        result = self.audit(arrays)
        self.assertIn('insufficient_continuous_standstill_before_release', result['failures'])

    def test_manifest_flags_do_not_establish_physics(self):
        arrays = fixture()
        arrays['success_demonstration'] = True
        del arrays['diagnostic_release_event']
        result = self.audit(arrays)
        self.assertFalse(result['passed'])
        self.assertIn('diagnostic_release_event', result['missing'])

    def test_raw_actions_masks_and_events_are_checked(self):
        for key, mutate, expected in [
            ('applied_action', lambda a: a.__setitem__((30, 1), 0.), 'commanded_applied_action_mismatch'),
            ('valid', lambda a: a.__setitem__((79, 0), True), 'future_mask_mismatch'),
            ('events', lambda a: a.__setitem__((10, 2), True), 'safety_event'),
            ('action_mask', lambda a: a.__setitem__(5, False), 'action_mask_mismatch'),
        ]:
            arrays = fixture(); mutate(arrays[key])
            self.assertIn(expected, self.audit(arrays)['failures'])

    def test_no_resume_and_fake_obstacle_fail(self):
        arrays = fixture(); arrays['states'][46:, 3] = 0.
        self.assertIn('insufficient_continuous_resume', self.audit(arrays)['failures'])
        arrays = fixture(); arrays['diagnostic_npc_clearance_m'][30, 0] = 5.
        self.assertIn('npc_gap_geometry_mismatch', self.audit(arrays)['failures'])

    def test_hold_requires_actual_brake(self):
        arrays = fixture(); arrays['action'][30, 1] = arrays['applied_action'][30, 1] = 0.
        self.assertIn('brake_not_held_during_standstill', self.audit(arrays)['failures'])

    def test_other_scenarios_not_accepted_from_manifest(self):
        result = audit_episode(fixture(), 'lateral_recovery_and_curve', EVENTS)
        self.assertFalse(result['passed'])
        self.assertIn('independent_behavior_audit_for_lateral_recovery_and_curve', result['missing'])


if __name__ == '__main__':
    unittest.main()
