"""CPU-only physical checks for targeted teacher demonstrations.

Diagnostics never enter learner inputs. A stop lasts N *completed intervals*,
which requires N+1 stationary observations. Manifest success flags are ignored.
"""
from dataclasses import dataclass
import numpy as np

STOP_CASES = ('lead_stop_and_resume', 'static_blockage_and_resume')


@dataclass
class StandstillGate:
    required_intervals: int = 25
    moved: bool = False
    holding: bool = False
    consecutive_intervals: int = 0
    longest_intervals: int = 0
    _previous_low: bool = False

    def observe(self, speed, displacement, blocker_present, blocker_speed, gap):
        self.moved |= speed >= 1.0 and displacement >= 2.0
        approaching = (self.moved and blocker_present and blocker_speed <= .05
                       and 2.0 <= gap <= 12.0)
        if approaching and speed <= .15:
            self.holding = True
        low = approaching and speed <= .05
        if low:
            self.consecutive_intervals = self.consecutive_intervals + 1 if self._previous_low else 0
        else:
            self.consecutive_intervals = 0
        self._previous_low = bool(low)
        self.longest_intervals = max(self.longest_intervals, self.consecutive_intervals)
        return bool(self.consecutive_intervals >= self.required_intervals)


def _longest_run(values):
    longest = count = 0
    for value in values:
        count = count + 1 if value else 0
        longest = max(longest, count)
    return longest


def audit_episode(arrays, case_type, event_names):
    """Recompute facts from step-aligned raw arrays; missing evidence never passes."""
    failures, missing, metrics = [], [], {}
    def check(condition, message):
        if not condition:
            failures.append(message)
    base = ('states', 'events', 'applied_action', 'action', 'valid', 'trajectory')
    missing.extend(k for k in base if k not in arrays)
    if missing:
        return dict(passed=False, failures=failures, missing=missing, metrics=metrics)
    states = np.asarray(arrays['states'])
    n = len(states) - 1
    if states.ndim != 2 or states.shape[1] < 4 or n < 1:
        return dict(passed=False, failures=['malformed_states'], missing=[], metrics={})
    check(np.isfinite(states).all(), 'nonfinite_states')
    events = np.asarray(arrays['events'])
    event_names = list(event_names)
    safety = ('crash', 'crash_vehicle', 'crash_object', 'crash_sidewalk', 'out_of_road')
    if not set(safety).issubset(event_names):
        missing.append('safety_event_names')
    elif events.shape != (n, len(event_names)):
        failures.append('event_shape')
    else:
        check(not events[:, [event_names.index(k) for k in safety]].any(), 'safety_event')
    commanded = np.asarray(arrays['action'])
    applied = np.asarray(arrays['applied_action'])
    check(commanded.shape == applied.shape == (n, 2), 'action_shape')
    if commanded.shape == applied.shape == (n, 2):
        check(np.isfinite(applied).all(), 'nonfinite_actions')
        check(np.allclose(commanded, applied, atol=1e-6), 'commanded_applied_action_mismatch')
    expected_valid = np.arange(n)[:, None] + 2 * np.arange(1, 11) <= n
    valid = np.asarray(arrays['valid'])
    check(valid.shape == expected_valid.shape and np.array_equal(valid, expected_valid), 'future_mask_mismatch')
    traj = np.asarray(arrays['trajectory'])
    if traj.shape != (n, 10, 4):
        failures.append('trajectory_shape')
    else:
        expected = np.zeros_like(traj)
        for t in range(n):
            indices = t + 2 * np.arange(1, 11)
            keep = indices <= n
            future = states[indices[keep]]
            delta = future[:, :2] - states[t, :2]
            c, s = np.cos(states[t, 2]), np.sin(states[t, 2])
            expected[t, keep, 0] = delta[:, 0] * c + delta[:, 1] * s
            expected[t, keep, 1] = -delta[:, 0] * s + delta[:, 1] * c
            expected[t, keep, 2] = future[:, 3]
            angle = future[:, 2] - states[t, 2]
            expected[t, keep, 3] = np.arctan2(np.sin(angle), np.cos(angle))
        check(np.allclose(traj, expected, atol=2e-4), 'trajectory_not_recomputed_from_states')
    speed = states[:-1, 3]
    moved = np.maximum.accumulate((speed >= 1.) & (np.linalg.norm(states[:-1, :2] - states[0, :2], axis=1) >= 2.))
    metrics['longest_stationary_intervals_after_motion'] = max(0, _longest_run(moved & (speed <= .05)) - 1)
    if case_type not in STOP_CASES:
        # This revision independently accepts stop/resume only. Other scenarios
        # retain measured diagnostics but cannot inherit manifest acceptance.
        if 'lateral_m' in arrays:
            metrics['max_abs_lateral_m'] = float(np.max(np.abs(arrays['lateral_m'])))
        missing.append('independent_behavior_audit_for_' + case_type)
        return dict(passed=False, failures=failures, missing=missing, metrics=metrics)
    check(metrics['longest_stationary_intervals_after_motion'] >= 25,
          'no_25_interval_standstill_after_motion')
    keys = ('diagnostic_npc_states', 'diagnostic_npc_present_before', 'diagnostic_npc_present_after',
            'diagnostic_npc_clearance_m', 'diagnostic_npc_lateral_m', 'diagnostic_release_event',
            'diagnostic_teacher_hold_brake', 'diagnostic_npc_length_m', 'diagnostic_ego_length_m',
            'diagnostic_road_heading_rad', 'step_id', 'time_s', 'transition_valid', 'action_mask')
    missing.extend(k for k in keys if k not in arrays)
    if missing:
        return dict(passed=False, failures=failures, missing=missing, metrics=metrics)
    check(np.array_equal(arrays['step_id'], np.arange(n)), 'step_sequence')
    check(np.allclose(arrays['time_s'], np.arange(n)*.1), 'time_sequence')
    check(np.asarray(arrays['transition_valid']).shape == (n,) and np.all(arrays['transition_valid']), 'transition_mask')
    noise = np.asarray(arrays.get('noise_active', np.zeros(n, bool)))
    unsafe = events[:, [event_names.index(k) for k in safety]].any(1) if events.shape == (n, len(event_names)) and set(safety).issubset(event_names) else np.ones(n, bool)
    check(np.array_equal(arrays['action_mask'], ~noise & ~unsafe), 'action_mask_mismatch')
    npc = np.asarray(arrays['diagnostic_npc_states'])
    before, after = (np.asarray(arrays[k], bool) for k in ('diagnostic_npc_present_before', 'diagnostic_npc_present_after'))
    gap, lat = (np.asarray(arrays[k]) for k in ('diagnostic_npc_clearance_m', 'diagnostic_npc_lateral_m'))
    if npc.ndim != 3 or npc.shape[0] != n or npc.shape[1] == 0 or npc.shape[2] != 6 or before.shape != npc.shape[:2] or after.shape != before.shape or gap.shape != before.shape or lat.shape != before.shape:
        failures.append('npc_diagnostic_shape')
        return dict(passed=False, failures=failures, missing=missing, metrics=metrics)
    heading = np.asarray(arrays['diagnostic_road_heading_rad'])
    delta = npc[:, :, :2] - states[:-1, None, :2]
    c, s = np.cos(heading)[:, None], np.sin(heading)[:, None]
    measured_gap = delta[:, :, 0]*c + delta[:, :, 1]*s - (np.asarray(arrays['diagnostic_npc_length_m']) + np.asarray(arrays['diagnostic_ego_length_m'])[:, None])/2
    measured_lat = -delta[:, :, 0]*s + delta[:, :, 1]*c
    check(np.allclose(gap[before], measured_gap[before], atol=.01), 'npc_gap_geometry_mismatch')
    check(np.allclose(lat[before], measured_lat[before], atol=.01), 'npc_lateral_geometry_mismatch')
    release = np.flatnonzero(arrays['diagnostic_release_event'])
    check(len(release) == 1, 'expected_exactly_one_release')
    if len(release) == 1:
        r = int(release[0]); metrics['release_step'] = r
        start = r
        blocker = before & (npc[:, :, 3] <= .05) & (gap >= 2.) & (gap <= 12.) & (np.abs(lat) <= 1.)
        eligible = moved & (speed <= .05) & blocker.any(1)
        while start > 0 and eligible[start-1]:
            start -= 1
        intervals = r-start if eligible[r] else 0
        metrics['stationary_intervals_before_release'] = intervals
        check(intervals >= 25, 'insufficient_continuous_standstill_before_release')
        check(not np.asarray(arrays['diagnostic_release_event'])[:r].any(), 'early_release')
        hold = np.asarray(arrays['diagnostic_teacher_hold_brake'], bool)
        if intervals >= 1:
            check(np.all(hold[start:r]) and np.all(applied[start:r, 1] <= -.99), 'brake_not_held_during_standstill')
        check(np.all(before[:r+1]), 'blocker_missing_before_release')
        if case_type == 'static_blockage_and_resume':
            check(not after[r:].any(), 'static_obstacle_not_removed_at_release')
        else:
            check(np.all(before) and np.all(after), 'lead_actor_disappeared')
            check(np.any(npc[:start, :, 3] >= .3), 'lead_never_moved_before_stop')
            check(np.any(npc[r+1:, :, 3] >= 1.), 'lead_did_not_resume_after_release')
        threshold = 3.5 if case_type == 'static_blockage_and_resume' else 1.5
        post = states[r+1:, 3]
        metrics['max_speed_after_release'] = float(post.max()) if len(post) else None
        metrics['resumed_consecutive_observations'] = _longest_run(post >= threshold)
        check(metrics['resumed_consecutive_observations'] >= 15, 'insufficient_continuous_resume')
    return dict(passed=not failures and not missing, failures=failures, missing=missing, metrics=metrics)
