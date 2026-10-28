"""Regression: tracker/PID must use the 10 Hz decision period by default."""

from controller.pure_pursuit import SpeedPID
from controller.trajectory_tracker import TrajectoryTracker


class _Car:
    FRONT_WHEELBASE = 1.4
    REAR_WHEELBASE = 1.4
    max_steering = 40.0


def test_tracker_default_is_10hz():
    tracker = TrajectoryTracker(_Car())
    assert tracker.pid.dt == 0.1


def test_tracker_explicit_dt_is_respected():
    tracker = TrajectoryTracker(_Car(), dt_s=0.02)
    assert tracker.pid.dt == 0.02


def test_speed_pid_default_is_10hz():
    assert SpeedPID().dt == 0.1
