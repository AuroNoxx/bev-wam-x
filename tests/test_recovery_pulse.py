"""Regression tests for the collection-time recovery pulse schedule."""

from evaluation.stage4_diagnostic import recovery_pulse


def test_recovery_pulse_matches_collection_schedule():
    cases = [(4300006, -1), (4300007, 1), (4300008, -1), (4300009, 1), (4300010, -1), (4300011, 1)]
    for seed, sign in cases:
        assert recovery_pulse(seed, 30, True) == (True, sign)
        assert recovery_pulse(seed, 35, True) == (True, sign)
        assert recovery_pulse(seed, 36, True) == (False, sign)
        assert recovery_pulse(seed, 190, True) == (True, sign)
        assert recovery_pulse(seed, 0, True) == (False, sign)
        assert recovery_pulse(seed, 30, False) == (False, sign)


def test_recovery_pulse_disabled_flag():
    assert recovery_pulse(4300006, 30, False) == (False, -1)
    assert recovery_pulse(4300000, 30, False) == (False, 1)
