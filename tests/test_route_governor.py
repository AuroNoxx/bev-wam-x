"""Regression tests for the route-aware sharp-turn fallback."""

import numpy as np

from controller.route_governor import route_plan, route_turn_rad, should_override


def _straight_plan():
    return np.array([[0.4 * i, 0.0, 4.0, 0.0] for i in range(1, 11)], dtype=np.float32)


def test_straight_route_does_not_override():
    route = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [15.0, 0.0]])
    override, info = should_override(_straight_plan(), route, np.ones(4, dtype=bool))
    assert not override
    assert abs(info["route_turn_deg"]) < 1.0


def test_sharp_route_overrides_straight_model_plan():
    route = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [10.0, 5.0]])
    assert route_turn_rad(route, np.ones(4, dtype=bool)) > np.deg2rad(45)
    override, info = should_override(_straight_plan(), route, np.ones(4, dtype=bool))
    assert override
    assert info["route_turn_deg"] > 45
    assert info["deviation_deg"] < -45


def test_route_plan_follows_turn_and_caps_speed():
    route = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [5.0, 10.0]])
    plan, info = route_plan(route, np.ones(4, dtype=bool), speed=4.0, horizon=2.0, steps=10)
    assert plan is not None and plan.shape == (10, 4)
    assert np.allclose(plan[:, 2], 4.0)
    assert plan[-1, 1] > 1.0
    assert info["route_turn_deg"] > 45


def test_route_plan_uses_model_lookahead_speed_below_cap():
    route = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [15.0, 0.0]])
    model = np.array([[1.0 * i, 0.0, 5.2, 0.0] for i in range(1, 11)], dtype=np.float32)
    plan, info = route_plan(route, np.ones(4, dtype=bool), speed=6.0, model_plan=model)
    assert np.allclose(plan[:, 2], 5.2)
    assert np.isclose(info["applied_speed_m_s"], 5.2)


def test_route_plan_uses_cap_when_model_is_faster():
    route = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [15.0, 0.0]])
    model = np.array([[1.0 * i, 0.0, 6.5, 0.0] for i in range(1, 11)], dtype=np.float32)
    plan, info = route_plan(route, np.ones(4, dtype=bool), speed=6.0, model_plan=model)
    assert np.allclose(plan[:, 2], 6.0)
    assert np.isclose(info["applied_speed_m_s"], 6.0)


def test_invalid_route_returns_none():
    plan, _ = route_plan(np.array([[0.0, 0.0], [0.0, 0.0]]), np.ones(2, dtype=bool), speed=4.0)
    assert plan is None
