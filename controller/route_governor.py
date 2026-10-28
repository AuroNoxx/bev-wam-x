"""Route-aware fallback trajectory for sharp turns (legal navigation observations only)."""

import numpy as np


def _wrap(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _heading(a, b):
    return float(np.arctan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0])))


def _route_points(route_points, route_valid):
    points = np.asarray(route_points, dtype=np.float64)
    valid = np.asarray(route_valid, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 2 or valid.shape != (len(points),):
        raise ValueError("route_points must be [N,2] and route_valid must be [N]")
    points = points[valid]
    if len(points) == 0:
        return points
    if np.linalg.norm(points[0]) > 1e-6:
        points = np.vstack([[0.0, 0.0], points])
    return points


def route_heading_rad(route_points, route_valid):
    """Heading of the final route segment in the ego frame."""
    points = _route_points(route_points, route_valid)
    return _heading(points[-2], points[-1]) if len(points) >= 2 else 0.0


def route_turn_rad(route_points, route_valid):
    """Net turn from the first to the last valid route segment."""
    points = _route_points(route_points, route_valid)
    if len(points) < 3:
        return 0.0
    return _wrap(_heading(points[-2], points[-1]) - _heading(points[0], points[1]))


def model_plan_heading_rad(plan):
    """Heading from the plan origin to its final waypoint."""
    arr = np.asarray(plan, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return 0.0
    return _heading(arr[0, :2], arr[-1, :2])


def model_target_speed_m_s(model_plan, lookahead=4.0):
    """Model speed at the tracker lookahead; None if unavailable."""
    if model_plan is None:
        return None
    arr = np.asarray(model_plan, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 3:
        return None
    distance = np.linalg.norm(arr[:, :2], axis=1)
    eligible = np.flatnonzero((distance >= float(lookahead)) & (arr[:, 0] > 0))
    index = int(eligible[0]) if len(eligible) else len(arr) - 1
    value = float(arr[index, 2])
    return value if np.isfinite(value) and value > 0.1 else None


def route_plan(route_points, route_valid, speed=4.5, horizon=2.0, steps=10, model_plan=None):
    """Resample the valid route into a [steps,4] (x,y,v,heading) plan.

    ``speed`` is the fallback cap; when ``model_plan`` is supplied, the plan
    uses min(cap, model speed at the lookahead) so long routes are not slowed.
    """
    points = _route_points(route_points, route_valid)
    model_speed = model_target_speed_m_s(model_plan)
    applied_speed = float(min(float(speed), model_speed)) if model_speed is not None else float(speed)
    info = dict(route_length_m=None, route_turn_deg=None, route_heading_deg=None,
                speed_m_s=float(speed), applied_speed_m_s=applied_speed,
                model_target_speed_m_s=model_speed, horizon_s=float(horizon), steps=int(steps))
    if len(points) < 2 or steps < 2 or speed <= 0 or horizon <= 0:
        return None, info
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cumulative[-1])
    if total <= 1e-9:
        return None, info
    ds = float(speed) * float(horizon) / (int(steps) - 1)
    targets = np.minimum(np.arange(1, int(steps) + 1, dtype=np.float64) * ds, total)
    plan = np.zeros((int(steps), 4), dtype=np.float32)
    for i, distance in enumerate(targets):
        j = int(np.clip(np.searchsorted(cumulative, distance, side="right") - 1, 0, len(points) - 2))
        length = float(seg[j])
        t = 0.0 if length <= 1e-9 else float((distance - cumulative[j]) / length)
        t = float(np.clip(t, 0.0, 1.0))
        plan[i, :2] = points[j] * (1.0 - t) + points[j + 1] * t
        plan[i, 2] = applied_speed
        plan[i, 3] = _heading(points[j], points[j + 1])
    info.update(route_length_m=total,
                route_turn_deg=float(np.degrees(route_turn_rad(route_points, route_valid))),
                route_heading_deg=float(np.degrees(route_heading_rad(route_points, route_valid))))
    return plan, info


def should_override(plan, route_points, route_valid, turn_threshold_deg=20.0, deviation_threshold_deg=20.0):
    """Override the model plan only when the route and the model disagree on a sharp turn."""
    turn = route_turn_rad(route_points, route_valid)
    route_h = route_heading_rad(route_points, route_valid)
    model_h = model_plan_heading_rad(plan)
    deviation = _wrap(model_h - route_h)
    info = dict(route_turn_deg=float(np.degrees(turn)),
                route_heading_deg=float(np.degrees(route_h)),
                model_plan_heading_deg=float(np.degrees(model_h)),
                deviation_deg=float(np.degrees(deviation)),
                turn_threshold_deg=float(turn_threshold_deg),
                deviation_threshold_deg=float(deviation_threshold_deg))
    override = bool(abs(turn) >= np.deg2rad(turn_threshold_deg) and
                    abs(deviation) >= np.deg2rad(deviation_threshold_deg))
    info["override"] = override
    return override, info
