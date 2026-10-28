"""New strict selector and measured-curvature braking; historical code unchanged."""
import numpy as np
from configs.integrated import IntegratedConfig
from planning.candidates import score_candidates


def route_progress(trajectory, points, valid):
    route = np.asarray(points)[np.asarray(valid, bool)]
    if len(route) < 2:
        raise ValueError('At least two valid navigation points required for scoring')
    delta = np.diff(route, axis=0)
    length = np.linalg.norm(delta, axis=1)
    if np.any(length < 1e-5):
        raise ValueError('Repeated navigation points')
    accumulated = np.r_[0., np.cumsum(length)]
    def project(point):
        fraction = np.clip(np.sum((point-route[:-1])*delta, axis=1)/(length**2), 0, 1)
        closest = route[:-1]+fraction[:, None]*delta
        i = int(np.argmin(np.linalg.norm(point-closest, axis=1)))
        return accumulated[i]+fraction[i]*length[i]
    return float(project(np.asarray(trajectory)[-1, :2])-project(np.zeros(2)))


def strict_score(candidates, bev, bevs, states, length, width, navigation,
                 probability=None, thresholds=None, config=IntegratedConfig(), maneuver_allowed=None):
    from configs.planner7 import Planner7Config
    old = score_candidates(candidates, bev, bevs, states, length, width,
                           probability, thresholds, Planner7Config())
    terms = old['terms'].copy()
    reasons = [list(row) for row in old['rejection_reasons']]
    for i, plan in enumerate(candidates):
        terms[i, 0] = -route_progress(plan, navigation.points, navigation.valid)/20.
        if terms[i, 4] > config.max_static_cost:
            reasons[i].append('static_geometry')
        if terms[i, 5] > config.max_dynamic_cost:
            reasons[i].append('dynamic_geometry')
        if maneuver_allowed is not None:
            reasons[i].extend(permission_rejections(plan, navigation, maneuver_allowed))
    weights = np.array([.3, 1., .5, .2, 5., 5., 10., 10.])
    scores = terms@weights
    eligible = np.array([not row for row in reasons])
    index = int(np.argmin(np.where(eligible, scores, np.inf))) if eligible.any() else -1
    return dict(old, terms=terms, scores=scores, rejection_reasons=reasons,
                selected_index=index, eligible=eligible)


def permission_rejections(trajectory, navigation, allowed):
    """Conservative route-relative lane displacement check, with shared map input.

    Turning along a curved route is not itself a lane change. Permission columns
    are toward smaller/larger lane indices (left/right on these PG roads).
    """
    allowed = np.asarray(allowed, bool)
    if allowed.shape != (2,):
        raise ValueError('Expected two explicitly observed static permissions')
    route = np.asarray(navigation.points)[navigation.valid]
    if len(route) < 2:
        return ['navigation_unavailable']
    delta = np.diff(route, axis=0); length = np.linalg.norm(delta, axis=-1)
    if np.any(length < 1e-5):
        return ['navigation_degenerate']
    def lateral(points):
        fraction = np.clip(np.sum((points[:, None]-route[:-1])*delta, axis=-1)/(length**2), 0, 1)
        projection = route[:-1]+fraction[..., None]*delta
        residual = points[:, None]-projection
        segment = np.linalg.norm(residual, axis=-1).argmin(1)
        offset = residual[np.arange(len(points)), segment]
        tangent = delta[segment]/length[segment, None]
        return tangent[:, 0]*offset[:, 1]-tangent[:, 1]*offset[:, 0]
    displacement = lateral(np.asarray(trajectory)[:, :2])-lateral(np.zeros((1, 2)))[0]
    threshold = float(navigation.lane_width_m)*.5
    reasons = []
    if displacement.max() > threshold and not allowed[0]:
        reasons.append('static_left_lane_permission')
    if displacement.min() < -threshold and not allowed[1]:
        reasons.append('static_right_lane_permission')
    return reasons


def curvature_braking(speed, yaw_rate, config=IntegratedConfig()):
    """Brake along observed curvature, not a guaranteed safe lane-following plan.

    Caller must check geometry/risk and log unavoidable/unknown fallback states.
    Curvature is bounded by declared lateral acceleration; heading stays
    continuous through stopping. No route oracle or traffic state is read.
    """
    if not np.isfinite([speed, yaw_rate]).all() or speed < 0:
        raise ValueError('Invalid measured motion')
    t = np.arange(1, config.trajectory_points+1)*config.waypoint_dt_s
    stop_t = np.minimum(t, speed/config.brake_m_s2)
    distance = speed*stop_t-.5*config.brake_m_s2*stop_t**2
    curvature = float(np.clip(yaw_rate/max(speed, .5),
                             -config.max_lateral_accel_m_s2/max(speed**2, .25),
                             config.max_lateral_accel_m_s2/max(speed**2, .25)))
    angle = curvature*distance
    x = distance*np.sinc(angle/np.pi)
    y = distance*(angle/2)*np.sinc(angle/(2*np.pi))**2
    return np.column_stack([x, y, np.maximum(speed-config.brake_m_s2*t, 0), angle]).astype(np.float32)

