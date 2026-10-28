"""Map-assisted nominal trajectory from the same declared route and own vehicle.

This is an explicit physical/control prior, not a learned behavior or a traffic
oracle. It uses no future simulator observation or other-vehicle state.
"""
import numpy as np


def navigation_reference(ego, route_points, route_valid, vehicle):
    ego, route, valid, vehicle = np.asarray(ego), np.asarray(route_points), np.asarray(route_valid, bool), np.asarray(vehicle)
    b = len(ego)
    if ego.shape != (b, 4, 2) or route.shape != (b, 8, 2) or valid.shape != (b, 8) or vehicle.shape != (b, 4):
        raise ValueError('Navigation reference input shapes differ from declared contract')
    if not np.isfinite(ego).all() or not np.isfinite(route).all() or not np.isfinite(vehicle).all() or np.any(vehicle <= 0):
        raise ValueError('Invalid reference observations/vehicle')
    segment_valid = valid[:, :-1] & valid[:, 1:]
    delta = np.diff(route, axis=1)
    length = np.where(segment_valid, np.linalg.norm(delta, axis=-1), 0.)
    cumulative = np.concatenate([np.zeros((b, 1)), np.cumsum(length, axis=1)], axis=1)
    xy = np.zeros((b, 2)); heading = np.zeros(b); speed = np.maximum(ego[:, -1, 0]*10, 0.)
    result = []; rows = np.arange(b); wheelbase = vehicle[:, 2]
    for step in range(20):
        fraction = np.clip(np.sum((xy[:, None]-route[:, :-1])*delta, axis=-1)/np.maximum(length**2, 1e-8), 0, 1)
        projection = route[:, :-1]+fraction[..., None]*delta
        distances = np.where(segment_valid & (length > 1e-6), np.linalg.norm(xy[:, None]-projection, axis=-1), np.inf)
        segment = distances.argmin(1)
        progress = cumulative[rows, segment]+fraction[rows, segment]*length[rows, segment]
        target_s = np.minimum(progress+np.maximum(4., .65*speed), cumulative[:, -1])
        target_segment = np.minimum((cumulative[:, 1:] < target_s[:, None]).sum(1), 6)
        local_s = target_s-cumulative[rows, target_segment]
        ratio = local_s/np.maximum(length[rows, target_segment], 1e-8)
        target = route[rows, target_segment]+ratio[:, None]*delta[rows, target_segment]-xy
        c, s = np.cos(heading), np.sin(heading)
        target_y = -target[:, 0]*s+target[:, 1]*c
        steering = np.arctan2(2*wheelbase*target_y, np.maximum((target**2).sum(1), 1e-4))
        steering = np.clip(steering, -vehicle[:, 3], vehicle[:, 3])
        curvature = np.tan(steering)/wheelbase
        desired = np.minimum(7., np.sqrt(3./np.maximum(np.abs(curvature), 1e-6)))
        desired = np.minimum(desired, np.sqrt(8*np.maximum(cumulative[:, -1]-progress, 0.)))
        desired = np.where(valid.sum(1) >= 2, desired, 0.)
        acceleration = np.clip((desired-speed)/.1, -4., 2.)
        next_speed = np.maximum(speed+.1*acceleration, 0.)
        midpoint_speed = .5*(speed+next_speed)
        yaw = midpoint_speed*curvature
        midpoint_heading = heading+.05*yaw
        xy += .1*midpoint_speed[:, None]*np.column_stack([np.cos(midpoint_heading), np.sin(midpoint_heading)])
        heading += .1*yaw; speed = next_speed
        if step % 2 == 1:
            result.append(np.column_stack([xy, speed, heading]).copy())
    return np.stack(result, axis=1).astype(np.float32)
