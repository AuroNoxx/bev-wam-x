"""Batched own-vehicle priors for the integrated planner, with actual wheelbase."""
import torch
from models.flow_matcher_transformer import kinematic_bicycle_rollout


def batched_vehicle_priors(ego, actions, vehicle):
    """ego[B,4,2], actions[K,2] or[B,K,2], vehicle[B,4] -> [B,K,10,4].

    Vehicle fields are length, width, wheelbase, maximum front steering radians.
    Input actions are commanded physical acceleration and front steering angle.
    """
    if ego.ndim != 3 or ego.shape[1:] != (4, 2) or vehicle.shape != (len(ego), 4):
        raise ValueError('Ego/vehicle shape mismatch')
    if actions.ndim == 2:
        actions = actions[None].expand(len(ego), -1, -1)
    if actions.ndim != 3 or actions.shape[0] != len(ego) or actions.shape[-1] != 2:
        raise ValueError('Action shape mismatch')
    b, k, _ = actions.shape
    repeated_ego = ego.repeat_interleave(k, dim=0)
    wheelbase = vehicle[:, 2].repeat_interleave(k)
    max_steer = vehicle[:, 3, None].expand(b, k).reshape(-1)
    steer = actions[..., 1].reshape(-1).clamp(-max_steer, max_steer)
    result = kinematic_bicycle_rollout(repeated_ego, a=actions[..., 0].reshape(-1),
                                     delta=steer, L=wheelbase)
    return result.reshape(b, k, 10, 4)
