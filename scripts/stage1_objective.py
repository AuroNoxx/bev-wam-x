"""Stage-one objectives and physical-unit diagnostics, independent of Trainer."""
import torch
from torch.nn import functional as F


def kinematic_residual(trajectory, state):
    """m/s residual of displacement versus trapezoidal heading/speed integration.

    trajectory is B,T,5 in the unchanged dataset scale. Include the observed
    origin, zero relative heading and current speed as the first boundary.
    This is a consistency constraint, not a calibrated vehicle dynamics model.
    """
    xy = trajectory[..., :2] * trajectory.new_tensor([50., 10.])
    direction = F.normalize(trajectory[..., [3, 2]], dim=-1, eps=1e-4)
    velocity = direction * (trajectory[..., 4:5] * 30.)
    initial_velocity = torch.stack([state[:, 0] * 30., torch.zeros_like(state[:, 0])], -1)[:, None]
    previous_xy = torch.cat([torch.zeros_like(xy[:, :1]), xy[:, :-1]], 1)
    previous_velocity = torch.cat([initial_velocity, velocity[:, :-1]], 1)
    return (xy - previous_xy) / .2 - (velocity + previous_velocity) * .5


def consistent_flow_loss(model, batch, config):
    physical_target = batch['trajectory'].float()
    target = model.to_flow_space(batch, physical_target) if hasattr(model, 'to_flow_space') else physical_target
    noise = torch.randn_like(target)
    t = torch.rand(len(target), 1, device=target.device)
    xt = (1 - t[:, :, None]) * noise + t[:, :, None] * target
    prediction = model(batch, xt, t).float()
    fm = F.mse_loss(prediction, target - noise)
    # Local endpoint estimate; auxiliary terms use the less ambiguous latter
    # half of the flow. Keep the standard FM objective over the entire interval.
    endpoint = xt + (1 - t[:, :, None]) * prediction
    if hasattr(model, 'from_flow_space'):
        endpoint = model.from_flow_space(batch, endpoint)
    gate = (t[:, 0] >= config['auxiliary_min_t']).float()
    gate = gate / (1 - config['auxiliary_min_t'])
    def gated(value):
        return (value.flatten(1).mean(1) * gate).mean()
    xy_error = (endpoint[..., :2] - physical_target[..., :2]) * endpoint.new_tensor([50., 10.])
    position = gated(F.smooth_l1_loss(xy_error / config['position_scale_m'], torch.zeros_like(xy_error), reduction='none'))
    residual = kinematic_residual(endpoint, batch['state'].float()) / config['velocity_scale_mps']
    kinematics = gated(F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction='none'))
    unit_heading = gated((endpoint[..., 2:4].square().sum(-1) - 1).square())
    negative_speed = gated(F.relu(-endpoint[..., 4] * 30. / config['velocity_scale_mps']).square())
    terms = {'fm': fm, 'position': position, 'kinematics': kinematics,
             'unit_heading': unit_heading, 'negative_speed': negative_speed}
    total = sum(config['weights'][name] * value for name, value in terms.items())
    return total, terms


def trajectory_metrics(pred, batch):
    """One value per window; all eight candidates count, no ground-truth selector."""
    target = batch['trajectory'].float()
    delta = (pred[..., :2] - target[:, None, :, :2]) * pred.new_tensor([50., 10.])
    error = delta.norm(dim=-1)
    ade = error.mean(-1)
    xy = pred[..., :2] * pred.new_tensor([50., 10.])
    derived_speed = (xy[:, :, 1:] - xy[:, :, :-1]).norm(dim=-1) / .2
    b, k = pred.shape[:2]
    residual = kinematic_residual(pred.flatten(0, 1), batch['state'].float().repeat_interleave(k, 0))
    target_residual = kinematic_residual(target, batch['state'].float())
    # RMS spread, not a quality score: collapsing all candidates can improve
    # mean ADE while removing useful alternatives, so retain this diagnostic.
    spread = (xy - xy.mean(1, keepdim=True)).square().sum(-1).mean((1, 2)).sqrt()
    return {
        'meanADE8_m': ade.mean(1), 'minADE8_m': ade.min(1).values,
        'meanFDE8_m': error[:, :, -1].mean(1), 'minFDE8_m': error[:, :, -1].min(1).values,
        'longitudinal_MAE_m': delta[..., 0].abs().mean((1, 2)),
        'lateral_MAE_m': delta[..., 1].abs().mean((1, 2)),
        'speed_RMSE_mps': ((pred[..., 4] - target[:, None, :, 4]) * 30.).square().mean((1, 2)).sqrt(),
        'speed_displacement_MAE_mps': (derived_speed - pred[:, :, 1:, 4] * 30.).abs().mean((1, 2)),
        'kinematic_residual_mps': residual.norm(dim=-1).reshape(b, k, -1).mean((1, 2)),
        'target_kinematic_residual_mps': target_residual.norm(dim=-1).mean(1),
        'candidate_spread_m': spread,
    }
