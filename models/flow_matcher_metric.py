"""Stage 7 v2 loss ablation; architecture and sampling stay unchanged."""
import torch

from models.flow_matcher_transformer import TransformerKinematicFlowMatcher


def physical_endpoint_losses(predicted, target, valid):
    """Masked ADE/FDE in metres, including only the actual 2 s endpoint for FDE.

    The caller supplies reconstructed clean trajectories, not ODE samples. Invalid
    targets are removed before arithmetic so NaNs cannot enter loss or gradients.
    """
    clean_target = torch.where(valid[..., None], target, predicted.detach())
    distance = torch.linalg.vector_norm(predicted[..., :2] - clean_target[..., :2], dim=-1)
    ade = (distance * valid).sum() / valid.sum().clamp_min(1)
    fde = (distance[:, -1] * valid[:, -1]).sum() / valid[:, -1].sum().clamp_min(1)
    return ade, fde


class MetricFlowMatcher(TransformerKinematicFlowMatcher):
    """Same v2 parameters; optional physical clean-endpoint reconstruction loss."""

    def compute_loss(self, history, ego, target, action_mask, waypoint_valid,
                     noise=None, t=None, ade_weight=0., fde_weight=0., return_parts=False):
        prior = self.compute_norm_prior(ego)
        std = self.residual_std.to(device=target.device, dtype=target.dtype).clamp_min(1e-6)
        valid = action_mask[:, None] & waypoint_valid
        # Sanitize before division/interpolation, including invalid NaN labels.
        clean = torch.where(valid[..., None], target, prior)
        residual = (clean - prior) / std
        source = torch.randn_like(target) if noise is None else noise
        time = torch.rand(len(target), device=target.device, dtype=target.dtype) if t is None else t
        endpoint = torch.where(valid[..., None], residual, source)
        time = time[:, None, None]
        z_t = (1 - time) * source + time * endpoint
        velocity = self.forward_vf(z_t, time[:, 0, 0], self.extract_context(history, ego))
        error = (velocity - (endpoint - source)).square() * valid[..., None]
        flow = error.sum() / (valid.sum() * 4).clamp_min(1)

        # Linear-path clean-endpoint estimator; this is NOT an integrated sample.
        # Adding its loss reweights the original regression in physical units/time.
        reconstructed_norm = prior + (z_t + (1 - time) * velocity) * std
        scale = torch.stack([self.trajectory_scale] * 3 + [self.trajectory_scale.new_tensor(1.)])
        predicted_si = (reconstructed_norm * self.trajectory_std + self.trajectory_mean) / scale
        target_si = (clean * self.trajectory_std + self.trajectory_mean) / scale
        ade, fde = physical_endpoint_losses(predicted_si, target_si, valid)
        loss = flow + ade_weight * ade + fde_weight * fde
        parts = dict(flow=flow, reconstructed_ade_m=ade, reconstructed_fde_m=fde, total=loss)
        return (loss, velocity, parts) if return_parts else (loss, velocity)
