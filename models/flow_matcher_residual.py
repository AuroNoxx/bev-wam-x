"""Linear-interpolation Conditional Residual Flow Matching trajectory head for multi-candidate generation."""
import math
import torch
from torch import nn
import torch.nn.functional as F

from models.flow_matcher import sinusoidal_time_embedding


class ConditionalResidualFlowMatcher(nn.Module):
    """Generates 10x4 trajectory as MSE baseline + flow-matched residual conditioned on BEV tokens."""

    def __init__(self, token_count=64, token_dim=256, condition_dim=256, hidden_dim=256, residual_std=None):
        super().__init__()
        self.token_count = token_count
        self.token_dim = token_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.time_emb_dim = 32

        # Temporal fusion over 4 history frames per spatial token
        self.history_proj = nn.Linear(4 * token_dim, condition_dim)
        self.ego_proj = nn.Linear(8, condition_dim)

        # Vector field network: noisy_z (40) + time_emb (32) + spatial_tokens (token_count * condition_dim)
        in_features = 40 + self.time_emb_dim + token_count * condition_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 40),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        if residual_std is None:
            residual_std = torch.ones(10, 4, dtype=torch.float32)
        elif not isinstance(residual_std, torch.Tensor):
            residual_std = torch.tensor(residual_std, dtype=torch.float32)
        self.register_buffer('residual_std', residual_std)

    @property
    def model_arch(self):
        return dict(
            token_count=self.token_count,
            token_dim=self.token_dim,
            condition_dim=self.condition_dim,
            hidden_dim=self.hidden_dim,
            time_emb_dim=self.time_emb_dim,
            model_type='residual',
        )

    def extract_condition(self, history, ego):
        """Extract condition representation preserving spatial token locations."""
        b, t, n, d = history.shape
        if t != 4 or n != self.token_count or d != self.token_dim:
            raise ValueError(f'Expected history of shape [B, 4, {self.token_count}, {self.token_dim}], got {history.shape}')
        temporal_flat = history.permute(0, 2, 1, 3).reshape(b, n, 4 * d)
        tokens = self.history_proj(temporal_flat)
        ego_emb = self.ego_proj(ego.reshape(b, 8))[:, None, :]
        fused = tokens + ego_emb
        return fused.reshape(b, n * self.condition_dim)

    def vector_field(self, z_t, t, condition):
        """Predict flow vector field v_t(z_t | condition)."""
        b = len(z_t)
        z_flat = z_t.reshape(b, 40)
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)
        inp = torch.cat([z_flat, t_emb, condition], dim=-1)
        return self.net(inp).reshape(b, 10, 4)

    def compute_loss(self, x_1, x_mse, condition, action_mask, waypoint_valid, noise=None, t=None):
        """Compute Flow Matching MSE loss on standardized residual:
        Target: z_1 = (x_1 - x_mse) / residual_std
        Interpolation: z_t = (1 - t) * z_0 + t * z_1
        Velocity field target: u_t = z_1 - z_0
        """
        b = len(x_1)
        device = x_1.device
        dtype = x_1.dtype

        std = self.residual_std.to(device=device, dtype=dtype)
        z_1_raw = (x_1 - x_mse) / std.clamp_min(1e-6)

        if noise is None:
            z_0 = torch.randn_like(z_1_raw)
        else:
            z_0 = noise

        if t is None:
            t_sample = torch.rand(b, device=device, dtype=dtype)
        else:
            t_sample = t

        point_mask = (action_mask[:, None] & waypoint_valid)[..., None]
        z_1_clean = torch.where(point_mask, z_1_raw, z_0)

        t_expand = t_sample[:, None, None]
        z_t = (1.0 - t_expand) * z_0 + t_expand * z_1_clean
        v_target = z_1_clean - z_0

        v_pred = self.vector_field(z_t, t_sample, condition)

        error_sq = (v_pred - v_target).square() * point_mask.float()
        denom = point_mask.float().sum() * 4.0
        loss = error_sq.sum() / denom.clamp_min(1.0)
        return loss, v_pred

    @torch.no_grad()
    def sample(self, condition, x_mse, num_steps=10, num_candidates=1, temperature=1.0, seed=None, device=None):
        """Batched Euler ODE integration from Gaussian noise to trajectory: t from 0 to 1.
        Final trajectory = x_mse + z_final * residual_std
        """
        b = len(condition)
        if device is None:
            device = condition.device

        k = num_candidates
        cond_rep = condition.repeat_interleave(k, dim=0)

        std = self.residual_std.to(device=device, dtype=condition.dtype)

        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)

        z = torch.randn(b * k, 10, 4, generator=gen, device=device, dtype=condition.dtype) * temperature
        dt = 1.0 / num_steps

        for s in range(num_steps):
            t_val = s * dt
            t_tensor = torch.full((b * k,), t_val, device=device, dtype=condition.dtype)
            v = self.vector_field(z, t_tensor, cond_rep)
            z = z + v * dt

        z_out = z.reshape(b, k, 10, 4)
        delta_x = z_out * std[None, None, :, :]

        x_mse_expanded = x_mse[:, None, :, :].expand(b, k, 10, 4)
        x_out = x_mse_expanded + delta_x

        if k == 1:
            return x_out[:, 0]
        return x_out
