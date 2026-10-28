"""Single Orin-X Kinematic-Flow-Planner (~12M) with Kinematic Bicycle Model Prior."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.flow_matcher import sinusoidal_time_embedding


def kinematic_bicycle_rollout(ego, a=None, delta=None, steps=10, dt=0.2, L=2.47, mu_g=8.3):
    """Numerically integrate Kinematic Bicycle Model for 10 steps (2.0s) at dt=0.2s.
    
    Adheres strictly to:
    - Wheelbase L = 2.47m (MetaDrive ego vehicle)
    - Longitudinal acceleration a in [-5.0, +2.0] m/s^2
    - Tire friction circle limit: sqrt(a_lon^2 + a_lat^2) <= mu*g (8.3 m/s^2)
    
    Args:
        ego: [B, 4, 2] or [B, 2] ego motion history ([v/10, yaw_rate])
        a: Optional [B] or scalar longitudinal acceleration (m/s^2)
        delta: Optional [B] or scalar front wheel steering angle (rad)
    Returns:
        [B, steps, 4] trajectory in ego-relative frame: (x, y, v, heading)
    """
    if ego.ndim == 2:
        v0 = ego[:, 0] * 10.0
        omega0 = ego[:, 1]
        a0 = torch.zeros_like(v0)
    elif ego.ndim == 3:
        v0 = ego[:, -1, 0] * 10.0
        v_prev = ego[:, -2, 0] * 10.0
        a0 = torch.clamp((v0 - v_prev) / 0.1, -5.0, 2.0)
        omega0 = ego[:, -1, 1]
    else:
        raise ValueError(f'Expected ego shape [B, 2] or [B, 4, 2], got {ego.shape}')

    b = v0.shape[0]
    device = ego.device
    dtype = ego.dtype

    if a is None:
        a_use = a0
    elif torch.is_tensor(a):
        a_use = a.expand(b).to(device=device, dtype=dtype)
    else:
        a_use = torch.full((b,), float(a), device=device, dtype=dtype)

    if delta is None:
        delta_use = torch.atan(L * omega0 / torch.clamp(v0, min=0.5))
    elif torch.is_tensor(delta):
        delta_use = delta.expand(b).to(device=device, dtype=dtype)
    else:
        delta_use = torch.full((b,), float(delta), device=device, dtype=dtype)

    xs, ys, vs, headings = [], [], [], []
    x = torch.zeros(b, device=device, dtype=dtype)
    y = torch.zeros(b, device=device, dtype=dtype)
    v = torch.clamp(v0, min=0.0)
    theta = torch.zeros(b, device=device, dtype=dtype)

    substeps = 5
    dt_sub = dt / substeps

    for step in range(steps):
        for _ in range(substeps):
            a_c = torch.clamp(a_use, -5.0, 2.0)
            v_next = torch.clamp(v + a_c * dt_sub, min=0.0, max=15.0)
            v_mid = 0.5 * (v + v_next)

            # Friction circle: a_lon^2 + a_lat^2 <= (mu*g)^2
            max_lat = torch.sqrt(torch.clamp(torch.tensor(mu_g**2, device=device, dtype=dtype) - a_c**2, min=0.0))
            raw_yaw = v_mid * torch.tan(delta_use) / L
            lat_accel = v_mid * raw_yaw
            clamped_lat = torch.clamp(lat_accel, -max_lat, max_lat)
            yaw_rate = torch.where(v_mid > 0.1, clamped_lat / v_mid.clamp_min(0.1), torch.zeros_like(v_mid))

            theta_mid = theta + 0.5 * yaw_rate * dt_sub
            x = x + v_mid * torch.cos(theta_mid) * dt_sub
            y = y + v_mid * torch.sin(theta_mid) * dt_sub
            theta = theta + yaw_rate * dt_sub
            v = v_next

        xs.append(x)
        ys.append(y)
        vs.append(v)
        headings.append(theta)

    return torch.stack([
        torch.stack(xs, dim=1),
        torch.stack(ys, dim=1),
        torch.stack(vs, dim=1),
        torch.stack(headings, dim=1)
    ], dim=-1)


def get_physical_candidate_actions(v: float = 7.0):
    """Dynamic speed-scaled candidate action primitives (acceleration m/s^2, steering angle rad).
    
    Guarantees physical lane changes (~3.5m lateral shift) and evasions across all speeds.
    """
    v_eff = max(float(v), 3.0)
    delta_lane = float(min(0.08, max(0.02, 3.5 * 2.47 / (v_eff**2 * 1.62))))
    return (
        (0.0, 0.0),              # 0: Nominal mode / Cruise
        (-2.0, 0.0),             # 1: Mild braking / Car-following
        (-4.5, 0.0),             # 2: Emergency heavy braking
        (0.0, delta_lane),       # 3: Left overtake / lane change
        (0.0, -delta_lane),      # 4: Right lane change
        (-2.5, delta_lane),      # 5: Brake + Left evasive
        (-2.5, -delta_lane),     # 6: Brake + Right evasive
        (1.0, 0.0),              # 7: Cruise acceleration
    )


PHYSICAL_CANDIDATE_ACTIONS = get_physical_candidate_actions(7.0)


class TransformerKinematicFlowMatcher(nn.Module):
    """Generates 10x4 trajectory from Kinematic Bicycle Model prior + Transformer Cross-Attention residual flow."""

    def __init__(
        self,
        token_count=64,
        token_dim=256,
        d_model=384,
        n_heads=6,
        n_layers=5,
        dim_feedforward=1536,
        residual_std=None,
        normalizer=None,
        **kwargs,
    ):
        super().__init__()
        self.token_count = token_count
        self.token_dim = token_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward

        # Context encoder: maintains 64 spatial token locations + 1 ego motion token
        self.history_proj = nn.Linear(4 * token_dim, d_model)
        self.ego_proj = nn.Linear(8, d_model)

        # Waypoint trajectory queries (10 waypoints at 0.2s)
        self.waypoint_proj = nn.Linear(4, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 10, d_model) * 0.02)
        self.time_proj = nn.Linear(32, d_model)

        # Cross-Attention Transformer Decoder
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            ) for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 4)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Buffers for kinematics and normalization
        if residual_std is None:
            residual_std = torch.ones(10, 4, dtype=torch.float32)
        elif not isinstance(residual_std, torch.Tensor):
            residual_std = torch.tensor(residual_std, dtype=torch.float32)
        self.register_buffer('residual_std', residual_std)

        if normalizer is not None:
            self.register_buffer('trajectory_mean', torch.tensor(normalizer.trajectory_mean, dtype=torch.float32))
            self.register_buffer('trajectory_std', torch.tensor(normalizer.trajectory_std, dtype=torch.float32))
            self.register_buffer('trajectory_scale', torch.tensor(float(normalizer.scale), dtype=torch.float32))
        else:
            self.register_buffer('trajectory_mean', torch.zeros(4, dtype=torch.float32))
            self.register_buffer('trajectory_std', torch.ones(4, dtype=torch.float32))
            self.register_buffer('trajectory_scale', torch.tensor(1.0, dtype=torch.float32))

    @property
    def model_arch(self):
        return dict(
            token_count=self.token_count,
            token_dim=self.token_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dim_feedforward=self.dim_feedforward,
            model_type='transformer_kinematic',
        )

    def compute_norm_prior(self, ego, a=None, delta=None):
        """Compute Kinematic Bicycle Model prior in normalized dataset coordinates."""
        if ego.ndim == 2:
            ego = ego.unsqueeze(0)
        raw_prior = kinematic_bicycle_rollout(ego.float(), a=a, delta=delta)  # [B, 10, 4] in SI meters
        scale = torch.stack([self.trajectory_scale] * 3 + [self.trajectory_scale.new_tensor(1.0)])
        norm_prior = (raw_prior * scale - self.trajectory_mean) / self.trajectory_std
        return norm_prior

    def extract_context(self, history, ego):
        """Extract spatial context tokens preserving 64 locations + 1 ego motion token."""
        if history.ndim == 3:
            history = history.unsqueeze(0)
        if ego.ndim == 2:
            ego = ego.unsqueeze(0)
        b, t, n, d = history.shape
        if t != 4 or n != self.token_count or d != self.token_dim:
            raise ValueError(f'Expected history of shape [B, 4, {self.token_count}, {self.token_dim}], got {history.shape}')
        flat_history = history.permute(0, 2, 1, 3).reshape(b, n, 4 * d)
        tokens = self.history_proj(flat_history)  # [B, 64, D]
        ego_token = self.ego_proj(ego.reshape(b, 8))[:, None, :]  # [B, 1, D]
        return torch.cat([ego_token, tokens], dim=1)  # [B, 65, D]

    def extract_condition(self, history, ego):
        """Compatibility alias for extract_context."""
        return self.extract_context(history, ego)

    def vector_field(self, z_t, t, context):
        """Compatibility alias for forward_vf."""
        return self.forward_vf(z_t, t, context)

    def forward_vf(self, z_t, t, context):
        """Predict vector field velocity v_t(z_t | t, context)."""
        b = len(z_t)
        t_emb = sinusoidal_time_embedding(t, 32)
        t_feat = self.time_proj(t_emb)[:, None, :]  # [B, 1, D]
        wp_feat = self.waypoint_proj(z_t) + self.pos_emb + t_feat  # [B, 10, D]

        h = wp_feat
        for layer in self.layers:
            h = layer(tgt=h, memory=context)
        return self.out_proj(self.out_norm(h))  # [B, 10, 4]

    def compute_loss(self, history, ego, target, action_mask, waypoint_valid, noise=None, t=None):
        """Compute Flow Matching MSE loss on residual relative to kinematic bicycle prior."""
        b = len(target)
        device = target.device
        dtype = target.dtype

        # 1. Kinematic bicycle prior
        norm_prior = self.compute_norm_prior(ego)

        # 2. Standardized residual target
        std = self.residual_std.to(device=device, dtype=dtype)
        z_1_raw = (target - norm_prior) / std.clamp_min(1e-6)

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

        context = self.extract_context(history, ego)
        v_pred = self.forward_vf(z_t, t_sample, context)

        error_sq = (v_pred - v_target).square() * point_mask.float()
        denom = point_mask.float().sum() * 4.0
        loss = error_sq.sum() / denom.clamp_min(1.0)
        return loss, v_pred

    @torch.no_grad()
    def sample(
        self,
        history,
        ego,
        num_steps=10,
        num_candidates=1,
        temperature=1.0,
        nominal_mode=False,
        seed=None,
        actions=None,
        device=None,
    ):
        """Batched Euler ODE integration from Gaussian noise to trajectory.
        Candidate = Kinematic Bicycle Prior(a_k, delta_k) + Standardized Residual * residual_std.

        Args:
            nominal_mode: If True, sets candidate 0 initial noise to 0 (deterministic mode),
                          while candidates 1..k-1 explore physical primitives with specified temperature.
        """
        if history.ndim == 3:
            history = history.unsqueeze(0)
        if ego.ndim == 2:
            ego = ego.unsqueeze(0)
        b = len(history)
        if device is None:
            device = history.device

        k = num_candidates

        # Extract context
        context = self.extract_context(history, ego)  # [B, 65, D]
        context_rep = context.repeat_interleave(k, dim=0)  # [B*K, 65, D]

        std = self.residual_std.to(device=device, dtype=history.dtype)

        # Compute kinematic bicycle priors
        if actions is not None:
            norm_priors = []
            for a_val, d_val in actions:
                p_k = self.compute_norm_prior(ego, a=a_val, delta=d_val)
                norm_priors.append(p_k)
            norm_priors = torch.stack(norm_priors, dim=1)  # [B, K, 10, 4]
        else:
            norm_prior = self.compute_norm_prior(ego)  # [B, 10, 4]
            norm_priors = norm_prior[:, None, :, :].expand(b, k, 10, 4)

        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)

        z = torch.randn(b, k, 10, 4, generator=gen, device=device, dtype=history.dtype) * temperature
        if nominal_mode and k > 0:
            z[:, 0] = 0.0
        z = z.reshape(b * k, 10, 4)
        dt = 1.0 / num_steps

        for s in range(num_steps):
            t_val = s * dt
            t_tensor = torch.full((b * k,), t_val, device=device, dtype=history.dtype)
            v = self.forward_vf(z, t_tensor, context_rep)
            z = z + v * dt

        z_out = z.reshape(b, k, 10, 4)
        delta_x = z_out * std[None, None, :, :]

        x_norm_out = norm_priors + delta_x

        if k == 1:
            return x_norm_out[:, 0]
        return x_norm_out


# Alias for BEV-JEPA-X naming contract
KinematicFlowPlanner = TransformerKinematicFlowMatcher
