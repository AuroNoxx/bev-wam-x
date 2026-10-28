"""BEV WA-JEPA Multi-Horizon Latent Flow Matching Model.

Implements:
1. Online BEV Spatiotemporal Encoder (processes history BEVs and ego motion)
2. EMA Target Teacher Encoder (processes future BEVs, updated via EMA tau=0.996)
3. Multi-Horizon Latent Flow Predictor (Transformer predicting future scene latent velocities across +0.5s, +1.0s, +1.5s, +2.0s)
"""

import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from configs.stage3 import Stage3Config
from models.latent import BEVEncoder
from models.flow_matcher import sinusoidal_time_embedding


class TransformerDecoderLayerSDPA(nn.Module):
    """Pre-norm SDPA Transformer Decoder Block with Self-Attention, Cross-Attention, and MLP."""

    def __init__(self, d_model=384, nhead=6, dim_feedforward=1536, dropout=0.0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Self-Attention
        self.norm_self = nn.LayerNorm(d_model)
        self.qkv_self = nn.Linear(d_model, d_model * 3)
        self.proj_self = nn.Linear(d_model, d_model)

        # Cross-Attention
        self.norm_cross_q = nn.LayerNorm(d_model)
        self.norm_cross_kv = nn.LayerNorm(d_model)
        self.q_cross = nn.Linear(d_model, d_model)
        self.kv_cross = nn.Linear(d_model, d_model * 2)
        self.proj_cross = nn.Linear(d_model, d_model)

        # MLP
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, tgt, memory, detach_action_for_scene=False, num_scene_tokens=256):
        """Args:
            tgt: [B, T_tgt, d_model]
            memory: [B, T_mem, d_model]
            detach_action_for_scene: bool (enforces asymmetric gradient contract)
            num_scene_tokens: int (number of leading scene tokens before trajectory tokens)
        """
        b, t_tgt, d = tgt.shape
        _, t_mem, _ = memory.shape

        # 1. Self-Attention with asymmetric gradient isolation
        qkv = self.qkv_self(self.norm_self(tgt)).reshape(b, t_tgt, 3, self.nhead, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [b, nhead, t_tgt, head_dim]

        if detach_action_for_scene and t_tgt > num_scene_tokens:
            q_s = q[:, :, :num_scene_tokens]
            q_t = q[:, :, num_scene_tokens:]
            k_s = k[:, :, :num_scene_tokens]
            k_t = k[:, :, num_scene_tokens:]
            v_s = v[:, :, :num_scene_tokens]
            v_t = v[:, :, num_scene_tokens:]

            # Scene queries attend to scene keys/values and DETACHED action keys/values
            # This completely blocks scene loss gradient from backpropagating into action inputs
            k_for_s = torch.cat([k_s, k_t.detach()], dim=2)
            v_for_s = torch.cat([v_s, v_t.detach()], dim=2)
            attended_s = F.scaled_dot_product_attention(q_s, k_for_s, v_for_s, dropout_p=0.0)

            # Trajectory queries attend to scene and action keys/values WITH FULL GRADIENTS
            # This allows action loss to propagate through the shared interactive representation
            attended_t = F.scaled_dot_product_attention(q_t, k, v, dropout_p=0.0)
            attended_self = torch.cat([attended_s, attended_t], dim=2)
        else:
            attended_self = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        tgt = tgt + self.proj_self(attended_self.transpose(1, 2).reshape(b, t_tgt, d))

        # 2. Cross-Attention
        q = self.q_cross(self.norm_cross_q(tgt)).reshape(b, t_tgt, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_cross(self.norm_cross_kv(memory)).reshape(b, t_mem, 2, self.nhead, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        attended_cross = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        tgt = tgt + self.proj_cross(attended_cross.transpose(1, 2).reshape(b, t_tgt, d))

        # 3. MLP
        tgt = tgt + self.mlp(self.norm_mlp(tgt))
        return tgt


class BEVOnlineEncoder(nn.Module):
    """Online BEV Spatiotemporal Encoder for history observations."""

    def __init__(self, cfg=None, encoder_ckpt=None):
        super().__init__()
        if cfg is None:
            cfg = Stage3Config()
        self.cfg = cfg
        self.bev_encoder = BEVEncoder(cfg)
        self.token_count = self.bev_encoder.token_count  # 64
        self.encoder_dim = cfg.encoder_dim  # 256

        # Temporal position embedding for 4 history frames
        self.temporal_pos = nn.Parameter(torch.randn(1, 4, 1, self.encoder_dim) * 0.02)

        # Ego motion projection: [B, 4, 2] -> [B, 4, 1, 256]
        self.ego_proj = nn.Sequential(
            nn.Linear(2, self.encoder_dim),
            nn.GELU(),
            nn.Linear(self.encoder_dim, self.encoder_dim),
        )

        if encoder_ckpt and Path(encoder_ckpt).exists():
            state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
            encoder_state = {k[8:]: v for k, v in state["model"].items() if k.startswith("encoder.")}
            self.bev_encoder.load_state_dict(encoder_state, strict=True)
            print(f"Loaded initial BEVEncoder weights from {encoder_ckpt}")

    def forward(self, history_bev, ego_motion):
        """Args:

        history_bev: [B, 4, 4, 128, 128] uint8 ego_motion:  [B, 4, 2] float32

        Returns:
            condition_tokens: [B, 260, 256] (256 spatial tokens + 4 ego tokens)
        """
        b = history_bev.shape[0]
        # Encode each history frame: [B*4, 4, 128, 128] -> [B*4, 64, 256] -> [B, 4, 64, 256]
        spatial = self.bev_encoder(history_bev.flatten(0, 1)).reshape(b, 4, self.token_count, self.encoder_dim)
        spatial = spatial + self.temporal_pos  # [B, 4, 64, 256]

        # Ego motion tokens: [B, 4, 1, 256]
        ego_tokens = self.ego_proj(ego_motion).unsqueeze(2)

        # Concatenate spatial and ego tokens along token dimension per frame: [B, 4, 65, 256]
        combined = torch.cat([spatial, ego_tokens], dim=2)
        # Flatten across history time: [B, 4 * 65, 256] = [B, 260, 256]
        return combined.flatten(1, 2)


class EMATargetEncoder(nn.Module):
    """EMA Target Teacher Encoder for future BEV target generation (eval mode, stop-grad)."""

    def __init__(self, online_encoder: BEVOnlineEncoder):
        super().__init__()
        cfg = online_encoder.cfg
        self.bev_encoder = BEVEncoder(cfg)
        self.token_count = self.bev_encoder.token_count
        self.encoder_dim = cfg.encoder_dim

        # Copy initial weights from online encoder
        self.bev_encoder.load_state_dict(online_encoder.bev_encoder.state_dict())
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def update_ema(self, online_encoder: BEVOnlineEncoder, tau: float = 0.996):
        """Update target weights via EMA: theta_ema <- tau * theta_ema + (1 - tau) * theta_online."""
        for p_target, p_online in zip(self.bev_encoder.parameters(), online_encoder.bev_encoder.parameters()):
            p_target.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)

    @torch.no_grad()
    def encode_future(self, future_bev):
        """Args:

        future_bev: [B, 4, 4, 128, 128] uint8 (horizons +0.5s, +1.0s, +1.5s, +2.0s)

        Returns:
            target_latents: [B, 4, 64, 256] float32
        """
        b = future_bev.shape[0]
        tokens = self.bev_encoder(future_bev.flatten(0, 1)).reshape(b, 4, self.token_count, self.encoder_dim)
        return tokens

    @torch.no_grad()
    def encode_current(self, current_bev):
        """Args:

        current_bev: [B, 4, 128, 128] uint8 (current frame t=0)

        Returns:
            current_latents: [B, 64, 256] float32
        """
        return self.bev_encoder(current_bev)


class MultiHorizonLatentFlowPredictor(nn.Module):
    """Transformer-based Multi-Horizon Latent Flow Predictor."""

    def __init__(
        self,
        token_count=64,
        latent_dim=256,
        d_model=384,
        nhead=6,
        nlayers=8,
        dim_feedforward=1536,
    ):
        super().__init__()
        self.token_count = token_count
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers

        # Projections
        self.in_proj = nn.Linear(latent_dim, d_model)
        self.cond_proj = nn.Linear(latent_dim, d_model)

        # Spatial position embedding for the 64 tokens
        self.pos_spatial = nn.Parameter(torch.randn(1, 1, token_count, d_model) * 0.02)
        # Horizon embedding for the 4 future horizons (+0.5s, +1.0s, +1.5s, +2.0s)
        self.pos_horizon = nn.Parameter(torch.randn(1, 4, 1, d_model) * 0.02)

        # Flow time embedding (continuous s in [0, 1])
        self.time_emb_dim = 32
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_emb_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Decoder layers with Pre-norm SDPA
        self.layers = nn.ModuleList([
            TransformerDecoderLayerSDPA(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            ) for _ in range(nlayers)
        ])

        # Output projection back to latent velocity
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, latent_dim)
        # Initialize output layer to zeros for identity start
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        # Preserve the flow-only initialization RNG stream when the branch is disabled.
        with torch.random.fork_rng(devices=[]):
            self.completion_head = nn.Linear(d_model, latent_dim)

    def forward(self, noisy_latents, s, condition_tokens):
        return self.forward_with_features(noisy_latents, s, condition_tokens)[0]

    def forward_with_features(self, noisy_latents, s, condition_tokens):
        """Args:

        noisy_latents:    [B, 4, 64, 256] s:                [B] continuous flow time in [0, 1] condition_tokens: [B, 260, 256]

        Returns:
            v_pred:           [B, 4, 64, 256] predicted velocity field
        """
        b = noisy_latents.shape[0]

        # 1. Project noisy latents: [B, 4, 64, d_model]
        x = self.in_proj(noisy_latents)
        # 2. Add spatial and physical horizon embeddings
        x = x + self.pos_spatial + self.pos_horizon
        # 3. Add flow time embedding: [B, 1, 1, d_model]
        time_emb = sinusoidal_time_embedding(s, self.time_emb_dim)  # [B, 32]
        time_feat = self.time_proj(time_emb)[:, None, None, :]  # [B, 1, 1, d_model]
        x = x + time_feat

        # Flatten future tokens: [B, 4 * 64, d_model] = [B, 256, d_model]
        tgt = x.flatten(1, 2)

        # Project condition tokens: [B, 260, d_model]
        memory = self.cond_proj(condition_tokens)

        # 4. Pass through Transformer Decoder layers
        for layer in self.layers:
            tgt = layer(tgt, memory)

        # 5. Output velocity projection: [B, 256, 256] -> [B, 4, 64, 256]
        features = self.out_norm(tgt)
        out = self.out_proj(features)
        v_pred = out.reshape(b, 4, self.token_count, self.latent_dim)
        return v_pred, features

    @torch.no_grad()
    def solve_euler(self, condition_tokens, num_steps=10, initial_noise=None):
        """Euler ODE integration from s=0 to s=1 to generate future latents.

        Args:
            condition_tokens: [B, 260, 256]
            num_steps:        int, number of ODE discretization steps (e.g. 10)
            initial_noise:    Optional [B, 4, 64, 256] Gaussian noise; if None, samples N(0, I).

        Returns:
            latents_pred:     [B, 4, 64, 256] generated future latents at s=1
        """
        b = condition_tokens.shape[0]
        device = condition_tokens.device
        dtype = condition_tokens.dtype

        if initial_noise is not None:
            z = initial_noise.clone()
        else:
            z = torch.randn(b, 4, self.token_count, self.latent_dim, device=device, dtype=dtype)

        dt = 1.0 / num_steps
        for i in range(num_steps):
            s_val = i * dt
            s_batch = torch.full((b,), s_val, device=device, dtype=torch.float32)
            v = self.forward(z, s_batch, condition_tokens)
            z = z + dt * v

        return z


class BEVWALatentFlowModel(nn.Module):
    """Complete Stage 3 System wrapping Online Encoder, Target Teacher, and Flow Predictor."""

    def __init__(self, cfg=None, encoder_ckpt=None):
        super().__init__()
        if cfg is None:
            cfg = Stage3Config()
        self.cfg = cfg
        if not 0.0 <= cfg.mask_ratio <= 1.0 or cfg.masked_completion_weight < 0:
            raise ValueError("mask_ratio must be in [0, 1] and masked_completion_weight nonnegative")
        if any(k not in (0, 1, 2, 3) for k in cfg.masked_horizons):
            raise ValueError("masked_horizons must contain only indices 0..3")
        self.online_encoder = BEVOnlineEncoder(cfg, encoder_ckpt=encoder_ckpt)
        self.target_encoder = EMATargetEncoder(self.online_encoder)
        self.predictor = MultiHorizonLatentFlowPredictor(
            token_count=64,
            latent_dim=cfg.encoder_dim,
            d_model=384,
            nhead=6,
            nlayers=8,
            dim_feedforward=1536,
        )

    def sample_completion_mask(self, future_bev_mask):
        """Independent Bernoulli patches per sample/horizon; exclude invalid targets."""
        eligible = future_bev_mask.bool().clone()
        if self.cfg.masked_horizons:
            selected = torch.zeros(4, dtype=torch.bool, device=eligible.device)
            selected[list(self.cfg.masked_horizons)] = True
            eligible &= selected[None]
        mask = torch.rand(*eligible.shape, self.predictor.token_count,
                          device=eligible.device) < self.cfg.mask_ratio
        return mask & eligible[..., None], eligible

    def forward_train(self, history_bev, ego_motion, future_bev, future_bev_mask):
        """Training step with Optimal Transport Flow Matching and gradient isolation.

        Returns:
            loss: scalar loss
            loss_dict: dict of loss components
        """
        b = history_bev.shape[0]
        device = history_bev.device

        # 1. Online encoder processes history
        condition_tokens = self.online_encoder(history_bev, ego_motion)  # [B, 260, 256]

        # 2. Target teacher processes future (strictly eval & no_grad)
        with torch.no_grad():
            target_latents = self.target_encoder.encode_future(future_bev)  # [B, 4, 64, 256]
            current_latents = self.target_encoder.encode_current(history_bev[:, -1])  # [B, 64, 256]

        # 2b. Optional residual target: predict the delta from the current latent.
        if self.cfg.residual_target:
            residual_horizons = tuple(self.cfg.residual_horizons) if self.cfg.residual_horizons else (0, 1, 2, 3)
            if len(residual_horizons) == 4:
                flow_target = target_latents - current_latents[:, None]
            else:
                flow_target = target_latents.clone()
                for k in residual_horizons:
                    flow_target[:, k] = target_latents[:, k] - current_latents
        else:
            flow_target = target_latents

        # 3. Flow Matching formulation: s ~ U[0, 1], eps ~ N(0, I)
        s = torch.rand(b, device=device, dtype=torch.float32)
        eps = torch.randn_like(flow_target)

        s_expand = s[:, None, None, None]
        z_s = (1.0 - s_expand) * eps + s_expand * flow_target
        target_velocity = flow_target - eps

        # 4. Predict velocity
        pred_velocity = self.predictor(z_s, s, condition_tokens)

        # 5. Masked Flow Matching loss, with optional dynamic-token weighting.
        mask = future_bev_mask.float()[:, :, None, None]  # [B, 4, 1, 1]
        if self.cfg.dynamic_loss_weight > 0:
            cur_bev = history_bev[:, -1].float() / 255.0
            fut_bev = future_bev.float() / 255.0
            dyn = (fut_bev - cur_bev[:, None]).abs().mean(dim=2)  # [B, 4, 128, 128]
            dyn_tok = F.adaptive_avg_pool2d(dyn.reshape(b * 4, 1, 128, 128), (8, 8)).reshape(b, 4, 64)
            dyn_tok = dyn_tok * future_bev_mask.float()[:, :, None]
            w = 1.0 + self.cfg.dynamic_loss_weight * dyn_tok / (dyn_tok.mean(dim=-1, keepdim=True) + 1e-6)
        else:
            dyn_tok = None
            w = torch.ones(b, 4, 64, device=device, dtype=torch.float32)
        w = w * future_bev_mask.float()[:, :, None]
        diff_sq = (pred_velocity - target_velocity).square() * mask
        weighted = diff_sq * w[..., None]
        norm_factor = torch.clamp(w.sum() * self.cfg.encoder_dim, min=1.0)
        flow_loss = weighted.sum() / norm_factor

        # 6. Token variance regularizer (checks representation health, avoids collapse)
        target_std = torch.sqrt(target_latents.var(dim=0) + 1e-4)  # [4, 64, 256]
        var_penalty = F.relu(1.0 - target_std).mean()

        loss_mask = flow_loss.new_zeros(())
        masked_count = eligible_count = 0
        if self.training and self.cfg.masked_completion_weight > 0:
            completion_mask, eligible = self.sample_completion_mask(future_bev_mask)
            masked_count = int(completion_mask.sum().item())
            eligible_count = int(eligible.sum().item()) * self.predictor.token_count
            if masked_count:
                masked_input = torch.where(completion_mask[..., None],
                                           self.predictor.mask_token, target_latents)
                _, features = self.predictor.forward_with_features(
                    masked_input, torch.ones(b, device=device), condition_tokens)
                completion = self.predictor.completion_head(features).reshape_as(target_latents)
                loss_mask = F.mse_loss(completion[completion_mask].float(),
                                       target_latents[completion_mask].float())

        total_loss = flow_loss + self.cfg.masked_completion_weight * loss_mask + 0.1 * var_penalty

        return total_loss, {
            "flow_loss": float(flow_loss.item()),
            "loss_mask": float(loss_mask.item()),
            "mask_ratio_actual": masked_count / max(eligible_count, 1),
            "masked_token_count": masked_count,
            "eligible_token_count": eligible_count,
            "var_penalty": float(var_penalty.item()),
            "total_loss": float(total_loss.item()),
            "dynamic_token_mean": float(dyn_tok.mean().item()) if dyn_tok is not None else 0.0,
            "residual_target": bool(self.cfg.residual_target),
        }

    def update_teacher(self, tau=0.996):
        self.target_encoder.update_ema(self.online_encoder, tau=tau)
