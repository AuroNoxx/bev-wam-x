"""Action-conditioned latent world model for Stage 3.2 dynamics rollout and risk scoring."""
import torch
from torch import nn
from torch.nn import functional as F

from .geometry import grid
from .perception_model import AttentionBlock, D, LATENT_TOKENS

# [v/30, yaw_rate, steer, throttle, brake] -> conditioning vector
KIN_DIM = 2
ACT_DIM = 3


class AdaLNBlock(nn.Module):
    """Pre-norm self-attention Transformer block with zero-initialized AdaLN conditioning."""

    def __init__(self, dim=D, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        with torch.no_grad():
            # Keep initial modulation residual small while ensuring non-zero step-0 gradient flow.
            self.ada[-1].bias[2 * dim:3 * dim] = 0.1
            self.ada[-1].bias[5 * dim:6 * dim] = 0.1

    def forward(self, x, cond):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada(cond).unsqueeze(1).chunk(6, dim=-1)
        scale_a = torch.tanh(scale_a)
        scale_m = torch.tanh(scale_m)
        gate_a = torch.tanh(gate_a)
        gate_m = torch.tanh(gate_m)
        h = self.norm1(x) * (1. + scale_a) + shift_a
        b, n, c = h.shape
        q, k, v = self.qkv(h).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind()
        q, k = self.q_norm(q), self.k_norm(k)
        attn = F.scaled_dot_product_attention(q, k, v)
        x = x + gate_a * self.proj(attn.transpose(1, 2).reshape(b, n, c))
        m = self.norm2(x) * (1. + scale_m) + shift_m
        return x + gate_m * self.mlp(m)


class ActionConditionedWorldModel(nn.Module):
    """Predict next Stage 3.1b perception latent z_{t+1}, ego kinematics, delta, risk, and THW."""

    def __init__(self, dim=D, layers=6, heads=8):
        super().__init__()
        self.dim = dim
        anchors = grid((16, 8)).reshape(-1, 2) / [80., 30.]
        self.register_buffer("anchor_xy", torch.tensor(anchors, dtype=torch.float32), persistent=False)
        self.pos_embed = nn.Linear(2, dim)
        self.cond_embed = nn.Sequential(
            nn.Linear(KIN_DIM + ACT_DIM, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.cond_token = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([AdaLNBlock(dim=dim, heads=heads) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(dim)
        self.delta_proj = nn.Linear(dim, dim)
        nn.init.normal_(self.delta_proj.weight, std=0.02)
        nn.init.zeros_(self.delta_proj.bias)
        self.next_norm = nn.LayerNorm(dim)

        self.readout_queries = nn.Parameter(torch.randn(1, 4, dim) * .02)
        self.readout_attn = AttentionBlock()
        feat_dim = 5 * dim
        self.readout_mlp = nn.Sequential(nn.Linear(feat_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.kinematics_head = nn.Linear(dim, 4)   # [v/30, yaw_rate, a_lon/10, a_lat/10]
        self.delta_head = nn.Linear(dim, 3)        # [dx, dy, dyaw]
        self.risk_head = nn.Linear(dim, 3)         # [collision, solid_crossing, offroute] logits
        self.inv_thw_head = nn.Linear(dim, 1)      # bounded via sigmoid in (0, 1.8)
        self.nav_head = nn.Linear(dim, 3)          # normalized nav progress state

    def step(self, z_t, kinematics_t, action_t):
        """Advance recurrent perception state z_t by one 0.1 s step under action (steer, throttle, brake)."""
        b, n, c = z_t.shape
        if n != LATENT_TOKENS or c != self.dim:
            raise ValueError(f"Expected (B, {LATENT_TOKENS}, {self.dim}) latent, got {tuple(z_t.shape)}")
        if kinematics_t.shape != (b, KIN_DIM) or action_t.shape != (b, ACT_DIM):
            raise ValueError("Invalid kinematics or action shape")

        if not torch.is_autocast_enabled(z_t.device.type):
            w_dtype = self.delta_proj.weight.dtype
            z_t = z_t.to(w_dtype)
            kinematics_t = kinematics_t.to(w_dtype)
            action_t = action_t.to(w_dtype)
        else:
            kinematics_t = kinematics_t.to(z_t.dtype)
            action_t = action_t.to(z_t.dtype)

        sa = torch.cat((kinematics_t, action_t), dim=-1)
        cond = self.cond_embed(sa)
        pos = self.pos_embed(self.anchor_xy).to(z_t.dtype).unsqueeze(0)
        tokens = z_t + pos + self.cond_token(cond).unsqueeze(1)
        for blk in self.blocks:
            tokens = blk(tokens, cond)

        h = self.out_norm(tokens)
        z_next = self.next_norm(z_t + self.delta_proj(h))

        rq = self.readout_queries.to(z_next.dtype).expand(b, -1, -1)
        pooled = self.readout_attn(rq, z_next).reshape(b, 4 * self.dim)
        feat = self.readout_mlp(torch.cat((pooled, cond.to(pooled.dtype)), dim=-1)).float()
        return dict(
            z_next=z_next,
            next_kinematics=self.kinematics_head(feat),
            next_delta=self.delta_head(feat),
            risk_logits=self.risk_head(feat),
            inv_thw=torch.sigmoid(self.inv_thw_head(feat)) * 1.8,
            nav_progress=self.nav_head(feat),
        )

    def rollout(self, z_0, kinematics_0, actions, *, gt_kinematics=None, teacher_forcing=0.0):
        """Autoregressively roll latent dynamics over actions (B, H, 3)."""
        horizon = actions.shape[1]
        z = z_0
        kin = kinematics_0
        buckets = {k: [] for k in ("z_next", "next_kinematics", "next_delta", "risk_logits", "inv_thw", "nav_progress")}
        for k in range(horizon):
            out = self.step(z, kin, actions[:, k])
            for name in buckets:
                buckets[name].append(out[name])
            z = out["z_next"]
            pred_kin = out["next_kinematics"][:, :KIN_DIM]
            if gt_kinematics is not None and teacher_forcing > 0.0 and self.training:
                use_gt = (torch.rand(z.shape[0], 1, device=z.device) < teacher_forcing).to(pred_kin.dtype)
                kin = use_gt * gt_kinematics[:, k + 1, :KIN_DIM] + (1.0 - use_gt) * pred_kin
            else:
                kin = pred_kin
        return {k: torch.stack(v, dim=1) for k, v in buckets.items()}
