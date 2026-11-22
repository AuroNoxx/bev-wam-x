"""Surround perception, perception-only recurrent latents, and a command-conditioned planner."""
import contextlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .geometry import CFG, H, W, NAMES, SIZES, HEIGHTS, grid, projection_tables
from .labels import MAX_BOX, MIN_BOX, OBJECT_CLASSES, OBJECT_SLOTS, VECTOR_X

D = 256
LATENT_TOKENS = 64
LIFT_SHAPE = (40, 30)
DECODE_SHAPE = (20, 15)
CLIP_LEN = 4


def sdpa_context(policy):
    if policy == "auto":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if policy == "aotriton":
        if not torch.version.hip:
            raise ValueError("AOTriton requires a ROCm build")
        torch.backends.cuda.preferred_rocm_fa_library("aotriton")
    if policy == "cuda_flash" and (not torch.cuda.is_available() or torch.version.hip):
        raise ValueError("cuda_flash requires a CUDA GPU")
    if policy not in ("math", "aotriton", "cuda_flash"):
        raise ValueError(f"Unknown SDPA backend: {policy}")
    return sdpa_kernel(SDPBackend.MATH if policy == "math" else SDPBackend.FLASH_ATTENTION)


def conv(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         nn.GroupNorm(8, cout), nn.SiLU())


class Residual(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.main = nn.Sequential(conv(cin, cout, stride), nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                                  nn.GroupNorm(8, cout))
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.GroupNorm(8, cout)))

    def forward(self, x):
        return F.silu(self.main(x) + self.skip(x))


class CameraFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                                  nn.GroupNorm(8, 64), nn.SiLU(), nn.MaxPool2d(3, stride=2, padding=1))
        widths, counts = (64, 128, 256, 512), (2, 2, 3, 2)
        self.stages = nn.ModuleList()
        for i, (width, count) in enumerate(zip(widths, counts)):
            self.stages.append(nn.Sequential(Residual(widths[max(0, i - 1)], width, 2 if i else 1),
                                             *[Residual(width, width, 1) for _ in range(count - 1)]))
        self.lateral = nn.ModuleList(nn.Conv2d(width, 64, 1) for width in widths[1:])
        self.out = conv(64, 64)

    def forward(self, x):
        x = self.stem(x)
        levels = []
        for stage in self.stages:
            x = stage(x)
            levels.append(x)
        a, b, c = (layer(x) for layer, x in zip(self.lateral, levels[1:]))
        b = b + F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(a + F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False))


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_norm = nn.LayerNorm(D)
        self.ctx_norm = nn.LayerNorm(D)
        self.q = nn.Linear(D, D)
        self.kv = nn.Linear(D, 2 * D)
        self.proj = nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, query, context=None):
        q = self.q(self.q_norm(query))
        kv = self.kv(self.ctx_norm(query if context is None else context))
        b, nq, _ = q.shape
        q = q.reshape(b, nq, 8, D // 8).transpose(1, 2)
        k, v = kv.reshape(b, -1, 2, 8, D // 8).permute(2, 0, 3, 1, 4).unbind()
        y = F.scaled_dot_product_attention(q, k, v)
        x = query + self.proj(y.transpose(1, 2).reshape(b, nq, D))
        return x + self.mlp(x)


class BEVWAMXPerception(nn.Module):
    """Accept only online sensor inputs and explicit recurrent state.

    rgb is an ordered tuple of six uint8 RGB BCHW tensors. lidar_bev is Bx6x160x120
    or None. cmd is Bx3 [follow, left, right], and reaches only the trajectory head.
    prev_latent is Bx64x256; ego_delta uses v2's previous-relative-to-current convention.
    """

    def __init__(self, lidar_dropout=.15):
        super().__init__()
        if not 0 <= lidar_dropout <= 1:
            raise ValueError("lidar_dropout must be in [0,1]")
        self.lidar_dropout = lidar_dropout
        self.camera = CameraFPN()
        self.lift_reduce = conv(64 * len(HEIGHTS), 128)
        self.lidar = nn.Sequential(conv(6, 48, 2), conv(48, 96, 2))
        self.fuse = nn.Sequential(conv(224, D), conv(D, D))
        self.bev_position = nn.Linear(2, D)
        self.resampler_queries = nn.Parameter(torch.randn(1, LATENT_TOKENS, D) * .02)
        self.delta_embed = nn.Sequential(nn.Linear(3, D), nn.SiLU(), nn.Linear(D, D))
        self.resample = AttentionBlock()
        self.latent_blocks = nn.Sequential(*[AttentionBlock() for _ in range(8)])
        self.latent_norm = nn.LayerNorm(D)

        self.dense_queries = nn.Parameter(torch.randn(1, DECODE_SHAPE[0] * DECODE_SHAPE[1], D) * .02)
        self.dense_attention = AttentionBlock()
        self.dense = nn.Sequential(conv(D, 128), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                   conv(128, 80), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                   conv(80, 48), nn.Conv2d(48, 17, 1))
        self.vector_queries = nn.Parameter(torch.randn(1, 7, D) * .02)
        self.vector_attention = AttentionBlock()
        self.vector = nn.Linear(D, len(VECTOR_X) * 2 + 1)
        self.object_queries = nn.Parameter(torch.randn(1, OBJECT_SLOTS, D) * .02)
        self.object_attention = AttentionBlock()
        self.objects = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 11 + len(OBJECT_CLASSES)))
        self.plan_attention = AttentionBlock()
        self.plan_query = nn.Parameter(torch.randn(1, 1, D) * .02)
        self.cmd_embed = nn.Linear(3, D)
        self.plan = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 75))

        coords, valid = projection_tables(shape=LIFT_SHAPE)
        for i, (width, height) in enumerate(SIZES):
            dimensions = np.ceil(np.array([width, height]) / 8)
            normalized = ((coords[:, i] / 8 + .5) * (2 / dimensions) - 1)
            self.register_buffer(f"uv_{i}", torch.from_numpy(normalized.astype(np.float32)), persistent=False)
        count = np.maximum(valid.sum(axis=1, keepdims=True), 1)
        self.register_buffer("view_weights", torch.from_numpy((valid / count).astype(np.float32)), persistent=False)
        xy = grid(LIFT_SHAPE).reshape(-1, 2) / [80., 30.]
        self.register_buffer("bev_xy", torch.from_numpy(xy.astype(np.float32)), persistent=False)
        self.register_buffer("box_min", torch.tensor(MIN_BOX, dtype=torch.float32), persistent=False)
        self.register_buffer("box_span", torch.tensor(MAX_BOX - MIN_BOX, dtype=torch.float32), persistent=False)
        self.register_buffer("center_scale", torch.tensor([60., 30., 4.]), persistent=False)
        self.register_buffer("center_offset", torch.tensor([20., 0., 1.]), persistent=False)
        self.register_buffer("heading_offset", torch.tensor([0., 1.]), persistent=False)
        self.register_buffer("path_scale", torch.tensor([15., 4.]), persistent=False)

    def lift(self, rgb):
        batch = rgb[0].shape[0]

        def features(images):
            return self.camera((images.float() / 255.).pow(CFG["rgb"]["preprocessing_gamma"]))

        front = features(rgb[0])
        sides = features(torch.cat(rgb[1:], dim=0)).split(batch)
        planes = None
        for i, feature in enumerate((front, *sides)):
            uv = getattr(self, f"uv_{i}").reshape(1, -1, LIFT_SHAPE[1], 2)
            sample = F.grid_sample(feature.float(), uv.expand(batch, -1, -1, -1),
                                   mode="bilinear", padding_mode="zeros", align_corners=False)
            sample = sample.reshape(batch, 64, len(HEIGHTS), *LIFT_SHAPE)
            weight = self.view_weights[:, i]
            contribution = (sample * weight).to(feature.dtype)
            planes = contribution if planes is None else planes + contribution
        return self.lift_reduce(planes.flatten(1, 2))

    def forward(self, rgb, lidar_bev=None, prev_latent=None, ego_delta=None, cmd=None):
        if len(rgb) != len(NAMES):
            raise ValueError("Expected six RGB tensors in config camera order")
        b = rgb[0].shape[0]
        visual = self.lift(rgb)
        if lidar_bev is None:
            lidar = visual.new_zeros(b, 96, *LIFT_SHAPE)
        else:
            lidar = self.lidar(lidar_bev.to(visual.dtype))
            if self.training and self.lidar_dropout > 0:
                retain = torch.rand((b, 1, 1, 1), device=lidar.device) >= self.lidar_dropout
                lidar = lidar * retain
        tokens = self.fuse(torch.cat((visual, lidar), dim=1)).flatten(2).transpose(1, 2)
        tokens = tokens + self.bev_position(self.bev_xy).to(tokens.dtype)
        queries = self.resampler_queries.to(tokens.dtype).expand(b, -1, -1)
        if prev_latent is not None:
            if prev_latent.shape != (b, LATENT_TOKENS, D) or ego_delta is None or ego_delta.shape != (b, 3):
                raise ValueError("prev_latent and ego_delta must have shapes Bx64x256 and Bx3")
            motion = self.delta_embed(ego_delta.to(tokens.dtype)).unsqueeze(1)
            tokens = torch.cat((tokens, prev_latent.to(tokens.dtype) + motion), dim=1)
        z_t = self.latent_norm(self.latent_blocks(self.resample(queries, tokens)))
        spatial = self.dense_attention(self.dense_queries.to(z_t.dtype).expand(b, -1, -1), z_t)
        spatial = spatial.transpose(1, 2).reshape(b, D, *DECODE_SHAPE)
        dense = F.interpolate(self.dense(spatial), size=(H, W), mode="bilinear", align_corners=False)
        vectors = self.vector(self.vector_attention(self.vector_queries.to(z_t.dtype).expand(b, -1, -1), z_t))
        obj = self.objects(self.object_attention(self.object_queries.to(z_t.dtype).expand(b, -1, -1), z_t)).float()
        center = obj[..., :3].tanh() * self.center_scale + self.center_offset
        size = self.box_min + self.box_span * obj[..., 3:6].sigmoid()
        heading = F.normalize(obj[..., 6:8] + self.heading_offset, dim=-1, eps=1e-6)
        boxes = torch.cat((center, size, heading, obj[..., 8:10] * 10), dim=-1)
        plan_query = self.plan_query.to(z_t.dtype).expand(b, -1, -1)
        if cmd is not None:
            if cmd.shape != (b, 3):
                raise ValueError("cmd must be Bx3")
            plan_query = plan_query + self.cmd_embed(cmd.to(z_t.dtype))[:, None]
        path = self.plan(self.plan_attention(plan_query, z_t)).float().reshape(b, 25, 3)
        path = torch.cat((path[..., :2] * self.path_scale,
                          F.softplus(path[..., 2:]) * 10.), dim=-1)
        return dict(z_t=z_t, lane_logits=dense[:, :4], occupancy_logits=dense[:, 4:12],
                    future_occupancy_logits=dense[:, 12:17], vector_exists_logits=vectors[..., -1],
                    vectors=vectors[..., :-1].reshape(b, 7, len(VECTOR_X), 2) * 10.,
                    boxes=boxes, cls_logits=obj[..., 10:-1], obj_logits=obj[..., -1], trajectory=path)
