"""Surround perception, perception-only recurrent latents, and a command-conditioned planner."""
import contextlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .geometry import CFG, DX, DY, H, HEIGHTS, NAMES, SIZES, W, X_MAX, Y_MAX, grid, projection_tables
from .labels import MAP_CLASSES, MAP_POINTS, MAP_SLOTS, MAX_BOX, MIN_BOX, OBJECT_CLASSES, OBJECT_SLOTS, VECTOR_X

D = 256
LATENT_TOKENS = 128
LIFT_SHAPE = (80, 60)
DECODE_SHAPE = (80, 60)
FINE_SHAPE = (H, W)
CLIP_LEN = 4
DN_GROUPS = 2
DN_SLOTS = 16


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


class FrozenBatchNorm2d(nn.Module):
    """Pure elementwise affine replacement for BatchNorm2d (ROCm/Windows MIOpen safe + batch-invariant)."""

    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__()
        self.register_buffer("weight", bn.weight.detach().clone())
        self.register_buffer("bias", bn.bias.detach().clone())
        self.register_buffer("running_mean", bn.running_mean.detach().clone())
        self.register_buffer("running_var", bn.running_var.detach().clone())
        self.eps = float(bn.eps)

    def forward(self, x):
        scale = self.weight * torch.rsqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale
        return x * scale.to(x.dtype).reshape(1, -1, 1, 1) + shift.to(x.dtype).reshape(1, -1, 1, 1)


def _freeze_bn(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, FrozenBatchNorm2d(child))
        else:
            _freeze_bn(child)


class PretrainedCameraFPN(nn.Module):
    """ImageNet-pretrained ResNet34 (FrozenBN) + stride-4 top-down FPN outputting 64 channels."""

    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import ResNet34_Weights, resnet34

        backbone = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        _freeze_bn(backbone)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(1, 3, 1, 1), persistent=False)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # stride 4, 64 ch
        self.layer2 = backbone.layer2  # stride 8, 128 ch
        self.layer3 = backbone.layer3  # stride 16, 256 ch
        self.lat1 = nn.Conv2d(64, 64, 1)
        self.lat2 = nn.Conv2d(128, 64, 1)
        self.lat3 = nn.Conv2d(256, 64, 1)
        self.out = conv(64, 64)

    def forward(self, x):
        x = (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        c1 = self.layer1(self.stem(x))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        p3 = self.lat3(c3)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(p1)


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


def _sample_bev_grid(bev_fine, xy_m):
    """Sample `(B, C, H, W)` BEV features at metric FLU `(B, ..., 2)` points using align_corners=False."""
    shape_prefix = xy_m.shape[:-1]
    xy_flat = xy_m.reshape(xy_m.shape[0], 1, -1, 2).float()
    u = 2.0 * ((Y_MAX - xy_flat[..., 1]) / DY) / W - 1.0
    v = 2.0 * ((X_MAX - xy_flat[..., 0]) / DX) / H - 1.0
    uv = torch.stack((u, v), dim=-1)
    sampled = F.grid_sample(bev_fine.float(), uv, mode="bilinear", padding_mode="zeros", align_corners=False)
    # sampled: (B, C, 1, M) -> (B, *shape_prefix[1:], C)
    sampled = sampled.squeeze(2).transpose(1, 2)
    return sampled.reshape(*shape_prefix, bev_fine.shape[1]).to(bev_fine.dtype)


class ObjectDecoderLayer(nn.Module):
    """Joint self-attention, cross-attention to BEV+latent context, and 3x3 fine-BEV local sampling."""

    def __init__(self, fine_ch=128):
        super().__init__()
        self.self_attn = AttentionBlock()
        self.cross_attn = AttentionBlock()
        self.local_norm = nn.LayerNorm(fine_ch * 9)
        self.local_proj = nn.Linear(fine_ch * 9, D)
        self.ffn = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, query, context, bev_fine, ref_xy, local_offsets, *, sa_groups=1, sa_slots=OBJECT_SLOTS):
        b, n_tot, d = query.shape
        if sa_groups > 1:
            # Pack [matching, dn_grp_0, dn_grp_1] along batch dim for FlashAttention isolation without attn_mask.
            q_sa = query.reshape(b * sa_groups, sa_slots, d)
            q_sa = self.self_attn(q_sa).reshape(b, n_tot, d)
        else:
            q_sa = self.self_attn(query)
        q = self.cross_attn(q_sa, context)
        pts = ref_xy[:, :, None, :] + local_offsets[None, None, :, :]
        sampled = _sample_bev_grid(bev_fine, pts).reshape(b, n_tot, -1)
        q = q + self.local_proj(self.local_norm(sampled.to(q.dtype)))
        return q + self.ffn(q)


class MapDecoderLayer(nn.Module):
    """MapTR vector instance decoder with self-attn, BEV cross-attn, and polyline vertex BEV pooling."""

    def __init__(self, fine_ch=128):
        super().__init__()
        self.self_attn = AttentionBlock()
        self.cross_attn = AttentionBlock()
        self.poly_conv = nn.Sequential(nn.Conv1d(fine_ch, fine_ch, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool1d(1))
        self.poly_proj = nn.Linear(fine_ch, D)
        self.pos_proj = nn.Linear(MAP_POINTS * 2, D)
        self.ffn = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, query, context, bev_fine, pts_m, xy_norm_scale):
        b, n_q, _ = query.shape
        q = self.self_attn(query)
        q = self.cross_attn(q, context)
        sampled = _sample_bev_grid(bev_fine, pts_m)  # (B, 24, 20, fine_ch)
        s_conv = self.poly_conv(sampled.reshape(b * n_q, MAP_POINTS, -1).transpose(1, 2).to(query.dtype)).squeeze(-1)
        pt_emb = self.pos_proj((pts_m / xy_norm_scale).reshape(b, n_q, -1).to(query.dtype))
        q = q + self.poly_proj(s_conv.reshape(b, n_q, -1)) + pt_emb
        return q + self.ffn(q)


class BEVWAMXPerception(nn.Module):
    """Accept only online sensor inputs and explicit recurrent state.

    rgb is an ordered tuple of six uint8 RGB BCHW tensors. lidar_bev is Bx6x160x120
    or None. Command Bx3, nav anchors Bx3x3 in FLU, and speed/yaw rate Bx2 in SI
    reach only the trajectory head.
    prev_latent is Bx128x256; ego_delta uses v2's previous-relative-to-current convention.
    """

    def __init__(self, lidar_dropout=.15, *, legacy=False, arch=None, pretrained=False):
        super().__init__()
        if not 0 <= lidar_dropout <= 1:
            raise ValueError("lidar_dropout must be in [0,1]")
        if arch is None:
            arch = "stage31" if legacy else "stage31b"
        if arch not in ("stage31", "stage31b", "stage31c"):
            raise ValueError(f"Unsupported perception architecture: {arch}")
        self.lidar_dropout = lidar_dropout
        self.arch = arch
        self.legacy = arch == "stage31"
        self.shape = (40, 30) if self.legacy else LIFT_SHAPE
        self.num_tokens = 64 if self.legacy else LATENT_TOKENS

        if self.arch == "stage31c":
            self.camera = PretrainedCameraFPN(pretrained=pretrained)
            self.lift_reduce = conv(64 * len(HEIGHTS), 128)
            self.lidar_fine = nn.Sequential(conv(6, 48, 1), conv(48, 48, 1))
            self.lidar = conv(48, 96, 2)
            self.fuse = nn.Sequential(conv(224, D), conv(D, D))
            self.lidar_skip = nn.Conv2d(48, 128, 1)
            self.bev_up = nn.Sequential(nn.Conv2d(D, 128, 1),
                                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.bev_fine_conv = conv(128, 128)
        else:
            self.camera = CameraFPN()
            self.lift_reduce = conv(64 * len(HEIGHTS), 128)
            self.lidar = nn.Sequential(conv(6, 48, 2), conv(48, 96, 2 if self.legacy else 1))
            self.fuse = nn.Sequential(conv(224, D), conv(D, D))

        self.bev_position = nn.Linear(2, D)
        self.resampler_queries = nn.Parameter(torch.randn(1, self.num_tokens, D) * .02)
        self.delta_embed = nn.Sequential(nn.Linear(3, D), nn.SiLU(), nn.Linear(D, D))
        self.resample = AttentionBlock()
        self.latent_blocks = nn.Sequential(*[AttentionBlock() for _ in range(8)])
        self.latent_norm = nn.LayerNorm(D)

        if self.legacy:
            self.dense_queries = nn.Parameter(torch.randn(1, 1200, D) * .02)
            self.dense_attention = AttentionBlock()
            self.dense = nn.Sequential(conv(D, 128), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                       conv(128, 80), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                       conv(80, 48), nn.Conv2d(48, 17, 1))
        elif self.arch == "stage31b":
            anchors = grid((16, 8)).reshape(-1, 2) / [80., 30.]
            self.register_buffer("anchor_xy", torch.tensor(anchors, dtype=torch.float32), persistent=False)
            self.dense = nn.Sequential(nn.Conv2d(D, 64, 1), conv(64, 64),
                                       nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                       conv(64, 32), nn.Conv2d(32, 17, 1))
        else:
            anchors = grid((16, 8)).reshape(-1, 2) / [80., 30.]
            self.register_buffer("anchor_xy", torch.tensor(anchors, dtype=torch.float32), persistent=False)
            self.dense = nn.Sequential(conv(128, 64), nn.Conv2d(64, 17, 1))
            hm_last = nn.Conv2d(64, len(OBJECT_CLASSES), 1)
            nn.init.constant_(hm_last.bias, -4.6)
            self.center_head = nn.Sequential(conv(128, 64), hm_last)

        self.vector_queries = nn.Parameter(torch.randn(1, 7, D) * .02)
        self.vector_attention = AttentionBlock()
        self.vector = nn.Linear(D, len(VECTOR_X) * 2 + 1)

        self.object_queries = nn.Parameter(torch.randn(1, OBJECT_SLOTS, D) * .02)
        if self.arch == "stage31c":
            self.prop_proj = nn.Linear(128, D)
            self.xy_embed = nn.Sequential(nn.Linear(2, D), nn.SiLU(), nn.Linear(D, D))
            self.dn_box_embed = nn.Sequential(nn.Linear(10 + len(OBJECT_CLASSES), D), nn.SiLU(), nn.Linear(D, D))
            self.obj_layers = nn.ModuleList([ObjectDecoderLayer(128) for _ in range(3)])
            self.obj_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 11 + len(OBJECT_CLASSES)))
                for _ in range(3)
            ])
            for head in self.obj_heads:
                nn.init.constant_(head[-1].bias[-1], -2.19)

            self.map_queries = nn.Parameter(torch.randn(1, MAP_SLOTS, D) * .02)
            self.map_init_pts = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, MAP_POINTS * 2))
            self.map_layers = nn.ModuleList([MapDecoderLayer(128) for _ in range(3)])
            self.map_pt_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, MAP_POINTS * 2)) for _ in range(3)
            ])
            self.map_cls_heads = nn.ModuleList([nn.Linear(D, len(MAP_CLASSES)) for _ in range(3)])
            self.map_obj_heads = nn.ModuleList([nn.Linear(D, 1) for _ in range(3)])
            for oh in self.map_obj_heads:
                nn.init.constant_(oh.bias, -2.19)
        else:
            self.object_attention = AttentionBlock()
            self.objects = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 11 + len(OBJECT_CLASSES)))

        self.plan_attention = AttentionBlock()
        self.plan_query = nn.Parameter(torch.randn(1, 1, D) * .02)
        self.cmd_embed = nn.Linear(3, D)
        self.nav_embed = nn.Linear(9, D)
        self.speed_embed = nn.Linear(2, D)
        self.plan = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 75))

        coords, valid = projection_tables(shape=self.shape)
        feat_stride = 4 if self.arch == "stage31c" else 8
        for i, (width, height) in enumerate(SIZES):
            dimensions = np.ceil(np.array([width, height]) / feat_stride)
            normalized = ((coords[:, i] / feat_stride + .5) * (2 / dimensions) - 1)
            self.register_buffer(f"uv_{i}", torch.from_numpy(normalized.astype(np.float32)), persistent=False)
        count = np.maximum(valid.sum(axis=1, keepdims=True), 1)
        self.register_buffer("view_weights", torch.from_numpy((valid / count).astype(np.float32)), persistent=False)
        xy = grid(self.shape).reshape(-1, 2) / [80., 30.]
        self.register_buffer("bev_xy", torch.from_numpy(xy.astype(np.float32)), persistent=False)
        self.register_buffer("fine_xy", torch.from_numpy(grid(FINE_SHAPE).reshape(-1, 2).astype(np.float32)), persistent=False)
        offsets = np.array([(ox * DX, oy * DY) for ox in (-1.0, 0.0, 1.0) for oy in (-1.0, 0.0, 1.0)], dtype=np.float32)
        self.register_buffer("local_offsets", torch.from_numpy(offsets), persistent=False)
        self.register_buffer("xy_norm_scale", torch.tensor([80., 30.], dtype=torch.float32), persistent=False)
        self.register_buffer("obj_xy_step", torch.tensor([6., 4.], dtype=torch.float32), persistent=False)
        self.register_buffer("box_min", torch.tensor(MIN_BOX, dtype=torch.float32), persistent=False)
        self.register_buffer("box_span", torch.tensor(MAX_BOX - MIN_BOX, dtype=torch.float32), persistent=False)
        self.register_buffer("center_scale", torch.tensor([60., 30., 4.]), persistent=False)
        self.register_buffer("center_offset", torch.tensor([20., 0., 1.]), persistent=False)
        self.register_buffer("heading_offset", torch.tensor([0., 1.]), persistent=False)
        self.register_buffer("path_scale", torch.tensor([15., 4.]), persistent=False)
        self.register_buffer("nav_scale", torch.tensor([36., 12., 6.]), persistent=False)
        self.register_buffer("speed_scale", torch.tensor([30., 1.]), persistent=False)

    def lift(self, rgb):
        batch = rgb[0].shape[0]

        def features(images):
            return self.camera((images.float() / 255.).pow(CFG["rgb"]["preprocessing_gamma"]))

        front = features(rgb[0])
        sides = features(torch.cat(rgb[1:], dim=0)).split(batch)
        planes = None
        for i, feature in enumerate((front, *sides)):
            uv = getattr(self, f"uv_{i}").reshape(1, -1, self.shape[1], 2)
            sample = F.grid_sample(feature.float(), uv.expand(batch, -1, -1, -1),
                                   mode="bilinear", padding_mode="zeros", align_corners=False)
            sample = sample.reshape(batch, 64, len(HEIGHTS), *self.shape)
            weight = self.view_weights[:, i]
            contribution = (sample * weight).to(feature.dtype)
            planes = contribution if planes is None else planes + contribution
        return self.lift_reduce(planes.flatten(1, 2))

    def forward(self, rgb, lidar_bev=None, prev_latent=None, ego_delta=None, cmd=None,
                nav_anchors=None, ego_speed=None, gt_boxes=None, *, return_bev_fine=False):
        if len(rgb) != len(NAMES):
            raise ValueError("Expected six RGB tensors in config camera order")
        b = rgb[0].shape[0]
        visual = self.lift(rgb)
        lidar_fine = None
        if lidar_bev is None:
            lidar = visual.new_zeros(b, 96, *self.shape)
            if self.arch == "stage31c":
                lidar_fine = visual.new_zeros(b, 48, *FINE_SHAPE)
        elif self.arch == "stage31c":
            lidar_fine = self.lidar_fine(lidar_bev.to(visual.dtype))
            lidar = self.lidar(lidar_fine)
            if self.training and self.lidar_dropout > 0:
                retain = (torch.rand((b, 1, 1, 1), device=lidar.device) >= self.lidar_dropout).to(visual.dtype)
                lidar = lidar * retain
                lidar_fine = lidar_fine * retain
        else:
            lidar = self.lidar(lidar_bev.to(visual.dtype))
            if self.training and self.lidar_dropout > 0:
                retain = torch.rand((b, 1, 1, 1), device=lidar.device) >= self.lidar_dropout
                lidar = lidar * retain

        bev = self.fuse(torch.cat((visual, lidar), dim=1))
        tokens = bev.flatten(2).transpose(1, 2)
        tokens = tokens + self.bev_position(self.bev_xy).to(tokens.dtype)
        queries = self.resampler_queries.to(tokens.dtype).expand(b, -1, -1)
        if not self.legacy:
            seed = F.adaptive_avg_pool2d(bev, (16, 8)).flatten(2).transpose(1, 2)
            queries = queries + seed + self.bev_position(self.anchor_xy).to(queries.dtype)
        if prev_latent is not None:
            if prev_latent.shape != (b, self.num_tokens, D) or ego_delta is None or ego_delta.shape != (b, 3):
                raise ValueError(f"Expected Bx{self.num_tokens}x{D} latent and Bx3 ego delta")
            motion = self.delta_embed(ego_delta.to(tokens.dtype)).unsqueeze(1)
            tokens = torch.cat((tokens, prev_latent.to(tokens.dtype) + motion), dim=1)
        z_t = self.latent_norm(self.latent_blocks(self.resample(queries, tokens)))

        if self.arch == "stage31c":
            decoded = self.decode_perception(z_t, bev, lidar_fine=lidar_fine, gt_boxes=gt_boxes,
                                             return_bev_fine=return_bev_fine)
        else:
            decoded = self.decode_perception(z_t, None if self.legacy else bev)
        return dict(z_t=z_t, **decoded, trajectory=self.plan_trajectory(z_t, cmd, nav_anchors, ego_speed))

    def _build_dn_queries(self, gt_boxes, dtype):
        """Construct positive and hard-negative DN-DETR denoising groups safely on GPU."""
        b = gt_boxes.shape[0]
        base = gt_boxes[:, :DN_SLOTS].float()
        base_valid = base[..., 10] > 0.5
        cls_idx = base[..., 9].long().clamp(0, len(OBJECT_CLASSES) - 1)
        cls_oh = F.one_hot(cls_idx, len(OBJECT_CLASSES)).float()

        dn_q_list, dn_ref_list = [], []
        pos_targets_list, pos_cls_list, pos_valid_list, neg_valid_list = [], [], [], []
        for _ in range(DN_GROUPS):
            p_xy = base[..., :2] + (torch.rand_like(base[..., :2]) * 2.0 - 1.0) * 0.5
            p_sz = (base[..., 3:6] * torch.empty_like(base[..., 3:6]).uniform_(0.85, 1.15)).clamp(min=0.3)
            p_yaw = base[..., 6] + (torch.rand_like(base[..., 6]) * 2.0 - 1.0) * 0.12
            p_vec = torch.cat((
                p_xy / self.xy_norm_scale, base[..., 2:3] / 4.0, p_sz / 10.0,
                p_yaw.sin()[..., None], p_yaw.cos()[..., None], base[..., 7:9] / 20.0,
            ), dim=-1)
            p_emb = self.dn_box_embed(torch.cat((p_vec, cls_oh), dim=-1).to(dtype))

            theta = torch.rand_like(base[..., 0]) * (2.0 * math.pi)
            rad = torch.empty_like(base[..., 0]).uniform_(1.8, 4.0)
            n_xy = base[..., :2] + torch.stack((theta.cos() * rad, theta.sin() * rad), dim=-1)
            n_sz = (base[..., 3:6] * torch.empty_like(base[..., 3:6]).uniform_(0.85, 1.15)).clamp(min=0.3)
            n_vec = torch.cat((
                n_xy / self.xy_norm_scale, base[..., 2:3] / 4.0, n_sz / 10.0,
                base[..., 6].sin()[..., None], base[..., 6].cos()[..., None], base[..., 7:9] / 20.0,
            ), dim=-1)
            n_emb = self.dn_box_embed(torch.cat((n_vec, cls_oh), dim=-1).to(dtype))

            v_mask = base_valid.unsqueeze(-1)
            p_emb = p_emb * v_mask.to(dtype)
            n_emb = n_emb * v_mask.to(dtype)
            pad_xy = base.new_full((b, DN_SLOTS, 2), -100.0)
            p_xy_masked = torch.where(v_mask, p_xy, pad_xy)
            n_xy_masked = torch.where(v_mask, n_xy, pad_xy)

            dn_q_list.append(torch.cat((p_emb, n_emb), dim=1))
            dn_ref_list.append(torch.cat((p_xy_masked, n_xy_masked), dim=1))
            tgt_enc = torch.cat((
                base[..., :6], base[..., 6:7].sin(), base[..., 6:7].cos(), base[..., 7:9],
            ), dim=-1)
            pos_targets_list.append(tgt_enc)
            pos_cls_list.append(cls_idx)
            pos_valid_list.append(base_valid)
            neg_valid_list.append(base_valid)

        dn_meta = dict(
            pos_targets=torch.cat(pos_targets_list, dim=1),  # (B, G*16, 10)
            pos_cls=torch.cat(pos_cls_list, dim=1),          # (B, G*16)
            pos_valid=torch.cat(pos_valid_list, dim=1),      # (B, G*16)
            neg_valid=torch.cat(neg_valid_list, dim=1),      # (B, G*16)
        )
        return torch.cat(dn_q_list, dim=1), torch.cat(dn_ref_list, dim=1), dn_meta

    def _decode_box_head(self, head_raw, ref_xy):
        center_xy = ref_xy + head_raw[..., :2].tanh() * self.obj_xy_step
        center_z = head_raw[..., 2:3].tanh() * 4.0 + 1.0
        center = torch.cat((center_xy, center_z), dim=-1)
        size = self.box_min + self.box_span * head_raw[..., 3:6].sigmoid()
        heading = F.normalize(head_raw[..., 6:8] + self.heading_offset, dim=-1, eps=1e-6)
        boxes = torch.cat((center, size, heading, head_raw[..., 8:10] * 10.0), dim=-1)
        return boxes, head_raw[..., 10:-1], head_raw[..., -1]

    def decode_perception(self, z_t, bev=None, lidar_fine=None, gt_boxes=None, *, return_bev_fine=False):
        """Decode latent alone for rollout, or fuse the observed BEV for online perception."""
        b = z_t.shape[0]
        context = z_t
        if self.legacy:
            spatial = self.dense_attention(self.dense_queries.to(z_t.dtype).expand(b, -1, -1), z_t)
            spatial = spatial.transpose(1, 2).reshape(b, D, *self.shape)
        elif self.arch == "stage31b":
            spatial = F.interpolate(z_t.transpose(1, 2).reshape(b, D, 16, 8), size=self.shape,
                                    mode="bilinear", align_corners=False)
            if bev is not None:
                if self.training:
                    bev = bev * (torch.rand(b, 1, 1, 1, device=bev.device) >= .2)
                spatial = spatial + bev
                local_keys = F.avg_pool2d(bev, 2).flatten(2).transpose(1, 2)
                pos = F.avg_pool2d(self.bev_xy.T.reshape(1, 2, *self.shape), 2).flatten(2).transpose(1, 2)
                context = torch.cat((z_t, local_keys + self.bev_position(pos).to(z_t.dtype)), dim=1)
        else:
            spatial = F.interpolate(z_t.transpose(1, 2).reshape(b, D, 16, 8), size=self.shape,
                                    mode="bilinear", align_corners=False)
            if bev is not None:
                spatial = spatial + bev
                local_keys = F.avg_pool2d(bev, 2).flatten(2).transpose(1, 2)
                pos = F.avg_pool2d(self.bev_xy.T.reshape(1, 2, *self.shape), 2).flatten(2).transpose(1, 2)
                context = torch.cat((z_t, local_keys + self.bev_position(pos).to(z_t.dtype)), dim=1)
            bev_up = self.bev_up(spatial)
            if lidar_fine is not None:
                bev_up = bev_up + self.lidar_skip(lidar_fine)
            bev_fine = self.bev_fine_conv(bev_up)

            dense = self.dense(bev_fine)
            center_hm = self.center_head(bev_fine).float()
            vectors = self.vector(self.vector_attention(self.vector_queries.to(z_t.dtype).expand(b, -1, -1), z_t))

            with torch.no_grad():
                hm_probs = center_hm.detach().sigmoid().max(dim=1, keepdim=True).values
                peaks = hm_probs == F.max_pool2d(hm_probs, kernel_size=3, stride=1, padding=1)
                scores_flat = (hm_probs * peaks.to(hm_probs.dtype)).flatten(1)
                topk_idx = scores_flat.topk(OBJECT_SLOTS, dim=1).indices
                ref_xy = self.fine_xy[topk_idx]  # (B, 32, 2)
            prop_feat = torch.gather(bev_fine.flatten(2).transpose(1, 2), 1, topk_idx.unsqueeze(-1).expand(-1, -1, 128))
            q_match = (
                self.prop_proj(prop_feat)
                + self.xy_embed((ref_xy / self.xy_norm_scale).to(z_t.dtype))
                + self.object_queries.to(z_t.dtype).expand(b, -1, -1)
            )

            use_dn = bool(self.training and gt_boxes is not None)
            if use_dn:
                dn_q, dn_ref, dn_meta = self._build_dn_queries(gt_boxes, z_t.dtype)
                q_obj = torch.cat((q_match, dn_q), dim=1)
                ref_all = torch.cat((ref_xy, dn_ref), dim=1)
                sa_groups = 1 + DN_GROUPS
            else:
                q_obj, ref_all, dn_meta = q_match, ref_xy, None
                sa_groups = 1

            aux_objects, dn_layers = [], []
            for layer, head in zip(self.obj_layers, self.obj_heads):
                q_obj = layer(q_obj, context, bev_fine, ref_all, self.local_offsets,
                              sa_groups=sa_groups, sa_slots=OBJECT_SLOTS)
                raw = head(q_obj).float()
                boxes_l, cls_l, obj_l = self._decode_box_head(raw, ref_all)
                ref_all = boxes_l[..., :2].detach()
                aux_objects.append(dict(
                    boxes=boxes_l[:, :OBJECT_SLOTS],
                    cls_logits=cls_l[:, :OBJECT_SLOTS],
                    obj_logits=obj_l[:, :OBJECT_SLOTS],
                ))
                if use_dn:
                    dn_b = boxes_l[:, OBJECT_SLOTS:].reshape(b, DN_GROUPS, 2 * DN_SLOTS, -1)
                    dn_c = cls_l[:, OBJECT_SLOTS:].reshape(b, DN_GROUPS, 2 * DN_SLOTS, -1)
                    dn_o = obj_l[:, OBJECT_SLOTS:].reshape(b, DN_GROUPS, 2 * DN_SLOTS)
                    dn_layers.append(dict(
                        pos_boxes=dn_b[:, :, :DN_SLOTS].reshape(b, DN_GROUPS * DN_SLOTS, -1),
                        pos_cls_logits=dn_c[:, :, :DN_SLOTS].reshape(b, DN_GROUPS * DN_SLOTS, -1),
                        pos_obj_logits=dn_o[:, :, :DN_SLOTS].reshape(b, DN_GROUPS * DN_SLOTS),
                        neg_obj_logits=dn_o[:, :, DN_SLOTS:].reshape(b, DN_GROUPS * DN_SLOTS),
                    ))

            q_map = self.map_queries.to(z_t.dtype).expand(b, -1, -1)
            pts_logit = self.map_init_pts(q_map).float().reshape(b, MAP_SLOTS, MAP_POINTS, 2)
            xy_scale, xy_offset = self.center_scale[:2], self.center_offset[:2]
            pts_m = pts_logit.tanh() * xy_scale + xy_offset
            aux_map = []
            for m_layer, pt_head, cls_head, obj_head in zip(
                self.map_layers, self.map_pt_heads, self.map_cls_heads, self.map_obj_heads
            ):
                q_map = m_layer(q_map, context, bev_fine, pts_m, self.xy_norm_scale)
                pts_logit = pts_logit + pt_head(q_map).float().reshape_as(pts_logit)
                pts_m = pts_logit.tanh() * xy_scale + xy_offset
                aux_map.append(dict(
                    map_polylines=pts_m,
                    map_cls_logits=cls_head(q_map).float(),
                    map_obj_logits=obj_head(q_map).float().squeeze(-1),
                ))

            out = dict(
                lane_logits=dense[:, :4],
                occupancy_logits=dense[:, 4:12],
                future_occupancy_logits=dense[:, 12:17],
                vector_exists_logits=vectors[..., -1],
                vectors=vectors[..., :-1].reshape(b, 7, len(VECTOR_X), 2) * 10.,
                center_heatmap_logits=center_hm,
                boxes=aux_objects[-1]["boxes"],
                cls_logits=aux_objects[-1]["cls_logits"],
                obj_logits=aux_objects[-1]["obj_logits"],
                aux_objects=aux_objects[:-1],
                map_polylines=aux_map[-1]["map_polylines"],
                map_cls_logits=aux_map[-1]["map_cls_logits"],
                map_obj_logits=aux_map[-1]["map_obj_logits"],
                aux_map=aux_map[:-1],
            )
            if use_dn:
                out["dn_outputs"] = dict(layers=dn_layers, meta=dn_meta)
            if return_bev_fine:
                out["bev_fine"] = bev_fine
            return out

        dense = self.dense(spatial)
        vectors = self.vector(self.vector_attention(self.vector_queries.to(z_t.dtype).expand(b, -1, -1), z_t))
        obj = self.objects(self.object_attention(self.object_queries.to(z_t.dtype).expand(b, -1, -1), context)).float()
        center = obj[..., :3].tanh() * self.center_scale + self.center_offset
        size = self.box_min + self.box_span * obj[..., 3:6].sigmoid()
        heading = F.normalize(obj[..., 6:8] + self.heading_offset, dim=-1, eps=1e-6)
        boxes = torch.cat((center, size, heading, obj[..., 8:10] * 10), dim=-1)
        return dict(lane_logits=dense[:, :4], occupancy_logits=dense[:, 4:12],
                    future_occupancy_logits=dense[:, 12:17], vector_exists_logits=vectors[..., -1],
                    vectors=vectors[..., :-1].reshape(b, 7, len(VECTOR_X), 2) * 10.,
                    boxes=boxes, cls_logits=obj[..., 10:-1], obj_logits=obj[..., -1])

    def plan_trajectory(self, z_t, cmd=None, nav_anchors=None, ego_speed=None):
        b = z_t.shape[0]
        plan_query = self.plan_query.to(z_t.dtype).expand(b, -1, -1)
        if cmd is not None:
            if cmd.shape != (b, 3):
                raise ValueError("cmd must be Bx3")
            plan_query = plan_query + self.cmd_embed(cmd.to(z_t.dtype))[:, None]
        if nav_anchors is not None:
            if nav_anchors.shape != (b, 3, 3):
                raise ValueError("nav_anchors must be Bx3x3")
            nav = (nav_anchors / self.nav_scale).reshape(b, 9).to(z_t.dtype)
            plan_query = plan_query + self.nav_embed(nav)[:, None]
        if ego_speed is not None:
            if ego_speed.shape != (b, 2):
                raise ValueError("ego_speed must be Bx2")
            motion = (ego_speed / self.speed_scale).to(z_t.dtype)
            plan_query = plan_query + self.speed_embed(motion)[:, None]
        path = self.plan(self.plan_attention(plan_query, z_t)).float().reshape(b, 25, 3)
        return torch.cat((path[..., :2] * self.path_scale, F.softplus(path[..., 2:]) * 10.), dim=-1)
