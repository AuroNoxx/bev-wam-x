"""Three-camera and LiDAR BEV with semantic map queries and a spatial planning query."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from v1_dual_system.fast_model import NativeBatchNorm
from .geometry import CAMS, H, W, X, Y, local, projection_tables


GAMMA = 0.5
NATIVE_C = ((768,448),(384,224),(384,224))
ROUTE_INSTRUCTIONS = ('follow', 'turn_left', 'turn_right', 'exit', 'enter_highway', 'enter_roundabout')
_ROUTE_INDEX = {name: i for i, name in enumerate(ROUTE_INSTRUCTIONS)}


def route_inst_vector(name):
    vec = np.zeros(len(ROUTE_INSTRUCTIONS), dtype=np.float32)
    vec[_ROUTE_INDEX.get(name, 0)] = 1.0
    return vec


def relative_ego_delta(pose_prev, pose_cur):
    dx, dy, _ = local(pose_prev[:3, 3], pose_cur)
    yaw_prev = math.atan2(float(pose_prev[1, 0]), float(pose_prev[0, 0]))
    yaw_cur = math.atan2(float(pose_cur[1, 0]), float(pose_cur[0, 0]))
    dyaw = (yaw_prev - yaw_cur + math.pi) % (2.0 * math.pi) - math.pi
    return np.asarray((dx, dy, dyaw), dtype=np.float32)


def encode_nav_rich(nav, cmd, route_inst=None, route_dist=None):
    batch = nav.shape[0]
    dtype = nav.dtype
    if route_inst is None:
        inst = nav.new_zeros((batch, len(ROUTE_INSTRUCTIONS)))
        inst[:, 0] = 1.0
    else:
        inst = route_inst.to(dtype=dtype)
    if route_dist is None:
        dist = nav.new_full((batch,), -1.0)
    else:
        dist = route_dist.to(dtype=dtype).reshape(batch)
    active = (dist >= 0.0).to(dtype)
    d_clip = dist.clamp(0.0, 450.0)
    far = active * (1.0 - d_clip / 450.0)
    near = active * (1.0 - dist.clamp(0.0, 150.0) / 150.0)
    imminent = active * torch.exp(-d_clip / 60.0)
    extra = torch.cat((inst, torch.stack((active, far, near, imminent), dim=1)), dim=1)
    return torch.cat((nav.flatten(1) / 60.0, cmd.to(dtype), extra), dim=1)


def mounts():
    transforms = []
    for x, y, z, yaw, pitch, _ in CAMS:
        yaw, pitch = np.deg2rad([yaw, pitch])
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        transforms.append([[cy*cp, -sy, -cy*sp, x], [sy*cp, cy, -sy*sp, y],
                           [sp, 0, cp, z], [0, 0, 0, 1]])
    return np.asarray(transforms)


def configure_sdpa(backend):
    if backend == 'auto':
        if torch.version.hip and torch.cuda.is_available():
            arch = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
            backend = 'aotriton' if arch == 'gfx1100' else 'math'
        elif torch.cuda.is_available():
            backend = 'cuda_flash'
        else:
            backend = 'math'
    if backend == 'aotriton':
        if not torch.version.hip or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx1100':
            raise RuntimeError('Phase 2 requires the verified gfx1100 ROCm device')
        library = torch.backends.cuda.preferred_rocm_fa_library('aotriton')
        if not str(library).endswith('AOTriton'):
            raise RuntimeError(f'Unexpected FlashAttention library: {library}')
    elif backend == 'cuda_flash':
        if torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError('cuda_flash requires a non-HIP CUDA device')
    elif backend != 'math':
        raise ValueError(f'Unsupported SDPA backend: {backend}')
    torch.backends.cuda.enable_flash_sdp(backend in ('aotriton', 'cuda_flash'))
    torch.backends.cuda.enable_math_sdp(backend in ('math', 'cuda_flash'))
    torch.backends.cuda.enable_mem_efficient_sdp(backend == 'cuda_flash')
    torch.backends.cuda.enable_cudnn_sdp(False)
    return backend


class ImagePyramid(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
                          norm_layer=NativeBatchNorm)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.stages = nn.ModuleList((resnet.layer2, resnet.layer3, resnet.layer4))
        self.laterals = nn.ModuleList(nn.Conv2d(c, 64, 1) for c in (128, 256, 512))
        self.out = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False), NativeBatchNorm(64), nn.SiLU())

    def forward(self, x):
        x = self.stem(x)
        lo = self.stages[0](x)
        mid = self.stages[1](lo)
        hi = self.stages[2](mid)
        mid = self.laterals[1](mid) + F.interpolate(self.laterals[2](hi), size=mid.shape[-2:], mode='nearest')
        return self.out(self.laterals[0](lo) + F.interpolate(mid, size=lo.shape[-2:], mode='nearest'))


class Lift(nn.Module):
    def __init__(self):
        super().__init__()
        uv, seen = projection_tables(mounts())
        uv = (uv + .5) * np.array([2/W, 2/H], dtype=np.float32) - 1
        grid = np.ascontiguousarray(uv.transpose(1, 0, 2, 3, 4).reshape(3, 640, 128, 2))
        seen = np.ascontiguousarray(seen.transpose(1, 0, 2, 3).reshape(3, 1, 640, 128))
        self.register_buffer('grid', torch.from_numpy(grid), persistent=False)
        self.register_buffer('visible', torch.from_numpy(seen), persistent=False)
        self.register_buffer('camera_count', self.visible.sum(0).clamp_min(1).float(), persistent=False)
        self.mix = nn.Sequential(nn.Conv2d(320, 64, 1, bias=False), NativeBatchNorm(64), nn.SiLU())

    def forward(self, features):
        batch = features.shape[0] // 3
        grid = self.grid.expand(batch, -1, -1, -1, -1).reshape(batch*3, 640, 128, 2)
        sampled = F.grid_sample(features, grid, align_corners=False).reshape(batch, 3, 64, 640, 128)
        total = (sampled * self.visible).sum(1) / self.camera_count
        return self.mix(total.reshape(batch, 320, 128, 128))


class NativeProjection(nn.Module):
    """Batch cameras of equal resolution, then fuse all observations in BEV."""
    def __init__(self, sizes):
        super().__init__()
        uv, visible = projection_tables(mounts(), sizes=sizes)
        self.groups = tuple(tuple(i for i,s in enumerate(sizes) if s==size) for size in dict.fromkeys(sizes))
        for j, indices in enumerate(self.groups):
            width,height=sizes[indices[0]]
            grid=(uv[:,indices]+.5)*np.array([2/width,2/height],np.float32)-1
            grid=grid.transpose(1,0,2,3,4).reshape(len(indices),640,128,2)
            mask=visible[:,indices].transpose(1,0,2,3).reshape(len(indices),1,640,128)
            self.register_buffer(f'grid{j}',torch.from_numpy(grid.copy()),persistent=False)
            self.register_buffer(f'mask{j}',torch.from_numpy(mask.copy()),persistent=False)
        count=visible.sum(1).clip(1).reshape(1,1,640,128).astype(np.float32)
        self.register_buffer('count',torch.from_numpy(count),persistent=False)

    def forward(self, pyramid, mix, views, gain=None, contrast=None):
        batch=views[0].shape[0]
        lifted=[]
        for j, indices in enumerate(self.groups):
            rgb=torch.stack([views[i] for i in indices],1)
            rgb=rgb_preprocess(rgb,gain,contrast).flatten(0,1).contiguous(memory_format=torch.channels_last)
            features=pyramid(rgb)
            grid=getattr(self,f'grid{j}').expand(batch,-1,-1,-1,-1).reshape(-1,640,128,2)
            sampled=F.grid_sample(features,grid,align_corners=False).reshape(batch,len(indices),64,640,128)
            lifted.append((sampled*getattr(self,f'mask{j}')).sum(1))
        return mix((sum(lifted)/self.count).reshape(batch,320,128,128))


def conv(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         NativeBatchNorm(cout), nn.SiLU())


def up(cin, skip, cout):
    return nn.Sequential(nn.Conv2d(cin + skip, cout, 1, bias=False), NativeBatchNorm(cout), nn.SiLU(), conv(cout, cout))


def spatial_encoding():
    t = torch.arange(16, dtype=torch.float32)
    scale = torch.exp(-math.log(10000) * torch.arange(64, dtype=torch.float32) / 64)
    phase = t[:, None]*scale
    axis = torch.stack((phase.sin(), phase.cos()), -1).reshape(16, 128)
    return torch.cat((axis[:, None].expand(-1, 16, -1), axis[None].expand(16, -1, -1)), -1).reshape(1, 256, 256)


class SensorBEV(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc0 = nn.Sequential(conv(70, 64), conv(64, 64))
        self.enc1 = nn.Sequential(conv(64, 96, 2), conv(96, 96))
        self.enc2 = nn.Sequential(conv(96, 128, 2), conv(128, 128))
        self.enc3 = nn.Sequential(conv(128, 256, 2), conv(256, 256))
        self.dec2 = up(256, 128, 128)
        self.dec1 = up(128, 96, 96)
        self.dec0 = up(96, 64, 64)
        self.register_buffer('position', spatial_encoding(), persistent=False)

    def forward(self, rgb_bev, lidar):
        x0 = self.enc0(torch.cat((rgb_bev, lidar.to(rgb_bev.dtype)), 1).contiguous(memory_format=torch.channels_last))
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        y2 = self.dec2(torch.cat((F.interpolate(x3, scale_factor=2, mode='nearest'), x2), 1))
        y1 = self.dec1(torch.cat((F.interpolate(y2, scale_factor=2, mode='nearest'), x1), 1))
        high = self.dec0(torch.cat((F.interpolate(y1, scale_factor=2, mode='nearest'), x0), 1))
        memory = x3.flatten(2).transpose(1, 2) + self.position.to(x3.dtype)
        return high, memory


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(256, 256)
        self.kv = nn.Linear(256, 512)
        self.out = nn.Linear(256, 256)

    def forward(self, queries, memory, map_plan_mask=False):
        batch, length, _ = queries.shape
        q = self.q(queries).reshape(batch, length, 8, 32).transpose(1, 2)
        kv = self.kv(memory).reshape(batch, memory.shape[1], 2, 8, 32)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        if map_plan_mask:
            # Block the planning key for the seven map rows. Two dense blocks also
            # work with fused ROCm SDPA kernels that reject a custom bias tensor.
            maps = F.scaled_dot_product_attention(q[:, :, :7], k[:, :, :7], v[:, :, :7])
            plan = F.scaled_dot_product_attention(q[:, :, 7:], k, v)
            x = torch.cat((maps, plan), dim=2)
        else:
            x = F.scaled_dot_product_attention(q, k, v)
        return self.out(x.transpose(1, 2).reshape(batch, length, 256))


class QueryLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_self = nn.LayerNorm(256)
        self.norm_cross = nn.LayerNorm(256)
        self.norm_ff = nn.LayerNorm(256)
        self.self_attn = Attention()
        self.cross_attn = Attention()
        self.ff = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))

    def forward(self, query, memory):
        x = self.norm_self(query)
        query = query + self.self_attn(x, x, map_plan_mask=True)
        query = query + self.cross_attn(self.norm_cross(query), memory)
        return query + self.ff(self.norm_ff(query))


def rgb_preprocess(rgb, gain=None, contrast=None):
    x = (rgb.float() / 255).pow(GAMMA)
    if gain is not None:
        x = ((x-.45)*contrast+.45)*gain
        x = x.clamp(0, 1)
    return (x-.45) / .25


def bev_grid(dy, yaw, xx, yy):
    c, s = yaw.cos(), yaw.sin()
    x = c*xx - s*yy
    y = s*xx + c*yy + dy
    return torch.stack((-y/19.2, (32.-x)/48.), -1)


def make_grid_axes():
    return (torch.from_numpy(X.astype(np.float32))[None], torch.from_numpy(Y.astype(np.float32))[None])


class Phase2(nn.Module):
    def __init__(self, pretrained=False, temporal=False, nav_rich=False):
        super().__init__()
        self.temporal = bool(temporal)
        self.nav_rich = bool(nav_rich)
        self.pyramid = ImagePyramid(pretrained)
        self.lift = Lift()
        self.native_projection=NativeProjection(NATIVE_C)
        self.sensor = SensorBEV()
        self.map_queries = nn.Parameter(torch.randn(1, 7, 256)*.025)
        self.plan_query = nn.Parameter(torch.randn(1, 1, 256)*.025)
        nav_dim = 19 if self.nav_rich else 9
        self.nav = nn.Sequential(nn.Linear(nav_dim, 128), nn.SiLU(), nn.Linear(128, 256))
        if self.temporal:
            self.motion_enc = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 256))
            self.temporal_q_norm = nn.LayerNorm(256)
            self.temporal_kv_norm = nn.LayerNorm(256)
            self.temporal_attn = Attention()
            self.temporal_gate = nn.Parameter(torch.zeros(1, 1, 256))
            self.register_buffer('ego_scale', torch.tensor([0.2, 0.2, 2.0]), persistent=False)
        self.decoder = nn.ModuleList(QueryLayer() for _ in range(3))
        self.norm = nn.LayerNorm(256)
        self.mask_head = nn.Conv2d(64, 4, 1)
        self.occ_head = nn.Conv2d(64, 5, 1)
        self.line_head = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 40))
        self.exists_head = nn.Linear(256, 1)
        self.path_head = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 40))
        nn.init.normal_(self.path_head[-1].weight, std=.003)
        nn.init.zeros_(self.path_head[-1].bias)
        self.register_buffer('line_stations', torch.linspace(0, 60, 20).reshape(1, 1, 20), persistent=False)

    def forward(self, rgb, lidar, nav, cmd, grid=None, gain=None, contrast=None,
                route_inst=None, route_dist=None, prev_map_tokens=None, ego_delta=None, has_prev=None):
        if isinstance(rgb,(tuple,list)):
            visual=self.native_projection(self.pyramid,self.lift.mix,rgb,gain,contrast)
        else:
            images = rgb_preprocess(rgb, gain, contrast).flatten(0, 1).contiguous(memory_format=torch.channels_last)
            visual = self.lift(self.pyramid(images))
        if grid is not None:
            visual = F.grid_sample(visual, grid, align_corners=False)
        return self.predict(visual, lidar, nav, cmd, route_inst=route_inst, route_dist=route_dist,
                            prev_map_tokens=prev_map_tokens, ego_delta=ego_delta, has_prev=has_prev)

    def predict(self, visual, lidar, nav, cmd, route_inst=None, route_dist=None,
                prev_map_tokens=None, ego_delta=None, has_prev=None):
        high, memory = self.sensor(visual, lidar)
        batch = high.shape[0]
        if self.nav_rich:
            intent = encode_nav_rich(nav, cmd, route_inst, route_dist)
        else:
            intent = torch.cat((nav.flatten(1)/60., cmd.to(nav.dtype)), 1)
        plan = self.plan_query + self.nav(intent).unsqueeze(1)
        map_q = self.map_queries.expand(batch, -1, -1)
        if self.temporal:
            if prev_map_tokens is None:
                prev_map_tokens = map_q.new_zeros(batch, 7, 256)
            if ego_delta is None:
                ego_delta = nav.new_zeros(batch, 3)
            if has_prev is None:
                has_prev = nav.new_zeros(batch)
            ego_s = ego_delta.to(map_q.dtype) * self.ego_scale.to(map_q.dtype)
            kv = self.temporal_kv_norm(prev_map_tokens.to(map_q.dtype)) + self.motion_enc(ego_s).unsqueeze(1)
            delta = self.temporal_attn(self.temporal_q_norm(map_q), kv, map_plan_mask=False)
            gate = self.temporal_gate.tanh().to(delta.dtype) * has_prev.reshape(batch, 1, 1).to(delta.dtype)
            map_q = map_q + gate * delta
        query = torch.cat((map_q, plan), 1)
        for layer in self.decoder:
            query = layer(query, memory)
        query = self.norm(query)
        vector = self.line_head(query[:, :7]).reshape(-1, 7, 20, 2).float()
        lines = torch.stack((self.line_stations+20*vector[..., 0], 12*vector[..., 1]), -1)
        theta = math.pi*self.path_head(query[:, 7]).float().tanh()
        path = 1.5*torch.stack((theta.cos(), theta.sin()), -1).cumsum(1)
        out = dict(path=path, lines=lines, exists=self.exists_head(query[:, :7]).squeeze(-1),
                   mask=self.mask_head(high), occ=self.occ_head(high))
        if self.temporal:
            out['map_tokens'] = query[:, :7]
        return out


def warm_start_from_v2(model, weights_path):
    """Initialize a Stage 2.6 Phase2 model from frozen v2 stage2.pt weights with zero extra-column nav."""
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=True)
    src = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    dst = model.state_dict()
    copied = 0
    for k, v in src.items():
        if k not in dst:
            continue
        if dst[k].shape == v.shape:
            dst[k].copy_(v)
            copied += 1
        elif k == 'nav.0.weight' and dst[k].shape[0] == v.shape[0] and dst[k].shape[1] >= v.shape[1]:
            dst[k].zero_()
            dst[k][:, :v.shape[1]].copy_(v)
            copied += 1
    if 'temporal_gate' in dst:
        dst['temporal_gate'].zero_()
    model.load_state_dict(dst, strict=True)
    return copied
