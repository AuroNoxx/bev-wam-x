"""ROCm autograd path using official PyTorch attention and native index_add.

Only operator dispatch changes, scoped to this process/context. No upstream
source or checkpoint is edited. Native CUDA JIT remains separately audited.
"""
from contextlib import contextmanager
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _fp32_batch_norm(self, x):
    # gfx1100 Windows MIOpen rejects this BF16 3D BN path. Use the shipped
    # native FP32 normalization kernel; keep original affine/buffer semantics.
    with torch.autocast(device_type=x.device.type, enabled=False), torch.backends.cudnn.flags(enabled=False):
        out = F.batch_norm(x.float(), self.running_mean, self.running_var,
                           self.weight, self.bias, self.training, self.momentum, self.eps)
    return out.to(x.dtype)


class FP32BatchNorm2d(nn.BatchNorm2d):
    forward = _fp32_batch_norm


class FP32BatchNorm3d(nn.BatchNorm3d):
    forward = _fp32_batch_norm


def native_fp32_batch_norm(module):
    for name, child in list(module.named_children()):
        cls = FP32BatchNorm3d if isinstance(child, nn.BatchNorm3d) else FP32BatchNorm2d if isinstance(child, nn.BatchNorm2d) else None
        if cls:
            replacement = cls(child.num_features, eps=child.eps, momentum=child.momentum,
                              affine=child.affine, track_running_stats=child.track_running_stats,
                              device=child.running_mean.device, dtype=torch.float32)
            replacement.load_state_dict(child.state_dict(), strict=True)
            replacement.train(child.training)
            setattr(module, name, replacement)
        else:
            native_fp32_batch_norm(child)


def align_deformable_output_dtype(module, args, kwargs, output):
    # Official spatial scatter allocates slots from query dtype. Under mixed
    # precision LayerNorm may leave query FP32 while value_proj outputs BF16.
    query = kwargs.get("query", args[0] if args else None)
    return output.to(query.dtype)


def deformable_attention(value, spatial_shapes, level_start_index, locations, weights, im2col_step):
    from qwen_drive_perception.layers import multi_scale_deformable_attn_pytorch
    with torch.autocast(device_type=value.device.type, enabled=False):
        fp_value, fp_locations, fp_weights = value.float(), locations.float(), weights.float()
        def sample(v, points, coefficients):
            with torch.autocast(device_type=v.device.type, enabled=False):
                return multi_scale_deformable_attn_pytorch(v, spatial_shapes, points, coefficients)
        if locations.shape[1] <= 256:
            out = sample(fp_value, fp_locations, fp_weights)
        else:
            # Queries are independent in this official operator. Tile only the
            # queries, retaining every camera, level, sample point and pixel.
            parts = []
            for start in range(0, locations.shape[1], 256):
                args = (fp_value, fp_locations[:,start:start+256], fp_weights[:,start:start+256])
                if torch.is_grad_enabled() and any(x.requires_grad for x in args):
                    parts.append(checkpoint(sample,*args,use_reentrant=False))
                else:
                    parts.append(sample(*args))
            out = torch.cat(parts,dim=1)
    return out.to(value.dtype)


def voxel_pool_depth(features, depth, coords, mask, B, X, Y, Z, *, collapse_camera_sum=False, contribution_chunk=8192):
    """Match upstream [B,Ncam,X,Y,Z,C] depth-weighted sum, FP32 accumulation.

    Source layout is [B,Nsweep,Ncam,D,H,W]. Sweeps sum into each camera's
    voxel volume, matching the official rank construction. index_add is a
    shipped PyTorch/ROCm operator with gradients to features and depth.
    """
    _, sweeps, cameras, D, H, W = mask.shape
    channels = features.shape[2]
    if features.shape != (B, sweeps * cameras, channels, H, W):
        raise ValueError("Feature/camera geometry mismatch")
    if depth.shape != (B * sweeps * cameras, D, H, W):
        raise ValueError("Depth geometry mismatch")
    indices = mask.reshape(-1).nonzero().flatten()
    selected = coords.reshape(-1, 4)[indices].long()
    camera = (indices // (D * H * W)) % cameras
    image = indices // (D * H * W)
    pixel = indices % (H * W)
    output_cameras = 1 if collapse_camera_sum else cameras
    camera_rank = torch.zeros_like(camera) if collapse_camera_sum else camera
    rank = (((selected[:, 0] * output_cameras + camera_rank) * X + selected[:, 1]) * Y + selected[:, 2]) * Z + selected[:, 3]
    with torch.autocast(device_type=features.device.type, enabled=False):
        feat = features.float().reshape(B * sweeps * cameras, channels, H * W).transpose(1, 2)
        flat_depth = depth.float().reshape(-1)
        out = features.new_zeros((B * output_cameras * X * Y * Z, channels), dtype=torch.float32)
        # Native feat_encoding immediately sums cameras before any convolution.
        # An explicit opt-in fuses that sum into the same native index_add, so
        # the World caller need not materialize eight dense camera volumes.
        # Chunks bound gather/product temporaries without dropping any rays.
        def weighted_gather(f, d, images, pixels, depths):
            return f[images,pixels] * d[depths,None]
        for start in range(0, max(1, len(indices)), contribution_chunk):
            span = slice(start, start+contribution_chunk)
            args = (feat,flat_depth,image[span],pixel[span],indices[span])
            weighted = (checkpoint(weighted_gather,*args,use_reentrant=False)
                        if torch.is_grad_enabled() and (feat.requires_grad or flat_depth.requires_grad)
                        else weighted_gather(*args))
            out.index_add_(0, rank[span], weighted)
        return out.reshape(B, output_cameras, X, Y, Z, channels).to(torch.promote_types(features.dtype, depth.dtype))


@contextmanager
def rocm_perception_ops(*, collapse_camera_sum=False):
    import qwen_drive_perception.attention as attention
    import qwen_drive_perception.ops as ops
    old_attention, old_pool = attention.multi_scale_deformable_attn_cuda, ops.voxel_pool_depth
    attention.multi_scale_deformable_attn_cuda = deformable_attention
    ops.voxel_pool_depth = partial(voxel_pool_depth, collapse_camera_sum=collapse_camera_sum)
    try:
        yield
    finally:
        attention.multi_scale_deformable_attn_cuda = old_attention
        ops.voxel_pool_depth = old_pool
