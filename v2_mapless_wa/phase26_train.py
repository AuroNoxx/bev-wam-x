"""Single-GPU BF16 Lightning training for Stage 2.6 (temporal + rich nav + clip sampling)."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

# Enable TunableOp before torch initialization (used on ROCm gfx1100; harmless on CUDA).
os.environ.setdefault('PYTORCH_TUNABLEOP_ENABLED', '1')

import cv2
import numpy as np
import psutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
import yaml

from v1_dual_system.checkpoint_io import VerifiedCheckpointIO
from .geometry import CAMS, NAMES, Z
from .phase2_model import (
    GAMMA,
    Phase2,
    bev_grid,
    configure_sdpa,
    make_grid_axes,
    relative_ego_delta,
    route_inst_vector,
    warm_start_from_v2,
)
from .phase2_objectives import change_frame, metrics_from_totals, multitask_loss, rejoin_path, validation_totals


LOCKED_CONFIG = Path(__file__).resolve().with_name('config.yaml')


def locked_tier(path: Path = LOCKED_CONFIG):
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    rendering, rgb = cfg['rendering'], cfg['rgb']
    tier = rendering['quality_level']
    expected = dict(quality_level=tier, renderer='dx12', windowed=True, window_width=640, window_height=360)
    if tier not in ('High', 'Medium') or rendering != expected or cfg['simulator'] != 'CARLA 1.0' or cfg['engine'] != 'UE 5.8':
        raise ValueError(f'Unsupported CARLA rendering configuration in {path}')
    sizes = tuple(tuple(rgb['sizes'][name]) for name in NAMES)
    if (sizes != ((768, 448), (384, 224), (384, 224)) or rgb['enable_postprocess_effects']
            or rgb['preprocessing_gamma'] != 0.5 or rgb['stored_rgb'] != 'raw'
            or tuple(rgb['projection_heights_m']) != Z):
        raise ValueError(f'RGB size or postprocessing differs from the measured settings in {path}')
    if (tuple(tuple(rgb[name]) for name in NAMES) != CAMS or cfg['lidar'] != dict(channels=64, rotation_hz=20)
            or cfg['fixed_delta_seconds'] != 0.05):
        raise ValueError(f'Sensor rig or timing differs from the measured settings in {path}')
    return tier


CLIP_BUCKETS = ('straight', 'curve', 'sharp_curve', 'exit', 'lane_change', 'recovery')


def check_running():
    pid = os.getpid()
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        if p.info['pid'] in (pid, os.getppid()):
            continue
        name = (p.info.get('name') or '').lower()
        text = ' '.join(p.info['cmdline'] or ())
        if 'python' in name and ('v2_mapless_wa.phase2_train' in text or 'v2_mapless_wa.phase26_train' in text):
            raise RuntimeError(f'Another training process is active: PID {p.info["pid"]}')


def acquire_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open('a+b')
    if os.name == 'nt':
        import msvcrt
        if stream.tell() == 0:
            stream.write(b'\0')
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return stream


def _frame_bucket_flags(row):
    inst = row.get('route_instruction', 'follow')
    kappa = abs(float(row.get('kappa_max', 0.0) or 0.0))
    return {
        'exit': inst == 'exit',
        'recovery': bool(row.get('recovery', False)),
        'lane_change': row.get('lane_change') is not None,
        'sharp_curve': bool(row.get('sharp_curve', False)) or kappa > 0.028,
        'curve': (not bool(row.get('straight', True))) or kappa > 0.008 or inst in ('enter_roundabout', 'turn_right', 'turn_left'),
    }


def clip_bucket(rows):
    flags = [_frame_bucket_flags(r) for r in rows]
    for name in ('exit', 'recovery', 'lane_change', 'sharp_curve', 'curve'):
        if any(f[name] for f in flags):
            return name
    return 'straight'


def _load_rgb(fd: Path, stem: str):
    webp = fd / f'{stem}.webp'
    path = webp if webp.is_file() else (fd / f'{stem}.png')
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f'Failed to load image: {path}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


class Stage26ClipDataset(Dataset):
    """Direct PNG + NPZ temporal clip dataset spanning one or more released corpora."""

    def __init__(self, roots, split: str, clip_len: int = 4, stride: int | None = None):
        self.roots = [Path(r) for r in roots]
        self.split = split
        self.clip_len = int(clip_len)
        self.stride = int(stride if stride is not None else (1 if split == 'train' else clip_len))
        self.episodes = {}
        self.clips = []
        self.manifest_digests = {}

        for root_idx, root in enumerate(self.roots):
            manifest = root / 'manifest.jsonl'
            if not manifest.is_file():
                raise FileNotFoundError(f'Missing manifest.jsonl in {root}')
            h = hashlib.sha256()
            by_ep = {}
            with manifest.open('rb') as bf:
                while True:
                    chunk = bf.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
            self.manifest_digests[str(root.resolve())] = h.hexdigest()

            with manifest.open('r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row['split'] != split:
                        continue
                    ep_key = (root_idx, row['episode'])
                    by_ep.setdefault(ep_key, []).append(row)

            for ep_key, rows in sorted(by_ep.items()):
                ep_dir = root / 'episodes' / ep_key[1]
                self.episodes[ep_key] = {'dir': ep_dir, 'rows': rows}
                n = len(rows)
                for start in range(0, n - self.clip_len + 1, self.stride):
                    window = rows[start:start + self.clip_len]
                    bkt = clip_bucket(window)
                    flags = [_frame_bucket_flags(r) for r in window]
                    self.clips.append({
                        'ep_key': ep_key,
                        'ep_id': f'{root_idx}:{ep_key[1]}',
                        'start': start,
                        'bucket': bkt,
                        'is_exit': np.asarray([f['exit'] for f in flags], dtype=np.bool_),
                        'is_curve': np.asarray([f['curve'] or f['sharp_curve'] for f in flags], dtype=np.bool_),
                    })

        if not self.clips:
            raise ValueError(f'No clips built for split={split!r} from {self.roots}')

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        spec = self.clips[idx]
        ep = self.episodes[spec['ep_key']]
        ep_dir = ep['dir']
        rows = ep['rows']
        start = spec['start']
        T = self.clip_len

        fronts, sides_l = [], []
        lidars, navs, cmds = [], [], []
        lines_l, valid_l, exists_l, rasters, paths = [], [], [], [], []
        insts, dists = [], []
        poses = []

        # Load prior frame pose when continuing mid-episode for exact relative ego motion.
        prev_pose = None
        if start > 0:
            prev_row = rows[start - 1]
            prev_fd = ep_dir / f"{int(prev_row['frame']):07d}"
            with np.load(prev_fd / 'index.npz') as iz:
                prev_pose = np.asarray(iz['world_from_ego'], dtype=np.float64)

        ego_deltas = np.zeros((T, 3), dtype=np.float32)
        has_prev = np.zeros(T, dtype=np.float32)

        for t in range(T):
            row = rows[start + t]
            fd = ep_dir / f"{int(row['frame']):07d}"
            fronts.append(_load_rgb(fd, 'front'))
            sides_l.append(np.stack((_load_rgb(fd, 'front_left'), _load_rgb(fd, 'front_right')), axis=0))

            with np.load(fd / 'lidar.npz') as lz:
                lidars.append(np.asarray(lz['bev'], dtype=np.float16))
            with np.load(fd / 'input.npz') as inp:
                navs.append(np.asarray(inp['sd_nav'], dtype=np.float32))
                cmds.append(np.asarray(inp['cmd_one_hot'], dtype=np.uint8))
            with np.load(fd / 'labels.npz') as lab:
                lines_l.append(np.asarray(lab['map_lines'], dtype=np.float32))
                valid_l.append(np.asarray(lab['point_valid'], dtype=np.bool_))
                exists_l.append(np.asarray(lab['exists'], dtype=np.bool_))
                rasters.append(np.concatenate((lab['masks'], lab['p_occ']), axis=0).astype(np.uint8))
                paths.append(np.asarray(lab['path_gt'], dtype=np.float32))
            with np.load(fd / 'index.npz') as iz:
                pose = np.asarray(iz['world_from_ego'], dtype=np.float64)
            poses.append(pose)

            if prev_pose is not None:
                ego_deltas[t] = relative_ego_delta(prev_pose, pose)
                has_prev[t] = 1.0
            prev_pose = pose

            insts.append(route_inst_vector(row.get('route_instruction', 'follow')))
            d_val = row.get('instruction_distance_m', -1.0)
            dists.append(float(-1.0 if d_val is None else d_val))

        return {
            'front': torch.from_numpy(np.stack(fronts, axis=0)),
            'sides': torch.from_numpy(np.stack(sides_l, axis=0)),
            'lidar': torch.from_numpy(np.stack(lidars, axis=0)),
            'nav': torch.from_numpy(np.stack(navs, axis=0)),
            'cmd': torch.from_numpy(np.stack(cmds, axis=0)),
            'lines': torch.from_numpy(np.stack(lines_l, axis=0)),
            'valid': torch.from_numpy(np.stack(valid_l, axis=0)),
            'exists': torch.from_numpy(np.stack(exists_l, axis=0)),
            'raster': torch.from_numpy(np.stack(rasters, axis=0)),
            'path': torch.from_numpy(np.stack(paths, axis=0)),
            'route_inst': torch.from_numpy(np.stack(insts, axis=0)),
            'route_dist': torch.tensor(dists, dtype=torch.float32),
            'ego_delta': torch.from_numpy(ego_deltas),
            'has_prev': torch.from_numpy(has_prev),
            'is_exit': torch.from_numpy(spec['is_exit']),
            'is_curve': torch.from_numpy(spec['is_curve']),
            'start': int(start),
            'ep_id': spec['ep_id'],
        }


class Stage26ClipSampler(Sampler):
    """Epoch-reseeded balanced 6-bucket sampler across temporal clips."""

    def __init__(self, clips, seed: int, clips_per_epoch: int | None = None):
        self.groups = {}
        for idx, c in enumerate(clips):
            self.groups.setdefault(c['bucket'], []).append(idx)
        self.bucket_names = sorted(self.groups)
        self.n_total = len(clips) if clips_per_epoch is None else int(clips_per_epoch)
        self.generator = torch.Generator().manual_seed(int(seed))

    def __len__(self):
        return self.n_total

    def __iter__(self):
        per_bucket = max(1, math.ceil(self.n_total / max(1, len(self.bucket_names))))
        draws = []
        for bkt in self.bucket_names:
            pool = torch.tensor(self.groups[bkt], dtype=torch.long)
            if len(pool) >= per_bucket:
                perm = torch.randperm(len(pool), generator=self.generator)[:per_bucket]
            else:
                perm = torch.randint(0, len(pool), (per_bucket,), generator=self.generator)
            draws.append(pool[perm])
        combined = torch.cat(draws, dim=0)
        order = torch.randperm(len(combined), generator=self.generator)
        selected = combined[order[:self.n_total]]
        return iter(selected.tolist())


def _transform_ego_delta(ego_delta, dy_p, yaw_p):
    """Transform relative pose deltas (prev_in_cur) identically to a clip-fixed BEV offset."""
    dx, dy_e, dyaw = ego_delta[..., 0], ego_delta[..., 1], ego_delta[..., 2]
    dy_bt = dy_p[:, None]
    px = dx - torch.sin(dyaw) * dy_bt
    py = dy_e + torch.cos(dyaw) * dy_bt
    pts = torch.stack((px, py), dim=-1)
    rot_pts = change_frame(pts, dy_p, yaw_p)
    return torch.stack((rot_pts[..., 0], rot_pts[..., 1], dyaw), dim=-1)


def make_clip_batch(batch, augment: bool, axes):
    """Apply clip-consistent mirror, BEV perturbation, photometric jitter, history dropout, and delta noise."""
    front = batch['front']          # (B, T, 3, 448, 768)
    sides = batch['sides']          # (B, T, 2, 3, 224, 384)
    lidar = batch['lidar']          # (B, T, 6, 128, 128)
    raster = batch['raster']        # (B, T, 9, 128, 128)
    nav = batch['nav']              # (B, T, 3, 2)
    cmd = batch['cmd']              # (B, T, 3)
    route_inst = batch['route_inst']  # (B, T, 6)
    route_dist = batch['route_dist']  # (B, T)
    ego_delta = batch['ego_delta']    # (B, T, 3)
    has_prev = batch['has_prev']      # (B, T)
    path = batch['path']            # (B, T, 40, 2)
    lines = batch['lines']          # (B, T, 7, 20, 2)
    valid = batch['valid']          # (B, T, 7, 20)
    exists = batch['exists']        # (B, T, 7)

    B, T = lidar.shape[:2]
    device = lidar.device
    xx, yy = axes

    grid, gain, contrast = None, None, None
    if augment:
        rand = torch.rand(B, 6, device=device)
        mirror = rand[:, 0] < 0.5
        cam = [0, 2, 1]
        order = [0, 2, 1, 4, 3, 6, 5]
        inst_order = [0, 2, 1, 3, 4, 5]

        m_img = mirror[:, None, None, None, None]
        front = torch.where(m_img, front.flip(-1), front)
        left_s, right_s = sides.unbind(2)
        left_m = torch.where(m_img, right_s.flip(-1), left_s)
        right_m = torch.where(m_img, left_s.flip(-1), right_s)
        sides = torch.stack((left_m, right_m), dim=2)

        lidar = torch.where(m_img, lidar.flip(-1), lidar)
        raster = torch.where(mirror[:, None, None, None, None], raster.flip(-1), raster)

        reflect = torch.stack((torch.ones_like(rand[:, 0]), 1.0 - 2.0 * mirror.float()), dim=-1)
        nav = nav * reflect[:, None, None, :]
        path = path * reflect[:, None, None, :]
        lines = torch.where(mirror[:, None, None, None, None], lines[:, :, order], lines) * reflect[:, None, None, None, :]
        valid = torch.where(mirror[:, None, None, None], valid[:, :, order], valid)
        exists = torch.where(mirror[:, None, None], exists[:, :, order], exists)
        cmd = torch.where(mirror[:, None, None], cmd[:, :, cam], cmd)
        route_inst = torch.where(mirror[:, None, None], route_inst[:, :, inst_order], route_inst)
        delta_reflect = torch.stack((torch.ones_like(rand[:, 0]), reflect[:, 1], reflect[:, 1]), dim=-1)
        ego_delta = ego_delta * delta_reflect[:, None, :]

        perturb = rand[:, 1] < 0.30
        dy_p = (2.0 * rand[:, 2] - 1.0) * 0.7 * perturb
        yaw_p = (2.0 * rand[:, 3] - 1.0) * math.radians(5.0) * perturb

        dy_bt = dy_p[:, None].expand(B, T).reshape(B * T)
        yaw_bt = yaw_p[:, None].expand(B, T).reshape(B * T)
        p_bt = perturb[:, None].expand(B, T).reshape(B * T)
        grid = bev_grid(dy_bt[:, None, None], yaw_bt[:, None, None], xx, yy)

        lidar = F.grid_sample(lidar.flatten(0, 1).float(), grid, mode='nearest', align_corners=False).reshape(B, T, 6, 128, 128).to(lidar.dtype)
        raster = F.grid_sample(raster.flatten(0, 1).float(), grid, mode='nearest', align_corners=False).reshape(B, T, 9, 128, 128)
        nav = change_frame(nav.flatten(0, 1), dy_bt, yaw_bt).reshape(B, T, 3, 2)
        lines = change_frame(lines.flatten(0, 1), dy_bt, yaw_bt).reshape(B, T, 7, 20, 2)
        path_rep = rejoin_path(path.flatten(0, 1), dy_bt, yaw_bt)
        path = torch.where(p_bt[:, None, None], path_rep, path.flatten(0, 1)).reshape(B, T, 40, 2)

        trans_delta = _transform_ego_delta(ego_delta, dy_p, yaw_p)
        ego_delta = torch.where(perturb[:, None, None], trans_delta, ego_delta)

        # Mid-clip history dropout (p=0.20 on t >= 1) and small ego motion noise
        if T > 1:
            drop = torch.rand(B, T - 1, device=device) < 0.20
            has_prev = torch.cat((has_prev[:, :1], torch.where(drop, torch.zeros_like(has_prev[:, 1:]), has_prev[:, 1:])), dim=1)
        noise = torch.randn_like(ego_delta) * ego_delta.new_tensor([0.03, 0.03, 0.005])
        ego_delta = (ego_delta + noise) * has_prev[..., None]

        gain = (0.85 + 0.30 * rand[:, 4])[:, None].expand(B, T).reshape(B * T, 1, 1, 1, 1)
        contrast = (0.80 + 0.40 * rand[:, 5])[:, None].expand(B, T).reshape(B * T, 1, 1, 1, 1)

    return {
        'front': front,
        'sides': sides,
        'lidar': lidar,
        'nav': nav,
        'cmd': cmd,
        'route_inst': route_inst,
        'route_dist': route_dist,
        'ego_delta': ego_delta,
        'has_prev': has_prev,
        'grid': grid.reshape(B, T, 128, 128, 2) if grid is not None else None,
        'gain': gain.reshape(B, T, 1, 1, 1, 1) if gain is not None else None,
        'contrast': contrast.reshape(B, T, 1, 1, 1, 1) if contrast is not None else None,
        'path': path,
        'lines': lines,
        'valid': valid,
        'exists': exists,
        'mask': raster[:, :, :4].float(),
        'occ': raster[:, :, 4:].float(),
    }


def curvature_weighted_tangent_loss(pred_path, gt_path):
    """Smooth L1 on unit trajectory tangents, upweighting curved segments."""
    gt_seg = gt_path[:, 1:] - gt_path[:, :-1]
    pred_seg = pred_path[:, 1:] - pred_path[:, :-1]
    gt_tan = gt_seg / gt_seg.norm(dim=-1, keepdim=True).clamp_min(1e-4)
    pred_tan = pred_seg / pred_seg.norm(dim=-1, keepdim=True).clamp_min(1e-4)
    dot = (gt_tan[:, 1:] * gt_tan[:, :-1]).sum(dim=-1).clamp(-0.999999, 0.999999)
    dtheta = torch.acos(dot)
    seg_curv = torch.cat((dtheta[:, :1], dtheta), dim=1)
    weights = 1.0 + 3.0 * (seg_curv > 0.025).float()
    loss_tan = F.smooth_l1_loss(pred_tan, gt_tan, reduction='none').mean(dim=-1)
    return (loss_tan * weights).mean() / weights.mean()


class Stage26Learner(pl.LightningModule):
    def __init__(self, config, sampler=None):
        super().__init__()
        self.model = Phase2(pretrained=False, temporal=True, nav_rich=True).to(memory_format=torch.channels_last)
        if config.get('warm_start'):
            warm_start_from_v2(self.model, config['warm_start'])
        self.config = config
        self.sampler = sampler
        xx, yy = make_grid_axes()
        self.register_buffer('xx', xx, persistent=False)
        self.register_buffer('yy', yy, persistent=False)
        self.totals = None
        self.val_exit_sum = None
        self.val_exit_cnt = None
        self.val_curve_sum = None
        self.val_curve_cnt = None
        self._val_ep = None
        self._val_prev_tokens = None
        self.start = time.perf_counter()

    def on_train_epoch_start(self):
        if self.sampler is not None:
            self.sampler.generator.manual_seed(int(self.config['seed']) + int(self.current_epoch))

    def _rollout_train(self, clip):
        B, T = clip['lidar'].shape[:2]
        prev_tokens = None
        total_loss = clip['path'].new_zeros(())
        for t in range(T):
            rgb = (clip['front'][:, t], *clip['sides'][:, t].unbind(1))
            grid_t = clip['grid'][:, t] if clip['grid'] is not None else None
            gain_t = clip['gain'][:, t] if clip['gain'] is not None else None
            contrast_t = clip['contrast'][:, t] if clip['contrast'] is not None else None
            has_p = clip['has_prev'][:, t]
            if prev_tokens is None:
                has_p = torch.zeros_like(has_p)
            out = self.model(
                rgb,
                clip['lidar'][:, t],
                clip['nav'][:, t],
                clip['cmd'][:, t],
                grid=grid_t,
                gain=gain_t,
                contrast=contrast_t,
                route_inst=clip['route_inst'][:, t],
                route_dist=clip['route_dist'][:, t],
                prev_map_tokens=prev_tokens,
                ego_delta=clip['ego_delta'][:, t],
                has_prev=has_p,
            )
            target_t = {
                'path': clip['path'][:, t],
                'lines': clip['lines'][:, t],
                'valid': clip['valid'][:, t],
                'exists': clip['exists'][:, t],
                'mask': clip['mask'][:, t],
                'occ': clip['occ'][:, t],
            }
            loss_t = multitask_loss(out, target_t, perception_priority=True)
            loss_t = loss_t + 0.5 * curvature_weighted_tangent_loss(out['path'], target_t['path'])
            total_loss = total_loss + loss_t
            prev_tokens = out['map_tokens'].detach()
        return total_loss / T

    def training_step(self, batch, batch_idx):
        clip = make_clip_batch(batch, True, (self.xx, self.yy))
        loss = self._rollout_train(clip)
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=len(batch['lidar']))
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        loss_t = outputs['loss']
        if not torch.isfinite(loss_t):
            raise FloatingPointError(f'Nonfinite loss at update {self.global_step}')
        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            print(
                f"epoch {self.current_epoch + 1} batch {batch_idx + 1} update {self.global_step} "
                f"loss {float(loss_t):.4f} elapsed {time.perf_counter() - self.start:.1f}s",
                flush=True,
            )

    def on_validation_epoch_start(self):
        self.totals = torch.zeros(36, device=self.device)
        self.val_exit_sum = torch.zeros((), device=self.device)
        self.val_exit_cnt = torch.zeros((), device=self.device)
        self.val_curve_sum = torch.zeros((), device=self.device)
        self.val_curve_cnt = torch.zeros((), device=self.device)
        self._val_ep = None
        self._val_prev_tokens = None

    def validation_step(self, batch, batch_idx):
        clip = make_clip_batch(batch, False, (self.xx, self.yy))
        B, T = clip['lidar'].shape[:2]
        ep_id = batch['ep_id'][0] if B == 1 else None
        is_exit = batch['is_exit']
        is_curve = batch['is_curve']
        if B != 1 or self._val_ep != ep_id:
            self._val_ep = ep_id
            self._val_prev_tokens = None
        prev_tokens = self._val_prev_tokens

        loss_sum = clip['path'].new_zeros(())
        for t in range(T):
            rgb = (clip['front'][:, t], *clip['sides'][:, t].unbind(1))
            has_p = clip['has_prev'][:, t]
            if prev_tokens is None:
                has_p = torch.zeros_like(has_p)
            out = self.model(
                rgb,
                clip['lidar'][:, t],
                clip['nav'][:, t],
                clip['cmd'][:, t],
                route_inst=clip['route_inst'][:, t],
                route_dist=clip['route_dist'][:, t],
                prev_map_tokens=prev_tokens,
                ego_delta=clip['ego_delta'][:, t],
                has_prev=has_p,
            )
            target_t = {
                'path': clip['path'][:, t],
                'lines': clip['lines'][:, t],
                'valid': clip['valid'][:, t],
                'exists': clip['exists'][:, t],
                'mask': clip['mask'][:, t],
                'occ': clip['occ'][:, t],
            }
            loss_t = multitask_loss(out, target_t, perception_priority=True)
            loss_t = loss_t + 0.5 * curvature_weighted_tangent_loss(out['path'], target_t['path'])
            loss_sum = loss_sum + loss_t
            self.totals += validation_totals(out, target_t)

            lat_near = (out['path'][:, :16, 1] - target_t['path'][:, :16, 1]).abs().mean(dim=1)
            ex_m = is_exit[:, t].float()
            cu_m = is_curve[:, t].float()
            self.val_exit_sum += (lat_near * ex_m).sum()
            self.val_exit_cnt += ex_m.sum()
            self.val_curve_sum += (lat_near * cu_m).sum()
            self.val_curve_cnt += cu_m.sum()
            prev_tokens = out['map_tokens'].detach()

        self._val_prev_tokens = prev_tokens if B == 1 else None
        self.log('val_loss', loss_sum / T, batch_size=B)

    def on_validation_epoch_end(self):
        result = metrics_from_totals(self.totals)
        exit_cnt = float(self.val_exit_cnt.item())
        curve_cnt = float(self.val_curve_cnt.item())
        val_exit_near_m = float((self.val_exit_sum / exit_cnt).item()) if exit_cnt > 0 else float(result['lateral_near_m'])
        val_curve_near_m = float((self.val_curve_sum / curve_cnt).item()) if curve_cnt > 0 else float(result['lateral_near_m'])
        score = 0.25 * float(result['lateral_near_m']) + 0.15 * val_exit_near_m - float(result['map_miou'])
        self.log('score', score)
        if not self.trainer.sanity_checking:
            result.update(
                val_exit_near_m=val_exit_near_m,
                val_exit_frames=int(exit_cnt),
                val_curve_near_m=val_curve_near_m,
                val_curve_frames=int(curve_cnt),
                epoch=self.current_epoch + 1,
                updates=self.global_step,
                score=score,
                lr=self.optimizers().param_groups[0]['lr'],
                peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                elapsed_s=time.perf_counter() - self.start,
            )
            with (Path(self.config['out']) / 'validation.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(result) + '\n')
            print(json.dumps(result), flush=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'], weight_decay=0.01, foreach=True)
        total_steps = max(2, int(self.config.get('planned_total_steps', 100)))
        warm_steps = max(1, min(int(self.config.get('updates_per_epoch', 1)), total_steps - 1))
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.05, end_factor=1.0, total_iters=warm_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - warm_steps), eta_min=1e-5)
        schedule = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cosine], milestones=[warm_steps])
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': schedule, 'interval': 'step'}}

    def on_save_checkpoint(self, checkpoint):
        checkpoint['task_state'] = self.config

    def on_load_checkpoint(self, checkpoint):
        original = checkpoint['task_state']
        for name in ('manifest_sha256', 'batch', 'accumulate', 'clip_len', 'seed', 'renderer_sha256', 'gamma', 'sdpa'):
            if original[name] != self.config[name]:
                raise ValueError(f'Checkpoint resume changes {name}')
        self.config['resumed_from_update'] = checkpoint['global_step']


def freeze(checkpoint_path, destination, config):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = {k.removeprefix('model.'): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
    Phase2(temporal=True, nav_rich=True).load_state_dict(state, strict=True)
    if not state or not all(torch.isfinite(v).all() for v in state.values()):
        raise RuntimeError('Invalid frozen state')
    payload = {
        'config': config,
        'model': state,
        'updates': int(checkpoint['global_step']),
        'epoch': int(checkpoint['epoch']) + 1,
    }
    pending = destination / 'stage26.pt.pending'
    final = destination / 'stage26.pt'
    torch.save(payload, pending)
    check = torch.load(pending, map_location='cpu', weights_only=True)
    for key, tensor in state.items():
        if not torch.equal(tensor, check['model'][key]):
            raise RuntimeError(f'Frozen serialization mismatch: {key}')
    with pending.open('rb+') as stream:
        os.fsync(stream.fileno())
    os.replace(pending, final)
    with final.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    record = {
        'frozen': str(final.resolve()),
        'selected_epoch': int(checkpoint['epoch']) + 1,
        'selected_updates': int(checkpoint['global_step']),
        'sha256': digest,
    }
    (destination / 'metadata.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def main(args):
    check_running()
    active_sdpa = configure_sdpa(args.sdpa)
    if torch.version.hip and not torch.cuda.tunable.is_enabled():
        raise RuntimeError('TunableOp did not enable on ROCm')
    if not torch.version.hip and torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    pl.seed_everything(args.seed, workers=True)
    tier = locked_tier()

    train = Stage26ClipDataset(args.datasets, 'train', clip_len=args.clip_len, stride=1)
    val = Stage26ClipDataset(args.datasets, 'val', clip_len=args.clip_len, stride=args.clip_len)

    sampler = Stage26ClipSampler(train.clips, args.seed, clips_per_epoch=args.clips_per_epoch)
    bucket_counts = dict(Counter(c['bucket'] for c in train.clips))
    draw = list(sampler)
    draw_counts = dict(Counter(train.clips[i]['bucket'] for i in draw))

    loader_extra = {'prefetch_factor': 2} if args.workers else {}
    train_loader = DataLoader(
        train,
        batch_size=args.batch,
        sampler=sampler,
        pin_memory=True,
        num_workers=args.workers,
        persistent_workers=bool(args.workers),
        drop_last=(len(sampler) % args.batch != 0),
        **loader_extra,
    )
    val_loader = DataLoader(
        val,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
        persistent_workers=bool(args.workers),
        **loader_extra,
    )
    updates_per_epoch = max(1, math.ceil(len(train_loader) / args.accumulate))
    planned_total_steps = args.max_steps if args.max_steps is not None else args.epochs * updates_per_epoch

    config = {
        'architecture': 'Phase26',
        'temporal': True,
        'nav_rich': True,
        'clip_len': args.clip_len,
        'clips_per_epoch': len(sampler),
        'datasets': [str(Path(d).resolve()) for d in args.datasets],
        'manifest_sha256': train.manifest_digests,
        'out': str(args.out.resolve()),
        'warm_start': str(args.warm_start.resolve()) if args.warm_start else None,
        'renderer': tier,
        'renderer_sha256': hashlib.sha256(LOCKED_CONFIG.read_bytes()).hexdigest(),
        'gamma': GAMMA,
        'seed': args.seed,
        'sdpa': active_sdpa,
        'precision': 'bf16-mixed',
        'epochs': args.epochs,
        'max_steps': args.max_steps,
        'batch': args.batch,
        'accumulate': args.accumulate,
        'lr': args.lr,
        'foreach': True,
        'updates_per_epoch': updates_per_epoch,
        'planned_total_steps': planned_total_steps,
        'train_clip_pool': bucket_counts,
        'first_epoch_draw': draw_counts,
    }
    learner = Stage26Learner(config, sampler)
    config['parameters'] = sum(p.numel() for p in learner.model.parameters())
    print(json.dumps(config), flush=True)
    (args.out / 'training-config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    checkpoints = ModelCheckpoint(
        dirpath=args.out / 'checkpoints',
        monitor='score',
        mode='min',
        filename='best',
        save_top_k=1,
        save_last=False,
        enable_version_counter=False,
        save_on_train_epoch_end=False,
    )
    latest = ModelCheckpoint(
        dirpath=args.out / 'checkpoints',
        monitor=None,
        filename='last',
        save_top_k=1,
        save_last=False,
        enable_version_counter=False,
        save_on_train_epoch_end=True,
    )
    trainer_kwargs = dict(
        accelerator='gpu',
        devices=1,
        precision='bf16-mixed',
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate,
        gradient_clip_val=2.0,
        callbacks=[checkpoints, latest],
        plugins=[VerifiedCheckpointIO()],
        logger=CSVLogger(args.out, name='lightning'),
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=20,
    )
    if args.max_steps is not None:
        trainer_kwargs['max_steps'] = int(args.max_steps)
        if int(args.max_steps) * int(args.accumulate) < len(train_loader):
            trainer_kwargs['limit_train_batches'] = int(args.max_steps) * int(args.accumulate)
    if args.limit_val_batches is not None:
        trainer_kwargs['limit_val_batches'] = int(args.limit_val_batches)

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(learner, train_loader, val_loader, ckpt_path=str(args.resume) if args.resume else None)

    saved_last = torch.load(args.out / 'checkpoints/last.ckpt', map_location='cpu', weights_only=False)
    if saved_last['global_step'] != trainer.global_step:
        raise RuntimeError('Latest verified checkpoint does not contain the last optimizer update')
    del saved_last
    if not checkpoints.best_model_path:
        raise RuntimeError('No validated best checkpoint')

    record = freeze(checkpoints.best_model_path, args.freeze, config)
    record.update(
        actual_updates=int(trainer.global_step),
        physical_batch=int(args.batch),
        gradient_accumulation=int(args.accumulate),
        clip_len=int(args.clip_len),
        wall_s=time.perf_counter() - learner.start,
    )
    (args.out / 'training-result.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--warm-start', type=Path, default=None)
    parser.add_argument('--clip-len', type=int, default=4)
    parser.add_argument('--clips-per-epoch', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--limit-val-batches', type=int, default=None)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--accumulate', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261121)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--sdpa', choices=('auto', 'aotriton', 'cuda_flash', 'math'), default='auto')
    parser.add_argument('--resume', type=Path, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with acquire_lock(args.out.parent / '.phase2-training.lock'):
        main(args)
