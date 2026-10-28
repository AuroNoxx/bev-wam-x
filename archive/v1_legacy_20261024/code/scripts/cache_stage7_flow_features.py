"""Extract and cache frozen Stage 4 BEV encoder tokens and anchor targets for Flow Matching."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from configs.stage7 import Stage7Config
from dataset.baseline import build_cache, validate_manifest
from mini_wa_jepa.geometry import wrap_angle
from mini_wa_jepa.normalization import Normalizer
from mini_wa_jepa.provenance import provenance, write_json
from scripts.evaluate_joint import load_joint


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_history_and_ego(states, episodes, steps, dt_s=0.1):
    """Compute 4-frame history indices and ego motion arrays following JointSequences."""
    n = len(states)
    positions = np.arange(n)
    history_indices = np.empty((n, 4), np.int64)
    ego = np.zeros((n, 2), np.float32)
    ego[:, 0] = states[:, 3] / 10.0

    previous = np.maximum(positions - 1, 0)
    consecutive = (positions > 0) & (episodes == episodes[previous]) & (steps - steps[previous] == 1)
    ego[consecutive, 1] = wrap_angle(states[consecutive, 2] - states[previous[consecutive], 2]) / dt_s

    starts = np.maximum.accumulate(np.where(consecutive, 0, positions))
    for slot, lag in enumerate((3, 2, 1, 0)):
        history_indices[:, slot] = np.maximum(positions - lag, starts)

    return history_indices, ego


@torch.no_grad()
def encode_all_frames(encoder, bev_mmap, output_file, max_frames=None, batch_size=128, device='cuda'):
    """Encode each unique BEV frame exactly once into float16 mmap tokens under SDPA MATH."""
    n_frames = len(bev_mmap) if max_frames is None else min(len(bev_mmap), max_frames)
    token_count = encoder.token_count
    token_dim = encoder.output.out_features

    output_file.parent.mkdir(parents=True, exist_ok=True)
    tokens_mmap = np.lib.format.open_memmap(
        output_file,
        mode='w+',
        dtype=np.float16,
        shape=(n_frames, token_count, token_dim),
    )

    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, n_frames, batch_size):
            end = min(start + batch_size, n_frames)
            batch_bev = torch.from_numpy(np.array(bev_mmap[start:end])).to(device)
            batch_tokens = encoder(batch_bev).half().cpu().numpy()
            tokens_mmap[start:end] = batch_tokens

    tokens_mmap.flush()
    del tokens_mmap
    return n_frames, token_count, token_dim


@torch.no_grad()
def compute_mse_predictions(model, frame_tokens_mmap, history_indices, ego_all, chosen_anchors, batch_size=128, device='cuda'):
    """Compute normalized Stage 4 MSE baseline predictions on chosen anchors under SDPA MATH."""
    n_chosen = len(chosen_anchors)
    predictions = np.empty((n_chosen, 10, 4), dtype=np.float32)

    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, n_chosen, batch_size):
            end = min(start + batch_size, n_chosen)
            anchor_batch = chosen_anchors[start:end]
            hist_batch = history_indices[anchor_batch]  # [B, 4]

            batch_tokens = torch.from_numpy(np.asarray(frame_tokens_mmap[hist_batch], dtype=np.float32)).to(device)
            batch_ego = torch.from_numpy(np.asarray(ego_all[hist_batch], dtype=np.float32)).to(device)

            result = model.forward_encoded(batch_tokens, batch_ego)
            pred = result['trajectory'].float().cpu().numpy()
            predictions[start:end] = pred

    return predictions


@torch.no_grad()
def measure_fp16_impact(model, bev_mmap, frame_tokens_mmap, history_indices, ego_all, device='cuda'):
    """Compare Stage 4 MSE output on first 16 frames: full FP32 encoder vs cached FP16 tokens."""
    probe_n = min(16, len(frame_tokens_mmap))
    if probe_n == 0:
        return dict(max_abs_diff=0.0, mse=0.0)

    # 1. FP32 through encoder
    with sdpa_kernel(SDPBackend.MATH):
        probe_bev = torch.from_numpy(np.array(bev_mmap[:probe_n])).to(device)
        fp32_tokens = model.encoder(probe_bev)  # [16, 64, 256]

        # Use probe_n as anchors
        hist = history_indices[:probe_n]
        ego = torch.from_numpy(np.asarray(ego_all[hist], dtype=np.float32)).to(device)

        # FP32 tokens forward
        fp32_batch = fp32_tokens[hist]
        pred_fp32 = model.forward_encoded(fp32_batch, ego)['trajectory'].float().cpu().numpy()

        # Cached FP16 tokens forward
        fp16_batch = torch.from_numpy(np.asarray(frame_tokens_mmap[hist], dtype=np.float32)).to(device)
        pred_fp16 = model.forward_encoded(fp16_batch, ego)['trajectory'].float().cpu().numpy()

    diff = np.abs(pred_fp32 - pred_fp16)
    return dict(
        max_abs_diff=float(np.max(diff)),
        mse=float(np.mean(diff ** 2)),
    )


def build_stage7_cache(data_root, output_dir, checkpoint_path, device='cuda', seed=20261021, smoke=False):
    if not getattr(torch.version, 'hip', None):
        raise RuntimeError('Native ROCm HIP is required for Stage 7 feature caching')

    torch.set_num_threads(4)
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    dataset_manifest = validate_manifest(data_root)
    cfg_data = dataset_manifest.get('config', {})
    if abs(cfg_data.get('dt_s', 0.0) - 0.1) > 1e-8:
        raise ValueError(f"Expected dt_s=0.1, got {cfg_data.get('dt_s')}")
    if abs(cfg_data.get('waypoint_dt_s', 0.0) - 0.2) > 1e-8:
        raise ValueError(f"Expected waypoint_dt_s=0.2, got {cfg_data.get('waypoint_dt_s')}")
    if cfg_data.get('future_steps', 0) != 10:
        raise ValueError(f"Expected future_steps=10, got {cfg_data.get('future_steps')}")

    norm_src = data_root / 'normalization.json'
    norm_dest = output_dir / 'normalization.json'
    shutil.copyfile(norm_src, norm_dest)
    normalizer = Normalizer(**json.loads(norm_dest.read_text()))

    # Load frozen Stage 4 Joint model
    model, model_config, saved_state = load_joint(checkpoint_path, device)
    model.eval()
    encoder = model.encoder.eval()
    model_param_count = sum(p.numel() for p in model.parameters())

    cfg = Stage7Config()
    report = dict(
        **provenance(cfg),
        stage=7,
        caching=True,
        completed=False,
        smoke=smoke,
        dataset_manifest_hash=file_sha256(data_root / 'manifest.json'),
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=file_sha256(checkpoint_path),
        model_parameter_count=model_param_count,
        splits={},
        array_hashes={},
    )
    write_json(output_dir / 'manifest.json', report)

    with tarfile.open(output_dir / 'source_snapshot.tar.gz', 'w:gz') as archive:
        for source in report['source_files']:
            archive.add(source, arcname=source)
        for doc in ('GOAL.md', 'AGENTS.md', 'docs/STAGE7_PROTOCOL.md'):
            archive.add(doc, arcname=doc)

    target_counts = {'train': 64 if smoke else 20000, 'val': 32 if smoke else 2048, 'test': 32 if smoke else 2048}
    max_frames_per_split = 256 if smoke else None

    for split in ('train', 'val', 'test'):
        split_started = time.perf_counter()
        raw_cache = build_cache(data_root, split)
        bev_mmap = np.load(raw_cache / 'bev.npy', mmap_mode='r')
        states = np.load(raw_cache / 'state.npy', mmap_mode='r')
        episodes = np.load(raw_cache / 'episode_id.npy', mmap_mode='r')
        steps = np.load(raw_cache / 'step_id.npy', mmap_mode='r')
        future_trajectories = np.load(raw_cache / 'future_trajectory.npy', mmap_mode='r')
        future_valid = np.load(raw_cache / 'future_valid.npy', mmap_mode='r')
        action_mask = np.load(raw_cache / 'action_mask.npy', mmap_mode='r')

        n_frames = len(bev_mmap)
        history_indices, ego_all = compute_history_and_ego(states, episodes, steps, dt_s=0.1)

        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=False)

        token_file = split_dir / 'frame_tokens.npy'
        actual_encoded_frames, token_count, token_dim = encode_all_frames(
            encoder, bev_mmap, token_file, max_frames=max_frames_per_split, batch_size=128, device=device
        )
        frame_tokens_mmap = np.load(token_file, mmap_mode='r')

        # Strictly filter eligible anchors within encoded frame range
        max_valid_anchor = actual_encoded_frames - 1
        eligible_mask = (action_mask == 1) & (future_valid.all(axis=-1)) & (np.arange(n_frames) <= max_valid_anchor)
        eligible_anchors = np.flatnonzero(eligible_mask)
        pre_count = int(actual_encoded_frames)
        post_count = int(len(eligible_anchors))

        target_n = target_counts[split]
        rng = np.random.default_rng(seed + (0 if split == 'train' else (1 if split == 'val' else 2)))
        if post_count > target_n:
            chosen = np.sort(rng.choice(eligible_anchors, target_n, replace=False))
        else:
            chosen = eligible_anchors.copy()

        anchor_hist_indices = history_indices[chosen]
        anchor_ego = ego_all[anchor_hist_indices]
        raw_trajs = future_trajectories[chosen]
        normalized_targets = normalizer.trajectory(raw_trajs)
        waypoint_valid = future_valid[chosen]
        chosen_action_mask = action_mask[chosen]

        mse_preds = compute_mse_predictions(
            model, frame_tokens_mmap, history_indices, ego_all, chosen, batch_size=128, device=device
        )

        # Save split arrays
        arrays_to_save = {
            'history_indices.npy': anchor_hist_indices,
            'ego.npy': anchor_ego,
            'targets.npy': normalized_targets,
            'waypoint_valid.npy': waypoint_valid,
            'action_mask.npy': chosen_action_mask,
            'anchors.npy': chosen,
            'episodes.npy': episodes[chosen],
            'steps.npy': steps[chosen],
            'mse_predictions.npy': mse_preds,
        }

        for arr_name, arr_data in arrays_to_save.items():
            arr_path = split_dir / arr_name
            np.save(arr_path, arr_data)
            report['array_hashes'][f'{split}/{arr_name}'] = file_sha256(arr_path)

        report['array_hashes'][f'{split}/frame_tokens.npy'] = file_sha256(token_file)

        split_info = dict(
            encoded_frames=actual_encoded_frames,
            eligible_anchors=post_count,
            chosen_count=int(len(chosen)),
            elapsed_seconds=round(time.perf_counter() - split_started, 2),
        )
        if split == 'train':
            impact = measure_fp16_impact(model, bev_mmap, frame_tokens_mmap, history_indices, ego_all, device=device)
            split_info['fp16_impact'] = impact

        report['splits'][split] = split_info
        write_json(output_dir / 'manifest.json', report)
        print(f"Cached {split}: {len(chosen)} anchors (encoded {actual_encoded_frames} frames)", flush=True)

    report['completed'] = True
    write_json(output_dir / 'manifest.json', report)
    print("Stage 7 feature caching completed successfully.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='Path to outputs/data/stage2_v2_100k')
    p.add_argument('--output', type=Path, required=True, help='Output cache directory')
    p.add_argument('--checkpoint', type=str, default='artifacts/restart_stage4/joint_001/best.pt', help='Stage 4 Joint checkpoint')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=20261021)
    p.add_argument('--smoke', action='store_true', help='Smoke mode: encode only 256 frames per split')
    args = p.parse_args()

    build_stage7_cache(
        data_root=args.data,
        output_dir=args.output,
        checkpoint_path=args.checkpoint,
        device=args.device,
        seed=args.seed,
        smoke=args.smoke,
    )


if __name__ == '__main__':
    main()
