"""Evaluation harness for Stage 7 Flow Matching candidate trajectory generation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import tarfile

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dataset.flow_features import FlowFeaturesDataset
from mini_wa_jepa.geometry import wrap_angle
from mini_wa_jepa.normalization import Normalizer
from mini_wa_jepa.provenance import provenance, write_json
from models.flow_matcher import ConditionalFlowMatcher
from configs.stage7 import Stage7Config
from scripts.evaluate_joint import load_joint


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_pairwise_diversity(candidates_si):
    """Average pairwise Euclidean distance across waypoints in meters for K candidates."""
    k = len(candidates_si)
    if k < 2:
        return 0.0
    pos = candidates_si[..., :2]  # [K, 10, 2]
    diffs = pos[:, None, :, :] - pos[None, :, :, :]  # [K, K, 10, 2]
    dist_matrix = np.linalg.norm(diffs, axis=-1).mean(axis=-1)  # [K, K]
    upper_tri = np.triu_indices(k, k=1)
    return float(dist_matrix[upper_tri].mean())


def evaluate_stage7_flow(data_dir, run_dir, output_dir, device='cuda'):
    torch.set_num_threads(4)
    if not getattr(torch.version, 'hip', None):
        raise RuntimeError('Native ROCm HIP is required for formal Stage 7 evaluation')

    data_dir = Path(data_dir)
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    evaluation_provenance=provenance(Stage7Config())
    with tarfile.open(output_dir/'source_snapshot.tar.gz','w:gz') as archive:
        for source in evaluation_provenance['source_files']:archive.add(source,arcname=source)
        for source in ('GOAL.md','AGENTS.md','docs/STAGE7_PROTOCOL.md'):archive.add(source,arcname=source)

    # 1. Enforce guard: only completed, non-smoke runs may be formally evaluated
    run_manifest = json.loads((run_dir / 'manifest.json').read_text())
    if not run_manifest.get('completed', False):
        raise RuntimeError(f"Run {run_dir} is incomplete; formal evaluation forbidden.")
    if run_manifest.get('smoke', False):
        raise RuntimeError(f"Run {run_dir} is a smoke test; formal test evaluation forbidden.")

    # 2. Load model checkpoint and restore architecture dynamically
    best_path = run_dir / 'best.pt'
    if not best_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {best_path}")
    checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)

    normalizer = Normalizer(**json.loads((data_dir / 'normalization.json').read_text()))

    model_arch = checkpoint.get('model_arch', dict(token_count=64, token_dim=256, condition_dim=256, hidden_dim=256))
    if model_arch.get('model_type') == 'transformer_kinematic':
        from models.flow_matcher_transformer import TransformerKinematicFlowMatcher
        model = TransformerKinematicFlowMatcher(
            **model_arch,
            normalizer=normalizer,
        ).to(device)
    else:
        model = ConditionalFlowMatcher(
            token_count=model_arch.get('token_count', 64),
            token_dim=model_arch.get('token_dim', 256),
            condition_dim=model_arch.get('condition_dim', 256),
            hidden_dim=model_arch.get('hidden_dim', 256),
        ).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Load Stage 4 Joint model for standalone head latency comparison
    stage4_ckpt_path = 'artifacts/restart_stage4/joint_001/best.pt'
    joint_model, _, _ = load_joint(stage4_ckpt_path, device)
    joint_model.eval()

    # 3. Load formal test split (2048 anchors)
    test_dataset = FlowFeaturesDataset(data_dir, split='test')
    n_test = len(test_dataset)

    # Metric accumulators
    mse_ades = []
    mse_fdes = []
    fm_c1_ades = []
    fm_c1_fdes = []
    oracle_min_ades = []
    diversities = []
    negative_speed_counts = 0
    over_15_speed_counts = 0
    over_8_accel_counts = 0
    heading_inconsistencies = []

    fm_latencies_ms = []
    mse_latencies_ms = []

    # Storage arrays for full raw artifact
    all_candidates_si = np.empty((n_test, 8, 10, 4), dtype=np.float32)
    all_gt_si = np.empty((n_test, 10, 4), dtype=np.float32)
    all_mse_si = np.empty((n_test, 10, 4), dtype=np.float32)
    all_waypoint_valids = np.empty((n_test, 10), dtype=bool)
    all_anchors = np.empty((n_test,), dtype=np.int64)
    all_episodes = np.empty((n_test,), dtype=np.int64)
    all_steps = np.empty((n_test,), dtype=np.int64)

    per_anchor_records = []

    print(f"Evaluating {n_test} test anchors on {device}...", flush=True)

    with sdpa_kernel(SDPBackend.MATH):
        for i in range(n_test):
            item = test_dataset[i]
            history = item['history'].unsqueeze(0).to(device)
            ego = item['ego'].unsqueeze(0).to(device)
            gt_norm = item['target'].numpy()  # [10, 4]
            gt_si = normalizer.inverse_trajectory(gt_norm)

            mse_pred_norm = item['mse_prediction'].numpy()  # [10, 4]
            mse_pred_si = normalizer.inverse_trajectory(mse_pred_norm)

            # Measure MSE head latency on cached tokens
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_mse_0 = time.perf_counter()
            with torch.no_grad():
                _ = joint_model.forward_encoded(history, ego)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            mse_lat_ms = (time.perf_counter() - t_mse_0) * 1000.0
            mse_latencies_ms.append(mse_lat_ms)

            # Measure Flow Matching head latency (10 Euler steps, 8 candidates batched)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_fm_0 = time.perf_counter()
            with torch.no_grad():
                if getattr(model, 'model_arch', {}).get('model_type') == 'transformer_kinematic':
                    candidates_norm = model.sample(
                        history,
                        ego,
                        num_steps=10,
                        num_candidates=8,
                        seed=42 + int(item['anchor']),
                        device=device,
                    )
                else:
                    condition = model.extract_condition(history, ego)
                    candidates_norm = model.sample(
                        condition,
                        num_steps=10,
                        num_candidates=8,
                        seed=42 + int(item['anchor']),
                        device=device,
                    )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fm_lat_ms = (time.perf_counter() - t_fm_0) * 1000.0
            fm_latencies_ms.append(fm_lat_ms)

            candidates_np = candidates_norm.squeeze(0).cpu().numpy()  # [8, 10, 4]
            if not np.isfinite(candidates_np).all():
                np.savez_compressed(output_dir / f'failed_sample_{i}.npz', history=history.cpu().numpy(), ego=ego.cpu().numpy(), candidates=candidates_np)
                raise FloatingPointError(f"Non-finite candidate output generated at anchor index {i}")

            candidates_si = normalizer.inverse_trajectory(candidates_np)

            # Store in raw arrays
            all_candidates_si[i] = candidates_si
            all_gt_si[i] = gt_si
            all_mse_si[i] = mse_pred_si
            all_waypoint_valids[i] = item['waypoint_valid'].numpy()
            all_anchors[i] = item['anchor']
            all_episodes[i] = item['episode']
            all_steps[i] = item['step']

            # MSE Baseline errors
            mse_err = np.linalg.norm(mse_pred_si[:, :2] - gt_si[:, :2], axis=-1)
            mse_ades.append(float(mse_err.mean()))
            mse_fdes.append(float(mse_err[-1]))

            # FM Candidate 1 errors
            c1_err = np.linalg.norm(candidates_si[0, :, :2] - gt_si[:, :2], axis=-1)
            fm_c1_ades.append(float(c1_err.mean()))
            fm_c1_fdes.append(float(c1_err[-1]))

            # Oracle min ADE of 8 candidates (clearly diagnostic only)
            all_c_errs = [np.linalg.norm(candidates_si[k, :, :2] - gt_si[:, :2], axis=-1).mean() for k in range(8)]
            oracle_min_ades.append(float(min(all_c_errs)))

            # Diversity
            div_m = compute_pairwise_diversity(candidates_si)
            diversities.append(div_m)

            # Kinematic feasibility across all 8 candidates
            speeds = candidates_si[..., 2]
            negative_speed_counts += int(np.sum(speeds < 0.0))
            over_15_speed_counts += int(np.sum(speeds > 15.0))

            accels = np.abs(np.diff(speeds, axis=-1)) / 0.2
            over_8_accel_counts += int(np.sum(accels > 8.0))

            # Heading vs derivative inconsistency
            dx = np.diff(candidates_si[..., 0], axis=-1)
            dy = np.diff(candidates_si[..., 1], axis=-1)
            pos_dist = np.sqrt(dx ** 2 + dy ** 2)
            moving = pos_dist > 0.1
            if np.any(moving):
                deriv_heading = np.arctan2(dy, dx)
                cand_heading = candidates_si[..., 1:, 3]
                h_diff = np.abs(wrap_angle(cand_heading - deriv_heading))
                heading_inconsistencies.extend(h_diff[moving].tolist())

            per_anchor_records.append(
                dict(
                    anchor=int(item['anchor']),
                    episode=int(item['episode']),
                    step=int(item['step']),
                    mse_ade=float(mse_ades[-1]),
                    mse_fde=float(mse_fdes[-1]),
                    fm_c1_ade=float(fm_c1_ades[-1]),
                    fm_c1_fde=float(fm_c1_fdes[-1]),
                    oracle_min_ade=float(oracle_min_ades[-1]),
                    diversity_m=div_m,
                    fm_latency_ms=fm_lat_ms,
                    mse_latency_ms=mse_lat_ms,
                )
            )

    total_candidate_points = n_test * 8 * 10
    total_candidate_accel_points = n_test * 8 * 9

    summary = dict(
        **evaluation_provenance,
        completed=True,
        evaluated_anchors=n_test,
        checkpoint_path=str(best_path),
        checkpoint_sha256=file_sha256(best_path),
        cache_dir=str(data_dir),
        cache_manifest_hash=file_sha256(data_dir / 'manifest.json'),
        mse_baseline=dict(
            mean_ade_m=float(np.mean(mse_ades)),
            mean_fde_m=float(np.mean(mse_fdes)),
        ),
        flow_matching_sample1=dict(
            mean_ade_m=float(np.mean(fm_c1_ades)),
            mean_fde_m=float(np.mean(fm_c1_fdes)),
        ),
        flow_matching_oracle_diagnostic=dict(
            mean_min_ade_m=float(np.mean(oracle_min_ades)),
            note='Oracle min-of-8 is a diagnostic upper bound and strictly not a deployable planner policy',
        ),
        diversity=dict(
            mean_pairwise_distance_m=float(np.mean(diversities)),
            p50_diversity_m=float(np.median(diversities)),
        ),
        kinematics=dict(
            negative_speed_rate=float(negative_speed_counts / total_candidate_points),
            over_15_speed_rate=float(over_15_speed_counts / total_candidate_points),
            over_8_accel_rate=float(over_8_accel_counts / total_candidate_accel_points),
            mean_heading_inconsistency_rad=float(np.mean(heading_inconsistencies)) if heading_inconsistencies else None,
        ),
        latency=dict(
            fm_head_p50_ms=float(np.percentile(fm_latencies_ms, 50)),
            fm_head_p95_ms=float(np.percentile(fm_latencies_ms, 95)),
            fm_head_mean_ms=float(np.mean(fm_latencies_ms)),
            mse_head_p50_ms=float(np.percentile(mse_latencies_ms, 50)),
            mse_head_p95_ms=float(np.percentile(mse_latencies_ms, 95)),
            mse_head_mean_ms=float(np.mean(mse_latencies_ms)),
            note='Latencies represent head-only forward execution given cached BEV tokens',
        ),
    )

    # Save full raw npz artifact for independent verification
    np.savez_compressed(
        output_dir / 'raw_eval.npz',
        candidates=all_candidates_si,
        ground_truth=all_gt_si,
        mse_predictions=all_mse_si,
        waypoint_valid=all_waypoint_valids,
        anchors=all_anchors,
        episodes=all_episodes,
        steps=all_steps,
        fm_latencies_ms=np.asarray(fm_latencies_ms, dtype=np.float32),
        mse_latencies_ms=np.asarray(mse_latencies_ms, dtype=np.float32),
    )

    write_json(output_dir / 'summary.json', summary)
    write_json(output_dir / 'per_anchor_metrics.json', per_anchor_records)
    print("Formal evaluation summary:\n", json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='Path to Stage 7 cached features directory')
    p.add_argument('--run', type=Path, required=True, help='Path to completed Flow Matching training run')
    p.add_argument('--output', type=Path, required=True, help='Path to evaluation output directory')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    evaluate_stage7_flow(
        data_dir=args.data,
        run_dir=args.run,
        output_dir=args.output,
        device=args.device,
    )


if __name__ == '__main__':
    main()
