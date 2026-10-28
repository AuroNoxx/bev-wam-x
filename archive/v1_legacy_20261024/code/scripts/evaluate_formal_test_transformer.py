#!/usr/bin/env python3
"""Formal blind test evaluation script for Stage 7 Transformer Flow Matcher.

Strictly runs on held-out test split (2,048 anchors).
Outputs complete formal audit artifacts:
- summary.json (Gate 1 verdict, MSE baseline, C1 ADE/FDE, diversity, kinematics, head P95 latency)
- per_anchor_metrics.json (per-sample breakdown for independent auditing)
- raw_eval.npz (full numpy arrays for candidates, ground truths, latencies)
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from dataset.flow_features import FlowFeaturesDataset
from mini_wa_jepa.provenance import provenance, write_json
from models.flow_matcher_transformer import TransformerKinematicFlowMatcher


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def wrap_angle(theta):
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def compute_pairwise_diversity(candidates_si):
    """Compute average pairwise Euclidean distance across 8 candidates over all 10 waypoints."""
    k = candidates_si.shape[0]
    pos = candidates_si[..., :2]  # [K, 10, 2]
    diffs = pos[:, None, :, :] - pos[None, :, :, :]  # [K, K, 10, 2]
    dists = np.linalg.norm(diffs, axis=-1).mean(axis=-1)  # [K, K]
    upper_tri = np.triu_indices(k, k=1)
    return float(dists[upper_tri].mean())


def check_kinematics(candidates_si):
    """Check kinematic feasibility on candidates (speed [0, 15], |a| <= 8, heading dev <= 0.35 rad)."""
    speeds = candidates_si[..., 2]
    speed_valid = (speeds >= 0.0) & (speeds <= 15.0)

    accels = np.zeros_like(speeds)
    accels[:, 1:] = np.abs(speeds[:, 1:] - speeds[:, :-1]) / 0.2
    accel_valid = accels <= 8.0

    headings = candidates_si[..., 3]
    dpos = np.zeros_like(candidates_si[..., :2])
    dpos[:, 1:] = candidates_si[:, 1:, :2] - candidates_si[:, :-1, :2]
    move_speed = np.linalg.norm(dpos, axis=-1) / 0.2
    motion_angle = np.arctan2(dpos[..., 1], dpos[..., 0])
    heading_diff = np.abs(wrap_angle(motion_angle - headings))
    heading_valid = (move_speed < 1.0) | (heading_diff <= 0.35)

    all_valid = speed_valid & accel_valid & heading_valid
    return float(all_valid.mean())


def evaluate_formal_test(data_dir: Path, ckpt_path: Path, output_dir: Path, temperature: float, nominal_mode: bool, device_str: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    print("=== Formal Test Set Evaluation for Stage 7 Transformer Flow Matcher ===")
    print(f"Data directory:    {data_dir}")
    print(f"Model Checkpoint:  {ckpt_path}")
    print(f"Output directory:  {output_dir}")
    print(f"Temperature tau*:  {temperature}")
    print(f"Nominal Mode (C0): {nominal_mode}")
    print(f"Device:            {device}")

    # Load test split strictly
    test_dataset = FlowFeaturesDataset(data_dir, split='test')
    normalizer = test_dataset.normalizer
    n_test = len(test_dataset)
    print(f"Loaded {n_test} held-out test anchors.", flush=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TransformerKinematicFlowMatcher(**ckpt['model_arch'], normalizer=normalizer).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Metric accumulators
    mse_ades = []
    mse_fdes = []
    fm_c1_ades = []
    fm_c1_fdes = []
    oracle_min_ades = []
    diversities = []
    kinematic_feasibilities = []
    fm_latencies_ms = []

    # Storage arrays for raw artifact
    all_candidates_si = np.empty((n_test, 8, 10, 4), dtype=np.float32)
    all_gt_si = np.empty((n_test, 10, 4), dtype=np.float32)
    all_mse_si = np.empty((n_test, 10, 4), dtype=np.float32)
    all_waypoint_valids = np.empty((n_test, 10), dtype=bool)
    all_anchors = np.empty((n_test,), dtype=np.int64)
    all_episodes = np.empty((n_test,), dtype=np.int64)
    all_steps = np.empty((n_test,), dtype=np.int64)

    per_anchor_records = []

    print(f"Running inference across {n_test} test anchors...", flush=True)

    with sdpa_kernel(SDPBackend.MATH):
        for i in range(n_test):
            item = test_dataset[i]
            history = item['history'].unsqueeze(0).to(device)
            ego = item['ego'].unsqueeze(0).to(device)
            gt_norm = item['target'].numpy()
            gt_si = normalizer.inverse_trajectory(gt_norm)

            mse_norm = item['mse_prediction'].numpy()
            mse_si = normalizer.inverse_trajectory(mse_norm)

            # Isolated Head Latency Measurement
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                cands_norm = model.sample(
                    history=history,
                    ego=ego,
                    num_steps=10,
                    num_candidates=8,
                    temperature=temperature,
                    nominal_mode=nominal_mode,
                    seed=42 + int(item['anchor']),
                    device=device,
                )

            if device.type == 'cuda':
                torch.cuda.synchronize()
            lat_ms = (time.perf_counter() - t0) * 1000.0
            fm_latencies_ms.append(lat_ms)

            cands_np = cands_norm.squeeze(0).cpu().numpy()
            if not np.isfinite(cands_np).all():
                raise FloatingPointError(f"Non-finite output generated at anchor index {i}")

            cands_si = normalizer.inverse_trajectory(cands_np)

            # Store raw
            all_candidates_si[i] = cands_si
            all_gt_si[i] = gt_si
            all_mse_si[i] = mse_si
            all_waypoint_valids[i] = item['waypoint_valid'].numpy()
            all_anchors[i] = item['anchor']
            all_episodes[i] = item['episode']
            all_steps[i] = item['step']

            # MSE Baseline errors
            mse_err = np.linalg.norm(mse_si[:, :2] - gt_si[:, :2], axis=-1)
            mse_ades.append(float(mse_err.mean()))
            mse_fdes.append(float(mse_err[-1]))

            # Flow Matching Candidate 1 errors
            c1_err = np.linalg.norm(cands_si[0, :, :2] - gt_si[:, :2], axis=-1)
            fm_c1_ades.append(float(c1_err.mean()))
            fm_c1_fdes.append(float(c1_err[-1]))

            # Oracle min ADE
            oracle_ades = [np.linalg.norm(cands_si[k, :, :2] - gt_si[:, :2], axis=-1).mean() for k in range(8)]
            oracle_min_ades.append(float(min(oracle_ades)))

            # Diversity & Kinematics
            div_m = compute_pairwise_diversity(cands_si)
            diversities.append(div_m)
            kin_val = check_kinematics(cands_si)
            kinematic_feasibilities.append(kin_val)

            per_anchor_records.append(dict(
                index=i,
                anchor=int(item['anchor']),
                episode=int(item['episode']),
                step=int(item['step']),
                mse_ade=float(mse_err.mean()),
                mse_fde=float(mse_err[-1]),
                fm_c1_ade=float(c1_err.mean()),
                fm_c1_fde=float(c1_err[-1]),
                oracle_min_ade=float(min(oracle_ades)),
                diversity_m=div_m,
                kinematic_feasibility=kin_val,
                head_latency_ms=lat_ms,
            ))

    mean_mse_ade = float(np.mean(mse_ades))
    mean_mse_fde = float(np.mean(mse_fdes))
    ade_threshold = 1.25 * mean_mse_ade
    fde_threshold = 1.25 * mean_mse_fde

    mean_c1_ade = float(np.mean(fm_c1_ades))
    mean_c1_fde = float(np.mean(fm_c1_fdes))
    mean_diversity = float(np.mean(diversities))
    mean_kinematics = float(np.mean(kinematic_feasibilities))
    p95_latency = float(np.percentile(fm_latencies_ms, 95))
    p50_latency = float(np.percentile(fm_latencies_ms, 50))
    mean_oracle_ade = float(np.mean(oracle_min_ades))

    pass_ade = mean_c1_ade <= ade_threshold
    pass_fde = mean_c1_fde <= fde_threshold
    pass_diversity = mean_diversity >= 0.1000
    pass_kinematics = mean_kinematics >= 0.9500
    pass_latency = p95_latency <= 30.00
    gate1_passed = pass_ade and pass_fde and pass_diversity and pass_kinematics and pass_latency

    prov = provenance()
    summary = dict(
        git_commit=prov['git_commit'],
        git_dirty=prov['git_dirty'],
        goal_sha256=prov['goal_sha256'],
        source_tree_sha256=prov['source_tree_sha256'],
        completed=True,
        evaluated_anchors=n_test,
        checkpoint_path=str(ckpt_path),
        checkpoint_sha256=file_sha256(ckpt_path),
        cache_dir=str(data_dir),
        temperature=temperature,
        nominal_mode=nominal_mode,
        mse_baseline=dict(
            mean_ade_m=mean_mse_ade,
            mean_fde_m=mean_mse_fde,
            ade_threshold_1_25x_m=ade_threshold,
            fde_threshold_1_25x_m=fde_threshold,
        ),
        flow_matching_sample1=dict(
            mean_ade_m=mean_c1_ade,
            mean_fde_m=mean_c1_fde,
            pass_ade=pass_ade,
            pass_fde=pass_fde,
        ),
        flow_matching_oracle_diagnostic=dict(
            mean_min_ade_m=mean_oracle_ade,
            note='Oracle min-of-8 is a diagnostic upper bound and strictly not a deployable planner policy',
        ),
        diversity=dict(
            mean_pairwise_distance_m=mean_diversity,
            p50_diversity_m=float(np.median(diversities)),
            threshold_m=0.1000,
            pass_diversity=pass_diversity,
        ),
        kinematics=dict(
            mean_feasibility=mean_kinematics,
            threshold=0.9500,
            pass_kinematics=pass_kinematics,
        ),
        latency=dict(
            fm_head_p50_ms=p50_latency,
            fm_head_p95_ms=p95_latency,
            threshold_p95_ms=30.00,
            pass_latency=pass_latency,
            note='Head latency measures isolated forward sampling given cached spatial tokens',
        ),
        gate1_verdict=dict(
            pass_ade=pass_ade,
            pass_fde=pass_fde,
            pass_diversity=pass_diversity,
            pass_kinematics=pass_kinematics,
            pass_latency=pass_latency,
            gate1_passed=gate1_passed,
        ),
    )

    # Save full raw npz artifact for independent verification
    raw_npz_path = output_dir / 'raw_eval.npz'
    np.savez_compressed(
        raw_npz_path,
        candidates=all_candidates_si,
        ground_truth=all_gt_si,
        mse_predictions=all_mse_si,
        waypoint_valid=all_waypoint_valids,
        anchors=all_anchors,
        episodes=all_episodes,
        steps=all_steps,
        fm_latencies_ms=np.asarray(fm_latencies_ms, dtype=np.float32),
    )
    print(f"Raw evaluation arrays saved to: {raw_npz_path}")

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Formal summary saved to:        {summary_path}")

    per_anchor_path = output_dir / 'per_anchor_metrics.json'
    with open(per_anchor_path, 'w') as f:
        json.dump(per_anchor_records, f, indent=2)
    print(f"Per-anchor records saved to:    {per_anchor_path}")

    print("\n=================== GATE 1 VERDICT ===================")
    print(f"C1 ADE:     {mean_c1_ade:.4f} m (<= {ade_threshold:.4f} m)  --> {'PASS' if pass_ade else 'FAIL'}")
    print(f"C1 FDE:     {mean_c1_fde:.4f} m (<= {fde_threshold:.4f} m)  --> {'PASS' if pass_fde else 'FAIL'}")
    print(f"Diversity:  {mean_diversity:.4f} m (>= 0.1000 m)  --> {'PASS' if pass_diversity else 'FAIL'}")
    print(f"Kinematics: {mean_kinematics*100:.2f}% (>= 95.00%)     --> {'PASS' if pass_kinematics else 'FAIL'}")
    print(f"Head P95:   {p95_latency:.2f} ms (<= 30.00 ms)   --> {'PASS' if pass_latency else 'FAIL'}")
    print(f"ALL PASS:   {gate1_passed}")
    print("======================================================")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('artifacts/restart_stage7/flow_features_001'))
    p.add_argument('--model-ckpt', type=Path, default=Path('artifacts/restart_stage7/ablation_step3_transformer/best.pt'))
    p.add_argument('--output', type=Path, default=Path('artifacts/restart_stage7/transformer_formal_test'))
    p.add_argument('--temperature', type=float, default=0.85)
    p.add_argument('--nominal-mode', action='store_true', default=True)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    evaluate_formal_test(
        data_dir=args.data,
        ckpt_path=args.model_ckpt,
        output_dir=args.output,
        temperature=args.temperature,
        nominal_mode=args.nominal_mode,
        device_str=args.device,
    )


if __name__ == '__main__':
    main()
