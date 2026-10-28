"""Calibrate and freeze temperature tau strictly on the 2,048-anchor Validation Split."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dataset.flow_features import FlowFeaturesDataset
from mini_wa_jepa.geometry import wrap_angle
from mini_wa_jepa.normalization import Normalizer
from models.flow_matcher_transformer import TransformerKinematicFlowMatcher


def compute_pairwise_diversity(candidates_si):
    k = len(candidates_si)
    if k < 2:
        return 0.0
    pos = candidates_si[..., :2]  # [K, 10, 2]
    diffs = pos[:, None, :, :] - pos[None, :, :, :]  # [K, K, 10, 2]
    dist_matrix = np.linalg.norm(diffs, axis=-1).mean(axis=-1)  # [K, K]
    upper_tri = np.triu_indices(k, k=1)
    return float(dist_matrix[upper_tri].mean())


def check_kinematics(candidates_si):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('artifacts/restart_stage7/flow_features_001'))
    p.add_argument('--model-ckpt', type=Path, default=Path('artifacts/restart_stage7/ablation_step3_transformer/best.pt'))
    p.add_argument('--output', type=Path, default=Path('artifacts/restart_stage7/temperature_calibration_val'))
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--nominal-mode', action='store_true', help='Set candidate 0 to deterministic mode (tau=0), while candidates 1..k-1 explore at tau')
    p.add_argument('--tau-grid', type=float, nargs='+', default=[0.6, 0.7, 0.8, 0.85, 0.9, 1.0])
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"=== Validating temperature strictly on VALIDATION split (2,048 anchors) ===")
    print(f"Data: {args.data}")
    print(f"Checkpoint: {args.model_ckpt}")
    print(f"Device: {device}")

    # Load validation split ONLY
    val_dataset = FlowFeaturesDataset(args.data, split='val')
    normalizer = val_dataset.normalizer
    n_val = len(val_dataset)
    print(f"Loaded {n_val} validation anchors. Test split is NOT touched.")

    ckpt = torch.load(args.model_ckpt, map_location=device, weights_only=False)
    model = TransformerKinematicFlowMatcher(**ckpt['model_arch'], normalizer=normalizer).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # 1. Compute validation MSE baseline ADE & FDE
    mse_ades = []
    mse_fdes = []
    for i in range(n_val):
        item = val_dataset[i]
        gt_si = normalizer.inverse_trajectory(item['target'].numpy())
        mse_si = normalizer.inverse_trajectory(item['mse_prediction'].numpy())
        err = np.linalg.norm(mse_si[:, :2] - gt_si[:, :2], axis=-1)
        mse_ades.append(float(err.mean()))
        mse_fdes.append(float(err[-1]))

    val_mse_ade = float(np.mean(mse_ades))
    val_mse_fde = float(np.mean(mse_fdes))
    ade_threshold = 1.25 * val_mse_ade
    fde_threshold = 1.25 * val_mse_fde

    print(f"\n--- Validation Baseline Reference (MSE) ---")
    print(f"Val MSE ADE: {val_mse_ade:.4f} m (Threshold <= 1.25x = {ade_threshold:.4f} m)")
    print(f"Val MSE FDE: {val_mse_fde:.4f} m (Threshold <= 1.25x = {fde_threshold:.4f} m)")
    print(f"Diversity Threshold: >= 0.1000 m")
    print(f"Kinematic Feasibility Threshold: >= 95.00%")
    print(f"Head Latency P95 Threshold: <= 30.00 ms")

    # 2. Grid sweep over candidate temperatures
    tau_grid = args.tau_grid
    results = {}

    for tau in tau_grid:
        c1_ades = []
        c1_fdes = []
        diversities = []
        kinematic_passes = []
        latencies_ms = []

        with sdpa_kernel(SDPBackend.MATH):
            for i in range(n_val):
                item = val_dataset[i]
                h = item['history'].unsqueeze(0).to(device)
                e = item['ego'].unsqueeze(0).to(device)
                gt_si = normalizer.inverse_trajectory(item['target'].numpy())

                # Measure isolated head-only sampling latency (10 steps, 8 candidates)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                with torch.no_grad():
                    cands_norm = model.sample(
                        history=h,
                        ego=e,
                        num_steps=10,
                        num_candidates=8,
                        temperature=tau,
                        nominal_mode=args.nominal_mode,
                        seed=42 + int(item['anchor']),
                        device=device,
                    )

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                lat_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(lat_ms)

                cands_np = cands_norm.squeeze(0).cpu().numpy()
                cands_si = normalizer.inverse_trajectory(cands_np)

                c1_err = np.linalg.norm(cands_si[0, :, :2] - gt_si[:, :2], axis=-1)
                c1_ades.append(float(c1_err.mean()))
                c1_fdes.append(float(c1_err[-1]))

                diversities.append(compute_pairwise_diversity(cands_si))
                kinematic_passes.append(check_kinematics(cands_si))

        mean_c1_ade = float(np.mean(c1_ades))
        mean_c1_fde = float(np.mean(c1_fdes))
        mean_div = float(np.mean(diversities))
        mean_kin = float(np.mean(kinematic_passes))
        p95_lat = float(np.percentile(latencies_ms, 95))

        pass_ade = mean_c1_ade <= ade_threshold
        pass_fde = mean_c1_fde <= fde_threshold
        pass_div = mean_div >= 0.10
        pass_kin = mean_kin >= 0.95
        pass_lat = p95_lat <= 30.0

        all_pass = pass_ade and pass_fde and pass_div and pass_kin and pass_lat

        res = dict(
            tau=tau,
            c1_ade=mean_c1_ade,
            c1_fde=mean_c1_fde,
            diversity=mean_div,
            kinematic_feasibility=mean_kin,
            head_latency_p95_ms=p95_lat,
            pass_ade=pass_ade,
            pass_fde=pass_fde,
            pass_diversity=pass_div,
            pass_kinematics=pass_kin,
            pass_latency=pass_lat,
            all_pass=all_pass,
        )
        results[str(tau)] = res

        status = "ALL PASS" if all_pass else "FAIL"
        print(f"tau={tau:.1f} | ADE: {mean_c1_ade:.4f}m ({'P' if pass_ade else 'F'}) | "
              f"FDE: {mean_c1_fde:.4f}m ({'P' if pass_fde else 'F'}) | "
              f"Div: {mean_div:.4f}m ({'P' if pass_div else 'F'}) | "
              f"Kin: {mean_kin*100:.1f}% ({'P' if pass_kin else 'F'}) | "
              f"LatP95: {p95_lat:.1f}ms ({'P' if pass_lat else 'F'}) => {status}")

    # 3. Decision rule: Select tau that passes all 5 criteria on validation set
    passing_taus = [r for r in results.values() if r['all_pass']]
    if passing_taus:
        # Choose passing tau with highest diversity (encouraging multi-modal exploration within safe bounds)
        best_choice = max(passing_taus, key=lambda x: x['diversity'])
        chosen_tau = best_choice['tau']
        print(f"\nSelection: tau* = {chosen_tau} passed all 5 criteria on the validation set.")
    else:
        # Fallback to closest Pareto optimal
        best_choice = min(results.values(), key=lambda x: x['c1_ade'] / ade_threshold + 0.1 / max(x['diversity'], 1e-4))
        chosen_tau = best_choice['tau']
        print(f"\nNo tau passed all 5 criteria. Best candidate: tau* = {chosen_tau}")

    summary = dict(
        data=str(args.data),
        model_checkpoint=str(args.model_ckpt),
        nominal_mode=args.nominal_mode,
        validation_anchors=n_val,
        val_mse_ade=val_mse_ade,
        val_mse_fde=val_mse_fde,
        ade_threshold_1_25x=ade_threshold,
        fde_threshold_1_25x=fde_threshold,
        chosen_tau=chosen_tau,
        chosen_metrics=best_choice,
        all_results=results,
    )

    out_file = args.output / 'calibration_summary.json'
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nCalibration artifact saved to {out_file}")


if __name__ == '__main__':
    main()
