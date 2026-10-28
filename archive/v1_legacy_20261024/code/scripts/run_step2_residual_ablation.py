"""Step 2 Ablation: Train and evaluate Residual Flow Matching with fixed 2-layer MLP architecture."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dataset.flow_features import FlowFeaturesDataset
from mini_wa_jepa.geometry import wrap_angle
from mini_wa_jepa.normalization import Normalizer
from models.flow_matcher_residual import ConditionalResidualFlowMatcher
from configs.stage7 import Stage7Config


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


def check_kinematics(candidates_si):
    """Compute waypoint-level kinematic compliance percentage.
    Speed in [0, 15] m/s, |accel| <= 8 m/s^2, heading deviation <= 0.35 rad on moving points.
    """
    # candidates_si: [K, 10, 4]
    speeds = candidates_si[..., 2]
    speed_valid = (speeds >= 0.0) & (speeds <= 15.0)

    # Waypoint accel along trajectory (dt = 0.2s)
    accels = np.zeros_like(speeds)
    accels[:, 1:] = np.abs(speeds[:, 1:] - speeds[:, :-1]) / 0.2
    accel_valid = accels <= 8.0

    # Heading consistency with velocity vector
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
    p.add_argument('--output', type=Path, default=Path('artifacts/restart_stage7/ablation_step2_residual'))
    p.add_argument('--seed', type=int, default=20261021)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"=== Step 2 Ablation: Residual Flow Matching on {device} ===")
    print(f"Loading data from {args.data}...")

    train_dataset = FlowFeaturesDataset(args.data, split='train')
    val_dataset = FlowFeaturesDataset(args.data, split='val')
    test_dataset = FlowFeaturesDataset(args.data, split='test')
    normalizer = train_dataset.normalizer

    # 1. Compute empirical residual std from train split
    train_targets = train_dataset.targets
    train_mse = train_dataset.mse_predictions
    diff = train_targets - train_mse
    residual_std_np = diff.std(axis=0)  # [10, 4]
    residual_std = torch.tensor(residual_std_np, dtype=torch.float32)
    np.save(args.output / 'residual_std.npy', residual_std_np)
    print(f"Residual std computed: min={residual_std_np.min():.4f}, max={residual_std_np.max():.4f}")

    # 2. Instantiate ConditionalResidualFlowMatcher
    model = ConditionalResidualFlowMatcher(
        token_count=64,
        token_dim=256,
        condition_dim=256,
        hidden_dim=256,
        residual_std=residual_std,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,} (Identical to Stage 7 baseline)")

    # 3. Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    best_val_loss = float('inf')
    t_train_0 = time.perf_counter()

    # Precompute fixed validation noise
    gen_val = torch.Generator().manual_seed(42)
    val_fixed_noise = [torch.randn(len(batch['target']), 10, 4, generator=gen_val) for batch in val_loader]
    val_fixed_t = [torch.rand(len(batch['target']), generator=gen_val) for batch in val_loader]

    print(f"\n--- Training for {args.epochs} epochs ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            history = batch['history'].to(device)
            ego = batch['ego'].to(device)
            target = batch['target'].to(device)
            mse_pred = batch['mse_prediction'].to(device)
            action_mask = batch['action_mask'].to(device)
            waypoint_valid = batch['waypoint_valid'].to(device)

            optimizer.zero_grad()
            with sdpa_kernel(SDPBackend.MATH):
                condition = model.extract_condition(history, ego)
                loss, _ = model.compute_loss(
                    x_1=target,
                    x_mse=mse_pred,
                    condition=condition,
                    action_mask=action_mask,
                    waypoint_valid=waypoint_valid,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for b_idx, batch in enumerate(val_loader):
                history = batch['history'].to(device)
                ego = batch['ego'].to(device)
                target = batch['target'].to(device)
                mse_pred = batch['mse_prediction'].to(device)
                action_mask = batch['action_mask'].to(device)
                waypoint_valid = batch['waypoint_valid'].to(device)

                with sdpa_kernel(SDPBackend.MATH):
                    condition = model.extract_condition(history, ego)
                    val_loss, _ = model.compute_loss(
                        x_1=target,
                        x_mse=mse_pred,
                        condition=condition,
                        action_mask=action_mask,
                        waypoint_valid=waypoint_valid,
                        noise=val_fixed_noise[b_idx].to(device),
                        t=val_fixed_t[b_idx].to(device),
                    )
                val_losses.append(val_loss.item())

        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses)

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            torch.save(dict(model=model.state_dict(), model_arch=model.model_arch, epoch=epoch, val_loss=best_val_loss), args.output / 'best.pt')
            marker = ' * (best)'
        else:
            marker = ''

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {mean_train:.5f} | Val Loss: {mean_val:.5f}{marker}")

    train_elapsed = time.perf_counter() - t_train_0
    print(f"Training completed in {train_elapsed:.2f}s. Best val loss: {best_val_loss:.5f}")

    # 4. Evaluation on formal 2048 test anchors
    print(f"\n--- Evaluating on formal test split (2,048 anchors) ---")
    best_ckpt = torch.load(args.output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model'])
    model.eval()

    n_test = len(test_dataset)

    for temp in [1.0, 0.5, 0.0]:
        c1_ades = []
        c1_fdes = []
        mean_ades = []
        mean_fdes = []
        mse_ades = []
        mse_fdes = []
        oracle_min_ades = []
        diversities = []
        kinematic_passes = []

        all_c1_errors = []
        all_mean_errors = []
        all_mse_errors = []

        with sdpa_kernel(SDPBackend.MATH):
            for i in range(n_test):
                item = test_dataset[i]
                history = item['history'].unsqueeze(0).to(device)
                ego = item['ego'].unsqueeze(0).to(device)
                gt_norm = item['target'].numpy()
                gt_si = normalizer.inverse_trajectory(gt_norm)

                mse_norm = item['mse_prediction'].unsqueeze(0).to(device)
                mse_si = normalizer.inverse_trajectory(item['mse_prediction'].numpy())

                with torch.no_grad():
                    condition = model.extract_condition(history, ego)
                    cands_norm = model.sample(
                        condition=condition,
                        x_mse=mse_norm,
                        num_steps=10,
                        num_candidates=8,
                        temperature=temp,
                        seed=42 + int(item['anchor']),
                        device=device,
                    )

                cands_np = cands_norm.squeeze(0).cpu().numpy()  # [8, 10, 4]
                cands_si = normalizer.inverse_trajectory(cands_np)

                # Errors
                mse_err = np.linalg.norm(mse_si[:, :2] - gt_si[:, :2], axis=-1)
                c1_err = np.linalg.norm(cands_si[0, :, :2] - gt_si[:, :2], axis=-1)
                mean_cand_si = cands_si.mean(axis=0)
                mean_err = np.linalg.norm(mean_cand_si[:, :2] - gt_si[:, :2], axis=-1)

                all_c_errs = [np.linalg.norm(cands_si[k, :, :2] - gt_si[:, :2], axis=-1).mean() for k in range(8)]

                c1_ades.append(float(c1_err.mean()))
                c1_fdes.append(float(c1_err[-1]))
                mean_ades.append(float(mean_err.mean()))
                mean_fdes.append(float(mean_err[-1]))
                mse_ades.append(float(mse_err.mean()))
                mse_fdes.append(float(mse_err[-1]))
                oracle_min_ades.append(float(min(all_c_errs)))

                all_c1_errors.append(c1_err)
                all_mean_errors.append(mean_err)
                all_mse_errors.append(mse_err)

                diversities.append(compute_pairwise_diversity(cands_si))
                kinematic_passes.append(check_kinematics(cands_si))

        mean_c1_by_step = np.mean(all_c1_errors, axis=0)
        mean_mean_by_step = np.mean(all_mean_errors, axis=0)
        mean_mse_by_step = np.mean(all_mse_errors, axis=0)

        print(f"\n========================================================")
        print(f" RESULTS FOR TEMPERATURE tau = {temp:.1f}")
        print(f"========================================================")
        print(f"Candidate 1 ADE / FDE:       {np.mean(c1_ades):.4f} m / {np.mean(c1_fdes):.4f} m")
        print(f"Mean Trajectory ADE / FDE:   {np.mean(mean_ades):.4f} m / {np.mean(mean_fdes):.4f} m")
        print(f"Oracle Min ADE (k=8):        {np.mean(oracle_min_ades):.4f} m")
        print(f"MSE Baseline ADE / FDE:      {np.mean(mse_ades):.4f} m / {np.mean(mse_fdes):.4f} m")
        print(f"Pairwise Diversity:          {np.mean(diversities):.4f} m")
        print(f"Kinematic Feasibility:       {np.mean(kinematic_passes)*100:.2f}%")
        print(f"\nWaypoint 1 (0.2s) Error: C1={mean_c1_by_step[0]:.4f}m | Mean={mean_mean_by_step[0]:.4f}m | MSE={mean_mse_by_step[0]:.4f}m")
        print(f"Waypoint 10 (2.0s) Error: C1={mean_c1_by_step[-1]:.4f}m | Mean={mean_mean_by_step[-1]:.4f}m | MSE={mean_mse_by_step[-1]:.4f}m")

        # Save metrics for this temperature
        metrics_dict = dict(
            temperature=temp,
            c1_ade=float(np.mean(c1_ades)),
            c1_fdes=float(np.mean(c1_fdes)),
            mean_ade=float(np.mean(mean_ades)),
            mean_fdes=float(np.mean(mean_fdes)),
            oracle_min_ade=float(np.mean(oracle_min_ades)),
            mse_ade=float(np.mean(mse_ades)),
            diversity=float(np.mean(diversities)),
            kinematic_feasibility=float(np.mean(kinematic_passes)),
            step_c1_errors=mean_c1_by_step.tolist(),
            step_mean_errors=mean_mean_by_step.tolist(),
            step_mse_errors=mean_mse_by_step.tolist(),
        )
        with open(args.output / f'metrics_temp_{temp:.1f}.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)


if __name__ == '__main__':
    main()
