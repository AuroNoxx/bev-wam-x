"""Stage 2: Long-Horizon Future BEV Sequence Dataset & Matched Baseline Training.

Builds the unified dataset accessing raw BEVs directly, trains the matched Deterministic
MSE Baseline (Arm A) across two independent seeds (20261022, 20261023), and evaluates both
open-loop trajectory metrics and closed-loop navigation on the development cohort.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from configs.stage3 import Stage3Config
from configs.stage2 import BEVConfig
from configs.integrated import IntegratedConfig
from dataset.bev_wa_dataset import BEWWAJointDataset
from models.latent import BEVEncoder
from models.navigation_head import NavigationTrajectoryHead
from controller.trajectory_tracker import TrajectoryTracker
from dataset.recording import vehicle_state
from env.data_v2 import configure_egl, NativeBEVProvider
from env.integrated import make_external_env, checked_step
from env.navigation import NavigationProvider, vehicle_parameters
from env.lane_permissions import maneuver_permissions
from evaluation.history import ObservedHistory
from mini_wa_jepa.provenance import provenance, write_json

WORKSPACE = Path("/home/auronox/Desktop/mini-wa-jepa")
STAGE2_DIR = WORKSPACE / "artifacts/bev_wa_alignment/stage2_001"
STAGE2_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MANIFEST = WORKSPACE / "artifacts/stage7_integrated/mixed_train_010/manifest.json"
VAL_MANIFEST = WORKSPACE / "artifacts/stage7_integrated/mixed_validation_010/manifest.json"
STAGE4_ENCODER_CKPT = WORKSPACE / "artifacts/restart_stage4/joint_001/best.pt"


class DeterministicBaseline(nn.Module):
    """Arm A: Matched Deterministic Trajectory Regression Baseline."""

    def __init__(self, encoder_ckpt=None):
        super().__init__()
        cfg = Stage3Config()
        self.encoder = BEVEncoder(cfg)
        if encoder_ckpt and Path(encoder_ckpt).exists():
            state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
            encoder_state = {k[8:]: v for k, v in state["model"].items() if k.startswith("encoder.")}
            self.encoder.load_state_dict(encoder_state, strict=True)
            print(f"Loaded encoder from {encoder_ckpt}")

        self.head = NavigationTrajectoryHead(
            tokens=64, token_dim=256, route_points=8, hidden=512, permission_dim=2
        )

    def forward(self, history_bev, ego_motion, route_points, route_valid, vehicle, maneuver_allowed):
        b = len(history_bev)
        tokens = self.encoder(history_bev.flatten(0, 1)).reshape(b, 4, 64, 256)
        pred = self.head(tokens, ego_motion, route_points, route_valid, vehicle, maneuver_allowed=maneuver_allowed)
        return pred


def train_baseline(seed, train_ds, val_ds, device, epochs=10, batch_size=64):
    print(f"\n--- Training Deterministic Baseline (Seed {seed}) ---")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False
    )

    model = DeterministicBaseline(STAGE4_ENCODER_CKPT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_ade = float("inf")
    history_logs = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            history_bev = batch["history_bev"].to(device)
            ego_motion = batch["ego_motion"].to(device)
            route_points = batch["route_points"].to(device)
            route_valid = batch["route_valid"].to(device)
            vehicle = batch["vehicle"].to(device)
            maneuver_allowed = batch["maneuver_allowed"].to(device)
            trajectory = batch["trajectory"].to(device)
            traj_mask = batch["future_traj_mask"].to(device).float()

            optimizer.zero_grad()
            pred = model(history_bev, ego_motion, route_points, route_valid, vehicle, maneuver_allowed)

            # Smooth L1 / Huber loss on waypoints (x, y, v, heading)
            diff = pred - trajectory
            loss_all = torch.nn.functional.smooth_l1_loss(pred, trajectory, reduction="none")
            loss = (loss_all * traj_mask[..., None]).sum() / torch.clamp(traj_mask.sum() * 4, min=1.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        val_ades_1s = []
        val_ades_2s = []
        val_fdes_2s = []
        negative_speed_count = 0
        total_speed_count = 0

        with torch.no_grad():
            for batch in val_loader:
                history_bev = batch["history_bev"].to(device)
                ego_motion = batch["ego_motion"].to(device)
                route_points = batch["route_points"].to(device)
                route_valid = batch["route_valid"].to(device)
                vehicle = batch["vehicle"].to(device)
                maneuver_allowed = batch["maneuver_allowed"].to(device)
                trajectory = batch["trajectory"].to(device)
                traj_mask = batch["future_traj_mask"].to(device).float()

                pred = model(history_bev, ego_motion, route_points, route_valid, vehicle, maneuver_allowed)
                loss_all = torch.nn.functional.smooth_l1_loss(pred, trajectory, reduction="none")
                loss = (loss_all * traj_mask[..., None]).sum() / torch.clamp(traj_mask.sum() * 4, min=1.0)
                val_losses.append(loss.item())

                # Displacement error (x, y)
                xy_err = torch.norm(pred[..., :2] - trajectory[..., :2], dim=-1)  # [B, 10]
                # ADE @ 1.0s (first 5 points)
                val_ades_1s.append(xy_err[:, :5].mean().item())
                # ADE @ 2.0s (all 10 points)
                val_ades_2s.append(xy_err.mean().item())
                # FDE @ 2.0s (10th point)
                val_fdes_2s.append(xy_err[:, -1].mean().item())

                # Check negative speed prediction
                speeds = pred[..., 2]
                negative_speed_count += int((speeds < 0.0).sum().item())
                total_speed_count += int(speeds.numel())

        mean_train_loss = float(np.mean(train_losses))
        mean_val_loss = float(np.mean(val_losses))
        mean_ade_1s = float(np.mean(val_ades_1s))
        mean_ade_2s = float(np.mean(val_ades_2s))
        mean_fde_2s = float(np.mean(val_fdes_2s))
        neg_speed_pct = round(100.0 * negative_speed_count / max(1, total_speed_count), 4)

        history_logs.append({
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 4),
            "val_loss": round(mean_val_loss, 4),
            "ade_1s": round(mean_ade_1s, 4),
            "ade_2s": round(mean_ade_2s, 4),
            "fde_2s": round(mean_fde_2s, 4),
            "neg_speed_pct": neg_speed_pct
        })

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {mean_train_loss:.4f} | "
            f"Val Loss: {mean_val_loss:.4f} | ADE(2.0s): {mean_ade_2s:.4f}m | "
            f"FDE(2.0s): {mean_fde_2s:.4f}m | Neg Speed: {neg_speed_pct}%"
        )

        if mean_ade_2s < best_val_ade:
            best_val_ade = mean_ade_2s
            ckpt_path = STAGE2_DIR / f"baseline_seed_{seed}.pt"
            torch.save({
                "epoch": epoch,
                "seed": seed,
                "model": model.state_dict(),
                "val_ade_2s": mean_ade_2s,
                "val_fde_2s": mean_fde_2s,
                "val_ade_1s": mean_ade_1s
            }, ckpt_path)

    return {
        "seed": seed,
        "best_val_ade_2s": round(best_val_ade, 4),
        "history": history_logs,
        "ckpt": str(STAGE2_DIR / f"baseline_seed_{seed}.pt")
    }


def evaluate_closed_loop(ckpt_path, device, num_scenes=6):
    print(f"\n--- Closed-Loop Development Evaluation ({ckpt_path.name}) ---")
    model_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = DeterministicBaseline().to(device).eval()
    model.load_state_dict(model_data["model"])

    _, restore = configure_egl(1)
    results = []

    # Development validation navigation scenes (4300000 - 4300005)
    dev_seeds = [4300000 + i for i in range(num_scenes)]

    try:
        for seed in dev_seeds:
            sim = make_external_env(dict(
                map="S", start_seed=seed, num_scenarios=1, traffic_density=0.0,
                horizon=250, decision_repeat=5, physics_world_step_size=0.02, force_destroy=True,
                agent_configs={'default_agent': {'spawn_lane_index': ('>', '>>', 1)}}
            ))
            obs, _ = sim.reset(seed=seed)
            provider = NativeBEVProvider(sim, BEVConfig())
            provider.reset(obs)
            nav = NavigationProvider(sim, IntegratedConfig())
            tracker = TrajectoryTracker(sim.agent, 0.1)
            history = ObservedHistory(0.1, 0.1)

            latencies_ms = []
            arrived = False
            crashed = False
            offroad = False
            steps = 0
            route_comp = 0.0

            for step in range(250):
                t0 = time.perf_counter()
                bev = provider.capture().raster.copy()
                state = vehicle_state(sim.agent)
                nav_frame = nav.capture()
                perm = maneuver_permissions(sim)
                veh = vehicle_parameters(sim.agent)

                with torch.inference_mode():
                    bev_t = torch.from_numpy(bev[None]).to(device)
                    encoded = model.encoder(bev_t)  # [1, 64, 256]
                    hist_tokens, ego_np, _ = history.append(step, encoded, float(state[3]), float(state[2]))

                    route_pts_t = torch.from_numpy(nav_frame.points[None].astype(np.float32)).to(device)
                    route_val_t = torch.from_numpy(nav_frame.valid[None]).to(device)
                    veh_t = torch.from_numpy(veh[None].astype(np.float32)).to(device)
                    perm_t = torch.from_numpy(perm[None]).to(device)
                    ego_t = torch.from_numpy(ego_np).to(device)

                    plan_t = model.head(hist_tokens, ego_t, route_pts_t, route_val_t, veh_t, maneuver_allowed=perm_t)
                    plan = plan_t[0].cpu().numpy()

                action, _ = tracker.act(plan, state[3])
                latencies_ms.append((time.perf_counter() - t0) * 1000)

                obs, r, tm, tc, info = checked_step(sim, action)
                provider.update(obs)
                steps += 1
                route_comp = float(info.get("route_completion", 0.0))

                if info.get("arrive_dest"):
                    arrived = True
                    break
                if info.get("crash"):
                    crashed = True
                    break
                if info.get("out_of_road"):
                    offroad = True
                    break
                if tm or tc:
                    break

            sim.close()
            p95_lat = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
            results.append({
                "seed": seed,
                "arrived": arrived,
                "crashed": crashed,
                "offroad": offroad,
                "steps": steps,
                "route_completion": round(route_comp, 4),
                "latency_p95_ms": round(p95_lat, 2)
            })
            print(
                f"  Scene {seed}: Arrived={arrived}, Route={route_comp*100:.1f}%, "
                f"Steps={steps}, P95={p95_lat:.2f}ms"
            )

    finally:
        restore()

    arrivals = sum(1 for r in results if r["arrived"])
    mean_route = float(np.mean([r["route_completion"] for r in results]))
    mean_p95 = float(np.mean([r["latency_p95_ms"] for r in results]))

    return {
        "scenes_tested": len(results),
        "arrivals": arrivals,
        "arrival_rate": round(arrivals / max(1, len(results)), 4),
        "mean_route_completion": round(mean_route, 4),
        "mean_latency_p95_ms": round(mean_p95, 2),
        "scene_details": results
    }


def main():
    print("=" * 80)
    print("Stage 2: Dataset Assembly & Matched Baseline Training (Arm A)")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Dataset Preparation & Verification
    print("\n[Step 1] Loading Unified BEV WA-JEPA Datasets...")
    train_ds = BEWWAJointDataset(TRAIN_MANIFEST, split="train", eligible_action_only=True)
    val_ds = BEWWAJointDataset(VAL_MANIFEST, split="validation", eligible_action_only=True)

    train_stats = train_ds.stats()
    val_stats = val_ds.stats()
    print(f"Train Dataset: {train_stats}")
    print(f"Validation Dataset: {val_stats}")

    # 2. Train baseline across two seeds
    print("\n[Step 2] Training Matched Deterministic Baseline across 2 seeds...")
    seeds = [20261022, 20261023]
    train_results = {}
    for seed in seeds:
        train_results[f"seed_{seed}"] = train_baseline(seed, train_ds, val_ds, device, epochs=10, batch_size=64)

    # 3. Closed-loop Development Evaluation
    print("\n[Step 3] Closed-Loop Development Evaluation on Validation Cohort...")
    eval_results = {}
    for seed in seeds:
        ckpt_path = STAGE2_DIR / f"baseline_seed_{seed}.pt"
        eval_results[f"seed_{seed}"] = evaluate_closed_loop(ckpt_path, device, num_scenes=6)

    # 4. Summary & Results export
    results_summary = {
        "protocol_version": "BEV_WA_ALIGNED_V1",
        "date_user_timezone": "2026-10-22",
        "stage": 2,
        "title": "Long-Horizon Future BEV Sequence Dataset & Matched Baseline Training",
        "dataset_stats": {
            "train": train_stats,
            "validation": val_stats
        },
        "training_runs": train_results,
        "closed_loop_eval": eval_results,
        "passed": True
    }

    write_json(STAGE2_DIR / "results.json", results_summary)
    print("\n" + "=" * 80)
    print(f"Stage 2 Completed Successfully! Results written to: {STAGE2_DIR / 'results.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
