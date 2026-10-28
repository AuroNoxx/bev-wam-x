"""Stage 4 training/evaluation utilities with corrected masked open-loop metrics.

The historical closed-loop function below is retained for evidence/reference only.
Use the independent reevaluate_bev_wa_stage4_open_loop entry point for existing
weights; closed-loop acceptance requires the dedicated classified-scenario runner.
Training requires an explicit mode and a fresh output directory. This script does
not establish matched four-arm input fairness or certify physical demonstrations.
"""

import argparse
from dataclasses import replace
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time

WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs.stage3 import Stage3Config
from configs.stage2 import BEVConfig
from configs.integrated import IntegratedConfig
from dataset.bev_wa_dataset import BEWWAJointDataset, extract_causal_actions
from models.bev_wa_joint_flow import (
    ArmBActionOnlyFlowModel,
    ArmCDecoupledFlowModel,
    ArmDJointFlowModel,
)
from evaluation.bev_wa_metrics import TrajectoryMetrics, render_stage4_report, format_metric
from models.latent import BEVEncoder
from models.navigation_head import NavigationTrajectoryHead
from env.data_v2 import configure_egl, NativeBEVProvider
from env.integrated import make_external_env, checked_step
from env.navigation import NavigationProvider, vehicle_parameters
from env.lane_permissions import maneuver_permissions
from env.traffic_teacher import spawn_slow_actor
from dataset.recording import vehicle_state
from controller.trajectory_tracker import TrajectoryTracker
from mini_wa_jepa.visualization.rerun_logger import RerunLogger
from mini_wa_jepa.provenance import provenance, write_json
from mini_wa_jepa.geometry import wrap_angle

WORKSPACE = Path("/home/auronox/Desktop/mini-wa-jepa")
STAGE4_REMEDIATED_DIR = WORKSPACE / "artifacts/bev_wa_alignment/stage4_remediated"
# Historical directory is read-only unless an explicit fresh output directory is selected.

STAGE4_LEGACY_DIR = WORKSPACE / "artifacts/bev_wa_alignment/stage4_001"
STAGE2_DIR = WORKSPACE / "artifacts/bev_wa_alignment/stage2_001"
STAGE3_DIR = WORKSPACE / "artifacts/bev_wa_alignment/stage3_001"
STAGE4_ENCODER_CKPT = WORKSPACE / "artifacts/restart_stage4/joint_001/best.pt"

TRAIN_MANIFEST = WORKSPACE / "artifacts/stage7_integrated/mixed_train_012/manifest.json"
VAL_MANIFEST = WORKSPACE / "artifacts/stage7_integrated/mixed_validation_012/manifest.json"


class DeterministicBaseline(nn.Module):
    """Arm A: Matched Deterministic Trajectory Regression Baseline."""

    def __init__(self, cfg=None, encoder_ckpt=None):
        super().__init__()
        if isinstance(cfg, (str, Path)) and encoder_ckpt is None:
            encoder_ckpt = cfg
            cfg = None
        if cfg is None:
            cfg = Stage3Config()
        self.encoder = BEVEncoder(cfg)
        if encoder_ckpt and Path(encoder_ckpt).exists():
            state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
            encoder_state = {k[8:]: v for k, v in state["model"].items() if k.startswith("encoder.")}
            self.encoder.load_state_dict(encoder_state, strict=True)

        self.head = NavigationTrajectoryHead(
            tokens=64, token_dim=256, route_points=8, hidden=512, permission_dim=2
        )

    def forward(self, history_bev, ego_motion, route_points, route_valid, vehicle, maneuver_allowed):
        b = len(history_bev)
        tokens = self.encoder(history_bev.flatten(0, 1)).reshape(b, 4, 64, 256)
        pred = self.head(tokens, ego_motion, route_points, route_valid, vehicle, maneuver_allowed=maneuver_allowed)
        return pred

    def forward_train(self, batch):
        h = batch["history_bev"]
        ego = batch["ego_motion"]
        rp = batch["route_points"]
        rv = batch["route_valid"]
        veh = batch["vehicle"]
        ma = batch["maneuver_allowed"]
        traj = batch["trajectory"]
        trajm = batch["future_traj_mask"]
        action_mask = batch["action_mask"]

        pred = self.forward(h, ego, rp, rv, veh, ma)
        valid_traj_mask = (trajm.float() * action_mask[:, None].float())[..., None]
        loss_traj_all = nn.functional.smooth_l1_loss(pred, traj, reduction="none") * valid_traj_mask
        loss = loss_traj_all.sum() / torch.clamp(valid_traj_mask.sum() * 4, min=1.0)
        return loss, {"loss_traj": float(loss.item()), "total_loss": float(loss.item())}

    @torch.no_grad()
    def solve_euler(self, history_bev, ego_motion, route_points, route_valid, vehicle, maneuver_allowed, **kwargs):
        return self.forward(history_bev, ego_motion, route_points, route_valid, vehicle, maneuver_allowed)


def evaluate_open_loop(model, val_loader, device, noise_seed=42, raw_path=None):
    model.eval()
    accumulator = TrajectoryMetrics()
    raw_batches = []
    sample_preds = []
    sample_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            h = batch["history_bev"].to(device)
            ego = batch["ego_motion"].to(device)
            rp = batch["route_points"].to(device)
            rv = batch["route_valid"].to(device)
            veh = batch["vehicle"].to(device)
            ma = batch["maneuver_allowed"].to(device)
            target_traj = batch["trajectory"].to(device)
            applied = batch.get("applied_action")
            action_valid = batch.get("action_valid")
            if applied is not None:
                applied = applied.to(device)
            if action_valid is not None:
                action_valid = action_valid.to(device)

            with (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                  if torch.device(device).type == "cuda" else nullcontext()):
                if isinstance(model, DeterministicBaseline):
                    pred_traj = model.forward(h, ego, rp, rv, veh, ma)
                else:
                    use_app = getattr(model, "condition_encoder", None) and model.condition_encoder.use_applied_action
                    res = model.solve_euler(
                        h, ego, rp, rv, veh, ma,
                        applied_action=applied if use_app else None,
                        action_valid=action_valid if use_app else None,
                        num_steps=5, noise_seed=noise_seed
                    )
                    pred_traj = res[0] if isinstance(res, tuple) else res

            pred_np = pred_traj.float().cpu().numpy()
            target_np = target_traj.float().cpu().numpy()
            valid = batch["future_traj_mask"].cpu().numpy()
            eligible = batch["action_mask"].cpu().numpy()
            accumulator.update(pred_np, target_np, valid, eligible)
            if raw_path is not None:
                raw_batches.append({"prediction": pred_np, "target": target_np, "valid": valid,
                                    "eligible": eligible, "episode_seed": batch["episode_seed"].cpu().numpy(),
                                    "step": batch["step"].cpu().numpy()})
            if batch_idx < 3:
                candidates = np.flatnonzero(eligible & valid.all(axis=1))
                if len(candidates):
                    i = candidates[0]
                    sample_preds.append(pred_np[i])
                    sample_targets.append(target_np[i])

    if raw_path is not None and raw_batches:
        with Path(raw_path).open("xb") as stream:
            np.savez_compressed(stream, **{key: np.concatenate([batch[key] for batch in raw_batches])
                                          for key in raw_batches[0]})
    return accumulator.result() | {"sample_preds": sample_preds, "sample_targets": sample_targets}


def evaluate_closed_loop_model(model, device, seeds=(20261022, 20261023), noise_seed=42):
    """Evaluates closed-loop driving across straight zero-traffic and interactive traffic scenarios."""
    model.eval()
    route_completions = []
    effective_rcs = []
    arrivals = []
    crashes = []
    offroads = []
    all_latencies_ms = []

    # 4 distinct scenarios per evaluation seed:
    # 1. Straight road zero traffic
    # 2. Straight road following slow vehicle
    # 3. Lead vehicle stop and resume
    # 4. Multi-lane overtake / navigation
    scenarios = [
        {"case": "straight_empty", "lanes": 1, "has_lead": False, "dynamic_lead": False},
        {"case": "following_slow", "lanes": 1, "has_lead": True, "dynamic_lead": False, "lead_dist": 22.0},
        {"case": "lead_stop_resume", "lanes": 1, "has_lead": True, "dynamic_lead": True, "lead_dist": 20.0},
        {"case": "multilane_empty", "lanes": 3, "has_lead": False, "dynamic_lead": False},
    ]

    for seed in seeds:
        for sc in scenarios:
            env_config = dict(
                map=3,
                map_config=dict(type="block_sequence", config="SSS", lane_num=sc["lanes"], exit_length=300),
                start_seed=seed,
                num_scenarios=1,
                traffic_density=0.0,
                horizon=250,
                decision_repeat=5,
                physics_world_step_size=0.02,
                force_destroy=True,
                random_lane_width=False,
                random_lane_num=False,
                random_agent_model=False,
                random_spawn_lane_index=False,
                agent_configs={"default_agent": {"spawn_lane_index": (">>", ">>>", 0 if sc["lanes"] == 1 else 1), "spawn_longitude": 5.0}},
            )
            sim = make_external_env(env_config)
            try:
                obs, _ = sim.reset(seed=seed)
                provider = NativeBEVProvider(sim, BEVConfig())
                provider.reset(obs)
                nav = NavigationProvider(sim, replace(IntegratedConfig(), preferred_lane=0 if sc["lanes"] == 1 else 1))
                tracker = TrajectoryTracker(sim.agent, dt_s=0.1)

                # Spawn interactive NPC if configured
                lead_npc = None
                if sc.get("has_lead"):
                    lane = sim.current_map.road_network.graph[sim.agent.lane.index[0]][sim.agent.lane.index[1]][sim.agent.lane.index[-1]]
                    lead_npc = spawn_slow_actor(
                        sim, lane, 5.0 + sc.get("lead_dist", 22.0), seed,
                        dynamic_stop_resume=sc.get("dynamic_lead", False),
                        stop_step=25, resume_step=90
                    )

                history_bev = []
                history_ego = []
                history_applied = []
                prev_heading = None

                arrived = False
                crashed = False
                offroad = False
                route_comp = 0.0

                for step in range(250):
                    t0 = time.perf_counter()
                    bev = provider.capture().raster.copy()
                    state = vehicle_state(sim.agent)
                    nav_frame = nav.capture()
                    perm = maneuver_permissions(sim)
                    veh = vehicle_parameters(sim.agent)

                    speed = float(state[3]) / 10.0
                    heading = float(state[2])
                    if prev_heading is None:
                        yaw_rate = 0.0
                    else:
                        yaw_rate = float(wrap_angle(heading - prev_heading) / 0.1)
                    prev_heading = heading

                    history_bev.append(bev)
                    history_ego.append(np.array([speed, yaw_rate], dtype=np.float32))

                    if len(history_bev) > 4:
                        history_bev.pop(0)
                        history_ego.pop(0)

                    h_bev = np.stack([history_bev[0]] * (4 - len(history_bev)) + history_bev, axis=0)
                    h_ego = np.stack([history_ego[0]] * (4 - len(history_ego)) + history_ego, axis=0)
                    h_app, h_app_valid = extract_causal_actions(history_applied, step, num_lags=4)

                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            h_bev_t = torch.from_numpy(h_bev[None]).to(device)
                            h_ego_t = torch.from_numpy(h_ego[None]).to(device)
                            rp_t = torch.from_numpy(nav_frame.points[None].astype(np.float32)).to(device)
                            rv_t = torch.from_numpy(nav_frame.valid[None]).to(device)
                            veh_t = torch.from_numpy(veh[None].astype(np.float32)).to(device)
                            perm_t = torch.from_numpy(perm[None]).to(device)
                            h_app_t = torch.from_numpy(h_app[None]).to(device)
                            h_app_valid_t = torch.from_numpy(h_app_valid[None]).to(device)

                            if isinstance(model, DeterministicBaseline):
                                plan_t = model(h_bev_t, h_ego_t, rp_t, rv_t, veh_t, perm_t)
                            else:
                                use_app = getattr(model, "condition_encoder", None) and model.condition_encoder.use_applied_action
                                res = model.solve_euler(
                                    h_bev_t, h_ego_t, rp_t, rv_t, veh_t, perm_t,
                                    applied_action=h_app_t if use_app else None,
                                    action_valid=h_app_valid_t if use_app else None,
                                    num_steps=5, noise_seed=noise_seed
                                )
                                plan_t = res[0] if isinstance(res, tuple) else res
                            plan = plan_t[0].cpu().float().numpy()

                    action, _ = tracker.act(plan, state[3])
                    all_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

                    obs, r, tm, tc, info = checked_step(sim, action)
                    provider.update(obs)
                    actual_applied = np.asarray(info["raw_action"], dtype=np.float32)
                    history_applied.append(actual_applied)

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

                effective_rc = 0.0 if (crashed or offroad) else route_comp
                route_completions.append(route_comp)
                effective_rcs.append(effective_rc)
                arrivals.append(arrived)
                crashes.append(crashed)
                offroads.append(offroad)
            finally:
                sim.close()

    overall_p95_lat = float(np.percentile(all_latencies_ms, 95)) if all_latencies_ms else 0.0

    return {
        "num_episodes": len(route_completions),
        "mean_route_completion": round(float(np.mean(route_completions)), 4),
        "mean_effective_route_completion": round(float(np.mean(effective_rcs)), 4),
        "arrival_rate": round(float(np.mean(arrivals)), 4),
        "crash_rate": round(float(np.mean(crashes)), 4),
        "offroad_rate": round(float(np.mean(offroads)), 4),
        "overall_latency_p95_ms": round(overall_p95_lat, 2),
    }


def train_and_eval_arm(arm_name, model_class, model_kwargs, seed, train_ds, val_ds, device, epochs=10, batch_size=64):
    if STAGE4_REMEDIATED_DIR.resolve() == (WORKSPACE / "artifacts/bev_wa_alignment/stage4_remediated").resolve():
        raise RuntimeError("Historical checkpoint directory is read-only; select a new output directory before any training")
    print(f"\n=======================================================")
    print(f"Training {arm_name} (Seed {seed})")
    print(f"=======================================================")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    tb_dir = STAGE4_REMEDIATED_DIR / "tb" / f"{arm_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}_seed_{seed}"
    writer = SummaryWriter(log_dir=str(tb_dir))

    cfg = Stage3Config(seed=seed)
    model = model_class(cfg=cfg, **model_kwargs).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_ade = float("inf")
    best_ckpt_path = STAGE4_REMEDIATED_DIR / f"{arm_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}_seed_{seed}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            batch_cuda = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, terms = model.forward_train(batch_cuda)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            if hasattr(model, "update_teacher"):
                model.update_teacher(tau=0.996)

            train_losses.append(terms["total_loss"])
            global_step = (epoch - 1) * len(train_loader) + step
            for k, val in terms.items():
                writer.add_scalar(f"train/{k}", val, global_step)

        scheduler.step()
        train_dt = time.time() - t0
        mean_train_loss = float(np.mean(train_losses))

        # Periodic validation
        val_metrics = evaluate_open_loop(model, val_loader, device, noise_seed=seed)
        score = val_metrics["val_ade_2s"]
        if score is None or not np.isfinite(score) or not val_metrics["open_loop_gate"]["complete_primary_labels"]:
            raise RuntimeError("No finite fully-labelled eligible validation ADE; refusing checkpoint selection")
        writer.add_scalar("val/ade_2s", val_metrics["val_ade_2s"], epoch)
        writer.add_scalar("val/fde_2s", val_metrics["val_fde_2s"], epoch)
        writer.add_scalar("val/neg_speed_pct", val_metrics["neg_speed_pct"], epoch)

        print(f"Epoch {epoch:02d}/{epochs:02d} [{train_dt:.1f}s]: Train Loss={mean_train_loss:.4f}, Val ADE(2s)={val_metrics['val_ade_2s']:.4f}m, FDE={val_metrics['val_fde_2s']:.4f}m")

        if score < best_val_ade:
            best_val_ade = val_metrics["val_ade_2s"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
                "seed": seed,
                "arm": arm_name,
            }, best_ckpt_path)

    writer.close()

    # Load best checkpoint for final evaluation
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"], strict=True)
    final_open_loop = evaluate_open_loop(model, val_loader, device, noise_seed=seed)
    final_closed_loop = evaluate_closed_loop_model(model, device, seeds=(seed,), noise_seed=seed)

    return {
        "val_ade_2s": final_open_loop["val_ade_2s"],
        "val_fde_2s": final_open_loop["val_fde_2s"],
        "neg_speed_pct": final_open_loop["neg_speed_pct"],
        "cohorts": final_open_loop["cohorts"],
        "open_loop_gate": final_open_loop["open_loop_gate"],
        "closed_loop": final_closed_loop,
        "sample_preds": final_open_loop["sample_preds"],
        "sample_targets": final_open_loop["sample_targets"],
        "ckpt": str(best_ckpt_path),
    }


def plot_prediction_vs_truth(sample_preds, sample_targets, save_path):
    """Generates visual comparison figures of open-loop trajectory predictions vs ground truth."""
    fig, axes = plt.subplots(1, min(3, len(sample_preds)), figsize=(15, 5))
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for idx, ax in enumerate(axes):
        pred = sample_preds[idx]
        target = sample_targets[idx]
        ax.plot(pred[:, 1], pred[:, 0], "r-o", label="Prediction (Arm D)", linewidth=2, markersize=5)
        ax.plot(target[:, 1], target[:, 0], "b--s", label="Ground Truth", linewidth=2, markersize=5)
        ax.set_title(f"Sample {idx+1}: 2.0s Trajectory")
        ax.set_xlabel("Lateral Y (m)")
        ax.set_ylabel("Longitudinal X (m)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved prediction vs truth figure to {save_path}")


def generate_report(results, report_path):
    Path(report_path).write_text(render_stage4_report(results), encoding="utf-8")


def main():
    global STAGE4_REMEDIATED_DIR
    parser = argparse.ArgumentParser(description="Run BEV WA-JEPA Stage 4 Remediated Training and Evaluation.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--eval-only", action="store_true", help="Evaluate existing checkpoints; never train missing ones")
    mode.add_argument("--train", action="store_true", help="Explicitly authorize this process to train (requires separate user stage authorization)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New, nonexistent artifact directory; never overwrite historical runs")
    parser.add_argument("--checkpoint-dir", type=Path, default=STAGE4_REMEDIATED_DIR)
    args = parser.parse_args()
    parser.error("Historical mixed-scene pipeline is retired: use reevaluate_bev_wa_stage4_open_loop.py for existing weights and the dedicated classified closed-loop runner. No training or simulation started.")
    STAGE4_REMEDIATED_DIR = args.output_dir.resolve()
    STAGE4_REMEDIATED_DIR.mkdir(parents=True, exist_ok=False)

    print("\n=======================================================")
    print("BEV WA-JEPA Stage 4 Remediated Mainline Execution")
    print("=======================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    _, restore = configure_egl(1)

    try:
        # Load datasets
        print("\n[Step 1] Loading Merged Datasets...")
        train_ds = BEWWAJointDataset(TRAIN_MANIFEST, split="train", eligible_action_only=True)
        val_ds = BEWWAJointDataset(VAL_MANIFEST, split="validation", eligible_action_only=False)

        train_stats = train_ds.stats()
        val_stats = val_ds.stats()
        print(f"Train Dataset: {train_stats}")
        print(f"Validation Dataset: {val_stats}")

        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
        seeds = [20261022, 20261023]

        # Step 2: Evaluate Historical References (Legacy Checkpoints)
        print("\n[Step 2] Evaluating Historical References (Legacy Checkpoints in Native Architecture)...")
        historical_references = {}

        # Legacy Arm A
        arm_a_hist = {"arm": "Arm A (Deterministic MSE, Legacy)", "type": "historical"}
        for seed in seeds:
            st2_ckpt = STAGE2_DIR / f"baseline_seed_{seed}.pt"
            if st2_ckpt.exists():
                model = DeterministicBaseline(STAGE4_ENCODER_CKPT).to(device)
                ckpt = torch.load(st2_ckpt, map_location=device, weights_only=False)
                model.load_state_dict(ckpt.get("model_state", ckpt.get("model", ckpt.get("model_state_dict"))), strict=True)
                metrics = evaluate_open_loop(model, val_loader, device, noise_seed=seed)
                cl = evaluate_closed_loop_model(model, device, seeds=(seed,), noise_seed=seed)
                arm_a_hist[f"seed_{seed}"] = {
                    "val_ade_2s": metrics["val_ade_2s"],
                    "val_fde_2s": metrics["val_fde_2s"],
                    "neg_speed_pct": metrics["neg_speed_pct"],
                    "cohorts": metrics["cohorts"],
                    "open_loop_gate": metrics["open_loop_gate"],
                    "closed_loop": cl,
                    "ckpt": str(st2_ckpt),
                }
                print(f"  Historical Arm A (Seed {seed}): Val ADE(2s)={metrics['val_ade_2s']:.4f}m, FDE(2s)={metrics['val_fde_2s']:.4f}m, Eff RC={cl['mean_effective_route_completion']*100:.1f}%")
        historical_references["Arm A (Legacy)"] = arm_a_hist

        # Legacy Arm B
        arm_b_hist = {"arm": "Arm B (Action-only Flow, Legacy)", "type": "historical"}
        for seed in seeds:
            ckpt_path = STAGE4_LEGACY_DIR / f"arm_b_(action-only_flow)_seed_{seed}.pt"
            if ckpt_path.exists():
                model = ArmBActionOnlyFlowModel(cfg=Stage3Config(seed=seed), encoder_ckpt=STAGE4_ENCODER_CKPT, use_applied_action=False).to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt.get("model_state", ckpt.get("model_state_dict")), strict=True)
                metrics = evaluate_open_loop(model, val_loader, device, noise_seed=seed)
                cl = evaluate_closed_loop_model(model, device, seeds=(seed,), noise_seed=seed)
                arm_b_hist[f"seed_{seed}"] = {
                    "val_ade_2s": metrics["val_ade_2s"],
                    "val_fde_2s": metrics["val_fde_2s"],
                    "neg_speed_pct": metrics["neg_speed_pct"],
                    "cohorts": metrics["cohorts"],
                    "open_loop_gate": metrics["open_loop_gate"],
                    "closed_loop": cl,
                    "ckpt": str(ckpt_path),
                }
                print(f"  Historical Arm B (Seed {seed}): Val ADE(2s)={metrics['val_ade_2s']:.4f}m, FDE(2s)={metrics['val_fde_2s']:.4f}m, Eff RC={cl['mean_effective_route_completion']*100:.1f}%")
        historical_references["Arm B (Legacy)"] = arm_b_hist

        # Legacy Arm C
        arm_c_hist = {"arm": "Arm C (Decoupled Flow, Legacy)", "type": "historical"}
        for seed in seeds:
            ckpt_path = STAGE4_LEGACY_DIR / f"arm_c_(decoupled_flow)_seed_{seed}.pt"
            if ckpt_path.exists():
                st3_ckpt = STAGE3_DIR / f"flow_latent_seed_{seed}.pt"
                model = ArmCDecoupledFlowModel(cfg=Stage3Config(seed=seed), encoder_ckpt=STAGE4_ENCODER_CKPT, stage3_ckpt=st3_ckpt, use_applied_action=False).to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt.get("model_state", ckpt.get("model_state_dict")), strict=True)
                metrics = evaluate_open_loop(model, val_loader, device, noise_seed=seed)
                cl = evaluate_closed_loop_model(model, device, seeds=(seed,), noise_seed=seed)
                arm_c_hist[f"seed_{seed}"] = {
                    "val_ade_2s": metrics["val_ade_2s"],
                    "val_fde_2s": metrics["val_fde_2s"],
                    "neg_speed_pct": metrics["neg_speed_pct"],
                    "cohorts": metrics["cohorts"],
                    "open_loop_gate": metrics["open_loop_gate"],
                    "closed_loop": cl,
                    "ckpt": str(ckpt_path),
                }
                print(f"  Historical Arm C (Seed {seed}): Val ADE(2s)={metrics['val_ade_2s']:.4f}m, FDE(2s)={metrics['val_fde_2s']:.4f}m, Eff RC={cl['mean_effective_route_completion']*100:.1f}%")
        historical_references["Arm C (Legacy)"] = arm_c_hist

        # Legacy Arm D (Un-isolated)
        arm_d_hist = {"arm": "Arm D (Joint WA-Flow, Legacy Un-isolated)", "type": "historical"}
        for seed in seeds:
            ckpt_path = STAGE4_LEGACY_DIR / f"arm_d_(joint_wa-flow)_seed_{seed}.pt"
            if ckpt_path.exists():
                model = ArmDJointFlowModel(cfg=Stage3Config(seed=seed), encoder_ckpt=STAGE4_ENCODER_CKPT, scene_loss_weight=1.0, use_applied_action=False).to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt.get("model_state", ckpt.get("model_state_dict")), strict=True)
                metrics = evaluate_open_loop(model, val_loader, device, noise_seed=seed)
                cl = evaluate_closed_loop_model(model, device, seeds=(seed,), noise_seed=seed)
                arm_d_hist[f"seed_{seed}"] = {
                    "val_ade_2s": metrics["val_ade_2s"],
                    "val_fde_2s": metrics["val_fde_2s"],
                    "neg_speed_pct": metrics["neg_speed_pct"],
                    "cohorts": metrics["cohorts"],
                    "open_loop_gate": metrics["open_loop_gate"],
                    "closed_loop": cl,
                    "ckpt": str(ckpt_path),
                }
                print(f"  Historical Arm D (Seed {seed}): Val ADE(2s)={metrics['val_ade_2s']:.4f}m, FDE(2s)={metrics['val_fde_2s']:.4f}m, Eff RC={cl['mean_effective_route_completion']*100:.1f}%")
        historical_references["Arm D (Legacy Un-isolated)"] = arm_d_hist

        # Step 3: Train and Evaluate Fair Comparison Arms
        print("\n[Step 3] Training and Evaluating Fair Comparison Arms on Merged Dataset 012...")
        arms_to_train = [
            ("Arm A (Deterministic MSE)", DeterministicBaseline, {"encoder_ckpt": STAGE4_ENCODER_CKPT}),
            ("Arm B (Action-only Flow)", ArmBActionOnlyFlowModel, {"encoder_ckpt": STAGE4_ENCODER_CKPT, "use_applied_action": True}),
            ("Arm D (Repaired Joint WA-Flow)", ArmDJointFlowModel, {"encoder_ckpt": STAGE4_ENCODER_CKPT, "scene_loss_weight": 1.0, "use_applied_action": True}),
            ("Ablation D-0 (Repaired L_scene=0)", ArmDJointFlowModel, {"encoder_ckpt": STAGE4_ENCODER_CKPT, "scene_loss_weight": 0.0, "use_applied_action": True}),
        ]

        fair_arms = {}
        for arm_name, model_class, base_kwargs in arms_to_train:
            arm_dict = {"arm": arm_name, "type": "fair_comparison"}
            for seed in seeds:
                best_ckpt_path = args.checkpoint_dir / f"{arm_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}_seed_{seed}.pt"
                if args.eval_only and not best_ckpt_path.is_file():
                    raise FileNotFoundError(f"Evaluation-only checkpoint missing: {best_ckpt_path}")
                if args.eval_only:
                    print(f"Loading pre-trained {arm_name} (Seed {seed}) from {best_ckpt_path}...")
                    cfg = Stage3Config(seed=seed)
                    model = model_class(cfg=cfg, **base_kwargs).to(device)
                    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
                    model.load_state_dict(best_ckpt["model_state"], strict=True)
                    final_open_loop = evaluate_open_loop(model, val_loader, device, noise_seed=seed)
                    final_closed_loop = evaluate_closed_loop_model(model, device, seeds=(seed,), noise_seed=seed)
                    res = {
                        "val_ade_2s": final_open_loop["val_ade_2s"],
                        "val_fde_2s": final_open_loop["val_fde_2s"],
                        "neg_speed_pct": final_open_loop["neg_speed_pct"],
                        "cohorts": final_open_loop["cohorts"],
                        "open_loop_gate": final_open_loop["open_loop_gate"],
                        "closed_loop": final_closed_loop,
                        "sample_preds": final_open_loop["sample_preds"],
                        "sample_targets": final_open_loop["sample_targets"],
                        "ckpt": str(best_ckpt_path),
                    }
                    print(f"  {arm_name} (Seed {seed}): Val ADE(2s)={res['val_ade_2s']:.4f}m, FDE(2s)={res['val_fde_2s']:.4f}m, Eff RC={final_closed_loop['mean_effective_route_completion']*100:.1f}%")
                else:
                    res = train_and_eval_arm(
                        arm_name=arm_name,
                        model_class=model_class,
                        model_kwargs=base_kwargs,
                        seed=seed,
                        train_ds=train_ds,
                        val_ds=val_ds,
                        device=device,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                    )
                arm_dict[f"seed_{seed}"] = res
            fair_arms[arm_name] = arm_dict

        # Step 4: Statistical Significance Analysis
        print("\n[Step 4] Computing Statistical Significance (Arm D vs Arm B)...")
        d_rcs = [
            fair_arms["Arm D (Repaired Joint WA-Flow)"]["seed_20261022"]["closed_loop"]["mean_effective_route_completion"],
            fair_arms["Arm D (Repaired Joint WA-Flow)"]["seed_20261023"]["closed_loop"]["mean_effective_route_completion"],
        ]
        b_rcs = [
            fair_arms["Arm B (Action-only Flow)"]["seed_20261022"]["closed_loop"]["mean_effective_route_completion"],
            fair_arms["Arm B (Action-only Flow)"]["seed_20261023"]["closed_loop"]["mean_effective_route_completion"],
        ]
        mean_d = float(np.mean(d_rcs))
        mean_b = float(np.mean(b_rcs))
        delta_rc = mean_d - mean_b

        t_stat, p_val = stats.ttest_rel(d_rcs, b_rcs) if len(d_rcs) > 1 else (0.0, 1.0)
        # Undefined statistics remain missing, never manufactured significance.

        comp_analysis = {
            "arm_d_repaired_mean_rc": round(mean_d, 4),
            "arm_b_mean_rc": round(mean_b, 4),
            "delta_rc": round(delta_rc, 4),
            "t_statistic": float(t_stat) if np.isfinite(t_stat) else None,
            "p_value": float(p_val) if np.isfinite(p_val) else None,
            "significant": None,
            "status": "diagnostic_only: mixed scenes and training-seed aggregates cannot establish joint-model gain",
        }

        # Step 5: Gate Checks
        d_ades = [
            fair_arms["Arm D (Repaired Joint WA-Flow)"]["seed_20261022"]["val_ade_2s"],
            fair_arms["Arm D (Repaired Joint WA-Flow)"]["seed_20261023"]["val_ade_2s"],
        ]
        d_fdes = [
            fair_arms["Arm D (Repaired Joint WA-Flow)"]["seed_20261022"]["val_fde_2s"],
            fair_arms["Arm D (Repaired Joint WA-Flow)"]["seed_20261023"]["val_fde_2s"],
        ]
        d_negs = [
            fair_arms["Arm D (Repaired Joint WA-Flow)"]["seed_20261022"]["neg_speed_pct"],
            fair_arms["Arm D (Repaired Joint WA-Flow)"]["seed_20261023"]["neg_speed_pct"],
        ]
        d_lats = [
            fair_arms["Arm D (Repaired Joint WA-Flow)"]["seed_20261022"]["closed_loop"]["overall_latency_p95_ms"],
            fair_arms["Arm D (Repaired Joint WA-Flow)"]["seed_20261023"]["closed_loop"]["overall_latency_p95_ms"],
        ]

        mean_ade = float(np.mean(d_ades))
        mean_fde = float(np.mean(d_fdes))
        mean_neg = float(np.mean(d_negs))
        mean_lat = float(np.mean(d_lats))

        gate1_passed = all(fair_arms["Arm D (Repaired Joint WA-Flow)"][f"seed_{seed}"]["open_loop_gate"]["passed"] for seed in seeds)
        gate2_passed = None  # Mixed navigation/following RC is not an acceptance criterion.
        gate3_passed = None  # Keep per-seed P95; its mean is not the pooled P95.

        gates = {
            "gate1_open_loop": {
                "ade_2s": round(mean_ade, 4),
                "fde_2s": round(mean_fde, 4),
                "neg_speed_pct": round(mean_neg, 4),
                "passed": gate1_passed,
            },
            "gate2_closed_loop": {
                "mean_effective_rc": round(mean_d, 4),
                "passed": gate2_passed,
            },
            "gate3_latency": {
                "per_seed_p95_latency_ms": dict(zip(map(str, seeds), d_lats)),
                "pooled_p95_latency_ms": None,
                "passed": gate3_passed,
            },
        }

        # Step 6: Plot Prediction vs Truth
        fig_path = STAGE4_REMEDIATED_DIR / "predictions_vs_truth.png"
        sample_preds = fair_arms["Arm D (Repaired Joint WA-Flow)"]["seed_20261022"]["sample_preds"]
        sample_targets = fair_arms["Arm D (Repaired Joint WA-Flow)"]["seed_20261022"]["sample_targets"]
        plot_prediction_vs_truth(sample_preds, sample_targets, fig_path)

        # Save results.json
        full_results = {
            "protocol_version": "BEV_WA_ALIGNED_REMEDIATED_V1",
            "date_user_timezone": "2026-10-23",
            "stage": 4,
            "title": "Stage 4 Remediated Mainline: Causal Temporal Alignment & Fair 4-Arm Comparison",
            "provenance": provenance(),
            "hardware": {
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            },
            "dataset_stats": {
                "train": train_stats,
                "validation": val_stats,
            },
            "historical_references": historical_references,
            "fair_arms": fair_arms,
            "comparative_analysis": comp_analysis,
            "gates": gates,
        }

        def sanitize_for_json(obj):
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items() if not k.startswith("sample_")}
            elif isinstance(obj, (list, tuple)):
                return [sanitize_for_json(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            elif isinstance(obj, (bool, str, int, float)) or obj is None:
                return obj
            return str(obj)

        results_file = STAGE4_REMEDIATED_DIR / "results.json"
        clean_results = sanitize_for_json(full_results)
        write_json(results_file, clean_results)
        print(f"Results saved to {results_file}")

        # Generate Markdown Report
        report_file = STAGE4_REMEDIATED_DIR / "REPORT_STAGE4_REMEDIATED.md"
        generate_report(full_results, report_file)
        print(f"Report generated at {report_file}")

    finally:
        restore()


if __name__ == "__main__":
    main()
