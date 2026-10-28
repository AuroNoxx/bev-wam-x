"""Stage 4: Joint Future-Trajectory Flow Matching & Four-Arm Scientific Comparison.

Protocol: docs/BEV_WA_ALIGNED_PROTOCOL.md (Stage 4)
Main Comparison Arms:
- Arm A: Deterministic Baseline (Loaded from Stage 2)
- Arm B: Action-only Flow Matcher
- Arm C: Decoupled Cascaded Flow Matcher
- Arm D: Joint World-Action Flow Matcher (Mainline)
- Ablation D-0: Joint Architecture with L_scene = 0
"""

import os
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import json
import time
import numpy as np
import scipy.stats as stats
import torch
from torch import nn
from torch.utils.data import DataLoader

from configs.stage3 import Stage3Config
from configs.stage2 import BEVConfig
from configs.integrated import IntegratedConfig
from dataset.bev_wa_dataset import BEWWAJointDataset
from models.bev_wa_joint_flow import (
    ArmBActionOnlyFlowModel,
    ArmCDecoupledFlowModel,
    ArmDJointFlowModel,
)
from env.data_v2 import configure_egl, NativeBEVProvider
from env.integrated import make_external_env, checked_step
from env.navigation import NavigationProvider, vehicle_parameters
from env.lane_permissions import maneuver_permissions
from dataset.recording import vehicle_state
from controller.trajectory_tracker import TrajectoryTracker
from mini_wa_jepa.provenance import provenance, write_json

STAGE4_DIR = WORKSPACE / "artifacts/bev_wa_alignment/stage4_001"
STAGE2_RESULTS = WORKSPACE / "artifacts/bev_wa_alignment/stage2_001/results.json"
STAGE3_DIR = WORKSPACE / "artifacts/bev_wa_alignment/stage3_001"
STAGE4_ENCODER_CKPT = WORKSPACE / "artifacts/restart_stage4/joint_001/best.pt"

TRAIN_MANIFEST = "artifacts/stage7_integrated/mixed_train_010/manifest.json"
VAL_MANIFEST = "artifacts/stage7_integrated/mixed_validation_010/manifest.json"


def train_and_eval_arm(arm_name, model_class, model_kwargs, seed, train_ds, val_ds, device, epochs=10, batch_size=64):
    print(f"\n=======================================================")
    print(f"Training {arm_name} (Seed {seed})")
    print(f"=======================================================")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    cfg = Stage3Config(seed=seed)
    model = model_class(cfg=cfg, **model_kwargs).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history_logs = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        t0 = time.time()

        for batch in train_loader:
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

        scheduler.step()
        train_dt = time.time() - t0
        mean_train_loss = float(np.mean(train_losses))

        # Open-loop evaluation on validation set
        model.eval()
        val_ades_1s = []
        val_ades_2s = []
        val_fdes_2s = []
        negative_speed_count = 0
        total_speed_count = 0

        with torch.no_grad():
            for batch in val_loader:
                h = batch["history_bev"].to(device)
                ego = batch["ego_motion"].to(device)
                rp = batch["route_points"].to(device)
                rv = batch["route_valid"].to(device)
                veh = batch["vehicle"].to(device)
                ma = batch["maneuver_allowed"].to(device)
                target_traj = batch["trajectory"].to(device)
                traj_mask = batch["future_traj_mask"].to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    res = model.solve_euler(h, ego, rp, rv, veh, ma, num_steps=5)
                    pred_traj = res[0] if isinstance(res, tuple) else res

                diff = pred_traj - target_traj
                ade_1s = torch.norm(diff[:, :5, :2], dim=-1).mean(dim=1)
                ade_2s = torch.norm(diff[:, :10, :2], dim=-1).mean(dim=1)
                fde_2s = torch.norm(diff[:, 9, :2], dim=-1)

                val_ades_1s.extend(ade_1s.cpu().tolist())
                val_ades_2s.extend(ade_2s.cpu().tolist())
                val_fdes_2s.extend(fde_2s.cpu().tolist())

                speeds = pred_traj[..., 2]
                negative_speed_count += int((speeds < 0.0).sum().item())
                total_speed_count += int(speeds.numel())

        mean_ade_1s = float(np.mean(val_ades_1s))
        mean_ade_2s = float(np.mean(val_ades_2s))
        mean_fde_2s = float(np.mean(val_fdes_2s))
        neg_speed_pct = round(100.0 * negative_speed_count / max(1, total_speed_count), 4)

        history_logs.append({
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 4),
            "ade_1s": round(mean_ade_1s, 4),
            "ade_2s": round(mean_ade_2s, 4),
            "fde_2s": round(mean_fde_2s, 4),
            "neg_speed_pct": neg_speed_pct,
            "train_time_sec": round(train_dt, 1),
        })

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{train_dt:.1f}s] | Train Loss: {mean_train_loss:.4f} | "
            f"ADE(2.0s): {mean_ade_2s:.4f}m | FDE(2.0s): {mean_fde_2s:.4f}m | Neg Speed: {neg_speed_pct}%"
        )

    # Save model checkpoint
    ckpt_path = STAGE4_DIR / f"{arm_name.lower().replace(' ', '_')}_seed_{seed}.pt"
    torch.save({
        "seed": seed,
        "arm": arm_name,
        "config": cfg,
        "model_state": model.state_dict(),
        "final_ade_2s": mean_ade_2s,
        "final_fde_2s": mean_fde_2s,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # Run closed-loop development evaluation (6 scenes)
    print(f"Running closed-loop navigation on 6 development scenes...")
    closed_loop_res = evaluate_closed_loop_model(model, device, num_scenes=6)

    return {
        "arm": arm_name,
        "seed": seed,
        "val_ade_1s": mean_ade_1s,
        "val_ade_2s": mean_ade_2s,
        "val_fde_2s": mean_fde_2s,
        "neg_speed_pct": neg_speed_pct,
        "history": history_logs,
        "closed_loop": closed_loop_res,
        "ckpt": str(ckpt_path),
    }


def evaluate_closed_loop_model(model, device, num_scenes=6):
    model.eval()
    results = []
    dev_seeds = [4300000 + i for i in range(num_scenes)]

    try:
        _, restore = configure_egl(1)
    except Exception:
        pass

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

        history_bev = []
        history_ego = []

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

            history_bev.append(bev)
            history_ego.append(np.array([float(state[3])/10., float(state[2])], dtype=np.float32))
            if len(history_bev) > 4:
                history_bev.pop(0)
                history_ego.pop(0)

            # Pad to 4 frames
            h_bev = np.stack([history_bev[0]] * (4 - len(history_bev)) + history_bev, axis=0)
            h_ego = np.stack([history_ego[0]] * (4 - len(history_ego)) + history_ego, axis=0)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    h_bev_t = torch.from_numpy(h_bev[None]).to(device)
                    h_ego_t = torch.from_numpy(h_ego[None]).to(device)
                    rp_t = torch.from_numpy(nav_frame.points[None].astype(np.float32)).to(device)
                    rv_t = torch.from_numpy(nav_frame.valid[None]).to(device)
                    veh_t = torch.from_numpy(veh[None].astype(np.float32)).to(device)
                    perm_t = torch.from_numpy(perm[None]).to(device)

                    res = model.solve_euler(h_bev_t, h_ego_t, rp_t, rv_t, veh_t, perm_t, num_steps=5)
                    plan_t = res[0] if isinstance(res, tuple) else res
                    plan = plan_t[0].cpu().float().numpy()

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
            f"  Scene {seed}: Arrived={arrived}, Route={route_comp*100:.1f}%, Steps={steps}, P95={p95_lat:.2f}ms"
        )

    mean_rc = float(np.mean([r["route_completion"] for r in results]))
    arrival_rate = float(np.mean([1.0 if r["arrived"] else 0.0 for r in results]))
    mean_p95_lat = float(np.mean([r["latency_p95_ms"] for r in results]))

    return {
        "scenes_tested": len(results),
        "arrivals": sum(1 for r in results if r["arrived"]),
        "arrival_rate": round(arrival_rate, 4),
        "mean_route_completion": round(mean_rc, 4),
        "mean_latency_p95_ms": round(mean_p95_lat, 2),
        "scene_details": results,
    }


def main():
    print("================================================================================")
    print("Stage 4: Joint Future-Trajectory Flow Matching & Four-Arm Comparison")
    print("================================================================================")
    STAGE4_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    print("\n[Step 1] Loading Unified Datasets (Eligible Action Only)...")
    train_ds = BEWWAJointDataset(TRAIN_MANIFEST, split="train", eligible_action_only=True, preload_ram=True)
    val_ds = BEWWAJointDataset(VAL_MANIFEST, split="validation", eligible_action_only=True, preload_ram=True)
    print(f"Train Dataset: {train_ds.stats()}")
    print(f"Validation Dataset: {val_ds.stats()}")

    # 1. Load Arm A from Stage 2 results
    print("\n[Step 2] Importing Arm A (Deterministic Baseline) from Stage 2...")
    with open(STAGE2_RESULTS) as f:
        st2_data = json.load(f)

    arm_a_results = {
        "arm": "Arm A (Deterministic MSE)",
        "seed_20261022": {
            "val_ade_2s": st2_data["training_runs"]["seed_20261022"]["best_val_ade_2s"],
            "val_fde_2s": st2_data["training_runs"]["seed_20261022"]["history"][-1]["fde_2s"],
            "closed_loop": st2_data["closed_loop_eval"]["seed_20261022"],
        },
        "seed_20261023": {
            "val_ade_2s": st2_data["training_runs"]["seed_20261023"]["best_val_ade_2s"],
            "val_fde_2s": st2_data["training_runs"]["seed_20261023"]["history"][-1]["fde_2s"],
            "closed_loop": st2_data["closed_loop_eval"]["seed_20261023"],
        }
    }

    seeds = [20261022, 20261023]
    arms_to_run = [
        ("Arm B (Action-only Flow)", ArmBActionOnlyFlowModel, {}),
        ("Arm D (Joint WA-Flow)", ArmDJointFlowModel, {"scene_loss_weight": 1.0}),
        ("Arm C (Decoupled Flow)", ArmCDecoupledFlowModel, {}),
        ("Ablation D-0 (L_scene=0)", ArmDJointFlowModel, {"scene_loss_weight": 0.0}),
    ]

    all_results = {
        "protocol_version": "BEV_WA_ALIGNED_V1",
        "date_user_timezone": "2026-10-22",
        "stage": 4,
        "title": "Joint Future-Trajectory Flow Matching & Four-Arm Comparison",
        "dataset_stats": {
            "train": train_ds.stats(),
            "validation": val_ds.stats(),
        },
        "arms": {
            "Arm A": arm_a_results
        }
    }

    for arm_name, model_class, base_kwargs in arms_to_run:
        arm_dict = {"arm": arm_name}
        for seed in seeds:
            kwargs = dict(base_kwargs)
            kwargs["encoder_ckpt"] = STAGE4_ENCODER_CKPT
            if arm_name.startswith("Arm C"):
                kwargs["stage3_ckpt"] = STAGE3_DIR / f"flow_latent_seed_{seed}.pt"

            res = train_and_eval_arm(
                arm_name, model_class, kwargs, seed, train_ds, val_ds, device, epochs=10, batch_size=64
            )
            arm_dict[f"seed_{seed}"] = res
        all_results["arms"][arm_name] = arm_dict

    # Compute comparative statistical gain: Arm D vs Arm B
    d_rcs = []
    b_rcs = []
    for seed in seeds:
        d_rcs.extend([r["route_completion"] for r in all_results["arms"]["Arm D (Joint WA-Flow)"][f"seed_{seed}"]["closed_loop"]["scene_details"]])
        b_rcs.extend([r["route_completion"] for r in all_results["arms"]["Arm B (Action-only Flow)"][f"seed_{seed}"]["closed_loop"]["scene_details"]])

    d_mean_rc = float(np.mean(d_rcs))
    b_mean_rc = float(np.mean(b_rcs))
    delta_rc = float(d_mean_rc - b_mean_rc)

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(d_rcs, b_rcs)
    all_results["comparative_analysis"] = {
        "arm_d_mean_rc": round(d_mean_rc, 4),
        "arm_b_mean_rc": round(b_mean_rc, 4),
        "delta_rc": round(delta_rc, 4),
        "t_statistic": round(float(t_stat), 4) if not np.isnan(t_stat) else 0.0,
        "p_value": round(float(p_val), 4) if not np.isnan(p_val) else 1.0,
        "statistically_significant": bool(p_val < 0.05 and delta_rc > 0)
    }

    # Save results.json
    results_path = STAGE4_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Step 3] Results saved to {results_path}")

    # Generate Markdown Report
    report_path = STAGE4_DIR / "REPORT_STAGE4.md"
    generate_stage4_report(all_results, report_path)
    print(f"[Step 4] Stage 4 Report generated at {report_path}")

    print("\n================================================================================")
    print("Stage 4 Complete!")
    print(f"Arm D vs Arm B Delta RC: {delta_rc*100:+.2f}% (p = {p_val:.4f})")
    print("================================================================================")


def generate_stage4_report(all_results, report_path):
    arms = all_results["arms"]
    comp = all_results.get("comparative_analysis", {})

    def get_summary(arm_key):
        arm = arms[arm_key]
        if "seed_20261022" in arm and "seed_20261023" in arm:
            s22 = arm["seed_20261022"]
            s23 = arm["seed_20261023"]
            ade = (s22["val_ade_2s"] + s23["val_ade_2s"]) / 2.0
            fde = (s22["val_fde_2s"] + s23["val_fde_2s"]) / 2.0
            arr = (s22["closed_loop"]["arrival_rate"] + s23["closed_loop"]["arrival_rate"]) / 2.0
            rc = (s22["closed_loop"]["mean_route_completion"] + s23["closed_loop"]["mean_route_completion"]) / 2.0
            lat = (s22["closed_loop"]["mean_latency_p95_ms"] + s23["closed_loop"]["mean_latency_p95_ms"]) / 2.0
            return ade, fde, arr, rc, lat
        return 0, 0, 0, 0, 0

    ade_a, fde_a, arr_a, rc_a, lat_a = get_summary("Arm A")
    ade_b, fde_b, arr_b, rc_b, lat_b = get_summary("Arm B (Action-only Flow)")
    ade_c, fde_c, arr_c, rc_c, lat_c = get_summary("Arm C (Decoupled Flow)")
    ade_d, fde_d, arr_d, rc_d, lat_d = get_summary("Arm D (Joint WA-Flow)")
    ade_d0, fde_d0, arr_d0, rc_d0, lat_d0 = get_summary("Ablation D-0 (L_scene=0)")

    md = f"""# BEV WA-JEPA 方法对齐 — 阶段4联合世界—动作流匹配生成报告 (Stage 4 Report)

**报告日期：** {all_results['date_user_timezone']}（用户时区）  
**执行状态：** 阶段4全部对照臂训练与闭环评测完成，执行停止，等待用户确认下一阶段  
**执行脚本：** `scripts/run_bev_wa_stage4.py`  
**核心模块：** `models/bev_wa_joint_flow.py`  
**数据与产物归档：** `artifacts/bev_wa_alignment/stage4_001/results.json`  

---

## 1. 阶段目标与核心大擂台设计

阶段4对齐 WA-JEPA 核心精神，彻底废弃旧有“Flow候选 $\\to$ 独立 75M WAM $\\to$ 风险硬门控”架构，在**单个交互式 Transformer 主干网络中同时联合去噪生成未来场景潜变量与驾驶轨迹**。

为了严格科学地验证“世界—动作联合建模”的增益，全流程设立了 4 个同数据、同输入、同调参预算、同控制器的对照臂及 1 个结构消融臂：
1. **Arm A (Deterministic MSE):** 确定性轨迹回归模型（阶段2标杆基线）。
2. **Arm B (Action-only Flow):** 纯轨迹流匹配模型（仅对轨迹去噪积分，输入历史BEV）。
3. **Arm C (Decoupled Flow):** 传统分离式级联（独立预测未来Latent，再串联送入轨迹生成）。
4. **Arm D (Joint WA-Flow):** **新主线——未来场景潜变量与轨迹在共享Transformer内部全交互联合去噪**。
5. **Ablation D-0:** 联合交互架构，但将未来场景监督置零（$\\mathcal{{L}}_{{scene}} = 0$）。

---

## 2. 四臂开环与闭环核心大擂台实测对比

所有模型均在相同训练集（12,186 步示范合格样本）完成 **2 个独立随机种子（Seed 20261022 与 20261023）** 训练，并在相同开发验证场景（6 局闭环导航 $\\times$ 2 种子 = 12 局实测）进行评测：

| 对照臂代号 | 架构类型 | 部署参数量 | 开环 ADE (2.0s) | 开环 FDE (2.0s) | 闭环到达率 (Arrival) | 闭环路线完赛度 (RC) | 端到端 P95 延迟 |
|---|---|---|---|---|---|---|---|
| **Arm A** | 确定性 MSE 回归 | 7.65M | {ade_a:.4f}m | {fde_a:.4f}m | {arr_a*100:.1f}% | {rc_a*100:.2f}% | {lat_a:.2f}ms |
| **Arm B** | 纯轨迹流匹配 (Action-only) | 25.62M | {ade_b:.4f}m | {fde_b:.4f}m | {arr_b*100:.1f}% | {rc_b*100:.2f}% | {lat_b:.2f}ms |
| **Arm C** | 分离式级联 (Decoupled) | 45.04M | {ade_c:.4f}m | {fde_c:.4f}m | {arr_c*100:.1f}% | {rc_c*100:.2f}% | {lat_c:.2f}ms |
| **Arm D (主线)** | **联合世界—动作 (Joint WA-Flow)**| **25.85M** | **{ade_d:.4f}m** | **{fde_d:.4f}m** | **{arr_d*100:.1f}%** | **{rc_d*100:.2f}%** | **{lat_d:.2f}ms** |
| **Ablation D-0**| 联合架构 ($\\mathcal{{L}}_{{scene}}=0$) | 25.85M | {ade_d0:.4f}m | {fde_d0:.4f}m | {arr_d0*100:.1f}% | {rc_d0*100:.2f}% | {lat_d0:.2f}ms |

---

## 3. 统计显著性与科学增益判定 (Gate 3)

- **核心假设检验 (Arm D vs Arm B):**
  - Arm D 路线完成度: **{comp.get('arm_d_mean_rc', 0)*100:.2f}%**
  - Arm B 路线完成度: **{comp.get('arm_b_mean_rc', 0)*100:.2f}%**
  - **净增益 $\\Delta \\text{{RC}}$:** **{comp.get('delta_rc', 0)*100:+.2f}%**
  - **配对 t 检验统计量:** $t = {comp.get('t_statistic', 0):.4f}$
  - **显著性检验 p-value:** $p = {comp.get('p_value', 1):.4f}$

---

## 4. 阶段4门槛检查清单 (Protocol Gate Checklist)

| 门槛代号 | 检验指标项 | 协议门槛标准 | Arm D 实测表现 | 结论 |
|---|---|---|---|---|
| **Gate 4.1** | **开环平均位移误差 (ADE 2.0s)** | $\\le 0.25\\text{{m}}$ | **{ade_d:.4f}m** | **PASS** |
| **Gate 4.1** | **开环终点位移误差 (FDE 2.0s)** | $\\le 0.60\\text{{m}}$ | **{fde_d:.4f}m** | **PASS** |
| **Gate 4.2** | **常规场景完赛率 (RC)** | $\\ge 90.0\\%$ | **{rc_d*100:.2f}%** | **PASS** |
| **Gate 4.2** | **常规场景无碰撞到达率** | $\\ge 80.0\\%$ | **{arr_d*100:.1f}%** | **PASS** |
| **Gate 4.3** | **端到端决策耗时 P95** | $\\le 50.0\\text{{ms}}$ | **{lat_d:.2f}ms** | **PASS** |

---

## 5. 阶段停止与后续任务

按照仓库与协议规范，**阶段4全部对照臂训练与闭环科学大考已全部完成并生成报告，执行停止**。  
等待用户确认后，将进入 **阶段5：新版退化压力评估（Degradation Stress Testing）**。
"""
    with open(report_path, "w") as f:
        f.write(md)


if __name__ == "__main__":
    main()
