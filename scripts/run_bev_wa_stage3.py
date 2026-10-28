"""Stage 3: BEV Future Latent Flow Matching Pretraining with EMA Target Encoder.

Protocol: docs/BEV_WA_ALIGNED_PROTOCOL.md (Stage 3)
Requirements:
1. Online BEV Spatiotemporal Encoder + EMA Target Teacher (tau=0.996)
2. Multi-Horizon Flow Matching (+0.5s, +1.0s, +1.5s, +2.0s)
3. Normalized Latent MSE E_norm(t) <= 0.85 across all horizons vs Copy-Current
4. Non-collapse checks: token variance >= 0.10, effective rank >= 16
5. Causal and gradient isolation (Gate 3)
6. Two independent seeds: 20261022, 20261023
"""

import os
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import argparse
import json
import time
import shlex
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from configs.stage3 import Stage3Config
from dataset.bev_wa_dataset import BEWWAJointDataset
from models.bev_wa_latent_flow import BEVWALatentFlowModel
from evaluation.latent import LatentMoments, spectrum_metrics
from mini_wa_jepa.provenance import provenance, write_json
from mini_wa_jepa.operators import sdpa_context, configure_operator_optimizations, verify_attention_fidelity
from torch.utils.tensorboard import SummaryWriter
from mini_wa_jepa.visualization.rerun_logger import RerunLogger, colorize_bev_4ch

WORKSPACE = Path("/home/auronox/Desktop/mini-wa-jepa")
STAGE3_DIR = WORKSPACE / "artifacts/bev_wa_alignment/stage3_001"
STAGE4_ENCODER_CKPT = WORKSPACE / "artifacts/restart_stage4/joint_001/best.pt"

TRAIN_MANIFEST = "artifacts/stage7_integrated/mixed_train_010/manifest.json"
VAL_MANIFEST = "artifacts/stage7_integrated/mixed_validation_010/manifest.json"


@torch.no_grad()
def evaluate_stage3(model, val_loader, device, num_eval_batches=64, num_ode_steps=10):
    """Evaluates ODE prediction vs Copy-Current baseline and representation health."""
    model.eval()
    moments = LatentMoments(64, 256, device)

    pred_errors = {0: [], 1: [], 2: [], 3: []}
    copy_errors = {0: [], 1: [], 2: [], 3: []}
    pred_errors_dyn = {0: [], 1: [], 2: [], 3: []}
    copy_errors_dyn = {0: [], 1: [], 2: [], 3: []}

    for batch_idx, batch in enumerate(val_loader):
        if num_eval_batches is not None and batch_idx >= num_eval_batches:
            break

        h = batch["history_bev"].to(device)
        ego = batch["ego_motion"].to(device)
        f = batch["future_bev"].to(device)
        m = batch["future_bev_mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            condition = model.online_encoder(h, ego)
            target_latents = model.target_encoder.encode_future(f)  # [B, 4, 64, 256]
            current_latent = model.target_encoder.encode_current(h[:, -1])  # [B, 64, 256]
            gen_latents = model.predictor.solve_euler(condition, num_steps=num_ode_steps)  # [B, 4, 64, 256]
            if getattr(model.cfg, "residual_target", False):
                residual_horizons = tuple(model.cfg.residual_horizons) if model.cfg.residual_horizons else (0, 1, 2, 3)
                for k in residual_horizons:
                    gen_latents[:, k] = gen_latents[:, k] + current_latent

        # Dynamic-token mask (top 25% of tokens where future differs from current)
        with torch.no_grad():
            cur_bev = h[:, -1].float() / 255.0
            fut_bev_n = f.float() / 255.0
            dyn = (fut_bev_n - cur_bev[:, None]).abs().mean(dim=2)  # [B,4,128,128]
            dyn_tok = F.adaptive_avg_pool2d(dyn.reshape(-1, 1, 128, 128), (8, 8)).reshape(h.shape[0], 4, 64)
            k_top = max(1, int(0.25 * 64))
            dyn_idx = dyn_tok.topk(k_top, dim=-1).indices
            dyn_mask = torch.zeros(h.shape[0], 4, 64, dtype=torch.bool, device=device)
            dyn_mask.scatter_(2, dyn_idx, True)

        # Update latent moments on target latents (at t=+1.0s or pooled)
        moments.update(target_latents[:, 1].float())

        # Measure MSE per horizon
        for k in range(4):
            valid_mask = m[:, k]
            if not valid_mask.any():
                continue
            diff_pred = (gen_latents[valid_mask, k].float() - target_latents[valid_mask, k].float()).square().mean(dim=(1, 2))
            diff_copy = (current_latent[valid_mask].float() - target_latents[valid_mask, k].float()).square().mean(dim=(1, 2))

            pred_errors[k].extend(diff_pred.cpu().tolist())
            copy_errors[k].extend(diff_copy.cpu().tolist())

            dm = dyn_mask[valid_mask, k]  # [n, 64]
            if dm.any():
                dp = (gen_latents[valid_mask, k].float() - target_latents[valid_mask, k].float()).square().mean(dim=-1)
                dc = (current_latent[valid_mask].float() - target_latents[valid_mask, k].float()).square().mean(dim=-1)
                pred_errors_dyn[k].extend(dp[dm].cpu().tolist())
                copy_errors_dyn[k].extend(dc[dm].cpu().tolist())

    health, _ = moments.finish()

    horizons = ["+0.5s", "+1.0s", "+1.5s", "+2.0s"]
    results_by_horizon = {}
    normalized_errors = {}

    for k in range(4):
        p_mse = float(np.mean(pred_errors[k])) if pred_errors[k] else float("nan")
        c_mse = float(np.mean(copy_errors[k])) if copy_errors[k] else float("nan")
        e_norm = p_mse / max(c_mse, 1e-8)
        results_by_horizon[horizons[k]] = {
            "pred_mse": p_mse,
            "copy_mse": c_mse,
            "normalized_mse": e_norm,
            "gate_passed": bool(e_norm <= 0.85),
        }
        normalized_errors[horizons[k]] = e_norm

    dynamic_normalized = {}
    for k in range(4):
        if not pred_errors_dyn[k]:
            dynamic_normalized[horizons[k]] = float("nan")
            results_by_horizon[horizons[k]]["dynamic_normalized_mse"] = float("nan")
            continue
        p_dyn = float(np.mean(pred_errors_dyn[k]))
        c_dyn = float(np.mean(copy_errors_dyn[k]))
        dynamic_normalized[horizons[k]] = p_dyn / max(c_dyn, 1e-8)
        results_by_horizon[horizons[k]]["dynamic_pred_mse"] = p_dyn
        results_by_horizon[horizons[k]]["dynamic_copy_mse"] = c_dyn
        results_by_horizon[horizons[k]]["dynamic_normalized_mse"] = p_dyn / max(c_dyn, 1e-8)
        results_by_horizon[horizons[k]]["dynamic_token_fraction"] = 0.25

    effective_rank = float(health["effective_rank"])
    token_variance = float(health["std_mean"])

    gate_rank_passed = bool(effective_rank >= 16.0)
    gate_var_passed = bool(token_variance >= 0.10)
    gate_norm_passed = all(res["gate_passed"] for res in results_by_horizon.values())

    all_passed = gate_rank_passed and gate_var_passed and gate_norm_passed

    return {
        "results_by_horizon": results_by_horizon,
        "dynamic_normalized_mse": dynamic_normalized,
        "effective_rank": effective_rank,
        "token_variance": token_variance,
        "gate_rank_passed": gate_rank_passed,
        "gate_var_passed": gate_var_passed,
        "gate_norm_passed": gate_norm_passed,
        "all_passed": all_passed,
    }


def train_stage3_seed(seed, train_ds, val_ds, device, epochs=10, batch_size=64,
                      residual_target=False, dynamic_loss_weight=0.0, residual_horizons=(),
                      rerun_save=False, rerun_live=False,
                      masked_completion_weight=0.0, mask_ratio=0.25):
    print(f"\n=======================================================")
    print(f"Training Stage 3 Multi-Horizon Flow Matcher (Seed {seed})")
    print(f"=======================================================")
    seed_start = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    cfg = Stage3Config(seed=seed, residual_target=residual_target, dynamic_loss_weight=dynamic_loss_weight,
                       residual_horizons=tuple(residual_horizons), epochs=epochs, batch_size=batch_size,
                       masked_completion_weight=masked_completion_weight, mask_ratio=mask_ratio)
    model = BEVWALatentFlowModel(cfg, encoder_ckpt=STAGE4_ENCODER_CKPT).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    tb_dir = STAGE3_DIR / "tensorboard" / f"seed_{seed}"
    writer = SummaryWriter(log_dir=str(tb_dir))
    rerun_logger = None
    if rerun_save or rerun_live:
        rerun_dir = STAGE3_DIR / "rerun"
        rerun_dir.mkdir(parents=True, exist_ok=True)
        rerun_logger = RerunLogger(app_id=f"stage3_seed_{seed}", spawn=rerun_live,
                                   save_path=str(rerun_dir / f"stage3_seed_{seed}.rrd"))
    epoch_log_path = STAGE3_DIR / f"epochs_seed_{seed}.jsonl"
    epoch_log_path.parent.mkdir(parents=True, exist_ok=True)

    history_logs = []

    for epoch in range(1, epochs + 1):
        model.train()
        model.target_encoder.eval()
        epoch_losses = []
        epoch_flow_losses = []
        epoch_mask_losses = []
        masked_tokens = eligible_tokens = 0
        t0 = time.time()

        for batch in train_loader:
            h = batch["history_bev"].to(device)
            ego = batch["ego_motion"].to(device)
            f = batch["future_bev"].to(device)
            m = batch["future_bev_mask"].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, terms = model.forward_train(h, ego, f, m)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            # Update EMA target encoder
            model.update_teacher(tau=0.996)

            epoch_losses.append(terms["total_loss"])
            epoch_flow_losses.append(terms["flow_loss"])
            epoch_mask_losses.append(terms["loss_mask"])
            masked_tokens += terms["masked_token_count"]
            eligible_tokens += terms["eligible_token_count"]

        scheduler.step()
        train_dt = time.time() - t0
        mean_loss = float(np.mean(epoch_losses))
        mean_flow = float(np.mean(epoch_flow_losses))
        mean_mask = float(np.mean(epoch_mask_losses))
        actual_ratio = masked_tokens / max(eligible_tokens, 1)

        # Quick validation evaluation (32 batches)
        eval_metrics = evaluate_stage3(model, val_loader, device, num_eval_batches=32, num_ode_steps=10)
        norm_summary = ", ".join([f"{h}: {res['normalized_mse']:.3f}" for h, res in eval_metrics["results_by_horizon"].items()])
        print(
            f"Epoch {epoch:02d}/{epochs} | flow={mean_flow:.4f} | mask={mean_mask:.4f} | "
            f"E_norm=[{norm_summary}] | dyn=["
            + ", ".join(f"{h}: {eval_metrics.get('dynamic_normalized_mse', {}).get(h, float('nan')):.3f}" for h in eval_metrics["results_by_horizon"])
            + f"] | rank={eval_metrics['effective_rank']:.1f} | var={eval_metrics['token_variance']:.3f} | gate={eval_metrics['all_passed']}",
            flush=True,
        )

        # TensorBoard: one writer per seed, one tag per metric; epoch is the x-axis.
        writer.add_scalar("train/loss", mean_loss, epoch)
        writer.add_scalar("train/flow_loss", mean_flow, epoch)
        writer.add_scalar("train/loss_mask", mean_mask, epoch)
        writer.add_scalar("train/mask_ratio_actual", actual_ratio, epoch)
        for h, res in eval_metrics["results_by_horizon"].items():
            writer.add_scalar(f"val/E_norm/{h}", res["normalized_mse"], epoch)
            dyn = eval_metrics.get("dynamic_normalized_mse", {}).get(h, float("nan"))
            if dyn == dyn:
                writer.add_scalar(f"val/dynamic_E_norm/{h}", dyn, epoch)
        writer.add_scalar("health/rank", eval_metrics["effective_rank"], epoch)
        writer.add_scalar("health/variance", eval_metrics["token_variance"], epoch)
        writer.add_scalar("gate/passed", float(eval_metrics["all_passed"]), epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.flush()

        epoch_record = {
            "epoch": epoch, "seed": seed, "train_loss": mean_loss, "train_flow_loss": mean_flow,
            "train_time_sec": train_dt,
            "flow_loss": mean_flow, "mask_loss": mean_mask, "mask_ratio_actual": actual_ratio,
            "masked_token_count": masked_tokens, "eligible_token_count": eligible_tokens,
            "rank": eval_metrics["effective_rank"], "var": eval_metrics["token_variance"],
            "gate": eval_metrics["all_passed"], "lr": optimizer.param_groups[0]["lr"],
            "time_s": time.time() - t0,
            "dynamic": eval_metrics["dynamic_normalized_mse"],
            "E_norm": {h: res["normalized_mse"] for h, res in eval_metrics["results_by_horizon"].items()},
            "dynamic_E_norm": eval_metrics.get("dynamic_normalized_mse", {}),
            "effective_rank": eval_metrics["effective_rank"], "token_variance": eval_metrics["token_variance"],
            "gate_passed": eval_metrics["all_passed"],
        }
        if rerun_logger is not None:
            # Monitoring must not consume the training/evaluation RNG stream.
            with torch.random.fork_rng(devices=[device.index or 0]), torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                torch.manual_seed(cfg.visualization_seed)
                vbatch = next(iter(val_loader))
                hb = vbatch["history_bev"][:1].to(device)
                eb = vbatch["ego_motion"][:1].to(device)
                fb = vbatch["future_bev"][:1].to(device)
                cond = model.online_encoder(hb, eb)
                tgt = model.target_encoder.encode_future(fb)
                cur = model.target_encoder.encode_current(hb[:, -1])
                gen = model.predictor.solve_euler(cond, num_steps=10)
                if getattr(model.cfg, "residual_target", False):
                    rh = tuple(model.cfg.residual_horizons) if model.cfg.residual_horizons else (0, 1, 2, 3)
                    for k in rh:
                        gen[:, k] = gen[:, k] + cur
            rerun_logger.set_step(epoch)
            rerun_logger.log_bev(hb[0, -1].cpu().numpy())
            # Separate paths preserve all four history and future frames at this epoch.
            for frame in range(4):
                rerun_logger.log_camera(colorize_bev_4ch(hb[0, frame].cpu().numpy()),
                                        name=f"history/frame_{frame}")
                rerun_logger.log_camera(colorize_bev_4ch(fb[0, frame].cpu().numpy()),
                                        name=f"future/horizon_{frame}")
            rerun_logger.log_context(epoch=epoch, sample_id=0, seed=seed,
                                      future_valid=vbatch["future_bev_mask"][0].tolist())
            save_stage3_snapshot(seed, epoch, hb, fb, gen, tgt, vbatch["future_bev_mask"][0])
            rerun_logger.log_latents(predicted=gen[0].float().cpu().numpy(), ground_truth=tgt[0].float().cpu().numpy())
            rerun_logger.log_metrics({
                **{f"E_norm/{h}": res["normalized_mse"] for h, res in eval_metrics["results_by_horizon"].items()},
                "flow_loss": mean_flow, "mask_loss": mean_mask, "mask_ratio_actual": actual_ratio,
                "rank": float(eval_metrics["effective_rank"]),
                "variance": float(eval_metrics["token_variance"]),
            })

        epoch_record["time_s"] = time.time() - t0
        epoch_record["norm_mse"] = epoch_record["E_norm"]
        with open(epoch_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
            handle.flush()
        history_logs.append(epoch_record)

    # Final thorough evaluation on full validation set
    print("\nRunning full validation set evaluation...")
    final_metrics = evaluate_stage3(model, val_loader, device, num_eval_batches=None, num_ode_steps=10)
    writer.close()
    if rerun_logger is not None:
        rerun_logger.close()

    ckpt_path = STAGE3_DIR / f"flow_latent_seed_{seed}.pt"
    torch.save({
        "seed": seed,
        "config": cfg,
        "online_encoder": model.online_encoder.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
        "predictor": model.predictor.state_dict(),
        "final_metrics": final_metrics,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}", flush=True)
    print(f"Seed {seed} summary | E_norm="
          f"{[round(v['normalized_mse'], 4) for v in final_metrics['results_by_horizon'].values()]}"
          f" | dyn={final_metrics['dynamic_normalized_mse']} | rank={final_metrics['effective_rank']:.1f}"
          f" | var={final_metrics['token_variance']:.3f} | gate={final_metrics['all_passed']}"
          f" | time_s={time.time() - seed_start:.1f}", flush=True)

    return {
        "seed": seed,
        "history": history_logs,
        "final_metrics": final_metrics,
        "ckpt": str(ckpt_path), "config": asdict(cfg),
        "time_s": time.time() - seed_start,
        "mask_ratio_actual": sum(r["masked_token_count"] for r in history_logs) / max(1, sum(r["eligible_token_count"] for r in history_logs)),
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def main():
    global STAGE3_DIR
    parser = argparse.ArgumentParser(description="Stage 3 BEV future-latent flow pretraining")
    parser.add_argument("--output-dir", type=Path, default=STAGE3_DIR)
    parser.add_argument("--residual", action="store_true", help="Predict future_latent - current_latent")
    parser.add_argument("--residual-horizons", nargs="+", type=int, choices=(0, 1, 2, 3), default=None,
                        help="Hybrid residual horizons (0=+0.5s, 1=+1.0s, 2=+1.5s, 3=+2.0s); omit = all")
    parser.add_argument("--dynamic-loss-weight", type=float, default=0.0, help="Upweight tokens where future differs from current")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rerun-save", action="store_true", help="Save per-seed Rerun .rrd timeline")
    parser.add_argument("--rerun", action="store_true", help="Spawn live Rerun viewer during training")
    parser.add_argument("--masked-completion-weight", type=float, default=0.0)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 <= args.mask_ratio <= 1 or args.masked_completion_weight < 0:
        parser.error("mask-ratio must be in [0,1]; masked-completion-weight must be nonnegative")
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("epochs >= 1 and batch-size >= 2 required")
    run_start = time.time()
    STAGE3_DIR = args.output_dir.resolve()
    print("================================================================================")
    print("Stage 3: BEV Future Latent Flow Matching Pretraining")
    print("================================================================================")
    print(f"Output: {STAGE3_DIR} | residual={args.residual} | dynamic_loss_weight={args.dynamic_loss_weight}")
    STAGE3_DIR.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("Stage 3 requires PyTorch ROCm on an AMD GPU")
    device = torch.device("cuda")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    print("\n[Step 1] Loading Datasets...")
    train_ds = BEWWAJointDataset(TRAIN_MANIFEST, split="train", eligible_action_only=False, preload_ram=True)
    val_ds = BEWWAJointDataset(VAL_MANIFEST, split="validation", eligible_action_only=False, preload_ram=True)
    print(f"Train Dataset: {train_ds.stats()}")
    print(f"Validation Dataset: {val_ds.stats()}")

    residual_horizons = tuple(args.residual_horizons) if args.residual_horizons else ()
    cfg = Stage3Config(residual_target=args.residual, dynamic_loss_weight=args.dynamic_loss_weight,
                       residual_horizons=residual_horizons, epochs=args.epochs, batch_size=args.batch_size,
                       masked_completion_weight=args.masked_completion_weight, mask_ratio=args.mask_ratio)
    operator_applied = configure_operator_optimizations(vendor="AMD", allow_tf32=False, cudnn_benchmark=False, deterministic=True)
    seeds = [20261022, 20261023]
    results = {
        "protocol_version": "BEV_WA_ALIGNED_V1",
        "date_user_timezone": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "provenance": provenance(cfg), "config": asdict(cfg),
        "torch_version": torch.__version__, "rocm_version": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "masked_completion_weight": args.masked_completion_weight, "mask_ratio": args.mask_ratio,
        "stage": 3,
        "title": "BEV Future Latent Flow Matching Pretraining",
        "dataset_stats": {
            "train": train_ds.stats(),
            "validation": val_ds.stats(),
        },
        "training_runs": {},
        "passed": True,
        "sdpa_backend": cfg.sdpa_backend,
        "operator_optimizations": operator_applied,
        "residual_target": args.residual,
        "residual_horizons": list(residual_horizons),
        "dynamic_loss_weight": args.dynamic_loss_weight,
        "output_dir": str(STAGE3_DIR),
        "tensorboard_dir": str(STAGE3_DIR / "tensorboard"),
        "rerun_dir": str(STAGE3_DIR / "rerun"),
    }

    write_json(STAGE3_DIR / "run_manifest.json", results)
    with sdpa_context(cfg.sdpa_backend, vendor="AMD"):
        for seed in seeds:
            run_res = train_stage3_seed(seed, train_ds, val_ds, device, epochs=args.epochs, batch_size=args.batch_size,
                                        residual_target=args.residual, dynamic_loss_weight=args.dynamic_loss_weight,
                                        residual_horizons=residual_horizons,
                                        rerun_save=args.rerun_save, rerun_live=args.rerun,
                                        masked_completion_weight=args.masked_completion_weight, mask_ratio=args.mask_ratio)
            run_res["sdpa_backend"] = cfg.sdpa_backend
            run_res["operator_optimizations"] = operator_applied
            results["training_runs"][f"seed_{seed}"] = run_res
            if not run_res["final_metrics"]["all_passed"]:
                results["passed"] = False
            write_json(STAGE3_DIR / "results_partial.json", results)

    results["sdpa_verification"] = verify_attention_fidelity(
        torch.randn(16, 4, 64, 64, device=device, dtype=torch.bfloat16),
        torch.randn(16, 4, 64, 64, device=device, dtype=torch.bfloat16),
        torch.randn(16, 4, 64, 64, device=device, dtype=torch.bfloat16),
        candidate_backend=cfg.sdpa_backend,
        vendor="AMD",
    )

    results["time_s"] = time.time() - run_start
    results["mask_ratio_actual"] = {k: v["mask_ratio_actual"] for k, v in results["training_runs"].items()}
    # Save results.json
    results_path = STAGE3_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Step 2] Results saved to {results_path}")

    # Generate Markdown Report
    report_path = STAGE3_DIR / "REPORT_STAGE3.md"
    generate_stage3_report(results, report_path)
    print(f"[Step 3] Stage 3 Report generated at {report_path}")

    print("\n================================================================================")
    print(f"Stage 3 Complete! All Gates Passed: {results['passed']}")
    print("================================================================================")


def generate_stage3_report(results, report_path):
    horizons = ["+0.5s", "+1.0s", "+1.5s", "+2.0s"]
    lines = ["# Stage 3 v3 — masked future patch completion", "",
             f"日期：{results['date_user_timezone']}；gate = **{results['passed']}**。本次执行仅 Stage 3。", "",
             "## 配置与判定口径", "", "```json", json.dumps(results["config"], indent=2), "```", "",
             f"SDPA = {results['sdpa_backend']}；operator flags = `{json.dumps(results['operator_optimizations'])}`。",
             f"PyTorch {results['torch_version']} / ROCm {results['rocm_version']} / {results['gpu']}。",
             "不使用 torch.compile、外部 flash-attn/xformers/Triton 或外部数据/权重；AUTO 可使用 PyTorch 自带 ROCm AOTriton 内核。",
             "Flow 保持 hybrid：horizons 0/1 residual，其余 absolute。Completion 重建 absolute EMA targets，仅训练时启用；EMA stop-grad。",
             "Mask 为每 sample/horizon/token 独立 Bernoulli，排除 future_bev_mask=False；actual ratio 的分母为有效且选中的 token。",
             "最终指标用完整验证集与 10-step Euler flow；每 epoch 趋势仅前 32 batches，不能替代最终结果。",
             "门槛：所有时域 E_norm <= 0.85，rank >= 16，var（历史字段实际为 std_mean）>= 0.10。动态子集为变化最大的 25% token。",
             "保留现有 variance penalty（来自 stop-grad EMA target，属于无梯度监测项）；没有改变该历史训练定义。", "",
             "## 最终完整验证集", "", "| Seed | +0.5s | +1.0s | +1.5s | +2.0s | rank | var/std_mean | gate |", "|---|---|---|---|---|---|---|---|"]
    for run in results["training_runs"].values():
        m = run["final_metrics"]
        vals = " | ".join(f"{m['results_by_horizon'][h]['normalized_mse']:.6f}" for h in horizons)
        lines.append(f"| {run['seed']} | {vals} | {m['effective_rank']:.3f} | {m['token_variance']:.6f} | {m['all_passed']} |")
    lines += ["", "## Dynamic 子集", "", "| Seed | +0.5s | +1.0s | +1.5s | +2.0s |", "|---|---|---|---|---|"]
    for run in results["training_runs"].values():
        vals = " | ".join(f"{run['final_metrics']['dynamic_normalized_mse'][h]:.6f}" for h in horizons)
        lines.append(f"| {run['seed']} | {vals} |")
    lines += ["", "## 历史对比（读取原 results.json，未重跑）", "",
              "| Experiment | Seed | Epochs | +0.5s | +1.0s | +1.5s | +2.0s | gate |", "|---|---|---|---|---|---|---|---|"]
    for name in ("stage3_001", "stage3_residual_001", "stage3_hybrid_001", STAGE3_DIR.name):
        data = results if name == STAGE3_DIR.name else json.loads((STAGE3_DIR.parent / name / "results.json").read_text())
        for run in data["training_runs"].values():
            m = run["final_metrics"]
            vals = " | ".join(f"{m['results_by_horizon'][h]['normalized_mse']:.6f}" for h in horizons)
            lines.append(f"| {name} | {run['seed']} | {len(run['history'])} | {vals} | {m['all_passed']} |")
    lines += ["", "旧 hybrid 的完整验证 +1.5s 为约 0.95/0.94，高于 0.85；此前‘+1.5s 已过’仅符合前 32 batches 的 epoch 指标，不能用于最终 gate。",
              "各实验 EMA teacher 随训练变化，E_norm 分母也变化；历史对比不等于固定 teacher 的严格因果消融。新增 mask 抽样改变 RNG 消耗；Rerun 抽样使用独立 RNG，不扰动训练。", "",
              "## 每 epoch 趋势（32 batches）", ""]
    for run in results["training_runs"].values():
        lines += [f"### Seed {run['seed']}", "", "| Epoch | flow | mask | +0.5s | +1.0s | +1.5s | +2.0s | ratio | rank | var | gate | seconds |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in run["history"]:
            vals = " | ".join(f"{r['E_norm'][h]:.4f}" for h in horizons)
            lines.append(f"| {r['epoch']} | {r['flow_loss']:.5f} | {r['mask_loss']:.5f} | {vals} | {r['mask_ratio_actual']:.5f} | {r['rank']:.2f} | {r['var']:.4f} | {r['gate']} | {r['time_s']:.1f} |")
        lines.append("")
    worst = max((m['normalized_mse'], run['seed'], h) for run in results['training_runs'].values() for h, m in run['final_metrics']['results_by_horizon'].items())
    lines += ["## 结论与下一步", "", f"Gate **{results['passed']}**；最差时域 {worst[2]}，seed {worst[1]}，E_norm={worst[0]:.6f}。",
              "若不通过：保留负结果，先做数据覆盖与采集器审计（短时域动态变化、时空对齐、有效 future 覆盖），再决定是否采集；不要继续盲目叠加损失。Completion loss 降低不证明 history-only flow 生成改善。",
              "目标/损失方向仅作为待验证假设：可见 future 辅助任务与无 future 的生成任务可能存在差异；本实验不能单独认定数据或目标为根因。若没有明确覆盖或目标缺陷，接受本轮负结果。",
              "本次停止于 Stage 3 报告；未启动 Stage 4/5–9。", "", "## 复现与交付", "", "```bash", results['command'], "```", "",
              f"源码 commit：`{results['provenance']['git_commit']}`（dirty 工作区；逐文件 hash 见 run_manifest.json）。",
              f"GOAL SHA256：`{results['provenance']['goal_sha256']}`。",
              f"总耗时：{results['time_s']:.1f}s。TensorBoard：`{results['tensorboard_dir']}`。Rerun：`{results['rerun_dir']}`。",
              f"SDPA fidelity：`{json.dumps(results['sdpa_verification'])}`。", ""]
    for run in results['training_runs'].values():
        lines += [f"- Seed {run['seed']}: checkpoint `{run['ckpt']}`；耗时 {run['time_s']:.1f}s；mask_ratio_actual {run['mask_ratio_actual']:.6f}；峰值 allocated VRAM {run['peak_vram_allocated_bytes']/2**30:.2f} GiB。",
                  f"  Epoch JSONL：`epochs_seed_{run['seed']}.jsonl`；RRD：`rerun/stage3_seed_{run['seed']}.rrd`；原始可视化数组/图：`figures/seed_{run['seed']}/`。"]
    report_path.write_text("\n".join(lines) + "\n")


def save_stage3_snapshot(seed, epoch, history, future, predicted, target, valid):
    """Actual latent prediction/target figures; never present latents as decoded BEV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = STAGE3_DIR / "figures" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    pred, tgt = predicted[0].float().cpu().numpy(), target[0].float().cpu().numpy()
    hist, fut = history[0].cpu().numpy(), future[0].cpu().numpy()
    np.savez_compressed(out / f"epoch_{epoch:02d}.npz", history=hist, future=fut,
                        predicted=pred, target=tgt, future_valid=valid.numpy())
    fig, axes = plt.subplots(3, 4, figsize=(12, 7), constrained_layout=True)
    for k in range(4):
        axes[0, k].imshow(colorize_bev_4ch(fut[k]))
        axes[0, k].set_title(f"Future +{(k+1)*0.5:.1f}s valid={bool(valid[k])}")
        lo, hi = min(pred[k].min(), tgt[k].min()), max(pred[k].max(), tgt[k].max())
        axes[1, k].imshow(pred[k], aspect="auto", vmin=lo, vmax=hi)
        axes[1, k].set_title("Predicted latent (token x channel)")
        axes[2, k].imshow(tgt[k], aspect="auto", vmin=lo, vmax=hi)
        axes[2, k].set_title("EMA target latent (token x channel)")
    fig.suptitle(f"Stage 3 masked seed={seed}, epoch={epoch}; fixed sample 0; latents are not decoded images")
    fig.savefig(out / f"epoch_{epoch:02d}.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
