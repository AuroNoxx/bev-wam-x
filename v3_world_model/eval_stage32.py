"""Offline acceptance evaluation, counterfactual probe, latency benchmark, and Car-OS 3D SR visual audit for Stage 3.2."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from v1_dual_system.checkpoint_io import VerifiedCheckpointIO

from .geometry import DT, H, W
from .perception_model import sdpa_context
from .sr_viewer import FontPack, SRViewer, prediction_scene
from .train_stage32 import LatentRolloutDataset, Stage32WorldModel

COUNTERFACTUAL_ACTIONS = {
    "straight": (0.0, 0.4, 0.0),
    "left": (-0.35, 0.3, 0.0),
    "right": (0.35, 0.3, 0.0),
    "brake": (0.0, 0.0, 0.8),
}


def load_stage32_module(ckpt_path, device):
    """Load Stage32WorldModel and metadata from a VerifiedCheckpointIO checkpoint without redundant disk I/O."""
    ck = VerifiedCheckpointIO().load_checkpoint(Path(ckpt_path), map_location="cpu")
    cfg = dict(ck["task_state"])
    module = Stage32WorldModel({**cfg, "perception_ckpt": None})
    module.load_state_dict(ck["state_dict"], strict=True)
    module = module.to(device).eval()
    return module, cfg, {
        "checkpoint": str(ckpt_path),
        "epoch": int(ck.get("epoch", -1)),
        "global_step": int(ck.get("global_step", -1)),
        "trainable_parameters": sum(p.numel() for p in module.world_model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in module.parameters()),
    }


def integrate_ego_delta(delta_seq):
    """Integrate (B, H, 3) v2 previous-in-current FLU deltas into forward trajectory (B, H, 2) and final yaw (B,)."""
    bsz, horizon, _ = delta_seq.shape
    x = torch.zeros(bsz, device=delta_seq.device, dtype=torch.float32)
    y = torch.zeros(bsz, device=delta_seq.device, dtype=torch.float32)
    yaw = torch.zeros(bsz, device=delta_seq.device, dtype=torch.float32)
    steps = []
    for k in range(horizon):
        dx, dy, dyaw = delta_seq[:, k, 0].float(), delta_seq[:, k, 1].float(), delta_seq[:, k, 2].float()
        yaw = yaw + dyaw
        c, s = torch.cos(yaw), torch.sin(yaw)
        x = x + (-dx) * c - (-dy) * s
        y = y + (-dx) * s + (-dy) * c
        steps.append(torch.stack((x, y), dim=-1))
    return torch.stack(steps, dim=1), yaw


def _per_sample_decode_metrics(pred_dec, tgt_dec):
    """Compute per-sample decode consistency loss and segmentation intersection/union counts."""
    pl = pred_dec["lane_logits"].float()
    tl = tgt_dec["lane_logits"].float()
    lane = F.smooth_l1_loss(pl, tl, reduction="none").mean(dim=(1, 2, 3)) + 2.0 * F.mse_loss(
        pl.sigmoid(), tl.sigmoid(), reduction="none"
    ).mean(dim=(1, 2, 3))

    po = pred_dec["occupancy_logits"].float()
    to = tgt_dec["occupancy_logits"].float()
    pfo = pred_dec["future_occupancy_logits"].float()
    tfo = tgt_dec["future_occupancy_logits"].float()
    occ = F.smooth_l1_loss(po, to, reduction="none").mean(dim=(1, 2, 3)) + F.smooth_l1_loss(
        pfo, tfo, reduction="none"
    ).mean(dim=(1, 2, 3))

    p_obj = pred_dec["obj_logits"].float()
    t_obj = tgt_dec["obj_logits"].float()
    box_w = (0.1 + t_obj.sigmoid()).unsqueeze(-1)
    box_loss = (F.smooth_l1_loss(pred_dec["boxes"].float(), tgt_dec["boxes"].float(), reduction="none") * box_w).mean(
        dim=(1, 2)
    )
    obj = (
        F.smooth_l1_loss(p_obj, t_obj, reduction="none").mean(dim=1)
        + box_loss
        + 0.5 * F.smooth_l1_loss(pred_dec["cls_logits"].float(), tgt_dec["cls_logits"].float(), reduction="none").mean(
            dim=(1, 2)
        )
    )
    total = lane + occ + obj

    stats = {"decode_loss": total}
    for idx, cname in ((0, "lane_drivable"), (1, "lane_dashed"), (2, "lane_solid")):
        p_m = pl[:, idx] > 0.0
        t_m = tl[:, idx] > 0.0
        stats[f"{cname}_inter"] = (p_m & t_m).sum(dim=(1, 2)).float()
        stats[f"{cname}_union"] = (p_m | t_m).sum(dim=(1, 2)).float()
    p_occ = po > 0.0
    t_occ = to > 0.0
    stats["occ_inter"] = (p_occ & t_occ).sum(dim=(1, 2, 3)).float()
    stats["occ_union"] = (p_occ | t_occ).sum(dim=(1, 2, 3)).float()
    return stats


def _finite_tensor_dict(*dicts):
    def _check(obj):
        if torch.is_tensor(obj):
            return bool(torch.isfinite(obj).all())
        if isinstance(obj, dict):
            return all(_check(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return all(_check(v) for v in obj)
        return True

    return all(_check(d) for d in dicts)


def _tag_validation_windows(val_ds):
    tags = []
    manifest_eps = {e["id"]: e for e in val_ds.manifest["episodes"]}
    for ep_id, start in val_ds.windows:
        ep = val_ds._load_episode(ep_id)
        is_junc = bool((ep["lane_state"][start + 1:start + val_ds.horizon + 1, 2] > 0).any())
        tags.append({
            "ep_id": ep_id,
            "start": int(start),
            "map": manifest_eps[ep_id]["map"],
            "is_junction": is_junc,
        })
    return tags


def _aggregate_group(records, mask):
    subset = [r for r, m in zip(records, mask) if m]
    if not subset:
        return {"count": 0}
    scalar_keys = [
        k for k in subset[0]
        if not k.endswith("_inter") and not k.endswith("_union") and k != "nonfinite"
    ]
    out = {
        "count": len(subset),
        "any_nonfinite": any(r["nonfinite"] for r in subset),
    }
    for k in scalar_keys:
        out[k] = float(np.mean([r[k] for r in subset]))
    for seg in ("lane_drivable_t1", "lane_drivable_t10", "lane_dashed_t1", "lane_dashed_t10",
                "lane_solid_t1", "lane_solid_t10", "occ_t1", "occ_t10"):
        inter = sum(r[f"{seg}_inter"] for r in subset)
        union = sum(r[f"{seg}_union"] for r in subset)
        out[f"{seg}_iou"] = float(inter / union) if union > 0 else 0.0
    return out


def evaluate_validation(module, val_ds, tags, *, batch_size=32, sdpa="aotriton", device="cuda"):
    """Run full validation split audit and break metrics down by map and junction status."""
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    records = []
    pos_weight = module.risk_pos_weight.float().to(device)
    offset = 0
    for batch in loader:
        bsz = batch["z"].shape[0]
        batch_gpu = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        z_seq = batch_gpu["z"]
        kin = batch_gpu["kinematics"]
        actions = batch_gpu["actions"]
        horizon = actions.shape[1]

        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")), sdpa_context(sdpa):
            out = module.world_model.rollout(z_seq[:, 0], kin[:, 0, :2], actions)

        t10 = horizon - 1
        t5 = min(4, t10)
        z_pred_f = out["z_next"].float()
        z_tgt_f = z_seq[:, 1:].float()
        cos_step = F.cosine_similarity(z_pred_f, z_tgt_f, dim=-1, eps=1e-4).mean(dim=-1)
        l1_step = (z_pred_f - z_tgt_f).abs().mean(dim=(-1, -2))

        speed_err = (out["next_kinematics"][..., 0].float() * 30.0 - kin[:, 1:, 0].float() * 30.0).abs()
        yaw_err = (out["next_kinematics"][..., 1].float() - kin[:, 1:, 1].float()).abs().mean(dim=1)
        delta_xy_err = torch.linalg.norm(
            out["next_delta"][..., :2].float() - batch_gpu["delta"][..., :2].float(), dim=-1
        ).mean(dim=1)
        pred_xy, _ = integrate_ego_delta(out["next_delta"])
        gt_xy, _ = integrate_ego_delta(batch_gpu["delta"])
        traj_dist = torch.linalg.norm(pred_xy - gt_xy, dim=-1)

        inv_thw_err = (out["inv_thw"].float() - batch_gpu["inv_thw"].float()).abs().mean(dim=(1, 2))
        nav_err = (out["nav_progress"].float() - batch_gpu["nav"].float()).abs().mean(dim=(1, 2))
        risk_bce = F.binary_cross_entropy_with_logits(
            out["risk_logits"].float(), batch_gpu["flags"].float(), pos_weight=pos_weight, reduction="none"
        ).mean(dim=(1, 2))
        risk_prob = out["risk_logits"].float().sigmoid().mean(dim=(1, 2))

        dec_stats = {}
        batch_finite = _finite_tensor_dict(out)
        with torch.no_grad(), torch.autocast(device.type, enabled=False), sdpa_context("math"):
            for tag_k, k in (("t1", 0), ("t10", t10)):
                p_dec = module.perception.decode_perception(out["z_next"][:, k].float(), bev=None)
                t_dec = module.perception.decode_perception(z_seq[:, k + 1].float(), bev=None)
                batch_finite = batch_finite and _finite_tensor_dict(p_dec, t_dec)
                dec_stats[tag_k] = _per_sample_decode_metrics(p_dec, t_dec)

        for i in range(bsz):
            d1, d10 = dec_stats["t1"], dec_stats["t10"]
            rec = {
                "nonfinite": not batch_finite,
                "latent_cos_t1": float(cos_step[i, 0]),
                "latent_cos_t5": float(cos_step[i, t5]),
                "latent_cos_t10": float(cos_step[i, t10]),
                "latent_cos_mean": float(cos_step[i].mean()),
                "latent_l1_t1": float(l1_step[i, 0]),
                "latent_l1_t10": float(l1_step[i, t10]),
                "latent_l1_mean": float(l1_step[i].mean()),
                "speed_mae_t1_mps": float(speed_err[i, 0]),
                "speed_mae_t5_mps": float(speed_err[i, t5]),
                "speed_mae_t10_mps": float(speed_err[i, t10]),
                "speed_mae_mean_mps": float(speed_err[i].mean()),
                "yaw_rate_mae_mean_radps": float(yaw_err[i]),
                "delta_xy_mae_mean_m": float(delta_xy_err[i]),
                "ade_1s_m": float(traj_dist[i].mean()),
                "fde_1s_m": float(traj_dist[i, t10]),
                "inv_thw_mae": float(inv_thw_err[i]),
                "nav_mae": float(nav_err[i]),
                "risk_bce": float(risk_bce[i]),
                "risk_mean_prob": float(risk_prob[i]),
                "decode_loss_t1": float(d1["decode_loss"][i]),
                "decode_loss_t10": float(d10["decode_loss"][i]),
            }
            for seg in ("lane_drivable", "lane_dashed", "lane_solid", "occ"):
                rec[f"{seg}_t1_inter"] = float(d1[f"{seg}_inter"][i])
                rec[f"{seg}_t1_union"] = float(d1[f"{seg}_union"][i])
                rec[f"{seg}_t10_inter"] = float(d10[f"{seg}_inter"][i])
                rec[f"{seg}_t10_union"] = float(d10[f"{seg}_union"][i])
            records.append(rec)
        offset += bsz

    assert len(records) == len(tags)
    return {
        "overall": _aggregate_group(records, [True] * len(tags)),
        "Town12": _aggregate_group(records, [t["map"] == "Town12" for t in tags]),
        "Town10HD_Opt": _aggregate_group(records, [t["map"] == "Town10HD_Opt" for t in tags]),
        "junction": _aggregate_group(records, [t["is_junction"] for t in tags]),
    }


def _select_counterfactual_indices(val_ds, tags):
    town12_idx, urban_junc_idx, urban_other_idx = [], [], []
    for idx, tag in enumerate(tags):
        ep = val_ds._load_episode(tag["ep_id"])
        v0 = float(ep["ego_state"][tag["start"], 0])
        if v0 <= 4.0:
            continue
        if tag["map"] == "Town10HD_Opt":
            (urban_junc_idx if tag["is_junction"] else urban_other_idx).append(idx)
        elif tag["map"] == "Town12":
            town12_idx.append(idx)
    urban_idx = (urban_junc_idx + urban_other_idx)[:32]
    return town12_idx[:32] + urban_idx


def _summarize_counterfactual(rows, mask):
    sub = [r for r, m in zip(rows, mask) if m]
    if not sub:
        return {"count": 0}
    out = {
        "count": len(sub),
        "any_nonfinite": any(r["nonfinite"] for r in sub),
    }
    for name in COUNTERFACTUAL_ACTIONS:
        out[f"v_end_{name}_mps"] = float(np.mean([r[f"v_end_{name}"] for r in sub]))
    for name in ("left", "straight", "right"):
        out[f"cum_dy_{name}_m"] = float(np.mean([r[f"cum_dy_{name}"] for r in sub]))
        out[f"cum_dyaw_{name}_rad"] = float(np.mean([r[f"cum_dyaw_{name}"] for r in sub]))
    out["brake_slows_frac"] = float(np.mean([r["brake_slows"] for r in sub]))
    out["lat_diverges_opposite_frac"] = float(np.mean([r["lat_diverges_opposite"] for r in sub]))
    out["yaw_diverges_opposite_frac"] = float(np.mean([r["yaw_diverges_opposite"] for r in sub]))
    out["both_diverge_opposite_frac"] = float(np.mean([r["both_diverge_opposite"] for r in sub]))
    out["latent_l2_left_vs_right_t10"] = float(np.mean([r["latent_l2_lr"] for r in sub]))
    out["latent_l2_brake_vs_straight_t10"] = float(np.mean([r["latent_l2_bs"] for r in sub]))
    out["drivable_l1_left_vs_right_t10"] = float(np.mean([r["drivable_l1_lr"] for r in sub]))
    return out


def evaluate_counterfactual(module, val_ds, tags, *, sdpa="aotriton", device="cuda"):
    """Roll out identical initial states under four constant counterfactual commands."""
    sel_indices = _select_counterfactual_indices(val_ds, tags)
    if not sel_indices:
        return {"overall": {"count": 0}}
    items = [val_ds[idx] for idx in sel_indices]
    sel_tags = [tags[idx] for idx in sel_indices]
    z0 = torch.stack([it["z"][0] for it in items]).to(device)
    kin0 = torch.stack([it["kinematics"][0, :2] for it in items]).to(device)
    bsz = z0.shape[0]
    horizon = val_ds.horizon

    outs, driv_masks = {}, {}
    nonfinite = False
    for name, act in COUNTERFACTUAL_ACTIONS.items():
        act_seq = torch.tensor(act, device=device, dtype=torch.float32).view(1, 1, 3).expand(bsz, horizon, 3)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")), sdpa_context(sdpa):
            out = module.world_model.rollout(z0, kin0, act_seq)
        nonfinite = nonfinite or (not _finite_tensor_dict(out))
        outs[name] = out
        if name in ("left", "right"):
            with torch.no_grad(), torch.autocast(device.type, enabled=False), sdpa_context("math"):
                dec = module.perception.decode_perception(out["z_next"][:, -1].float(), bev=None)
            nonfinite = nonfinite or (not _finite_tensor_dict(dec))
            driv_masks[name] = dec["lane_logits"][:, 0].float().sigmoid()

    xy_by, yaw_by = {}, {}
    for name in ("left", "straight", "right"):
        xy, yaw = integrate_ego_delta(outs[name]["next_delta"])
        xy_by[name] = xy[:, -1]
        yaw_by[name] = yaw

    lat_lr = torch.linalg.norm(
        outs["left"]["z_next"][:, -1].float() - outs["right"]["z_next"][:, -1].float(), dim=-1
    ).mean(dim=-1)
    lat_bs = torch.linalg.norm(
        outs["brake"]["z_next"][:, -1].float() - outs["straight"]["z_next"][:, -1].float(), dim=-1
    ).mean(dim=-1)
    driv_diff = (driv_masks["left"] - driv_masks["right"]).abs().mean(dim=(-1, -2))

    rows = []
    for i in range(bsz):
        row = {
            "nonfinite": nonfinite,
            "latent_l2_lr": float(lat_lr[i]),
            "latent_l2_bs": float(lat_bs[i]),
            "drivable_l1_lr": float(driv_diff[i]),
        }
        for name in COUNTERFACTUAL_ACTIONS:
            row[f"v_end_{name}"] = float(outs[name]["next_kinematics"][i, -1, 0] * 30.0)
        for name in ("left", "straight", "right"):
            row[f"cum_dy_{name}"] = float(xy_by[name][i, 1])
            row[f"cum_dyaw_{name}"] = float(yaw_by[name][i])
        row["brake_slows"] = row["v_end_brake"] < row["v_end_straight"]
        row["lat_diverges_opposite"] = (
            row["cum_dy_left"] > row["cum_dy_straight"] and row["cum_dy_right"] < row["cum_dy_straight"]
        )
        row["yaw_diverges_opposite"] = (
            row["cum_dyaw_left"] > row["cum_dyaw_straight"] and row["cum_dyaw_right"] < row["cum_dyaw_straight"]
        )
        row["both_diverge_opposite"] = row["lat_diverges_opposite"] and row["yaw_diverges_opposite"]
        rows.append(row)

    return {
        "overall": _summarize_counterfactual(rows, [True] * len(sel_tags)),
        "Town12": _summarize_counterfactual(rows, [t["map"] == "Town12" for t in sel_tags]),
        "Town10HD_Opt": _summarize_counterfactual(rows, [t["map"] == "Town10HD_Opt" for t in sel_tags]),
    }


def benchmark_inference(module, *, sdpa="aotriton", device="cuda", warmup=10, iters=50):
    """Benchmark RX 7900 XT world-model single-step and batched 10-step rollout latencies + peak VRAM."""
    if device.type != "cuda":
        return {"skipped": "CUDA device unavailable"}

    def measure(bsz, horizon, call_fn):
        z0 = torch.randn(bsz, 128, 256, device=device, dtype=torch.bfloat16)
        k0 = torch.tensor([[0.5, 0.0]], device=device, dtype=torch.bfloat16).expand(bsz, 2)
        act = torch.zeros(bsz, horizon, 3, device=device, dtype=torch.bfloat16)
        act[:, :, 1] = 0.3
        for _ in range(warmup):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), sdpa_context(sdpa):
                call_fn(z0, k0, act)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), sdpa_context(sdpa):
                call_fn(z0, k0, act)
            torch.cuda.synchronize()
            times.append((perf_counter() - t0) * 1000.0)
        return float(np.median(times)), float(np.quantile(times, 0.95))

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    step_med, step_p95 = measure(1, 1, lambda z, k, a: module.world_model.step(z, k, a[:, 0]))
    r1_med, r1_p95 = measure(1, 10, lambda z, k, a: module.world_model.rollout(z, k, a))
    r32_med, r32_p95 = measure(32, 10, lambda z, k, a: module.world_model.rollout(z, k, a))
    r64_med, r64_p95 = measure(64, 10, lambda z, k, a: module.world_model.rollout(z, k, a))

    return {
        "device": torch.cuda.get_device_name(0),
        "sdpa": sdpa,
        "step_b1_ms": step_med,
        "step_b1_p95_ms": step_p95,
        "rollout10_b1_ms": r1_med,
        "rollout10_b1_p95_ms": r1_p95,
        "rollout10_b32_ms": r32_med,
        "rollout10_b32_p95_ms": r32_p95,
        "rollout10_b64_ms": r64_med,
        "rollout10_b64_p95_ms": r64_p95,
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024 ** 2)),
    }


def _select_visual_windows(module, val_ds, tags, *, device="cuda"):
    """Pick one busy Town12 highway window and one Town10HD_Opt urban intersection window."""
    highway_cand, urban_cand = [], []
    for idx, tag in enumerate(tags):
        ep = val_ds._load_episode(tag["ep_id"])
        v0 = float(ep["ego_state"][tag["start"], 0])
        if tag["map"] == "Town12" and v0 > 12.0:
            highway_cand.append((idx, tag, ep, v0))
        elif tag["map"] == "Town10HD_Opt" and (tag["is_junction"] or v0 > 3.0):
            urban_cand.append((idx, tag, ep, v0))

    def score_pick(cands, *, prefer_junction=False, min_speed=0.0):
        best = None
        sampled = cands[::max(1, len(cands) // 48)][:48]
        for idx, tag, ep, v0 in sampled:
            z0 = ep["z"][tag["start"]].unsqueeze(0).to(device)
            with torch.no_grad(), torch.autocast(device.type, enabled=False), sdpa_context("math"):
                dec = module.perception.decode_perception(z0.float(), bev=None)
            vis = int((dec["obj_logits"][0].float().sigmoid() > 0.5).sum().item())
            key = (
                int(prefer_junction and tag["is_junction"]),
                int(v0 > min_speed),
                min(vis, 6),
                v0,
            )
            if best is None or key > best[0]:
                best = (key, idx, tag, vis)
        return best

    hw = score_pick(highway_cand, min_speed=12.0)
    ub = score_pick(urban_cand, prefer_junction=True, min_speed=3.0)
    if hw is None or ub is None:
        raise RuntimeError(f"Could not find visual sample windows (hw={hw is not None}, ub={ub is not None})")
    return hw[1], ub[1]


def _make_trajectory_from_delta(delta_25, kin_25):
    """Build (1, 25, 3) viewer trajectory tensor from integrated delta and speeds."""
    xy, _ = integrate_ego_delta(delta_25.unsqueeze(0))
    speed = (kin_25[:, 0] * 30.0).clamp(min=0.0).view(1, 25, 1)
    return torch.cat((xy.cpu(), speed.cpu()), dim=-1)


def _gt_scene_for_frame(dataset_dir, ep, ep_id, frame_abs_idx):
    """Load exact ground-truth frame dict matching SRViewer schema if raw dataset exists."""
    targets_path = dataset_dir / ep_id / "targets.npz"
    if not targets_path.is_file():
        return None
    with np.load(targets_path, allow_pickle=False) as archive:
        lanes_bits = archive["lanes"][frame_abs_idx]
        lanes = np.unpackbits(lanes_bits, count=4 * H * W, bitorder="little").reshape(4, H, W).astype(np.float32)
        focc_bits = archive["future_occupancy"][frame_abs_idx]
        focc = np.unpackbits(focc_bits, count=5 * H * W, bitorder="little").reshape(5, H, W).astype(bool)
        boxes = archive["boxes"][frame_abs_idx].astype(np.float32)
        trajectory = archive["trajectory"][frame_abs_idx].astype(np.float32)
        ego = archive["ego_state"][frame_abs_idx].astype(np.float32)
        thw = float(archive["thw"][frame_abs_idx])
    cmd = ep["cmd"][frame_abs_idx].numpy()
    intent = ("Follow", "Left Change", "Right Change")[int(np.argmax(cmd))]
    return dict(
        lanes=lanes,
        boxes=boxes,
        trajectory=trajectory,
        future_occupancy=focc,
        speed=float(ego[0]),
        accel_lon=float(ego[1]),
        steer=float(ego[4]),
        throttle=float(ego[5]),
        brake=float(ego[6]),
        thw=thw,
        cruise=float(ego[0]),
        limit=80.0,
        intent=intent,
        mode="GT",
        fit_geometry=False,
    )


def _banner_tile(bgr_img, title, subtitle, fonts):
    """Prepend a dark glassmorphic header banner cleanly onto a BGR viewer tile."""
    h, w, _ = bgr_img.shape
    bar_h = 54
    bar = Image.new("RGB", (w, bar_h), (12, 18, 28))
    draw = ImageDraw.Draw(bar)
    draw.text((14, 7), title, font=fonts.h2, fill=(232, 240, 255))
    draw.text((14, 30), subtitle, font=fonts.body, fill=(142, 188, 242))
    draw.line([(0, bar_h - 1), (w, bar_h - 1)], fill=(48, 72, 108), width=1)
    bar_bgr = np.asarray(bar, dtype=np.uint8)[:, :, ::-1]
    return np.vstack((bar_bgr, bgr_img))


def _wm_pred_and_scene(module, z_frame, kin_frame, action_row, *, horizon_traj_actions, intent, nav_hint, sdpa, device):
    """Decode z_frame cleanly and attach a 25-step WM-imagined trajectory ribbon."""
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")), sdpa_context(sdpa):
        traj_roll = module.world_model.rollout(z_frame, kin_frame[:, :2], horizon_traj_actions)
    traj_tensor = _make_trajectory_from_delta(traj_roll["next_delta"][0], traj_roll["next_kinematics"][0])
    with torch.no_grad(), torch.autocast(device.type, enabled=False), sdpa_context("math"):
        dec = module.perception.decode_perception(z_frame.float(), bev=None)
    pred = {**dec, "trajectory": traj_tensor.to(dec["boxes"].device)}
    v_mps = float(kin_frame[0, 0] * 30.0)
    return prediction_scene(
        pred,
        speed=v_mps,
        cruise=max(v_mps, 1.0),
        intent=intent,
        steer=float(action_row[0]),
        throttle=float(action_row[1]),
        brake=float(action_row[2]),
        nav_hint=nav_hint,
        clean=True,
        mode="OFFLINE",
    ), dec


def _render_window_comparison(viewer, module, val_ds, dataset_dir, tag, out_path, *, sdpa, device):
    ep_id, s = tag["ep_id"], tag["start"]
    ep = val_ds._load_episode(ep_id)
    ep_len = ep["z"].shape[0]
    item = val_ds[val_ds.windows.index((ep_id, s))]
    z0 = item["z"][0:1].to(device)
    kin_gt = item["kinematics"].to(device)
    actions = item["actions"].to(device)

    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")), sdpa_context(sdpa):
        roll = module.world_model.rollout(z0, kin_gt[0:1, :2], actions.unsqueeze(0))

    fonts = FontPack(0.85)
    key_steps = (0, 5, 10)
    gt_tiles, wm_tiles = [], []
    cmd_idx = int(ep["cmd"][s].argmax().item())
    intent = ("Follow", "Left Change", "Right Change")[cmd_idx]
    junc_txt = "Junction" if tag["is_junction"] else "Road"

    for k in key_steps:
        abs_idx = s + k
        ego_k = ep["ego_state"][abs_idx]
        act_k = ego_k[4:7].numpy()
        v_gt = float(ego_k[0])
        yr_gt = float(ego_k[3])

        # Row 1: True GT from targets.npz when available, else decoded target latent
        gt_scene = _gt_scene_for_frame(dataset_dir, ep, ep_id, abs_idx) if dataset_dir else None
        if gt_scene is None:
            pad_act = ep["ego_state"][abs_idx:min(abs_idx + 25, ep_len), 4:7]
            if len(pad_act) < 25:
                pad_act = torch.cat((pad_act, pad_act[-1:].expand(25 - len(pad_act), 3)), dim=0)
            k_abs = torch.stack((ego_k[0] / 30., ego_k[3])).unsqueeze(0).to(device)
            gt_scene, _ = _wm_pred_and_scene(
                module, ep["z"][abs_idx:abs_idx + 1].to(device), k_abs, act_k,
                horizon_traj_actions=pad_act.unsqueeze(0).to(device),
                intent=intent, nav_hint=f"{tag['map']} · {junc_txt}", sdpa=sdpa, device=device,
            )
        gt_img = viewer.render(gt_scene)
        gt_tiles.append(_banner_tile(
            gt_img,
            f"GT  t+{k * DT:.1f}s   [{ep_id} · {tag['map']} · {junc_txt}]",
            f"v={v_gt * 3.6:.1f} km/h ({v_gt:.2f} m/s)  |  yaw_rate={yr_gt:+.3f} rad/s",
            fonts,
        ))

        # Row 2: WM Dream (k=0 is initial latent z_0; k>0 is rollout z_hat_k)
        if k == 0:
            z_wm = z0
            k_wm = kin_gt[0:1, :2]
            v_wm, yr_wm = v_gt, yr_gt
            cos_k = 1.0
            act_wm = actions[0].cpu().numpy()
            rem_act = actions.unsqueeze(0)
        else:
            z_wm = roll["z_next"][:, k - 1]
            k_wm = roll["next_kinematics"][:, k - 1, :2]
            v_wm = float(k_wm[0, 0] * 30.0)
            yr_wm = float(k_wm[0, 1])
            cos_k = float(F.cosine_similarity(z_wm.float(), ep["z"][abs_idx:abs_idx + 1].to(device).float(), dim=-1).mean())
            act_wm = actions[k - 1].cpu().numpy()
            rem_act = actions[k - 1:].unsqueeze(0)
        if rem_act.shape[1] < 25:
            rem_act = torch.cat((rem_act, rem_act[:, -1:].expand(1, 25 - rem_act.shape[1], 3)), dim=1)
        wm_scene, _ = _wm_pred_and_scene(
            module, z_wm, k_wm, act_wm,
            horizon_traj_actions=rem_act,
            intent=intent, nav_hint=f"WM Dream cos={cos_k:.3f}", sdpa=sdpa, device=device,
        )
        wm_img = viewer.render(wm_scene)
        wm_label = "WM  t+0.0s (Latent Init)" if k == 0 else f"WM Dream  t+{k * DT:.1f}s"
        wm_tiles.append(_banner_tile(
            wm_img,
            wm_label,
            f"v={v_wm * 3.6:.1f} km/h ({v_wm:.2f} m/s)  |  yaw_rate={yr_wm:+.3f}  |  z_cos={cos_k:.4f}",
            fonts,
        ))

    grid_img = np.vstack((np.hstack(gt_tiles), np.hstack(wm_tiles)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(cv2.imencode(".png", grid_img)[1].tobytes())


def _render_counterfactual_strip(viewer, module, val_ds, tag, out_path, *, sdpa, device):
    ep_id, s = tag["ep_id"], tag["start"]
    item = val_ds[val_ds.windows.index((ep_id, s))]
    z0 = item["z"][0:1].to(device)
    kin0 = item["kinematics"][0:1, :2].to(device)
    fonts = FontPack(0.85)
    tiles = []

    for label, key in (
        ("WM Dream t+1.0s · Left Steer (-0.35)", "left"),
        ("WM Dream t+1.0s · Straight (0.00)", "straight"),
        ("WM Dream t+1.0s · Right Steer (+0.35)", "right"),
    ):
        act_vec = COUNTERFACTUAL_ACTIONS[key]
        act_10 = torch.tensor(act_vec, device=device, dtype=torch.float32).view(1, 1, 3).expand(1, 10, 3)
        act_25 = torch.tensor(act_vec, device=device, dtype=torch.float32).view(1, 1, 3).expand(1, 25, 3)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")), sdpa_context(sdpa):
            roll = module.world_model.rollout(z0, kin0, act_10)
        xy, yaw = integrate_ego_delta(roll["next_delta"])
        dy_m = float(xy[0, -1, 1])
        dyaw_deg = float(np.rad2deg(yaw[0].item()))
        v_end = float(roll["next_kinematics"][0, -1, 0] * 30.0)
        intent = "Left Change" if key == "left" else ("Right Change" if key == "right" else "Follow")
        scene, _ = _wm_pred_and_scene(
            module, roll["z_next"][:, -1], roll["next_kinematics"][:, -1, :2], act_vec,
            horizon_traj_actions=act_25, intent=intent,
            nav_hint=f"dy={dy_m:+.2f}m yaw={dyaw_deg:+.1f}deg", sdpa=sdpa, device=device,
        )
        img = viewer.render(scene)
        tiles.append(_banner_tile(
            img,
            label,
            f"v={v_end * 3.6:.1f} km/h  |  1s lat shift dy={dy_m:+.2f} m  |  heading={dyaw_deg:+.1f} deg",
            fonts,
        ))

    strip = np.hstack(tiles)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(cv2.imencode(".png", strip)[1].tobytes())


def render_visual_suite(module, val_ds, tags, dataset_dir, render_dir, *, sdpa="aotriton", device="cuda"):
    """Render highway, urban junction, and counterfactual Car-OS 3D SR imagination previews."""
    hw_idx, ub_idx = _select_visual_windows(module, val_ds, tags, device=device)
    render_dir = Path(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    viewer = SRViewer(size=(800, 450), backend="auto", window=False)
    paths = {}
    try:
        hw_path = render_dir / "wm_eval_highway_town12.png"
        _render_window_comparison(viewer, module, val_ds, dataset_dir, tags[hw_idx], hw_path, sdpa=sdpa, device=device)
        paths["highway"] = str(hw_path)

        ub_path = render_dir / "wm_eval_urban_town10.png"
        _render_window_comparison(viewer, module, val_ds, dataset_dir, tags[ub_idx], ub_path, sdpa=sdpa, device=device)
        paths["urban"] = str(ub_path)

        cf_path = render_dir / "wm_eval_counterfactual.png"
        _render_counterfactual_strip(viewer, module, val_ds, tags[hw_idx], cf_path, sdpa=sdpa, device=device)
        paths["counterfactual"] = str(cf_path)
    finally:
        viewer.close()
    return {
        "highway_window": tags[hw_idx],
        "urban_window": tags[ub_idx],
        "files": paths,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--latents", type=Path, default=Path(".cache/stage32/latents"))
    parser.add_argument("--dataset", type=Path, default=Path(".cache/stage31b/dataset"))
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--render-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module, cfg, meta = load_stage32_module(args.ckpt, device)
    sdpa = cfg.get("sdpa", "aotriton") if (device.type == "cuda" and torch.version.hip) else "auto"
    val_ds = LatentRolloutDataset(args.latents, "val", horizon=cfg.get("horizon", 10), stride=10, urban_weight=1.0)
    tags = _tag_validation_windows(val_ds)

    report = {
        "model": meta,
        "validation": evaluate_validation(module, val_ds, tags, batch_size=args.batch_size, sdpa=sdpa, device=device),
        "counterfactual": evaluate_counterfactual(module, val_ds, tags, sdpa=sdpa, device=device),
    }
    if args.benchmark:
        report["benchmark"] = benchmark_inference(module, sdpa=sdpa, device=device)
    if args.render_dir is not None:
        dataset_dir = args.dataset if args.dataset.is_dir() else None
        report["renders"] = render_visual_suite(module, val_ds, tags, dataset_dir, args.render_dir, sdpa=sdpa, device=device)

    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
