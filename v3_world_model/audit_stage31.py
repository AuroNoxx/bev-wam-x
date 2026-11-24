"""Offline Stage 3.1b failure-mode audit: distance-bucketed detection, duplicates, ghosts, lane IoU, and 7-vector error."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader, Subset

from v1_dual_system.checkpoint_io import VerifiedCheckpointIO

from .data import Clips
from .geometry import CONFIG_SHA256, H, W, grid, pixels
from .labels import OBJECT_CLASSES, VECTOR_NAMES
from .pack_colab_v3 import sha256
from .perception_model import BEVWAMXPerception, CLIP_LEN, sdpa_context
from .sr_postprocess import SRPostProcessor

DIST_BUCKETS = (("0-20m", 0.0, 20.0), ("20-40m", 20.0, 40.0), ("40-60m", 40.0, 60.0), ("60m+", 60.0, 1e9))
LANE_NAMES = ("drivable", "dashed", "solid", "edge")
_DILATE_KERNEL = np.ones((3, 3), dtype=np.uint8)


def _hungarian_match(pred_xy, gt_xy, radius=3.0):
    """Return matched (pred_indices, gt_indices, distances) under < radius 1-to-1 assignment."""
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        empty_i = np.empty((0,), dtype=np.int64)
        return empty_i, empty_i, np.empty((0,), dtype=np.float32)
    distances = np.linalg.norm(pred_xy[:, None, :2] - gt_xy[None, :, :2], axis=-1)
    penalty = (min(distances.shape) + 1) * radius
    costs = np.where(distances < radius, distances, penalty)
    rows, cols = linear_sum_assignment(costs)
    keep = distances[rows, cols] < radius
    return rows[keep], cols[keep], distances[rows[keep], cols[keep]]


def _is_zero_lidar_ghost(xy, lidar_bev):
    """Check whether predicted XY lacks elevated LiDAR support in a +-2 px BEV window."""
    if not len(xy):
        return np.empty((0,), dtype=bool)
    uv = np.rint(pixels(xy)).astype(np.int64)
    out = np.ones(len(xy), dtype=bool)
    for i, (col, row) in enumerate(uv):
        if not (0 <= col < W and 0 <= row < H):
            continue
        c0, c1 = max(0, col - 2), min(W, col + 3)
        r0, r1 = max(0, row - 2), min(H, row + 3)
        hi_z = float(lidar_bev[1, r0:r1, c0:c1].max())
        log_cnt = float(lidar_bev[5, r0:r1, c0:c1].max())
        if hi_z > 0.35 and log_cnt > 0.5:
            out[i] = False
    return out


def _dist_bucket(dists):
    out = np.zeros(len(dists), dtype=np.int64)
    for b_idx, (_, lo, hi) in enumerate(DIST_BUCKETS):
        out[(dists >= lo) & (dists < hi)] = b_idx
    return out


def _empty_accumulator():
    n_cls = len(OBJECT_CLASSES)
    n_vec = len(VECTOR_NAMES)
    return dict(
        frames=0,
        raw_tp=0, raw_pred=0, raw_gt=0,
        clean_tp=0, clean_pred=0,
        dup_fp=0, ghost_fp=0, non_gt_fp=0,
        bucket_gt=np.zeros(len(DIST_BUCKETS), dtype=np.int64),
        bucket_tp_gt=np.zeros(len(DIST_BUCKETS), dtype=np.int64),
        bucket_pred=np.zeros(len(DIST_BUCKETS), dtype=np.int64),
        bucket_tp_pred=np.zeros(len(DIST_BUCKETS), dtype=np.int64),
        cls_gt=np.zeros(n_cls, dtype=np.int64),
        cls_tp=np.zeros(n_cls, dtype=np.int64),
        cls_correct=np.zeros(n_cls, dtype=np.int64),
        lane_inter=np.zeros(4, dtype=np.int64),
        lane_union=np.zeros(4, dtype=np.int64),
        lane_dil_inter=np.zeros(3, dtype=np.int64),
        lane_dil_union=np.zeros(3, dtype=np.int64),
        vec_valid_cnt=np.zeros(n_vec, dtype=np.int64),
        vec_lat_abs_sum=np.zeros(n_vec, dtype=np.float64),
        vec_xy_abs_sum=np.zeros(n_vec, dtype=np.float64),
        vec_ex_tp=np.zeros(n_vec, dtype=np.int64),
        vec_ex_fp=np.zeros(n_vec, dtype=np.int64),
        vec_ex_fn=np.zeros(n_vec, dtype=np.int64),
        vec_ex_tn=np.zeros(n_vec, dtype=np.int64),
        map_tp=0, map_pred=0, map_gt=0, map_l1_sum=0.0,
    )


def _finalize(acc):
    def prf(tp, pred, gt):
        p = float(tp / max(1, pred))
        r = float(tp / max(1, gt))
        f1 = float(2 * p * r / max(1e-8, p + r)) if (p + r) > 0 else 0.0
        return dict(tp=int(tp), predicted=int(pred), truth=int(gt),
                    precision_3m=p, recall_3m=r, f1_3m=f1)

    buckets = {}
    for bi, (label, _, _) in enumerate(DIST_BUCKETS):
        r = float(acc["bucket_tp_gt"][bi] / max(1, acc["bucket_gt"][bi]))
        p = float(acc["bucket_tp_pred"][bi] / max(1, acc["bucket_pred"][bi]))
        buckets[label] = dict(
            truth=int(acc["bucket_gt"][bi]), tp_gt=int(acc["bucket_tp_gt"][bi]), recall_3m=r,
            predicted=int(acc["bucket_pred"][bi]), tp_pred=int(acc["bucket_tp_pred"][bi]), precision_3m=p,
        )

    per_class = {}
    for ci, cname in enumerate(OBJECT_CLASSES):
        per_class[cname] = dict(
            truth=int(acc["cls_gt"][ci]),
            tp_3m=int(acc["cls_tp"][ci]),
            recall_3m=float(acc["cls_tp"][ci] / max(1, acc["cls_gt"][ci])),
            cls_match_given_tp=float(acc["cls_correct"][ci] / max(1, acc["cls_tp"][ci])),
        )

    exact_iou = {name: float(acc["lane_inter"][i] / max(1, acc["lane_union"][i])) for i, name in enumerate(LANE_NAMES)}
    dilated_iou = {name: float(acc["lane_dil_inter"][i] / max(1, acc["lane_dil_union"][i]))
                   for i, name in enumerate(LANE_NAMES[1:])}

    tot_vec_valid = int(acc["vec_valid_cnt"].sum())
    ex_tp = int(acc["vec_ex_tp"].sum())
    ex_fp = int(acc["vec_ex_fp"].sum())
    ex_fn = int(acc["vec_ex_fn"].sum())
    ex_tn = int(acc["vec_ex_tn"].sum())
    ex_p = float(ex_tp / max(1, ex_tp + ex_fp))
    ex_r = float(ex_tp / max(1, ex_tp + ex_fn))
    per_vec = {}
    for vi, vname in enumerate(VECTOR_NAMES):
        cnt = int(acc["vec_valid_cnt"][vi])
        tp_v, fp_v, fn_v, tn_v = int(acc["vec_ex_tp"][vi]), int(acc["vec_ex_fp"][vi]), int(acc["vec_ex_fn"][vi]), int(acc["vec_ex_tn"][vi])
        per_vec[vname] = dict(
            valid_points=cnt,
            lateral_l1_m=float(acc["vec_lat_abs_sum"][vi] / max(1, cnt)),
            xy_l1_m=float(acc["vec_xy_abs_sum"][vi] / max(1, cnt)),
            exists_acc=float((tp_v + tn_v) / max(1, tp_v + tn_v + fp_v + fn_v)),
        )

    raw_fp = max(0, acc["raw_pred"] - acc["raw_tp"])
    map_p = float(acc["map_tp"] / max(1, acc["map_pred"]))
    map_r = float(acc["map_tp"] / max(1, acc["map_gt"]))
    return dict(
        frames=int(acc["frames"]),
        detection_raw=prf(acc["raw_tp"], acc["raw_pred"], acc["raw_gt"]),
        detection_clean_lidar=prf(acc["clean_tp"], acc["clean_pred"], acc["raw_gt"]),
        failure_breakdown=dict(
            total_fp=int(raw_fp),
            duplicate_fp=int(acc["dup_fp"]),
            duplicate_fp_rate_of_fp=float(acc["dup_fp"] / max(1, raw_fp)),
            non_gt_fp=int(acc["non_gt_fp"]),
            zero_lidar_ghost_fp=int(acc["ghost_fp"]),
            zero_lidar_ghost_rate_of_fp=float(acc["ghost_fp"] / max(1, raw_fp)),
            missed_fn=int(max(0, acc["raw_gt"] - acc["raw_tp"])),
        ),
        distance_buckets=buckets,
        per_class_recall=per_class,
        lane_iou_roi_exact=exact_iou,
        lane_iou_roi_dilated_3x3=dilated_iou,
        maptr=dict(
            tp_1_5m=int(acc["map_tp"]), predicted=int(acc["map_pred"]), truth=int(acc["map_gt"]),
            precision=map_p, recall=map_r,
            f1=float(2 * map_p * map_r / max(1e-8, map_p + map_r)) if (map_p + map_r) > 0 else 0.0,
            mean_point_l1_m=float(acc["map_l1_sum"] / max(1, acc["map_tp"])),
        ),
        vector_head=dict(
            valid_points=tot_vec_valid,
            lateral_l1_m=float(acc["vec_lat_abs_sum"].sum() / max(1, tot_vec_valid)),
            xy_l1_m=float(acc["vec_xy_abs_sum"].sum() / max(1, tot_vec_valid)),
            exists_acc=float((ex_tp + ex_tn) / max(1, ex_tp + ex_tn + ex_fp + ex_fn)),
            exists_precision=ex_p,
            exists_recall=ex_r,
            exists_f1=float(2 * ex_p * ex_r / max(1e-8, ex_p + ex_r)) if (ex_p + ex_r) > 0 else 0.0,
            by_channel=per_vec,
        ),
    )


def _update_accumulators(acc, *, geom, cls_id, scores, lbev, lane_pred, vec_pred, vec_ex_pred,
                         gt_boxes_all, gt_lanes, gt_vec, gt_vec_valid, gt_vec_ex, roi_mask,
                         map_pts=None, map_scores=None, gt_map_pts=None, gt_map_valid=None):
    acc["frames"] += 1
    yaw = np.arctan2(geom[:, 6], geom[:, 7])
    pred_valid = scores > 0.5
    raw_boxes = np.column_stack((geom[:, :6], yaw, geom[:, 8:10], cls_id, pred_valid)).astype(np.float32)

    sel_idx = np.flatnonzero(pred_valid)
    pred_sel = raw_boxes[sel_idx]
    gt_valid = gt_boxes_all[gt_boxes_all[:, -1] > 0.5]

    acc["raw_pred"] += len(pred_sel)
    acc["raw_gt"] += len(gt_valid)

    gt_dists = np.hypot(gt_valid[:, 0], gt_valid[:, 1]) if len(gt_valid) else np.empty((0,), np.float32)
    pred_dists = np.hypot(pred_sel[:, 0], pred_sel[:, 1]) if len(pred_sel) else np.empty((0,), np.float32)
    gt_b = _dist_bucket(gt_dists)
    pred_b = _dist_bucket(pred_dists)
    for bi in range(len(DIST_BUCKETS)):
        acc["bucket_gt"][bi] += int((gt_b == bi).sum())
        acc["bucket_pred"][bi] += int((pred_b == bi).sum())
    if len(gt_valid):
        gt_cls = np.clip(np.rint(gt_valid[:, 9]).astype(np.int64), 0, len(OBJECT_CLASSES) - 1)
        for ci in range(len(OBJECT_CLASSES)):
            acc["cls_gt"][ci] += int((gt_cls == ci).sum())
    else:
        gt_cls = np.empty((0,), dtype=np.int64)

    m_p, m_g, _ = _hungarian_match(pred_sel[:, :2] if len(pred_sel) else pred_sel, gt_valid[:, :2] if len(gt_valid) else gt_valid, radius=3.0)
    acc["raw_tp"] += len(m_p)
    for p_i, g_i in zip(m_p, m_g):
        acc["bucket_tp_gt"][gt_b[g_i]] += 1
        acc["bucket_tp_pred"][pred_b[p_i]] += 1
        g_c = int(gt_cls[g_i])
        acc["cls_tp"][g_c] += 1
        if int(pred_sel[p_i, 9]) == g_c:
            acc["cls_correct"][g_c] += 1

    # Analyze unmatched false positives: duplicate detections near GT vs zero-LiDAR empty-road ghosts.
    matched_mask = np.zeros(len(pred_sel), dtype=bool)
    matched_mask[m_p] = True
    unmatched = pred_sel[~matched_mask]
    if len(unmatched):
        if len(gt_valid):
            d_any = np.linalg.norm(unmatched[:, None, :2] - gt_valid[None, :, :2], axis=-1).min(axis=1)
        else:
            d_any = np.full(len(unmatched), np.inf, dtype=np.float32)
        is_dup = d_any < 3.0
        acc["dup_fp"] += int(is_dup.sum())
        non_gt = unmatched[~is_dup]
        acc["non_gt_fp"] += len(non_gt)
        if len(non_gt):
            acc["ghost_fp"] += int(_is_zero_lidar_ghost(non_gt[:, :2], lbev).sum())

    # Compare against SRPostProcessor.clean_frame with LiDAR BEV.
    clean_boxes, _ = SRPostProcessor.clean_frame(raw_boxes, scores, lidar_bev=lbev)
    acc["clean_pred"] += len(clean_boxes)
    c_p, _, _ = _hungarian_match(clean_boxes[:, :2] if len(clean_boxes) else clean_boxes,
                                 gt_valid[:, :2] if len(gt_valid) else gt_valid, radius=3.0)
    acc["clean_tp"] += len(c_p)

    # Lane IoU (exact & 3x3 tolerance dilated on line/edge channels).
    pred_m = lane_pred > 0.0
    gt_m = gt_lanes.astype(bool)
    acc["lane_inter"] += (pred_m & gt_m & roi_mask[None]).sum(axis=(1, 2))
    acc["lane_union"] += ((pred_m | gt_m) & roi_mask[None]).sum(axis=(1, 2))
    for ch in range(3):
        p_dil = cv2.dilate(pred_m[ch + 1].astype(np.uint8), _DILATE_KERNEL).astype(bool)
        g_dil = cv2.dilate(gt_m[ch + 1].astype(np.uint8), _DILATE_KERNEL).astype(bool)
        acc["lane_dil_inter"][ch] += int((p_dil & g_dil & roi_mask).sum())
        acc["lane_dil_union"][ch] += int(((p_dil | g_dil) & roi_mask).sum())

    # Vector head evaluation (valid points only + existence).
    diff = np.abs(vec_pred - gt_vec)
    v_cnt = gt_vec_valid.sum(axis=1)
    acc["vec_valid_cnt"] += v_cnt
    acc["vec_lat_abs_sum"] += (diff[..., 1] * gt_vec_valid).sum(axis=1)
    acc["vec_xy_abs_sum"] += (diff.sum(axis=-1) * gt_vec_valid).sum(axis=1)
    pred_ex = vec_ex_pred > 0.0
    gt_ex = gt_vec_ex.astype(bool)
    acc["vec_ex_tp"] += (pred_ex & gt_ex).astype(np.int64)
    acc["vec_ex_fp"] += (pred_ex & (~gt_ex)).astype(np.int64)
    acc["vec_ex_fn"] += ((~pred_ex) & gt_ex).astype(np.int64)
    acc["vec_ex_tn"] += ((~pred_ex) & (~gt_ex)).astype(np.int64)

    if map_pts is not None and gt_map_pts is not None and gt_map_valid is not None:
        psel = map_pts[map_scores > 0.5]
        gsel = gt_map_pts[gt_map_valid.astype(bool)]
        acc["map_pred"] += len(psel)
        acc["map_gt"] += len(gsel)
        if len(psel) and len(gsel):
            df = np.abs(psel[:, None] - gsel[None]).mean(axis=(-2, -1))
            dr = np.abs(psel[:, None] - gsel[None, :, ::-1]).mean(axis=(-2, -1))
            dmin = np.minimum(df, dr)
            ri, ci = linear_sum_assignment(dmin)
            hit = dmin[ri, ci] < 1.5
            acc["map_tp"] += int(hit.sum())
            if hit.any():
                acc["map_l1_sum"] += float(dmin[ri[hit], ci[hit]].sum())


@torch.inference_mode()
def audit(dataset, checkpoint, *, clips=None, sdpa="aotriton", batch_size=8, workers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA/ROCm GPU is required for Stage 3.1 audit")

    state = VerifiedCheckpointIO().load_checkpoint(checkpoint, map_location="cpu")
    if state["task_state"]["config_sha256"] != CONFIG_SHA256:
        raise ValueError("Checkpoint sensor contract mismatch")
    fingerprint = sha256(Path(dataset) / "manifest.json")
    if state["task_state"]["pack_sha256"] != fingerprint:
        raise ValueError("Checkpoint and validation manifest fingerprints differ")

    step = int(state["global_step"])
    arch = state["task_state"].get("architecture", "stage31b")
    model = BEVWAMXPerception(lidar_dropout=0.0, arch=arch, pretrained=False).to(device).eval()
    model.load_state_dict({k.removeprefix("model."): v for k, v in state["state_dict"].items()})
    del state

    is_31c = arch == "stage31c"
    data = Clips(dataset, "val", load_maptr=is_31c, strong_aug=False)
    tail_skip = {}
    if clips is None:
        for episode, entry in data.entries.items():
            length = len(entry["frame_ids"])
            remainder = length % CLIP_LEN
            if remainder:
                tail_skip[len(data.starts)] = CLIP_LEN - remainder
                data.starts.append((episode, length - CLIP_LEN))
        indices = list(range(len(data)))
    else:
        if clips <= 0:
            raise ValueError("clips must be positive")
        indices = np.linspace(0, len(data) - 1, min(clips, len(data))).astype(int).tolist()

    subset = Subset(data, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )

    g = grid()
    roi_mask = (g[..., 0] >= 0.0) & (g[..., 0] <= 40.0) & (np.abs(g[..., 1]) <= 18.0)

    overall = _empty_accumulator()
    by_map = {}
    clip_cursor = 0
    for batch_idx, batch in enumerate(loader):
        inputs, targets = batch["inputs"], batch["targets"]
        bsz = inputs["lidar_bev"].shape[0]
        batch_indices = indices[clip_cursor:clip_cursor + bsz]
        clip_cursor += bsz

        latent = None
        for t in range(CLIP_LEN):
            rgb_t = tuple(x[:, t].to(device, non_blocking=True) for x in inputs["rgb"])
            lbev_t = inputs["lidar_bev"][:, t].to(device, non_blocking=True)
            delta_t = inputs["ego_delta"][:, t].to(device, non_blocking=True) if latent is not None else None
            cmd_t = inputs["cmd"][:, t].to(device, non_blocking=True)
            nav_t = inputs["nav_anchors"][:, t].to(device, non_blocking=True)
            spd_t = inputs["ego_speed"][:, t].to(device, non_blocking=True)

            with sdpa_context(sdpa), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                pred = model(rgb_t, lidar_bev=lbev_t, prev_latent=latent, ego_delta=delta_t,
                             cmd=cmd_t, nav_anchors=nav_t, ego_speed=spd_t)
            latent = pred["z_t"]

            geoms = pred["boxes"].float().cpu().numpy()
            cls_ids = pred["cls_logits"].float().cpu().numpy().argmax(axis=-1)
            scores_b = pred["obj_logits"].float().sigmoid().cpu().numpy()
            lbev_np = inputs["lidar_bev"][:, t].float().numpy()
            lane_b = pred["lane_logits"].float().cpu().numpy()
            vec_b = pred["vectors"].float().cpu().numpy()
            vec_ex_b = pred["vector_exists_logits"].float().cpu().numpy()
            map_pts_b = pred["map_polylines"].float().cpu().numpy() if "map_polylines" in pred else None
            map_sc_b = pred["map_obj_logits"].float().sigmoid().cpu().numpy() if "map_obj_logits" in pred else None

            gt_boxes_b = targets["boxes"][:, t].numpy()
            gt_lanes_b = targets["lanes_cont"][:, t].numpy() if is_31c else targets["lanes"][:, t].numpy()
            gt_vec_b = targets["vectors"][:, t].numpy()
            gt_vv_b = targets["vector_valid"][:, t].numpy()
            gt_vex_b = targets["vector_exists"][:, t].numpy()
            gt_mpts_b = targets["map_polylines"][:, t].numpy() if is_31c else None
            gt_mv_b = targets["map_valid"][:, t].numpy() if is_31c else None

            for b, ds_idx in enumerate(batch_indices):
                if t < tail_skip.get(ds_idx, 0):
                    continue
                ep_id, _ = data.starts[ds_idx]
                town = data.entries[ep_id]["map"]
                map_acc = by_map.setdefault(town, _empty_accumulator())
                kwargs = dict(
                    geom=geoms[b], cls_id=cls_ids[b], scores=scores_b[b], lbev=lbev_np[b],
                    lane_pred=lane_b[b], vec_pred=vec_b[b], vec_ex_pred=vec_ex_b[b],
                    gt_boxes_all=gt_boxes_b[b], gt_lanes=gt_lanes_b[b],
                    gt_vec=gt_vec_b[b], gt_vec_valid=gt_vv_b[b], gt_vec_ex=gt_vex_b[b],
                    roi_mask=roi_mask,
                    map_pts=map_pts_b[b] if map_pts_b is not None else None,
                    map_scores=map_sc_b[b] if map_sc_b is not None else None,
                    gt_map_pts=gt_mpts_b[b] if gt_mpts_b is not None else None,
                    gt_map_valid=gt_mv_b[b] if gt_mv_b is not None else None,
                )
                _update_accumulators(overall, **kwargs)
                _update_accumulators(map_acc, **kwargs)

        if (batch_idx + 1) % 25 == 0 or batch_idx + 1 == len(loader):
            print(f"Stage 3.1 audit: {min(clip_cursor, len(indices))}/{len(indices)} clips", flush=True)

    return dict(
        checkpoint=str(checkpoint),
        checkpoint_step=step,
        architecture=arch,
        pack_sha256=fingerprint,
        split="val",
        clips=len(indices),
        sdpa=sdpa,
        summary=_finalize(overall),
        by_map={town: _finalize(acc) for town, acc in sorted(by_map.items())},
    )


def _print_summary(report):
    s = report["summary"]
    d_raw, d_cln = s["detection_raw"], s["detection_clean_lidar"]
    fb = s["failure_breakdown"]
    print(f"\n=== Stage 3.1 Audit ({report['architecture']}, step={report['checkpoint_step']}, frames={s['frames']}) ===")
    print(f"Raw Det@0.5  : P={d_raw['precision_3m']:.2%}  R={d_raw['recall_3m']:.2%}  F1={d_raw['f1_3m']:.2%} "
          f"(TP={d_raw['tp']} / Pred={d_raw['predicted']} / GT={d_raw['truth']})")
    print(f"Clean+LiDAR  : P={d_cln['precision_3m']:.2%}  R={d_cln['recall_3m']:.2%}  F1={d_cln['f1_3m']:.2%} "
          f"(TP={d_cln['tp']} / Pred={d_cln['predicted']})")
    print(f"FP Breakdown : total_FP={fb['total_fp']} | dup_near_GT={fb['duplicate_fp']} ({fb['duplicate_fp_rate_of_fp']:.1%}) "
          f"| zero_LiDAR_ghost={fb['zero_lidar_ghost_fp']} ({fb['zero_lidar_ghost_rate_of_fp']:.1%}) | missed_FN={fb['missed_fn']}")
    print("Distance R/P : " + " | ".join(
        f"{k}: R={v['recall_3m']:.1%} P={v['precision_3m']:.1%} (n={v['truth']})"
        for k, v in s["distance_buckets"].items()
    ))
    print("Per-class R  : " + " | ".join(
        f"{k}: {v['recall_3m']:.1%} ({v['tp_3m']}/{v['truth']})"
        for k, v in s["per_class_recall"].items()
    ))
    print("Lane IoU 1px : " + " | ".join(f"{k}={v:.3f}" for k, v in s["lane_iou_roi_exact"].items()))
    print("Lane IoU 3x3 : " + " | ".join(f"{k}={v:.3f}" for k, v in s["lane_iou_roi_dilated_3x3"].items()))
    mt = s.get("maptr", {})
    if mt.get("truth", 0) > 0:
        print(f"MapTR Vector : P={mt['precision']:.2%}  R={mt['recall']:.2%}  F1={mt['f1']:.2%}  "
              f"mean_L1={mt['mean_point_l1_m']:.3f}m (TP={mt['tp_1_5m']}/{mt['predicted']}/{mt['truth']})")
    vh = s["vector_head"]
    print(f"7-Vector Head: lat_L1={vh['lateral_l1_m']:.3f}m  xy_L1={vh['xy_l1_m']:.3f}m  "
          f"exists_acc={vh['exists_acc']:.1%}  exists_F1={vh['exists_f1']:.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(".cache/stage31b/dataset"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/v3_stage31b_step7968/best.ckpt"))
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--clips", type=int, default=None)
    scope.add_argument("--all", action="store_true", help="Evaluate every validation frame once (default)")
    parser.add_argument("--sdpa", choices=("aotriton", "math", "cuda_flash"), default="aotriton")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path(".cache/stage31b/audit_31b_baseline.json"))
    args = parser.parse_args()

    clips = None if (args.all or args.clips is None) else args.clips
    report = audit(args.dataset, args.checkpoint, clips=clips, sdpa=args.sdpa,
                   batch_size=args.batch, workers=args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.part")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(args.out)
    _print_summary(report)
    print(f"\nSaved audit report -> {args.out}")


if __name__ == "__main__":
    main()
