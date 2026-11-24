"""Offline MapTR polyline instance and continuous-divider target builder for Stage 3.1c."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import tarfile
from time import perf_counter

import cv2
import numpy as np

from .geometry import CONFIG_SHA256, H, W, pixels
from .labels import MAP_CLASSES, MAP_POINTS, MAP_SLOTS, exclusive_markings
from .pack_colab_v3 import BIT_PACK, bit_pack, read_manifest, sha256
from .qc_v3 import load_npz
from .sr_geometry import (
    _arc_length,
    _merge_collinear_paths,
    _resample_open,
    _split_ring_on_roi,
    _surface_contours,
    centerlines,
    clean_drivable_mask,
    solid_lane_paths,
)

OVERLAY_FORMAT = "bev-wam-x-stage31c-maptr-v1"


def canonical_resample(pts, n_points=MAP_POINTS):
    """Canonicalize polyline orientation and resample uniformly in arc length to (n_points, 2)."""
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 2:
        return None
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.r_[True, seg > 1e-4]]
    if len(pts) < 2:
        return None
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    total = float(arc[-1])
    if total < 2.0:
        return None
    dx = float(pts[-1, 0] - pts[0, 0])
    dy = float(pts[-1, 1] - pts[0, 1])
    if dx < -1.0 or (abs(dx) <= 1.0 and dy < 0.0):
        pts = pts[::-1]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        total = float(arc[-1])
    s_new = np.linspace(0.0, total, n_points, dtype=np.float32)
    return np.column_stack((np.interp(s_new, arc, pts[:, 0]), np.interp(s_new, arc, pts[:, 1]))).astype(np.float32)


def _densify_ring(ring, step=1.5):
    """Subdivide long straight ApproxPolyDP edges before ROI clipping so long straight curbs stay intact."""
    pts = np.asarray(ring, dtype=np.float32)
    if len(pts) < 3:
        return pts
    closed = np.vstack((pts, pts[0]))
    out = []
    for a, b in zip(closed[:-1], closed[1:]):
        dist = float(np.linalg.norm(b - a))
        n_seg = max(1, int(np.ceil(dist / step)))
        for k in range(n_seg):
            t = k / n_seg
            out.append(a + t * (b - a))
    return np.asarray(out, dtype=np.float32)


def _extract_dashed_paths(dashed_mask, clean_road):
    """Bridge 3m/9m dashed lane dots into continuous polylines across both straights and tight curves."""
    dil_road = cv2.dilate((clean_road >= 0.5).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    binary = (dashed_mask >= 0.5) & (dil_road > 0)
    if not binary.any():
        return []
    raw = centerlines(binary.astype(np.float32))
    open_p = [np.asarray(p, np.float32) for p in raw if not (len(p) >= 3 and np.array_equal(p[0], p[-1]))]
    closed_p = [np.asarray(p, np.float32) for p in raw if (len(p) >= 3 and np.array_equal(p[0], p[-1]))]
    merged = _merge_collinear_paths(open_p, max_gap=11.5, cos_thresh=0.80, max_lat=1.35)
    return [_resample_open(p, step=0.5) for p in merged if _arc_length(p) >= 2.0] + [
        p for p in closed_p if _arc_length(p) >= 2.0
    ]


def _extract_edge_paths(clean_road):
    """Extract continuous curb/road-boundary polylines without approxPolyDP losing straight borders."""
    rings, hier, _ = _surface_contours(clean_road, fit=False)
    if hier is None:
        return []
    out = []
    for i, (_, _, _, parent) in enumerate(hier):
        ring = rings[i]
        if len(ring) < 3:
            continue
        if parent >= 0:
            x, y = ring[:, 0], ring[:, 1]
            if 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) < 18.0:
                continue
        dense = _densify_ring(ring, step=1.5)
        out.extend(_split_ring_on_roi(dense, min_arc_len=3.0, fit=False))
    return out


def extract_frame_maptr(lanes_mask):
    """Extract MapTR polylines (24, 20, 2) + continuous lane masks (4, 160, 120) from raw GT lanes."""
    lanes = np.asarray(lanes_mask, dtype=np.float32)
    clean_road = clean_drivable_mask(lanes[0])
    dashed_paths = _extract_dashed_paths(lanes[1], clean_road)
    solid_paths = solid_lane_paths(lanes[2], clean_road, lanes[1], fit=False)
    edge_paths = _extract_edge_paths(clean_road)

    candidates = []
    anchor = np.array([15.0, 0.0], dtype=np.float32)
    for cls_id, paths, bonus in ((0, dashed_paths, -3.0), (1, solid_paths, -3.0), (2, edge_paths, 0.0)):
        for raw in paths:
            pts = canonical_resample(raw, MAP_POINTS)
            if pts is None:
                continue
            arc_len = _arc_length(pts)
            min_d = float(np.linalg.norm(pts - anchor[None, :], axis=1).min())
            priority = min_d - 0.08 * arc_len + bonus
            candidates.append((priority, cls_id, pts))

    candidates.sort(key=lambda item: item[0])
    selected = candidates[:MAP_SLOTS]

    polylines = np.zeros((MAP_SLOTS, MAP_POINTS, 2), dtype=np.float16)
    classes = np.zeros((MAP_SLOTS,), dtype=np.uint8)
    valid = np.zeros((MAP_SLOTS,), dtype=bool)
    for idx, (_, cls_id, pts) in enumerate(selected):
        polylines[idx] = pts.astype(np.float16)
        classes[idx] = cls_id
        valid[idx] = True

    # Build dense continuous divider raster (thickness=2 at 0.6 m/px) for auxiliary dense supervision.
    cont = np.zeros((4, H, W), dtype=np.uint8)
    cont[0] = (lanes[0] >= 0.5).astype(np.uint8)
    for path in dashed_paths:
        arr = np.asarray(path, dtype=np.float32)
        if len(arr) >= 2:
            cv2.polylines(cont[1], [np.rint(pixels(arr) * 16).astype(np.int32)], False, 1, 2, shift=4)
    cont[2] = (lanes[2] >= 0.5).astype(np.uint8)
    for path in solid_paths:
        arr = np.asarray(path, dtype=np.float32)
        if len(arr) >= 2:
            cv2.polylines(cont[2], [np.rint(pixels(arr) * 16).astype(np.int32)], False, 1, 2, shift=4)
    cont[3] = (lanes[3] >= 0.5).astype(np.uint8)
    cont = exclusive_markings(cont)
    return polylines, classes, valid, cont


def _process_episode(args):
    dataset_root, ep_id = args
    cv2.setNumThreads(1)
    ep_dir = dataset_root / ep_id
    out_path = ep_dir / "maptr_targets.npz"
    raw = load_npz(ep_dir / "targets.npz")
    packed_lanes = raw["lanes"]
    n_frames = int(packed_lanes.shape[0])

    polylines = np.zeros((n_frames, MAP_SLOTS, MAP_POINTS, 2), dtype=np.float16)
    classes = np.zeros((n_frames, MAP_SLOTS), dtype=np.uint8)
    valid = np.zeros((n_frames, MAP_SLOTS), dtype=bool)
    lanes_cont = np.zeros_like(packed_lanes)
    flat_size = 4 * H * W
    for t in range(n_frames):
        mask = np.unpackbits(packed_lanes[t], count=flat_size, bitorder="little").reshape(4, H, W)
        p, c, v, cont = extract_frame_maptr(mask)
        polylines[t] = p
        classes[t] = c
        valid[t] = v
        lanes_cont[t] = bit_pack(cont)

    tmp = ep_dir / ".maptr_targets.npz.part"
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, map_polylines=polylines, map_classes=classes,
                            map_valid=valid, lanes_cont=lanes_cont)
    tmp.replace(out_path)
    return ep_id, n_frames, int(valid.sum())


def _render_preview(dataset_root, ep_id, frame_idx, out_path):
    raw = load_npz(dataset_root / ep_id / "targets.npz")
    ov = load_npz(dataset_root / ep_id / "maptr_targets.npz")
    flat_size = 4 * H * W
    raw_lanes = np.unpackbits(raw["lanes"][frame_idx], count=flat_size, bitorder="little").reshape(4, H, W)
    cont_lanes = np.unpackbits(ov["lanes_cont"][frame_idx], count=flat_size, bitorder="little").reshape(4, H, W)
    polys = ov["map_polylines"][frame_idx].astype(np.float32)
    cls_ids = ov["map_classes"][frame_idx]
    valids = ov["map_valid"][frame_idx]

    scale = 4
    canvas = np.zeros((H * scale, W * 2 * scale, 3), dtype=np.uint8)

    # Left: raw sparse GT lanes
    left = canvas[:, :W * scale]
    up_raw = np.repeat(np.repeat(raw_lanes, scale, axis=1), scale, axis=2)
    left[up_raw[0] > 0] = (46, 54, 68)
    left[up_raw[1] > 0] = (230, 235, 245)
    left[up_raw[2] > 0] = (60, 195, 245)
    left[up_raw[3] > 0] = (220, 180, 70)

    # Right: continuous raster + crisp MapTR polylines overlaid
    right = canvas[:, W * scale:]
    up_cont = np.repeat(np.repeat(cont_lanes, scale, axis=1), scale, axis=2)
    right[up_cont[0] > 0] = (46, 54, 68)
    right[up_cont[1] > 0] = (95, 105, 120)
    right[up_cont[2] > 0] = (45, 95, 120)

    colors = ((245, 248, 255), (60, 210, 255), (235, 210, 75))
    for i in np.flatnonzero(valids):
        uv = (pixels(polys[i]) + 0.5) * scale
        pts = np.rint(uv).astype(np.int32).reshape(-1, 1, 2)
        col = colors[int(cls_ids[i])]
        cv2.polylines(right, [pts], False, col, 2, lineType=cv2.LINE_AA)
        for px, py in pts.reshape(-1, 2)[::4]:
            cv2.circle(right, (int(px), int(py)), 2, col, -1)

    cv2.putText(canvas, f"{ep_id} t={frame_idx} | LEFT: Raw GT Lanes", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 240, 250), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"RIGHT: MapTR Polylines ({int(valids.sum())} inst)", (W * scale + 12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 240, 250), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(cv2.imencode(".png", canvas)[1].tobytes())


def pack_overlay(dataset_root, manifest, out_dir):
    """Package per-episode maptr_targets.npz into a single verifiable overlay tar for Colab."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / "stage31c-maptr-overlay.tar"
    tmp_tar = out_dir / ".stage31c-maptr-overlay.tar.part"
    with tarfile.open(tmp_tar, "w") as tar:
        for entry in manifest["episodes"]:
            src = dataset_root / entry["id"] / "maptr_targets.npz"
            if not src.is_file():
                raise FileNotFoundError(src)
            tar.add(src, arcname=f"{entry['id']}/maptr_targets.npz")
    tmp_tar.replace(tar_path)

    doc = dict(
        format=OVERLAY_FORMAT,
        config_sha256=CONFIG_SHA256,
        pack_sha256=sha256(dataset_root / "manifest.json"),
        episodes=len(manifest["episodes"]),
        total_frames=int(manifest["total_frames"]),
        map_slots=MAP_SLOTS,
        map_points=MAP_POINTS,
        map_classes=list(MAP_CLASSES),
        tar_name=tar_path.name,
        tar_bytes=tar_path.stat().st_size,
        tar_sha256=sha256(tar_path),
    )
    mpath = out_dir / "overlay_manifest.json"
    mtmp = out_dir / ".overlay_manifest.json.part"
    mtmp.write_text(json.dumps(doc, indent=2, allow_nan=False), encoding="utf-8")
    mtmp.replace(mpath)
    return tar_path, mpath, doc


def unpack_overlay(overlay_dir, dataset_root):
    """Verify overlay SHA-256 against dataset manifest and extract all maptr_targets.npz files."""
    mpath = overlay_dir / "overlay_manifest.json"
    doc = json.loads(mpath.read_text(encoding="utf-8"))
    if doc["format"] != OVERLAY_FORMAT or doc["config_sha256"] != CONFIG_SHA256:
        raise ValueError("Incompatible MapTR overlay manifest or config SHA-256")
    ds_sha = sha256(dataset_root / "manifest.json")
    if doc["pack_sha256"] != ds_sha:
        raise ValueError(f"Overlay pack_sha256 {doc['pack_sha256']} != dataset {ds_sha}")
    tar_path = overlay_dir / doc["tar_name"]
    if tar_path.stat().st_size != doc["tar_bytes"] or sha256(tar_path) != doc["tar_sha256"]:
        raise ValueError("MapTR overlay archive failed SHA-256/size verification")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(dataset_root, filter="data")
    manifest = read_manifest(dataset_root / "manifest.json", check_shards=False)
    for entry in manifest["episodes"]:
        target = dataset_root / entry["id"] / "maptr_targets.npz"
        if not target.is_file():
            raise FileNotFoundError(f"Missing unpacked overlay file: {target}")
    print(f"Verified and unpacked {len(manifest['episodes'])} MapTR overlay files into {dataset_root}", flush=True)


def build_all(dataset_root, *, workers=8, pack_out=None, preview_dir=None, skip_existing=False):
    manifest = read_manifest(dataset_root / "manifest.json", check_shards=False)
    ep_ids = [e["id"] for e in manifest["episodes"]
              if not (skip_existing and (dataset_root / e["id"] / "maptr_targets.npz").is_file())]
    if ep_ids:
        t0 = perf_counter()
        tot_frames, tot_inst = 0, 0
        tasks = [(dataset_root, eid) for eid in ep_ids]
        with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
            for idx, (eid, nf, ni) in enumerate(pool.map(_process_episode, tasks, chunksize=4), 1):
                tot_frames += nf
                tot_inst += ni
                if idx % 50 == 0 or idx == len(tasks):
                    print(f"MapTR targets: {idx}/{len(tasks)} episodes ({tot_frames} frames, "
                          f"{tot_inst / max(1, tot_frames):.2f} inst/frame, {perf_counter() - t0:.1f}s)", flush=True)

    if preview_dir is not None:
        t12 = [e for e in manifest["episodes"] if e["map"] == "Town12"]
        t10 = [e for e in manifest["episodes"] if e["map"] == "Town10HD_Opt"]
        picks = []
        if t12:
            hw = next((e for e in t12 if e.get("scenario") in ("multi_car", "slow_overtake")), t12[0])
            ramp = next((e for e in t12 if e.get("scenario") in ("ramp", "sharp_curve", "curve")), t12[-1])
            picks.append((hw["id"], min(20, len(hw["frame_ids"]) - 1), "preview_town12_a.png"))
            picks.append((ramp["id"], min(25, len(ramp["frame_ids"]) - 1), "preview_town12_b.png"))
        if t10:
            urb = next((e for e in t10 if e.get("scenario") in ("multi_car", "curve")), t10[0])
            picks.append((urb["id"], min(20, len(urb["frame_ids"]) - 1), "preview_town10_a.png"))
        for eid, fidx, fname in picks:
            _render_preview(dataset_root, eid, fidx, preview_dir / fname)
        print(f"Saved {len(picks)} preview images -> {preview_dir}", flush=True)

    if pack_out is not None:
        tar_path, mpath, doc = pack_overlay(dataset_root, manifest, pack_out)
        print(f"Packed MapTR overlay: {tar_path} ({doc['tar_bytes'] / (1024 ** 2):.2f} MiB, "
              f"sha256={doc['tar_sha256'][:12]})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(".cache/stage31b/dataset"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pack-out", type=Path, default=None)
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--unpack-overlay", type=Path, default=None,
                        help="Directory containing overlay_manifest.json + stage31c-maptr-overlay.tar")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.unpack_overlay is not None:
        unpack_overlay(args.unpack_overlay, args.dataset)
    else:
        build_all(args.dataset, workers=args.workers, pack_out=args.pack_out,
                  preview_dir=args.preview_dir, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
