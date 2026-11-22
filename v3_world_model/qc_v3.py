"""Validate the raw Stage 3 contract and audit independently separated maneuver events."""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .geometry import (CONFIG_SHA256, DT, H, W, Z, NAMES, SIZES, camera_mounts, delta_error,
                       project, wrap)
from .labels import MAX_BOX, MIN_BOX, OBJECT_CLASSES, VECTOR_X, footprint
from .observations import AUDIT_SENSORS, command_vector, validate_observation
from .sim import require_collection_map

LABEL_SHAPES = dict(lanes=(4, H, W), vectors=(7, len(VECTOR_X), 2),
                    vector_valid=(7, len(VECTOR_X)), vector_exists=(7,), boxes=(32, 11), box_ids=(32,),
                    occupancy=(Z, H, W), future_occupancy=(5, H, W), trajectory=(25, 3),
                    ego_state=(7,), flags=(3,), thw=(), thw_actor_id=(), nav_state=(3,),
                    lane_state=(3,))
BINARY = ("lanes", "vector_valid", "vector_exists", "occupancy", "future_occupancy", "flags")
EVENTS = ("straight", "curve", "sharp_curve", "left_change", "right_change", "slow_overtake", "ramp")


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def read_image(path):
    img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Unreadable image: {path}")
    return img


def episode_index(roots):
    for root in roots:
        if not root.is_dir() or any(root.glob(".partial-*")):
            raise ValueError(f"Missing or partially collected dataset: {root}")
        paths = sorted(root.glob("*/episode.json"))
        if not paths:
            raise ValueError(f"No episodes at {root}")
        episode_dirs = {p.parent for p in paths}
        if any(p.is_dir() and p.name.startswith("episode-") and p not in episode_dirs for p in root.iterdir()):
            raise ValueError(f"Incomplete episode in {root}")
        for path in paths:
            info = json.loads(path.read_text(encoding="utf-8"))
            town = require_collection_map(info["map"])
            if (info["config_sha256"] != CONFIG_SHA256 or info["dt"] != DT or
                    info["future_tail_ticks"] != 25 or not info["server"].startswith("1.0")):
                raise ValueError(f"Incompatible capture contract: {path}")
            if any(not info[k].startswith(f"{town}/") or len(info[k].split("/")) != 2 or
                   not info[k].split("/")[1] for k in ("route_id", "initial_state_id")):
                raise ValueError(f"Invalid grouping metadata: {path}")
            yield path.parent, info


def check(condition, message):
    if not condition:
        raise ValueError(message)


def check_finite(*groups):
    for group in groups:
        for key, item in group.items():
            check(np.issubdtype(item.dtype, np.number) or item.dtype == bool, f"Invalid dtype for {key}")
            check(np.isfinite(item).all(), f"NaN/Inf in {key}")


def box_evidence(scan, boxes, views, mounts):
    """Cross-project independent in-box LiDAR returns onto RGB for manual spot checks."""
    observed = []
    for box in boxes:
        distance = np.linalg.norm(box[:2])
        if not 4 <= distance <= 40:
            continue
        # A 30 cm tolerance allows for bounding-box approximation and lidar hits on the shell.
        offset = scan[:, :3] - box[:3]
        c, s = np.cos(box[6]), np.sin(box[6])
        along = offset[:, 0] * c + offset[:, 1] * s
        across = -offset[:, 0] * s + offset[:, 1] * c
        inside = ((abs(along) <= box[3] / 2 + .30) & (abs(across) <= box[4] / 2 + .30) &
                  (abs(offset[:, 2]) <= box[5] / 2 + .30) & (scan[:, 2] > box[2] - box[5] / 2 + .3))
        for name, mount, image in zip(NAMES, mounts, views):
            center, visible, _ = project(box[:3], name, mount)
            if not visible:
                continue
            corners = footprint(box)
            corners = np.vstack((np.column_stack((corners, np.full(4, box[2] - box[5] / 2))),
                                 np.column_stack((corners, np.full(4, box[2] + box[5] / 2)))))
            box_uv, _, depths = project(corners, name, mount)
            if min(depths) <= 0:
                continue
            raw_lower, raw_upper = box_uv.min(axis=0), box_uv.max(axis=0)
            # Exclude clipped and tiny boxes. QC reviewers get visible, interpretable evidence.
            ww, hh = image.shape[1], image.shape[0]
            if np.any(raw_lower < 0) or np.any(raw_upper >= (ww, hh)) or min(raw_upper - raw_lower) < 6:
                continue
            uv, valid, _ = project(scan[inside, :3], name, mount)
            lower, upper = raw_lower - 3, raw_upper + 3
            matched = uv[valid & ((uv >= lower) & (uv <= upper)).all(axis=1)]
            if len(matched) < 3:
                continue
            annotated = image.copy()
            cv2.circle(annotated, tuple(np.rint(center).astype(int)), 3, (25, 255, 25), -1)
            for x, y in np.rint(matched[::max(1, len(matched) // 150)]).astype(int):
                cv2.circle(annotated, (x, y), 1, (10, 180, 255), -1)
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                         (0, 4), (1, 5), (2, 6), (3, 7)):
                cv2.line(annotated, tuple(np.rint(box_uv[a]).astype(int)),
                         tuple(np.rint(box_uv[b]).astype(int)), (25, 255, 25), 1)
            observed.append((name, len(matched), annotated))
            break
    return observed


def event_count(mask, min_frames, separation=5):
    steps = np.flatnonzero(mask)
    if not len(steps):
        return 0
    return sum(len(r) >= min_frames for r in np.split(steps, np.flatnonzero(np.diff(steps) > separation) + 1))


def maneuvers(states, steps, scenario):
    yaw = np.array([np.arctan2(p[1, 0], p[0, 0]) for p in steps])
    speed = np.array([s["ego_state"][0] for s in states])
    rate = abs(wrap(np.diff(yaw, prepend=yaw[0]))) / DT
    curvature = rate / np.maximum(speed, 2.)
    counts = dict(straight=event_count((speed >= 3) & (curvature < .005), 10),
                  curve=event_count((speed >= 3) & (curvature >= .005) & (curvature < .028), 5),
                  sharp_curve=event_count((speed >= 3) & (curvature >= .028), 4),
                  left_change=0, right_change=0, slow_overtake=0,
                  ramp=event_count(np.array([s["lane_state"][2] for s in states]) != 0, 5)
                  if scenario == "ramp" else 0)
    last = -10
    for i in range(1, len(states) - 4):
        old, new = states[i - 1]["lane_state"], states[i]["lane_state"]
        if i - last < 10 or old[0] != new[0] or old[2] or new[2] or old[1] * new[1] <= 0:
            continue
        if abs(old[1] - new[1]) != 1 or not all(np.array_equal(new, s["lane_state"]) for s in states[i:i + 5]):
            continue
        lateral = np.dot(steps[min(i + 5, len(steps) - 1)][:3, 3] - steps[max(0, i - 5)][:3, 3],
                         -steps[max(0, i - 5)][:3, 1])
        if abs(lateral) < .5:
            continue
        last = i
        counts["left_change" if lateral > 0 else "right_change"] += 1
        blockers = {int(s["thw_actor_id"]) for s in states[max(0, i - 20):i]
                    if 0 <= s["thw"] < 2.4 and s["thw_actor_id"] > 0}
        if scenario == "slow_overtake" and blockers:
            # Credit the event only when the exact blocked actor is passed.
            for s in states[i + 4:]:
                b = s["boxes"]
                passed = ((b[:, -1] == 1) & (b[:, 0] < -2.3) &
                          (abs(b[:, 1]) > 1.9) & (abs(b[:, 1]) < 6) & (np.cos(b[:, 6]) > .8))
                if blockers.intersection(s["box_ids"][passed]):
                    counts["slow_overtake"] += 1
                    break
    return counts


def audit_episode(folder, info, out, episode_number):
    frames = info["frame_ids"]
    expected = [str(f) for f in frames]
    check(len(frames) >= 4 and all(type(f) is int for f in frames) and np.all(np.diff(frames) == 1),
          f"Missing or nonconsecutive frames: {folder}")
    check({p.name for p in folder.iterdir() if p.is_dir()} == set(expected), f"Unindexed frames: {folder}")
    change = info["lane_change"]
    requested, ended = change["requested_frame"], change["ended_frame"]
    scenario_side = {"left_change": "left", "right_change": "right", "slow_overtake": "left"}.get(info["scenario"])
    if requested is None:
        check(ended is None and change["side"] == "follow" and change["status"] == "unrequested",
              f"Invalid unrequested command: {folder}")
    else:
        check(requested in frames[12:] and change["side"] == scenario_side, f"Invalid command request: {folder}")
        check((ended is None and change["status"] == "executing") or
              (ended in frames and ended > requested and change["status"] in {"completed", "timed_out"}),
              f"Invalid command end: {folder}")
    stamps, poses, states, painted, evidence = [], [], [], 0, []
    prev = None
    for idx, fid in enumerate(frames):
        path = folder / str(fid)
        obs, audit, labels = [load_npz(path / f"{n}.npz") for n in ("obs", "audit", "labels")]
        validate_observation(obs)
        check_finite(audit, labels)
        check(set(labels) == set(LABEL_SHAPES) and all(labels[n].shape == s for n, s in LABEL_SHAPES.items()),
              f"Invalid target shapes: {path}")
        check(all(np.isin(labels[n], (0, 1)).all() for n in BINARY), f"Non-binary masks: {path}")
        check(np.array_equal(labels["vector_exists"], labels["vector_valid"].any(axis=1)),
              f"Vector existence inconsistent: {path}")
        check(np.array_equal(obs["ego_speed"], labels["ego_state"][[0, 3]]), f"Speed or yaw rate mismatch: {path}")
        active_cmd = requested is not None and fid >= requested and (ended is None or fid < ended)
        cmd = command_vector(change["side"] if active_cmd else "follow")
        check(np.array_equal(obs["cmd"], cmd), f"Command outside the execution window: {path}")
        check(audit["sensor_ids"].shape == (len(AUDIT_SENSORS),) and np.all(audit["sensor_ids"] == fid),
              f"Sensor frame mismatch: {path}")
        ts = audit["timestamps"]
        check(ts.shape == (len(AUDIT_SENSORS),) and np.ptp(ts) <= .001, f"Sensor timestamp mismatch: {path}")
        stamps.append(ts)
        pose = audit["pose"]
        scan = audit["lidar_points"]
        check(pose.shape == (4, 4) and audit["sensor_poses"].shape == (len(AUDIT_SENSORS), 4, 4) and
              np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6) and
              np.array_equal(pose[3], [0., 0., 0., 1.]) and scan.ndim == 2 and
              scan.shape[1] == 4 and len(scan) > 0, f"Invalid pose or LiDAR: {path}")
        check((np.max(abs(obs["ego_delta"])) == 0 if prev is None else
               delta_error(obs["ego_delta"], prev, pose) < 1e-5), f"ego_delta mismatch: {path}")
        prev = pose
        poses.append(pose)
        masks = labels["lanes"]
        check(not np.any(masks[1] & masks[2]), f"Dashed and solid marking overlap: {path}")
        painted += bool(masks[1:3].any())
        boxes = labels["boxes"]
        check(np.isin(boxes[:, -1], (0, 1)).all() and np.all(np.diff(boxes[:, -1]) <= 0),
              f"Invalid detection slots: {path}")
        ids = labels["box_ids"]
        active = boxes[:, -1] == 1
        check(np.all(ids[~active] == -1) and np.all(ids[active] > 0) and
              len(np.unique(ids[active])) == active.sum(), f"Invalid actor identifiers: {path}")
        boxes = boxes[boxes[:, -1] == 1]
        norm_error = abs(np.hypot(np.sin(boxes[:, 6]), np.cos(boxes[:, 6])) - 1)
        check(np.all(boxes[:, 3:6] >= MIN_BOX) and np.all(boxes[:, 3:6] <= MAX_BOX) and
              np.all(abs(boxes[:, 6]) <= np.pi + 1e-6) and np.all(norm_error < 1e-4) and
              np.isin(boxes[:, 9], np.arange(len(OBJECT_CLASSES))).all() and
              np.all(np.diff(np.linalg.norm(boxes[:, :3], axis=1)) >= 0), f"Invalid box geometry: {path}")
        check(0 <= labels["nav_state"][1] < 8 and labels["nav_state"][2] > 0,
              f"Navigation offset or opposite direction: {path}")
        views = []
        for name, (w, h) in zip(NAMES, SIZES):
            view = read_image(path / f"{name}.png")
            check(view.shape == (h, w, 3) and view.dtype == np.uint8, f"Sensor shape: {path}/{name}")
            views.append(view)
        if idx % 12 == 0 and len(evidence) < 3:
            mounts = np.linalg.solve(pose, audit["sensor_poses"][:6])
            check(np.allclose(mounts, camera_mounts(), atol=.001, rtol=0), f"RGB extrinsic mismatch: {path}")
            for name, hits, img in box_evidence(scan, boxes, views, mounts)[:3 - len(evidence)]:
                destination = f"box-{episode_number:05d}-{fid}-{len(evidence)}-{name}.png"
                (out / destination).write_bytes(cv2.imencode(".png", img)[1].tobytes())
                evidence.append(dict(frame=fid, camera=name, points=hits, image=destination))
        states.append({k: labels[k] for k in ("ego_state", "boxes", "box_ids", "lane_state", "thw", "thw_actor_id")})
    check(np.max(abs(np.diff(stamps, axis=0) - DT)) <= .001, f"10 Hz timebase drift: {folder}")
    check(painted / len(frames) >= .90, f"Lane marking coverage {painted / len(frames):.1%}: {folder}")
    return dict(episode=folder.name, map=info["map"], frames=len(frames), marking_fraction=painted / len(frames),
                evidence=evidence), maneuvers(states, poses, info["scenario"])


def run_qc(roots, out, require_evidence=True):
    entries = list(episode_index(roots))  # Reject a prohibited map before auditing any frame.
    out.mkdir(exist_ok=True, parents=True)
    reports, per_episode, totals = [], [], Counter({k: 0 for k in EVENTS})
    for i, (path, info) in enumerate(entries):
        report, count = audit_episode(path, info, out, i)
        reports.append(report)
        per_episode.append(dict(episode=path.name, map=info["map"], scenario=info["scenario"], counts=count))
        totals.update(count)
    warnings = [f"No independent events recorded for {key}" for key in EVENTS if not totals[key]]
    evidence_count = sum(len(r["evidence"]) for r in reports)
    passed = evidence_count > 0 or not require_evidence
    qc = dict(passed=passed, frames=sum(r["frames"] for r in reports), episodes=reports, box_samples=evidence_count,
              sensor_frame_offset=0, dropped_frame_rate=0, nonfinite_count=0,
              dashed_solid_overlap=0, nav_ge_8m=0, nav_wrong_direction=0, town04_opt_frames=0)
    for name, report in (("qc", qc), ("maneuvers", dict(totals=totals, warnings=warnings, episodes=per_episode))):
        (out / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not passed:
        raise ValueError("No visible vehicle box could be cross-checked with LiDAR and RGB")
    return qc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_qc(args.source, args.out)
    print(f"QC passed on {report['frames']} frames, {report['box_samples']} box overlays")


if __name__ == "__main__":
    main()
