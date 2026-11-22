"""Measure surround blind spots, actual mount error and cross-sensor road-line reprojection."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .geometry import (CFG, CONFIG_SHA256, FLU, HEIGHTS, NAMES, camera_mounts,
                       focal_length_px, project, projection_tables)
from .sim import Rig, bgr, connect, matrix, require_collection_map, synchronous

RULES = CFG["calibration"]
SEMANTIC = np.dtype([("xyz", "<f4", (3,)), ("cos", "<f4"), ("actor", "<u4"), ("tag", "<u4")])


def coverage(mounts):
    radii = np.linspace(*RULES["ground_annulus_m"], 51)
    azimuth = np.deg2rad(np.arange(360))
    x, y = radii[:, None] * np.cos(azimuth), radii[:, None] * np.sin(azimuth)
    pts = np.stack((x, y, np.full_like(x, HEIGHTS[0])), axis=-1)
    visible = np.any([project(pts, name, tf)[1] for name, tf in zip(NAMES, mounts)], axis=0)
    max_gap = 0
    for ring in visible:
        run = 0
        for gap in np.tile(~ring, 2):
            run = run + 1 if gap else 0
            max_gap = max(max_gap, min(run, 360))
    return float(visible.mean()), max_gap


def metrics_pass(report):
    try:
        fraction = report["coverage_fraction"]
        return (report["frames"] >= RULES["minimum_frames"] and
                RULES["min_coverage_fraction"] <= fraction <= 1 and
                0 <= report["largest_angular_gap_deg"] <= RULES["max_gap_degrees"] and
                0 <= report["worst_mount_abs_error"] <= RULES["max_mount_error"] and
                0 <= report["cameras"]["back"]["max_ego_occlusion_fraction"] <=
                RULES["max_back_ego_occlusion_fraction"] and
                all(report["cameras"][name]["matches"] >= RULES["min_lane_matches_per_camera"] and
                    0 <= report["cameras"][name]["lane_p95_px"] <= RULES["max_lane_p95_px"]
                    for name in NAMES))
    except (KeyError, TypeError):
        return False


def require_pass(path, town, server):
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("passed") is not True or report.get("map") != town or
            report.get("server") != server or report.get("config_sha256") != CONFIG_SHA256 or
            not metrics_pass(report)):
        raise ValueError(f"Calibration gate failed for {town}, CARLA {server}: {path}")


def ego_silhouette(name, mount, shape, box_min, box_max):
    """Ray/box intersection in CARLA rig axes, used to distinguish ego from other cars."""
    h, w = shape
    v, u = np.indices(shape)
    focal = focal_length_px(name)
    rays = np.stack((np.ones(shape), (u - w / 2) / focal, -(v - h / 2) / focal), axis=-1)
    rays = rays @ mount[:3, :3].T
    origin = mount[:3, 3]
    inv = 1 / np.where(np.abs(rays) < 1e-10, 1e-10, rays)
    a, b = (box_min - origin) * inv, (box_max - origin) * inv
    near = np.minimum(a, b).max(axis=-1)
    far = np.maximum(a, b).min(axis=-1)
    return far >= np.maximum(near, 1e-6)


def lane_reprojection(xyz_flu, name, mount, labels, ego_mask, road_tag):
    radius = np.linalg.norm(xyz_flu[:, :2], axis=1)
    inner, outer = RULES["ground_annulus_m"]
    in_ring = xyz_flu[(inner <= radius) & (radius <= outer)]
    uv, visible, depth = project(in_ring, name, mount)
    max_depth = focal_length_px(name) * RULES["nominal_marking_width_m"] / RULES["min_marking_width_px"]
    subpixel = visible & (depth > max_depth)
    uv = uv[visible & ~subpixel]
    x, y = np.rint(uv).astype(np.int32).T
    blocked = ego_mask[y, x]
    yy, xx = np.nonzero((labels == road_tag) & ~ego_mask)
    if not len(xx):
        # If RoadLines is absent despite visible projected returns, count opaque misses.
        distances = np.full((~blocked).sum(), np.hypot(*labels.shape))
    else:
        distances, _ = cKDTree(np.column_stack((xx, yy))).query(uv[~blocked])
    return distances, int(blocked.sum()), int(subpixel.sum())


def mosaic(images, mounts, ego_masks):
    uv, mask = projection_tables(mounts)
    ground = HEIGHTS.index(-.2)
    sums = np.zeros((*mask.shape[-2:], 3), np.float32)
    count = np.zeros((*mask.shape[-2:], 1), np.float32)
    for cam, (image, ego_mask) in enumerate(zip(images, ego_masks, strict=True)):
        coordinates = uv[ground, cam]
        u, v = coordinates[..., 0], coordinates[..., 1]
        occluded = cv2.remap(ego_mask.astype(np.uint8), u, v, cv2.INTER_NEAREST) > 0
        keep = (mask[ground, cam] & ~occluded)[..., None]
        sums += cv2.remap(image, u, v, cv2.INTER_LINEAR) * keep
        count += keep
    rgb = sums / (255 * np.maximum(count, 1))
    return np.rint(255 * rgb ** CFG["rgb"]["preprocessing_gamma"]).astype(np.uint8)


def capture(carla, world, count):
    road = world.get_map()
    starts = road.get_spawn_points()
    if not starts:
        raise RuntimeError("Map has no calibration spawn points")
    picks = np.unique(np.linspace(0, len(starts) - 1, min(5, count), dtype=int))
    road_tag = int(carla.CityObjectLabel.RoadLines)
    car_tag = int(carla.CityObjectLabel.Car)
    errors = {name: [] for name in NAMES}
    occlusions = {name: [] for name in NAMES}
    exclusions = {name: np.zeros(2, dtype=int) for name in NAMES}
    fractions, gaps, mount_errors, images = [], [], [], []
    for index, spawn_idx in enumerate(picks):
        bp = world.get_blueprint_library().find("vehicle.lincoln.mkz")
        ego = world.spawn_actor(bp, starts[spawn_idx])
        try:
            corners = np.asarray([[p.x, p.y, p.z] for p in ego.bounding_box.get_local_vertices()])
            box_min, box_max = corners.min(axis=0) - .10, corners.max(axis=0) + .10
            ego.apply_control(carla.VehicleControl(hand_brake=True))
            with Rig(carla, world, ego) as rig:
                rig.cameras()
                rig.cameras("semantic_segmentation")
                rig.lidar(semantic=True)
                for _ in range(8):
                    rig.tick()
                for tick in range(count // len(picks) + (index < count % len(picks))):
                    snap, bundle = rig.tick()
                    pose = matrix(snap.find(ego.id).get_transform())
                    mounts = np.stack([np.linalg.solve(pose, matrix(bundle[f"rgb/{n}"].transform)) for n in NAMES])
                    mount_errors.append(float(np.max(np.abs(mounts - camera_mounts()))))
                    frac, gap = coverage(mounts)
                    fractions.append(frac)
                    gaps.append(gap)
                    lidar = bundle["semantic_lidar"]
                    lidar_pose = np.linalg.solve(pose, matrix(lidar.transform))
                    points = np.frombuffer(lidar.raw_data, SEMANTIC)
                    xyz = points["xyz"][points["tag"] == road_tag] @ lidar_pose[:3, :3].T + lidar_pose[:3, 3]
                    xyz = xyz * FLU
                    ego_masks = []
                    for name, mount in zip(NAMES, mounts):
                        labels = bgr(bundle[f"semantic_segmentation/{name}"])[..., 2]
                        ego_mask = ((labels == car_tag) &
                                    ego_silhouette(name, mount, labels.shape, box_min, box_max))
                        ego_masks.append(ego_mask)
                        occlusions[name].append([ego_mask.mean(), ego_mask[-30:].mean()])
                        distances, blocked, subpixel = lane_reprojection(
                            xyz, name, mount, labels, ego_mask, road_tag)
                        errors[name].extend(distances.tolist())
                        exclusions[name] += (blocked, subpixel)
                    if tick == 0:
                        images.append(mosaic([bgr(bundle[f"rgb/{n}"]) for n in NAMES], mounts, ego_masks))
        finally:
            ego.destroy()
    cameras = {}
    for name, values in errors.items():
        whole, bottom = np.asarray(occlusions[name]).T
        cameras[name] = dict(matches=len(values), lane_p95_px=float(np.quantile(values, .95)) if values else None,
                             ego_occlusion_fraction=float(whole.mean()),
                             max_ego_occlusion_fraction=float(whole.max()),
                             bottom30_ego_occlusion_fraction=float(bottom.mean()),
                             max_bottom30_ego_occlusion_fraction=float(bottom.max()),
                             ego_occluded_returns=int(exclusions[name][0]),
                             subpixel_returns=int(exclusions[name][1]))
    samples = [v for values in errors.values() for v in values]
    return dict(frames=count, cameras=cameras, coverage_fraction=min(fractions),
                largest_angular_gap_deg=max(gaps), worst_mount_abs_error=max(mount_errors),
                lane_p95_px=float(np.quantile(samples, .95)) if samples else None), images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="Town10HD_Opt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2460)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    town = require_collection_map(args.map)
    if args.frames < RULES["minimum_frames"]:
        parser.error(f"At least {RULES['minimum_frames']} calibration frames required")
    carla, _, world, server = connect(args.host, args.port, town)
    args.out.mkdir(parents=True, exist_ok=True)
    with synchronous(world):
        report, images = capture(carla, world, args.frames)
    report.update(map=town, server=server, config_sha256=CONFIG_SHA256, passed=metrics_pass(report),
                  ground_annulus_m=RULES["ground_annulus_m"], thresholds=RULES)
    for i, img in enumerate(images):
        path = args.out / f"surround_bev_{i:02d}.png"
        path.write_bytes(cv2.imencode(".png", cv2.resize(img, None, fx=4, fy=4))[1].tobytes())
    (args.out / "calibration.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("Calibration gate below threshold. Inspect calibration.json and surround mosaics.")


if __name__ == "__main__":
    main()
