"""Collect contiguous Stage 3.1 episodes on the training map allowlist."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .calibration_probe import require_pass
from .geometry import CFG, CONFIG_SHA256, DT, FLU, FUTURE_TICKS, NAMES, ego_delta, lidar_bev, local
from .labels import (EVENT_FIELDS, SOLID, ActorState, RoadSurfaces, future_targets, lane_vectors,
                     local_objects, mark_type, neighbor, object_slots, occupancy, same_direction, xyz)
from .sim import Rig, bgr, connect, depth_cm, matrix, require_collection_map, synchronous

SCENARIOS = ("straight", "multi_car", "curve", "sharp_curve", "left_change",
             "right_change", "slow_overtake", "ramp")
NPCS = {"vehicle.mini.cooper": 0, "vehicle.dodge.charger": 0, "vehicle.nissan.patrol": 1,
        "vehicle.carlamotors.carlacola": 2, "vehicle.harley-davidson.low_rider": 3}
SENSOR_NAMES = tuple(f"rgb/{n}" for n in NAMES) + ("lidar",) + tuple(f"depth/{n}" for n in NAMES)


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def choose_path(start, turn_bias=0):
    path = [start]
    for _ in range(100):
        cur = path[-1]
        options = cur.next(3.0)
        if not options:
            break
        fwd = cur.transform.get_forward_vector()
        right = cur.transform.get_right_vector()
        best = max(options, key=lambda w: fwd.dot(w.transform.get_forward_vector()) +
                   turn_bias * right.dot(w.transform.get_forward_vector()))
        path.append(best)
    return path


def catalog(road, carla):
    result = []
    for spawn in road.get_spawn_points():
        wp = road.get_waypoint(spawn.location, lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        for turn_bias in (0, -1.25, 1.25):
            path = choose_path(wp, turn_bias)
            if len(path) < 76:
                continue
            xyzs = np.stack([xyz(x.transform.location) for x in path])
            angles = np.unwrap(np.deg2rad([x.transform.rotation.yaw for x in path]))
            bend = float(np.rad2deg(np.max(np.abs(angles[4:40] - angles[:36]))))
            # A connector must link lane sections with different lane bundles.
            # This rejects ordinary same-width intersection turns as ramp candidates.
            lane_counts = [lane_bundle(w) for w in (path[2], path[27])]
            ramp = any(x.is_junction for x in path[4:20]) and bend > 16 and min(lane_counts) < max(lane_counts)
            side = tuple(same_direction(w, neighbor(w, s)) for s in (-1, 1) for w in path[5:12])
            run = (all(side[:7]), all(side[7:]))
            route_key = [x.road_id for x in path]
            topology = [key for i, key in enumerate(route_key) if i == 0 or key != route_key[i - 1]]
            result.append((spawn, path, xyzs, bend, ramp, run, digest(topology)))
    if not result:
        raise RuntimeError("No continuous Stage 3 routes found")
    return result


def lane_bundle(wp):
    count = 1
    for side in (-1, 1):
        current = wp
        for _ in range(12):
            other = neighbor(current, side)
            if not same_direction(current, other):
                break
            count += 1
            current = other
    return count


def select_route(routes, scenario, sequence, rng, used):
    def rank(row):
        _, _, _, bend, ramp, sides, route = row
        reuse = (scenario, route) in used
        if scenario == "ramp":
            return (not ramp, reuse, abs(bend - 31))
        if scenario in ("left_change", "right_change", "slow_overtake"):
            side_ok = sides[0 if scenario != "right_change" else 1]
            return (not side_ok, reuse, bend)
        target = {"curve": 16, "sharp_curve": 32}.get(scenario, 0)
        return (False, reuse, abs(bend - target))

    eligible = sorted(routes, key=rank)
    if rank(eligible[0])[0]:
        raise RuntimeError(f"No route supports {scenario}")
    candidates = [route for route in eligible if rank(route) < (False, True, 0)]
    pool = candidates or eligible
    top = [route for route in pool[:24] if not rank(route)[0]]
    # Stable seeded choice within the first few equally suitable route variants.
    top = top[:min(6 + sequence // len(SCENARIOS), len(top))]
    selected = top[int(rng.integers(len(top)))]
    used.add((scenario, selected[-1]))
    return selected


class RouteNav:
    """Coarse route anchors from the commanded SD centerline, independently audited against driving."""

    def __init__(self, line):
        self.points = line[::3]
        self.start = self.points[:-1, :2]
        self.delta = np.diff(self.points[:, :2], axis=0)
        self.length = np.maximum(np.linalg.norm(self.delta, axis=1), .001)
        self.arc = np.r_[0., np.cumsum(self.length)]
        self.segment = 0

    def state(self, pose):
        lo, hi = max(0, self.segment - 2), min(len(self.length), self.segment + 8)
        delta = self.delta[lo:hi]
        diff = pose[:2, 3] - self.start[lo:hi]
        t = np.clip(np.einsum("ij,ij->i", diff, delta) / self.length[lo:hi] ** 2, 0., 1.)
        idx = int(np.argmin(np.linalg.norm(diff - t[:, None] * delta, axis=1)))
        i = self.segment = lo + idx
        cross = delta[idx, 0] * diff[idx, 1] - delta[idx, 1] * diff[idx, 0]
        alignment = np.dot(delta[idx], pose[:2, 0]) / self.length[i]
        progress = self.arc[i] + t[idx] * self.length[i]
        anchors = np.column_stack([np.interp(progress + [12., 24., 36.], self.arc, v)
                                   for v in self.points.T])
        return local(anchors, pose).astype(np.float32), np.array(
            [progress, abs(cross) / self.length[i], alignment], dtype=np.float32)


def actors_at(actors, snapshot):
    results = []
    for actor in actors:
        snap = snapshot.find(actor.id)
        if snap is None:
            continue
        tf = snap.get_transform()
        bbox = actor.bounding_box
        pose = matrix(tf)
        bbox_pose = pose @ matrix(type(tf)(bbox.location, bbox.rotation))
        results.append(ActorState(bbox_pose[:3, 3], bbox_pose[:3, 0], xyz(snap.get_velocity()),
                                  2 * xyz(bbox.extent), NPCS[actor.type_id], actor.id))
    return results


def spawn_traffic(carla, world, tm, ego, route, scenario, rng, npcs):
    spawn, path, *_ = route
    library = world.get_blueprint_library()
    options = [bp for bp in library.filter("vehicle.*") if bp.id in NPCS]
    if not options:
        raise RuntimeError("No supported CARLA NPC blueprints")
    if scenario in ("slow_overtake", "multi_car"):
        tf = path[9].transform
        tf.location.z += .40
        leader = world.spawn_actor(options[0], tf)
        npcs.append(leader)
        leader.set_autopilot(True, tm.get_port())
        tm.set_path(leader, [wp.transform.location for wp in path[10:]])
        tm.set_desired_speed(leader, 19 if scenario == "slow_overtake" else 65)
        tm.auto_lane_change(leader, False)
    others = sorted(world.get_map().get_spawn_points(),
                    key=lambda tf: tf.location.distance(spawn.location))
    limit = 14 if scenario == "multi_car" else 7
    for tf in others:
        if len(npcs) >= limit:
            break
        if tf.location.distance(spawn.location) < 35:
            continue
        other = world.try_spawn_actor(options[int(rng.integers(len(options)))], tf)
        if other is None:
            continue
        npcs.append(other)
        other.set_autopilot(True, tm.get_port())
        tm.set_desired_speed(other, float(rng.uniform(36, 84)))
        tm.auto_lane_change(other, False)
    ego.set_autopilot(True, tm.get_port())
    tm.set_path(ego, [wp.transform.location for wp in path[1:]])
    tm.set_desired_speed(ego, {"multi_car": 82, "ramp": 55, "sharp_curve": 46}.get(scenario, 64))
    tm.auto_lane_change(ego, False)


def front_vehicle(objects, ids, speed):
    ahead = (objects[:, 0] > 0) & (abs(objects[:, 1]) < 1.8) & (np.cos(objects[:, 6]) > .5)
    if not ahead.any() or speed < .5:
        return -1., -1
    gaps = objects[ahead, 0] - objects[ahead, 3] / 2 - 2.3
    nearest = np.argmin(gaps)
    return float(max(0., gaps[nearest]) / speed), ids[ahead][nearest]


def write_observation(folder, fid, bundle, pose, prev, scan, sensor_poses, lidar_angle):
    frame = folder / str(fid)
    frame.mkdir()
    for name in NAMES:
        ok, data = cv2.imencode(".png", bgr(bundle[f"rgb/{name}"]), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise IOError(f"PNG write failed for {name}")
        (frame / f"{name}.png").write_bytes(data.tobytes())
    np.savez(frame / "obs.npz", lidar_bev=lidar_bev(scan),
             ego_delta=np.zeros(3, np.float32) if prev is None else ego_delta(prev, pose))
    np.savez_compressed(frame / "depth.npz", **{n: depth_cm(bundle[f"depth/{n}"]) for n in NAMES})
    np.savez(frame / "audit.npz", pose=pose, sensor_poses=sensor_poses, lidar_points=scan,
             sensor_ids=np.asarray([bundle[n].frame for n in SENSOR_NAMES], np.int64),
             timestamps=np.asarray([bundle[n].timestamp for n in SENSOR_NAMES], np.float64),
             lidar_angle=np.float64(lidar_angle))


def episode(carla, world, tm, surfaces, route, scenario, count, rng, folder, town, server, seed, index):
    spawn, path, xy, _, _, _, route_hash = route
    ego_bp = world.get_blueprint_library().find("vehicle.lincoln.mkz")
    ego_bp.set_attribute("role_name", "hero")
    ego = world.spawn_actor(ego_bp, spawn)
    npcs, trace, pending, retained = [], [], [], []
    event_ids = {"collision": set(), "solid_crossing": set()}
    nav = RouteNav(xy)
    road = world.get_map()
    previous = None
    forced = False
    folder.mkdir()
    try:
        world.set_weather(carla.WeatherParameters.ClearNoon)
        spawn_traffic(carla, world, tm, ego, route, scenario, rng, npcs)
        with Rig(carla, world, ego) as rig:
            rig.cameras()
            rig.cameras("depth")
            rig.lidar()
            rig.attach("collision", "sensor.other.collision", carla.Transform(), {},
                       lambda data: event_ids["collision"].add(data.frame))
            rig.attach("lane", "sensor.other.lane_invasion", carla.Transform(), {},
                       lambda data: event_ids["solid_crossing"].add(data.frame)
                       if any(mark_type(m) in SOLID for m in data.crossed_lane_markings) else None)
            for _ in range(10):
                rig.tick()
            for tick in range(count + FUTURE_TICKS[-1]):
                snap, bundle = rig.tick()
                fid = snap.frame
                ego_snap = snap.find(ego.id)
                tf = ego_snap.get_transform()
                pose = matrix(tf)
                speed = float(np.linalg.norm(xyz(ego_snap.get_velocity())[:2]))
                moving = actors_at(npcs, snap)
                trace.append((pose, speed, moving))
                if tick >= count:
                    continue
                current, ids = local_objects(moving, pose)
                anchors, nav_state = nav.state(pose)
                waypoint = road.get_waypoint(tf.location, lane_type=carla.LaneType.Driving)
                lidar = bundle["lidar"]
                scan = np.frombuffer(lidar.raw_data, dtype="<f4").reshape(-1, 4).copy()
                sensor_poses = np.stack([matrix(bundle[n].transform) for n in SENSOR_NAMES])
                scan[:, :3] = local(scan[:, :3] @ sensor_poses[6, :3, :3].T + sensor_poses[6, :3, 3], pose)
                retained.append(fid)
                write_observation(folder, fid, bundle, pose, previous, scan, sensor_poses, lidar.horizontal_angle)
                previous = pose
                vec, vec_valid, vec_exists = lane_vectors(waypoint, pose)
                lon, lat, _ = (xyz(ego_snap.get_acceleration()) @ pose[:3, :3]) * FLU
                control = ego.get_control()
                state = np.array([speed, lon, lat, -np.deg2rad(ego_snap.get_angular_velocity().z),
                                  control.steer, control.throttle, control.brake], np.float32)
                thw, thw_id = front_vehicle(current, ids, speed)
                if scenario in ("left_change", "right_change", "slow_overtake") and not forced:
                    side = 1 if scenario == "right_change" else -1
                    due = tick >= 12 and (scenario != "slow_overtake" or 0 <= thw < 2.4)
                    if due and waypoint and same_direction(waypoint, neighbor(waypoint, side)):
                        tm.force_lane_change(ego, side > 0)
                        forced = True
                state_lane = np.array([waypoint.road_id, waypoint.lane_id, int(waypoint.is_junction)],
                                      dtype=np.int64) if waypoint else np.zeros(3, np.int64)
                boxes, box_ids = object_slots(current, ids)
                targets = dict(lanes=surfaces.masks(pose), vectors=vec, vector_valid=vec_valid,
                               vector_exists=vec_exists, boxes=boxes, box_ids=box_ids, occupancy=occupancy(scan, current),
                               ego_state=state, thw=np.float32(thw), nav_anchors=anchors,
                               nav_state=nav_state, lane_state=state_lane, thw_actor_id=np.int64(thw_id))
                pending.append(targets)
        for tick, (fid, targets) in enumerate(zip(retained, pending, strict=True)):
            future = trace[tick + 1:tick + 1 + FUTURE_TICKS[-1]]
            poses, speeds, moving = zip(*future)
            trajectory, dynamic = future_targets(trace[tick][0], poses, speeds, moving)
            targets.update(trajectory=trajectory, future_occupancy=dynamic, flags=np.array(
                [fid in event_ids[field] for field in EVENT_FIELDS[:2]] +
                [targets["nav_state"][1] >= 5 or targets["lane_state"][1] == 0], dtype=bool))
            np.savez_compressed(folder / str(fid) / "labels.npz", **targets)
        route_id = f"{town}/{route_hash}"
        start_id = f"{town}/{digest(np.round(matrix(spawn), 3).tolist())}"
        write_json(folder / "episode.json", dict(map=town, server=server, route_id=route_id,
                   initial_state_id=start_id, scenario=scenario, frame_ids=retained, dt=DT,
                   config_sha256=CONFIG_SHA256, seed=seed, episode=index, traffic_count=len(npcs),
                   forced_change=forced, future_tail_ticks=FUTURE_TICKS[-1]))
    finally:
        for actor in reversed(npcs):
            actor.destroy()
        ego.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="Town10HD_Opt")
    parser.add_argument("--port", type=int, default=2460)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--seed", type=int, default=714)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--pilot-frames", type=int)
    modes.add_argument("--frames", type=int)
    args = parser.parse_args()
    town = require_collection_map(args.map)
    scenarios = tuple(dict.fromkeys(args.scenarios or SCENARIOS))
    count = args.pilot_frames if args.pilot_frames is not None else args.frames
    if count < len(scenarios) * 4:
        parser.error("Allocate at least four usable frames to every scenario")
    carla, client, world, server = connect(args.host, args.port, town)
    require_pass(args.calibration, town, server)
    args.out.mkdir(parents=True, exist_ok=False)
    road = world.get_map()
    rng = np.random.default_rng(args.seed)
    tm = client.get_trafficmanager(args.port + 5)
    used = set()
    with synchronous(world, tm, args.seed):
        routes = catalog(road, carla)
        surfaces = RoadSurfaces(road)
        n = max(len(scenarios), (count + 119) // 120)
        for index in range(n):
            scenario = scenarios[index % len(scenarios)]
            chosen = select_route(routes, scenario, index, rng, used)
            print(f"Episode {index + 1}/{n}: {scenario}, route {chosen[-1]}", flush=True)
            stage = args.out / f".partial-{index:05d}"
            episode(carla, world, tm, surfaces, chosen, scenario, count // n + (index < count % n),
                    rng, stage, town, server, args.seed, index)
            stage.rename(args.out / f"episode-{index:05d}")


if __name__ == "__main__":
    main()
