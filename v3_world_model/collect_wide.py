"""Collect and audit contiguous, dense Town12 motorway clips with the locked rig."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .calibration_probe import require_pass
from .collect_v3 import (NPCS, SpawnUnavailable, choose_path, digest, episode, raised_spawn,
                         write_json)
from .labels import RoadSurfaces, neighbor, xyz
from .qc_v3 import audit_episode
from .sim import connect, synchronous


def motorway_sections(road):
    limits = {}
    for segment in ET.fromstring(road.to_opendrive()).findall("road"):
        if segment.get("junction") != "-1":
            continue
        intervals = sorted((float(t.get("s")), t.find("speed")) for t in segment.findall("type"))
        for idx, (start, speed) in enumerate(intervals):
            if speed is None or speed.get("unit") not in ("km/h", "kmh") or speed.get("max") != "120":
                continue
            end = intervals[idx + 1][0] if idx + 1 < len(intervals) else float(segment.get("length"))
            limits.setdefault(int(segment.get("id")), []).append((start, end))
    return limits


def lanes_at(wp, limits):
    if (wp is None or wp.is_junction or wp.lane_width < 2.7 or
            not any(lo <= wp.s < hi for lo, hi in limits.get(wp.road_id, ()))):
        return []
    lanes = [wp]
    for side in (-1, 1):
        current = wp
        while (adjacent := neighbor(current, side)) is not None:
            if (adjacent.road_id != wp.road_id or adjacent.lane_id * wp.lane_id <= 0 or
                    adjacent.lane_width < 2.7 or adjacent.lane_id in {x.lane_id for x in lanes} or
                    adjacent.transform.get_forward_vector().dot(wp.transform.get_forward_vector()) < .95):
                break
            lanes.append(adjacent)
            current = adjacent
    return lanes


def traffic(carla, world, tm, ego, route, scenario, rng, npcs):
    _, path, *_ = route
    options = [bp for bp in world.get_blueprint_library().filter("vehicle.*")
               if bp.id in NPCS and NPCS[bp.id] <= 1]
    for offset in (6, 13, 21):
        for lane in lanes_at(path[offset + int(rng.integers(-2, 3))], {path[0].road_id: [(0, float("inf"))]}):
            if offset == 6 and lane.lane_id == path[0].lane_id:
                continue
            other = world.try_spawn_actor(options[int(rng.integers(len(options)))],
                                          raised_spawn(carla, lane.transform, .4))
            if other is not None:
                npcs.append(other)
                other.set_autopilot(True, tm.get_port())
                tm.set_desired_speed(other, float(rng.uniform(48, 82)))
                tm.auto_lane_change(other, False)
    if len(npcs) < 3:
        raise SpawnUnavailable("Fewer than three same-carriageway NPCs could spawn")
    ego.set_autopilot(True, tm.get_port())
    tm.set_path(ego, [wp.transform.location for wp in path[1:]])
    tm.set_desired_speed(ego, 74)
    tm.auto_lane_change(ego, False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=5000)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=2460)
    parser.add_argument("--seed", type=int, default=31024)
    args = parser.parse_args()
    if args.frames < 4:
        parser.error("At least one four-frame clip is required")
    carla, client, world, server = connect("127.0.0.1", args.port, "Town12")
    require_pass(args.calibration, "Town12", server)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    work = out.with_name(out.name + "-attempts")
    work.mkdir(exist_ok=True)
    road = world.get_map()
    limits = motorway_sections(road)
    candidates = defaultdict(list)
    for wp in road.generate_waypoints(60):
        if len(lanes_at(wp, limits)) >= 3:
            candidates[(wp.road_id, np.sign(wp.lane_id))].append(wp)
    rng = np.random.default_rng(args.seed)
    groups = list(candidates)
    rng.shuffle(groups)
    if not groups:
        raise ValueError("No eligible motorway candidates on active map")
    print(f"Town12 motorway candidate carriageways: {len(groups)}", flush=True)
    tm = client.get_trafficmanager(args.port + 5)
    n = (args.frames + 119) // 120
    with synchronous(world, tm, args.seed):
        surfaces = RoadSurfaces(road)
        for idx in range(n):
            size = args.frames // n + (idx < args.frames % n)
            dest = out / f"episode-{idx:05d}"
            if dest.exists():
                info = json.loads((dest / "episode.json").read_text())
                if not info.get("motorway_audit") or len(info["frame_ids"]) != size or info["seed"] != args.seed:
                    raise ValueError(f"Incompatible existing capture {dest}")
                continue
            for attempt in range(150):
                pool = candidates[groups[(idx + attempt) % len(groups)]]
                start = pool[int(rng.integers(len(pool)))]
                path = choose_path(start)
                if len(path) < 101 or any(w.road_id != start.road_id or len(lanes_at(w, limits)) < 3 for w in path):
                    continue
                route = (start.transform, path, np.stack([xyz(w.transform.location) for w in path]),
                         0., False, (), digest([start.road_id]))
                stage = work / f"capture-{idx}-{attempt}"
                if stage.exists():
                    continue
                counts = []

                def audit(wp, boxes, fid):
                    width = len(lanes_at(wp, limits))
                    same = (boxes[:, -1] == 1) & (np.cos(boxes[:, 6]) > .9) & (abs(boxes[:, 2]) < 3)
                    cars = int(same.sum())
                    if width < 3 or cars < 2:
                        raise SpawnUnavailable(f"Frame {fid}: {width} lanes, {cars} same-direction boxes")
                    counts.append([width, cars])

                try:
                    episode(carla, world, tm, surfaces, route, "multi_car", size, rng, stage,
                            "Town12", server, args.seed, idx, traffic, audit)
                    info = json.loads((stage / "episode.json").read_text())
                    audit_episode(stage, info, work, idx)
                except (SpawnUnavailable, ValueError) as exc:
                    print(f"Reject episode {idx}, road {start.road_id}: {exc}", flush=True)
                    world.tick(45)
                    continue
                info["motorway_audit"] = dict(road=start.road_id, speed_limit_kmh=120,
                                              columns=["same_direction_lanes", "same_direction_boxes"],
                                              counts=counts)
                write_json(stage / "episode.json", info)
                stage.rename(dest)
                print(f"Accepted {idx+1}/{n}, {size} frames, road {start.road_id}, "
                      f"minimum lanes/cars {np.min(counts, axis=0).tolist()}", flush=True)
                break
            else:
                raise RuntimeError(f"Could not capture eligible episode {idx}")


if __name__ == "__main__":
    main()
