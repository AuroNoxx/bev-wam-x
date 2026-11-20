"""Recover metric route anchors without using a future observation or expert trajectory."""
import argparse
import hashlib
import json
import math
from collections import defaultdict

import carla
import numpy as np

from .collect_fast_data import path_from
from .diagnose_gt_controller import Reference
from .navigation import anchors, FILM_STATIONS
from .sensors import DATA, LOG, ROOT, write_json


def save_array(folder, name, values):
    path = folder / (name + ".npy")
    np.save(path, values, allow_pickle=False)
    return dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), count=len(values))


def focused_routes(maps, out):
    from .collect_spatial import forks
    root = DATA / "spatial-navigation"
    result = {}
    for town, map_ in maps.items():
        choices = {f"{town}-fork-{road}-{lane}": [approach + path_from(b,360) for b in branches]
                   for (road,lane),(approach,branches) in forks(map_)}
        edges = {w.id:w for w in map_.generate_waypoints(28)}
        for meta_path in sorted(root.glob(town + "-*/complete.json")):
            folder = meta_path.parent
            meta = json.loads(meta_path.read_bytes())
            case = meta["case"]
            paths = choices.get(meta["group"])
            route = paths[case == "right"] if paths else path_from(edges[meta["route_start"]],720)
            assert route[0].id == meta["route_start"]
            with np.load(folder/"labels.npz") as labels, np.load(folder/"stream.npz") as stream:
                frames, poses, old = labels["frame"], stream["poses"], labels["nav"]
                station = labels["arc"] - meta["branch_station"]
            def project(path):
                ref = Reference(path)
                return np.stack([anchors(ref,p,FILM_STATIONS) for p in poses[:max(frames)+1]])[frames]
            values = project(route)
            np.testing.assert_allclose(values[:,:4],old,atol=.015)
            result[folder.name] = save_array(out,folder.name,values)
            if folder.name == "Town04-fork-50-6-left":
                idx = np.flatnonzero((station>=-30.) & (station<0.))
                if len(idx)!=15:
                    raise ValueError("Held-out approach frame count changed")
                left,right = values[idx],project(paths[1])[idx]
                np.savez(out/"probe.npz",indices=idx,left=left,right=right,station=station[idx])
    return result


def main(film=False):
    distances = FILM_STATIONS if film else (30.,60.)
    manifest = json.loads((DATA / "manifest-corrective.json").read_bytes())
    out = DATA / ("film-navigation" if film else "navigation") / manifest["identity"]
    out.mkdir(exist_ok=True, parents=True)
    maps, starts = {}, {}
    desc = dict(manifest=manifest["identity"], splits={}, dagger={}, maps={})
    for town in ("Town04", "Town06"):
        path = ROOT / ".cache/v1-dual-system/carla-client/0.9.15/Carla/Maps/OpenDrive" / (town+".xodr")
        raw = path.read_bytes()
        maps[town] = carla.Map(town, raw.decode())
        starts[town] = {w.id:w for w in maps[town].generate_waypoints(12)}
        desc["maps"][town] = hashlib.sha256(raw).hexdigest()
    for split in ("train", "validation"):
        entries = manifest[split]
        groups = defaultdict(list)
        for idx, (episode, i) in enumerate(entries):
            groups[episode].append((idx,i))
        values = np.empty((len(entries),2*len(distances)), np.float32)
        for episode, pairs in groups.items():
            folder = DATA / episode
            meta = json.loads((folder/"complete.json").read_bytes())
            route = meta["route"]
            start = maps[meta["town"]].get_waypoint_xodr(*route[0])
            path = path_from(start)
            if len(path[::8])!=len(route):raise ValueError("Recorded route length mismatch")
            for w, (road, lane, s) in zip(path[::8],route):
                if (w.road_id,w.lane_id)!=(road,lane) or abs(w.s-s)>.1:
                    raise ValueError(f"Route reconstruction differs in {episode}")
            ref = Reference(path)
            with np.load(folder/"labels.npz") as labels, np.load(folder/"stream.npz") as stream:
                frames, poses = labels["frame"], stream["poses"]
                indices = {int(frames[i]):idx for idx,i in pairs}
                if len(indices)!=len(pairs):raise ValueError("Duplicate source frames")
                for f in range(max(indices)+1):
                    if f not in indices:
                        ref.project(poses[f])
                        continue
                    value = anchors(ref,poses[f],distances)
                    if np.linalg.norm(value[2:4])>67.5:
                        raise ValueError(f"Route projection too far at {episode}/{f}: {value}")
                    values[indices[f]] = value
        desc["splits"][split] = save_array(out,split,values)
        desc["splits"][split]["order"] = hashlib.sha256(json.dumps(entries,separators=(",",":")).encode()).hexdigest()
    logs = {}
    for path in LOG.glob("closed-loop-*-dagger/summary.json"):
        m = json.loads(path.read_bytes())
        logs[m["town"],m["seed"],m["route_start"]] = path.parent
    for path in sorted((DATA/"dagger-epoch26").glob("*/*/complete.json")):
        m = json.loads(path.read_bytes())
        town,seed,start = m["town"],m["seed"],m["route_start"]
        trace = logs[town,seed,start]/"frames.jsonl"
        rows = {r["frame"]:r for s in trace.read_text().splitlines() if (r:=json.loads(s))}
        ref = Reference(path_from(starts[town][start],distance=700))
        with np.load(path.parent/"labels.npz") as lab:
            frames=lab["frame"]
        values = []
        for i,f in enumerate(frames):
            if f not in rows:continue
            row = rows[f]
            angle=math.radians(row["heading_deg"])
            pose=np.eye(4)
            pose[:2,0]=[math.cos(angle),math.sin(angle)]
            pose[:2,3]=row["world_xy"]
            values.append([i,*anchors(ref,pose,distances)])
        desc["dagger"][str(path.parent.relative_to(DATA).as_posix())] = save_array(out,f"{town}-{seed}",np.float32(values))
    if film:
        old = DATA / "navigation" / manifest["identity"]
        for key, skip in (("splits",0),("dagger",1)):
            for item in desc[key].values():
                values = np.load(out/item["file"])
                np.testing.assert_allclose(values[:,:4+skip],np.load(old/item["file"]),atol=.015)
        desc["focus"] = focused_routes(maps,out)
        desc["stations_m"] = distances
    write_json(out/"identity.json",desc)
    print({k:v["count"] for k,v in desc["splits"].items()},sum(v["count"] for v in desc["dagger"].values()),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--film",action="store_true")
    main(parser.parse_args().film)
