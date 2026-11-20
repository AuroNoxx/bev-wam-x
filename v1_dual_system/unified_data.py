"""Paths and sidecars for the current-frame, route-conditioned driver."""
import io
import json
import math
from pathlib import Path

import numpy as np

from .sensors import DATA, ROOT, write_json
from .unified_grid import lidar_raster

BASE=DATA/"unified-ipm"
ART=ROOT/"artifacts/v1-dual-system/unified-ipm"
S=np.arange(-18,117,2,dtype=np.float64)


def route_polyline(ref,pose,station):
    # Outside the route, repeat its physical endpoint; never invent a continuation.
    xyz=np.stack([np.interp(station+S,ref.arc,axis) for axis in ref.xyz.T],axis=1)
    xyz=(xyz-pose[:3,3])@pose[:3,:3]
    return (xyz[:,:2]*[1.,-1.]).astype(np.float32)


def prepare_original(split):
    import carla
    from collections import defaultdict
    from .collect_fast_data import path_from
    from .diagnose_gt_controller import Reference
    source=json.loads((DATA/"manifest-corrective.json").read_bytes())
    groups=defaultdict(list)
    for e,i in source[split]:groups[e].append(i)
    indices=[]
    maps={}
    for episode,selection in groups.items():
        old=DATA/episode
        out=BASE/"original"/episode
        out.mkdir(parents=True,exist_ok=True)
        if not (out/"complete.json").exists():
            meta=json.loads((old/"complete.json").read_bytes())
            town=meta["town"]
            if town not in maps:
                raw=ROOT/".cache/v1-dual-system/carla-client/0.9.15/Carla/Maps/OpenDrive"/(town+".xodr")
                maps[town]=carla.Map(town,raw.read_text(encoding="utf8"))
            path=path_from(maps[town].get_waypoint_xodr(*meta["route"][0]))
            if len(path[::8])!=len(meta["route"]) or any((w.road_id,w.lane_id)!=(r,l) or abs(w.s-s)>.1 for w,(r,l,s) in zip(path[::8],meta["route"])):
                raise ValueError(episode+" route changed")
            ref=Reference(path)
            with np.load(old/"stream.npz") as stream,np.load(old/"labels.npz") as labels:
                poses=stream["poses"]
                rgb=stream["rgb"]
                lidar=stream["lidar"]
                selected=np.asarray(selection)
                frames=labels["frame"][selected]
                positions={int(f):i for i,f in enumerate(frames)}
                nav=np.zeros((len(frames),len(S),2),np.float32)
                raster=np.zeros((len(frames),3,128,128),np.uint8)
                locked=np.zeros(len(frames),np.float32)
                with (old/"lidar.bin").open("rb") as scans:
                    for f,pose in enumerate(poses):
                        arc,lat,_=ref.project(pose)
                        if f not in positions:continue
                        i=positions[f]
                        scans.seek(int(lidar[f]))
                        raw=scans.read(int(lidar[f+1]-lidar[f]))
                        if not raw or rgb[f]==rgb[f+1]:raise ValueError(f"Missing current sensor {episode}/{f}")
                        raster[i]=lidar_raster(np.frombuffer(raw,np.float32).reshape(-1,4))
                        nav[i]=route_polyline(ref,pose,arc)
                        if meta["scene"] in ("left","right"):
                            direction=1 if meta["scene"]=="left" else -1
                            width=path[ref.source_index[ref.index]].lane_width
                            # Explicit collection manoeuvre, including its post-change KEEP windows.
                            changing=labels["intent"][selected[i]]!=0
                            if changing or direction*lat>.75*width:locked[i]=direction*width
                np.save(out/"lidar.npy",raster)
                np.savez(out/"labels.npz",v=labels["state"][selected,0],xy=labels["xy"][selected],
                         cmd=labels["intent"][selected],curve=labels["curvature"][selected],
                         nav=nav,shift=locked,rgb=np.stack((rgb[frames],rgb[frames+1]),axis=1),
                         frame=stream["frame"][frames])
            write_json(out/"complete.json",dict(rgb_source=str((old/"rgb.bin").relative_to(DATA)),n=len(frames),
                      occupancy_valid=False,group=episode,split=split,fork=False,town=town))
        indices.append(str(out.relative_to(BASE)))
    write_json(BASE/(split+"-original.json"),indices)
    print(split,len(indices),len(source[split]),flush=True)


if __name__=="__main__":
    for name in ("train","validation"):prepare_original(name)
