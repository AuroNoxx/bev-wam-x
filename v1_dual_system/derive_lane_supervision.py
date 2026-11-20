"""Derive current-frame lane supervision from recorded poses and local OpenDRIVE."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
import math

import carla
import numpy as np

from .sensors import CAL_ID, DATA, ROOT, write_json


def lane_target(lane_map, pose):
    location=carla.Location(*pose[:3,3].tolist())
    wp=lane_map.get_waypoint(location,project_to_road=True,lane_type=carla.LaneType.Driving)
    if wp is None:raise ValueError(f"No driving lane at {location}")
    tf=wp.transform
    theta=math.radians(tf.rotation.yaw)
    lat=(location.x-tf.location.x)*math.sin(theta)-(location.y-tf.location.y)*math.cos(theta)
    # CARLA yaw is clockwise in projected coordinates. FLU yaw is positive left.
    delta=theta-math.atan2(pose[1,0],pose[0,0])
    yaw=math.atan2(math.sin(delta),math.cos(delta))
    inside=lane_map.get_waypoint(location,project_to_road=False,lane_type=carla.LaneType.Driving)
    return (lat,yaw),wp,inside


def derive(corrective):
    suffix="-corrective" if corrective else ""
    manifest=json.loads((DATA/f"manifest{suffix}.json").read_bytes())
    if manifest["calibration"]!=CAL_ID:raise ValueError("Unexpected calibration")
    out=DATA/"lane-supervision"/manifest["identity"]
    out.mkdir(parents=True,exist_ok=True)
    desc=dict(version=1,manifest=manifest["identity"],calibration=CAL_ID,
              convention="nearest Driving lane, ego left-positive [metres, radians]",maps={},splits={})
    maps={}
    for split in ("train","validation"):
        entries=manifest[split]
        groups=defaultdict(list)
        for j,(episode,i) in enumerate(entries):groups[episode].append((j,i))
        truth=np.full((len(entries),2),np.nan,np.float32)
        junction=edge=outside=0
        for episode,rows in groups.items():
            folder=DATA/episode
            meta=json.loads((folder/"complete.json").read_bytes())
            town=meta["town"]
            if town not in maps:
                path=ROOT/".cache/v1-dual-system/carla-client/0.9.15/Carla/Maps/OpenDrive"/f"{town}.xodr"
                if not path.exists():
                    path=ROOT/"runtime/CARLA-0.9.15/WindowsNoEditor/CarlaUE4/Content/Carla/Maps/OpenDrive"/path.name
                raw=path.read_bytes()
                maps[town]=carla.Map(town,raw.decode())
                desc["maps"][town]=hashlib.sha256(raw).hexdigest()
            with np.load(folder/"labels.npz") as labels,np.load(folder/"stream.npz") as stream:
                frames,history=labels["frame"],labels["history"]
                poses,ticks=stream["poses"],stream["frame"]
                for j,i in rows:
                    f=frames[i]
                    if ticks[f]-ticks[history[i]]!=6:raise ValueError(f"Broken temporal alignment at {episode}/{i}")
                    val,wp,inside=lane_target(maps[town],poses[f])
                    truth[j]=val
                    junction+=int(wp.is_junction)
                    edge+=int(abs(abs(val[0])-wp.lane_width/2)<.15)
                    outside+=int(inside is None)
        if not np.isfinite(truth).all():raise ValueError("Missing lane targets")
        stream=io.BytesIO()
        np.save(stream,truth,allow_pickle=False)
        raw=stream.getvalue()
        digest=hashlib.sha256(raw).hexdigest()
        filename=f"{split}-{digest}.npy"
        temp=out/(filename+".pending")
        temp.write_bytes(raw)
        temp.replace(out/filename)
        order=hashlib.sha256(json.dumps(entries,separators=(",",":")).encode()).hexdigest()
        desc["splits"][split]=dict(file=filename,sha256=digest,order_sha256=order,count=len(truth),
                                  junction=junction,within_15cm_lane_edge=edge,outside_driving_lane=outside,
                                  max_abs=np.max(np.abs(truth),axis=0).tolist())
    write_json(out/"identity.json",desc)
    print(json.dumps(desc))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--corrective",action="store_true")
    derive(parser.parse_args().corrective)
