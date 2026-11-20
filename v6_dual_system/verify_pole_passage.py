"""Measure the same physical pole in both CARLA runs using passive LiDAR observations."""
import argparse
import json
from pathlib import Path

import numpy as np

from .export_crash_bev import clearance


def rows(file):
    return [json.loads(line) for line in file.read_bytes().splitlines() if line]


def world_points(xyz,pose):
    return (xyz*[1,-1,1])@pose[:3,:3].T+pose[:3,3]


def verify(original,run):
    reference=original/"evidence"
    meta=json.loads((reference/"capture.json").read_bytes())
    with np.load(reference/f"frame-{meta['crash_frame']}.npz") as impact:
        points=impact["semantic_xyz"]
        in_box=np.all(abs(points-np.array(meta["box_center"]))<=np.array(meta["extent"])+.02,axis=1)
        colliding=(impact["semantic_tags"]==meta["tags"]["Poles"]) & in_box
        if not colliding.any():raise ValueError("Cannot identify original contact")
        centre=world_points(points[colliding],impact["sensor_pose"]).mean(0)
    samples=[]
    for file in reference.glob("frame-*.npz"):
        with np.load(file) as frame:
            pole=frame["semantic_xyz"][frame["semantic_tags"]==meta["tags"]["Poles"]]
            xyz=world_points(pole,frame["sensor_pose"])
            samples.extend(xyz[np.linalg.norm(xyz[:,:2]-centre[:2],axis=1)<.45])
    obs=run/"evidence"
    telemetry=rows(run/"frames.jsonl")
    with np.load(obs/"pole-passages.npz") as new:
        static=np.unique(np.round(samples,3),axis=0)
        ext=new["extent"];box=new["box_center"];rotation=np.deg2rad(new["box_yaw_deg"])
        n=new["indices"];frames=new["frames"];poses=new["poses"]
        # Include the newly observed faces of the same pole in the offline distance calculation.
        extra=[]
        points=new["points"]
        for i,pose in enumerate(poses):
            xyz=world_points(points[n[i]:n[i+1]],pose)
            extra.extend(xyz[np.linalg.norm(xyz[:,:2]-centre[:2],axis=1)<.45])
        if extra:static=np.unique(np.round(np.vstack((static,extra)),3),axis=0)
        distances=[];longitudinal=[]
        for pose in poses:
            local=((static-pose[:3,3])@pose[:3,:3])*[1,-1,1]
            local=local[abs(local[:,2]-box[2])<=ext[2]]
            distances.append(clearance(local,box[:2],rotation,ext[:2]) if len(local) else np.inf)
            longitudinal.append(float((centre-pose[:3,3])@pose[:3,0]))
    nearest=int(np.argmin(distances))
    if not np.isfinite(distances[nearest]):raise ValueError("No height overlap with original pole")
    by_frame={r["frame"]:r for r in telemetry}
    index=lambda f:min(by_frame,key=lambda n:abs(n-int(f)))
    close=by_frame[index(frames[nearest])]
    t0=min(telemetry,key=lambda r:abs(r["sim_s"]-20.70))
    at=int(np.argmin(abs(frames-t0["frame"])))
    passed=min(longitudinal[nearest:]) < -6 and np.linalg.norm(poses[nearest,:2,3]-centre[:2])<12
    speeds=[by_frame[index(f)]["kmh"] for f,d in zip(frames,distances) if d<=12.]
    contacts=rows(run/"collisions.jsonl") if (run/"collisions.jsonl").exists() else []
    meta2=json.loads((obs/"run-meta.json").read_bytes())
    out=dict(run=run.name,original=original.name,same_checkpoint=meta["sha256"]==meta2["sha256"],
             pole_xy=centre[:2].tolist(),pole_surface_points=len(static),new_observations=len(extra),
             sensor_ticks=len(frames),missing_sensor_ticks=int(frames[-1]-frames[0]+1-len(frames)),
             passed_location=bool(passed),min_clearance_m=float(distances[nearest]),
             closest_sim_s=close["sim_s"],closest_speed_kmh=close["kmh"],
             at_original_time=dict(time=t0["sim_s"],speed=t0["kmh"],clearance=distances[at]),
             passing_speed_p50_kmh=float(np.median(speeds)) if speeds else None,
             collisions=contacts,elapsed_sim_s=telemetry[-1]["sim_s"],controls=len(telemetry),
             guard_frames=sum(r["corridor_guard"] for r in telemetry),static_braking_frames=sum(r["static_brake"] for r in telemetry))
    (run/"pole-passage.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("original",type=Path);p.add_argument("run",type=Path)
    args=p.parse_args();verify(args.original,args.run)
