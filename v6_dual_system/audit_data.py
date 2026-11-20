"""Validate complete causal windows, matched raw streams and route-level splits."""
import hashlib
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

from .collect_fast_data import MIX, SCENES, TOWNS
from .sensors import CALIBRATION, CAL_ID, DATA, write_json


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit(corrective=False):
    check(json.loads((DATA/"calibration.json").read_bytes())==CALIBRATION, "sensor calibration")
    index, summary, failures = {}, {}, []
    base=json.loads((DATA/"manifest.json").read_bytes()) if corrective else None
    if corrective:
        from .collect_corrections import REPLACE
    for town,(total,split) in TOWNS.items():
        grouped = {s:[] for s in SCENES}
        for f in sorted((DATA/town).glob("*/complete.json")):
            meta = json.loads(f.read_bytes())
            check((meta["town"],meta["split"],meta["calibration"],meta["dt"])==(town,split,CAL_ID,.05),str(f))
            if meta["count"]:
                grouped[meta["scene"]].append((f.parent,meta))
        chosen, speeds, intents, recovery, accel, rec_curve = [], [], [], [], [], []
        route_ids=set()
        for scene,fraction in zip(SCENES,MIX):
            candidates = []
            for d,meta in grouped[scene]:
                with np.load(d/"labels.npz") as a:
                    candidates.extend((d,i) for i in range(len(a["frame"])))
            quota=round(total*fraction)
            if corrective:
                active={p for p,_ in grouped[scene]}
                candidates=[(DATA/d,i) for d,i in base[split] if DATA/d in active]
                if split=="train" and scene in REPLACE:
                    staged=[]
                    for f in sorted((DATA/"Town04-corrections").glob(scene+"-*/complete.json")):
                        meta=json.loads(f.read_bytes())
                        check((meta["town"],meta["split"],meta["calibration"],meta["dt"])==(town,split,CAL_ID,.05),str(f))
                        staged.extend((f.parent,i) for i in range(meta["count"]))
                    count=REPLACE[scene]
                    check(len(staged)>=count,f"Insufficient corrective {scene}: {len(staged)}/{count}")
                    old_ids=np.linspace(0,len(candidates)-1,quota-count,dtype=int)
                    new_ids=np.linspace(0,len(staged)-1,count,dtype=int)
                    candidates=[candidates[i] for i in old_ids]+[staged[i] for i in new_ids]
            if len(candidates)<quota:
                failures.append(f"{town}/{scene}: {len(candidates)}/{quota}")
            ids=np.linspace(0,len(candidates)-1,min(quota,len(candidates)),dtype=int)
            selected={}
            for j in ids:
                path,i=candidates[j]
                selected.setdefault(path,[]).append(i)
            for d,ix in selected.items():
                with np.load(d/"labels.npz") as a, np.load(d/"stream.npz") as stream:
                    meta=json.loads((d/"complete.json").read_bytes())
                    route_ids.add(meta["route_start"])
                    frame=a["frame"][ix]
                    hist=a["history"][ix]
                    stamp=stream["frame"]
                    check(np.all(stamp[frame+80]-stamp[hist]==86),f"{d}: frame continuity")
                    check(np.all(np.diff(stamp)>0) and np.all(frame-hist==6),str(d))
                    poses=stream["poses"]
                    inv=np.linalg.inv(poses[frame])
                    future=poses[frame[:,None]+np.arange(2,81,2),:3,3]
                    xy=(future @ inv[:,:3,:3].transpose(0,2,1)+inv[:,None,:3,3])[:,:,:2]*[1,-1]
                    check(np.allclose(xy,a["xy"][ix],atol=1e-4),f"{d}: trajectory labels")
                    check(a["occ"].shape==(meta["count"],4,128,16),f"{d}: occupancy shape")
                    for key in ("xy","state","target","offset","curvature"):
                        check(np.isfinite(a[key]).all(),f"{d}: nonfinite {key}")
                    n=meta["raw_frames"]
                    check((d/"bev.bin").stat().st_size==n*3*128*128,f"{d}: BEV size")
                    for name in ("rgb","lidar"):
                        offsets=stream[name]
                        check(len(offsets)==n+1 and offsets[0]==0 and np.all(np.diff(offsets)>=0)
                              and offsets[-1]==(d/(name+".bin")).stat().st_size,f"{d}: {name} offsets")
                    raw=np.memmap(d/"rgb.bin",dtype=np.uint8,mode="r")
                    scan=np.memmap(d/"lidar.bin",dtype=np.float32,mode="r")
                    rgb_offsets, lidar_offsets = stream["rgb"], stream["lidar"]
                    for j in np.unique(np.r_[hist,frame]):
                        lo,hi=rgb_offsets[j:j+2]
                        image=cv2.imdecode(raw[lo:hi],cv2.IMREAD_COLOR)
                        check(image is not None and image.shape==(360,640,3) and image.min()<image.max(),f"{d}: RGB {j}")
                        begin,end=lidar_offsets[j:j+2]
                        check(end>begin and (end-begin)%16==0 and np.isfinite(scan[begin//4:end//4]).all(),f"{d}: LiDAR {j}")
                    v=a["state"][ix,0]*3.6
                    speeds.extend(v)
                    intents.extend(a["intent"][ix].tolist())
                    recovery.extend(a["recovery"][ix].tolist())
                    if scene=="recovery": rec_curve.extend(a["curvature"][ix].tolist())
                    if scene=="braking": accel.extend(a["state"][ix,1].tolist())
                    rel=d.relative_to(DATA).as_posix()
                    chosen.extend([rel,i] for i in ix)
        v=np.asarray(speeds)
        bins=np.histogram(v,bins=np.r_[np.arange(0,120,5),np.inf])[0]
        turns=np.bincount(intents,minlength=3)
        if len(chosen)!=total or not np.all(bins>0) or not len(v) or not np.any((v>=119.5)&(v<=120.5)):
            failures.append(f"{town}: speed/count coverage")
        if len(v) and (min(v)>.5 or abs(sum(recovery)/len(v)-.15)>.001 or min(turns)==0 or min(accel,default=0)>-3):
            failures.append(f"{town}: standstill/recovery/change/brake evidence")
        if not (any(x<.001 for x in rec_curve) and any(x>=.001 for x in rec_curve)):
            failures.append(f"{town}: recovery needs both straight and curved segments")
        summary[split]=dict(town=town,samples=len(v),episodes=len(set(p for p,_ in chosen)),distinct_starts=len(route_ids),
                            speed_range_kmh=[min(v),max(v)] if len(v) else [],speed_bins_5_kmh=bins.tolist(),
                            intents=turns.tolist(),recovery=int(sum(recovery)),hard_brake_frames=sum(x<-3 for x in accel))
        index[split]=chosen
    suffix="-corrective" if corrective else ""
    if corrective:
        check(index["validation"]==base["validation"],"Paired independent validation set changed")
        index["parent_identity"]=base["identity"]
        families={}
        for f in (DATA/"Town04-corrections").glob("*/complete.json"):
            m=json.loads(f.read_bytes())
            if m["adaptation"]["route_kind"]=="paired":
                families.setdefault(m["adaptation"]["route_seed"],[]).append((f.parent,m))
        worst=0.
        for members in families.values():
            check({m["scene"] for _,m in members}=={"highway","left","right"},"incomplete physical branch family")
            check(len({m["route_start"] for _,m in members})==1,"paired start mismatch")
            traces=[]
            for folder,m in members:
                with np.load(folder/"stream.npz") as stream:
                    frames=stream["frame"]-stream["frame"][0]
                    poses=stream["poses"]
                traces.append({int(t):p[:3,3] for t,p in zip(frames,poses) if t<m["adaptation"]["event_tick"]-2})
            common=sorted(set.intersection(*(set(t) for t in traces)))
            check(len(common)>20,"insufficient shared pre-command history")
            worst=max(worst,max(float(np.linalg.norm(t[k]-traces[0][k])) for t in traces[1:] for k in common))
        summary["paired_branches"]=dict(training_families=len(families),max_precommand_center_delta_m=worst)
    write_json(DATA/f"audit{suffix}.json",dict(splits=summary,failures=failures))
    print(json.dumps(summary,indent=2),flush=True)
    if failures: raise ValueError(failures)
    index["calibration"]=CAL_ID
    index["identity"]=hashlib.sha256(json.dumps(index,sort_keys=True).encode()).hexdigest()
    write_json(DATA/f"manifest{suffix}.json",index)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--corrective",action="store_true")
    audit(p.parse_args().corrective)
