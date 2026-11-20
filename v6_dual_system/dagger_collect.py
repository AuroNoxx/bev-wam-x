"""Offline map teacher attached to observations from the unmodified closed-loop pilot."""
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

from .collect_fast_data import neighbor, path_from
from .derive_lane_supervision import lane_target
from .diagnose_gt_controller import Reference
from .sensors import CAL_ID, DATA, LOG, write_json

BASE=DATA/"dagger-epoch26"
ROW_BYTES=6*360*640+3*128*128


def protected_roads(town):
    roads=set()
    manifest=json.loads((DATA/"manifest-corrective.json").read_bytes())
    for episode in {entry[0] for entry in manifest["validation"]}:
        meta=json.loads((DATA/episode/"complete.json").read_bytes())
        if meta["town"]==town:roads.update(row[0] for row in meta["route"])
    for path in LOG.glob("closed-loop-*/summary.json"):
        run=json.loads(path.read_bytes())
        if run["town"]!=town or run.get("scenario")=="dagger":continue
        with (path.parent/"frames.jsonl").open() as stream:
            roads.update(json.loads(line)["road"] for line in stream)
    return roads


def collection_route(map_,town,seed):
    blocked=protected_roads(town)
    starts=[w for w in map_.generate_waypoints(12) if w.road_id not in blocked and
            not w.is_junction and w.lane_width>=3 and (neighbor(w,True) or neighbor(w,False))]
    random.Random(seed).shuffle(starts)
    longest=[]
    for w in starts[:400]:
        available=[]
        for p in path_from(w,distance=700):
            if p.road_id in blocked:break
            available.append(p)
        if len(available)>len(longest):longest=available
        if len(available)>=260:return available,blocked
    if len(longest)<90:raise ValueError(f"{town} has no route with enough disjoint teacher lookahead")
    return longest,blocked


class DaggerRecorder:
    def __init__(self,town,route,blocked,map_,seed,quota,checkpoint):
        import time
        self.folder=BASE/town/f"{seed}-{time.time_ns()}"
        self.folder.mkdir(parents=True)
        self.map=map_
        self.ref=Reference(route)
        self.blocked=blocked
        self.roads=set()
        self.quota=quota
        self.file=(self.folder/"observations.bin").open("xb")
        self.hasher=hashlib.sha256()
        self.rows=[]
        self.meta=dict(town=town,seed=seed,route_start=int(route[0].id),calibration=CAL_ID,
                       protected_roads=sorted(blocked),teacher_roads=sorted({w.road_id for w in route}),
                       checkpoint=checkpoint,empty_dynamic_scene=True)

    def add(self,frame,pose,rgb,bev,state,target):
        aux,wp,_=lane_target(self.map,pose)
        arc,offset,_=self.ref.project(pose)
        if (wp.road_id in self.blocked or arc+140>=self.ref.arc[-1]
                or abs(offset)>2*wp.lane_width):return True
        if self.rows and frame<=self.rows[-1][0]:raise ValueError("Repeated source frame")
        self.roads.add(wp.road_id)
        ref=self.ref
        heading=ref.heading[ref.index]
        yaw=heading-math.atan2(pose[1,0],pose[0,0])
        speed=max(0.,state[0]*math.cos(yaw))
        lateral_speed=state[0]*math.sin(yaw)
        dt=.1
        times=np.arange(1,41)*dt
        # A quintic in route-normal coordinates starts at the learner's actual
        # displacement and normal velocity and meets the designated lane smoothly.
        duration=min(4.,max(2.,math.sqrt(2.0*abs(offset))+.35*abs(lateral_speed)))
        u=np.minimum(1.,times/duration)
        residual=offset*(1-10*u**3+15*u**4-6*u**5)
        residual+=lateral_speed*duration*(u-6*u**3+8*u**4-3*u**5)
        s=arc
        stations=[]
        for _ in times:
            look=s+max(5.,speed*1.5)
            bends=np.interp([look-5,look+5],ref.arc[:-1],ref.heading)
            cap=min(target/3.6,math.sqrt(3./max(abs(bends[1]-bends[0])/10.,1e-4)))
            next_speed=speed+np.clip(cap-speed,-4.*dt,2.*dt)
            s+=.5*(speed+next_speed)*dt
            stations.append(s)
            speed=next_speed
        xyz=np.column_stack([np.interp(stations,ref.arc,axis) for axis in ref.xyz.T])
        angle=np.interp(stations,ref.arc[:-1],ref.heading)
        xyz[:,0]+=residual*np.sin(angle)
        xyz[:,1]-=residual*np.cos(angle)
        local=(xyz-pose[:3,3])@pose[:3,:3]
        xy=(local[:,:2]*[1,-1]).astype(np.float32)
        if not np.isfinite(xy).all():raise ValueError("Invalid teacher geometry")
        # A teacher that turns onto an unrelated arm cannot supervise lane recovery.
        if xy[-1,0]<=0 or abs(math.atan2(xy[-1,1],xy[-1,0]))>math.radians(50):
            return True
        for data in (rgb,bev):
            raw=memoryview(np.ascontiguousarray(data)).cast("B")
            self.file.write(raw)
            self.hasher.update(raw)
        self.rows.append((frame,state,target,xy,aux,offset,yaw))
        return len(self.rows)>=self.quota

    def close(self,success):
        self.file.close()
        if not success or not self.rows:return
        frame,state,target,xy,aux,offset,yaw=zip(*self.rows)
        labels=self.folder/"labels.npz"
        np.savez(labels,frame=frame,state=np.float32(state),target=np.float32(target),xy=np.float32(xy),
                 lane=np.float32(aux),offset=offset,heading=yaw)
        if (self.folder/"observations.bin").stat().st_size!=len(frame)*ROW_BYTES:raise ValueError("Partial observations")
        self.meta.update(count=len(frame),visited_roads=sorted(self.roads),observations_sha256=self.hasher.hexdigest(),
                         labels_sha256=hashlib.sha256(labels.read_bytes()).hexdigest())
        write_json(self.folder/"complete.json",self.meta)
