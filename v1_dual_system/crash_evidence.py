"""Passive, collision-triggered capture. Semantic labels never enter the driving pipeline."""
from collections import deque
import hashlib
import json
from threading import Lock

import carla
import numpy as np

from .sensors import CALIBRATION, DT


class CrashEvidence:
    def __init__(self,world,ego,folder,checkpoint):
        self.folder=folder/"evidence"
        self.folder.mkdir()
        self.frames=deque(maxlen=84)
        self.semantic={}
        self.pole_observations=[]
        self.lock=Lock()
        labels=carla.CityObjectLabel
        self.tags={key:int(getattr(labels,key)) for key in ("Walls","Fences","GuardRail","Poles")}
        self.tag_ids=list(self.tags.values())
        box=ego.bounding_box
        self.meta=dict(tags=self.tags,box_center=[box.location.x,-box.location.y,box.location.z],
                       extent=[box.extent.x,box.extent.y,box.extent.z],
                       box_yaw_deg=-box.rotation.yaw,dt=DT,checkpoint=str(checkpoint),
                       sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        bp=world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
        _,mount,settings=CALIBRATION["lidar"]
        for k,v in settings.items():
            if bp.has_attribute(k):bp.set_attribute(k,v)
        self.sensor=world.spawn_actor(bp,carla.Transform(carla.Location(*mount)),attach_to=ego)
        self.sensor.listen(self.on_semantic)

    def on_semantic(self,measurement):
        values=np.frombuffer(measurement.raw_data,np.float32).reshape(-1,6)
        tags=values.view(np.uint32)[:,5]
        keep=np.isin(tags,self.tag_ids)
        pts=values[keep,:3].copy()
        pts[:,1]*=-1
        pts[:,2]+=CALIBRATION["lidar"][1][2]
        pose=np.asarray(measurement.transform.get_matrix(),dtype=float)
        pose[:3,3]-=pose[:3,:3]@np.asarray(CALIBRATION["lidar"][1])
        near=(tags[keep]==self.tags["Poles"])&(np.abs(pts[:,0])<16)&(np.abs(pts[:,1])<12)
        height=self.meta["box_center"][2]
        half=self.meta["extent"][2]
        near&=(pts[:,2]>=height-half)&(pts[:,2]<=height+half)
        with self.lock:
            self.semantic[measurement.frame]=(pts,tags[keep].astype(np.uint8))
            self.pole_observations.append((measurement.frame,pose,pts[near].copy()))
            self.semantic.pop(measurement.frame-105,None)

    def record(self,row,rgb,lidar,nav,plane,xy,sensor_pose,control_pose,route,scan):
        self.frames.append((dict(row),dict(rgb=rgb.copy(),lidar=lidar.copy(),nav=nav.copy(),
            plane=np.array(plane,np.float32),raw_xy=np.array(xy),sensor_pose=sensor_pose.copy(),
            control_pose=control_pose.copy(),route=route.copy(),scan=scan.copy())))

    def finish(self,hits):
        self.sensor.stop()
        self.sensor.destroy()
        samples=sorted(self.pole_observations,key=lambda item:item[0])
        if samples:
            np.savez_compressed(self.folder/"pole-passages.npz",frames=[s[0] for s in samples],
                poses=[s[1] for s in samples],indices=np.r_[0,np.cumsum([len(s[2]) for s in samples])],
                points=np.concatenate([s[2] for s in samples]),box_center=self.meta["box_center"],
                extent=self.meta["extent"],box_yaw_deg=self.meta["box_yaw_deg"])
        (self.folder/"run-meta.json").write_text(json.dumps(self.meta,indent=2))
        if not hits:return
        crash,actor=min(hits)
        self.meta.update(crash_frame=crash,actor=actor)
        (self.folder/"capture.json").write_text(json.dumps(self.meta,indent=2))
        with (self.folder/"commands.jsonl").open("w") as log:
            for row,data in self.frames:
                if not crash-54<=row["frame"]<=crash:continue
                with self.lock:points=self.semantic.get(row["sensor_frame"])
                if points is None:raise RuntimeError(f"Missing semantic LiDAR for {row['sensor_frame']}")
                np.savez(self.folder/f"frame-{row['frame']}.npz",**data,semantic_xyz=points[0],semantic_tags=points[1])
                log.write(json.dumps(row)+"\n")
