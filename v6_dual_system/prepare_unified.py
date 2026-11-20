"""Prepare the route channel and inspect five geometrically calibrated training samples."""
import io
import json
from pathlib import Path

import cv2
import numpy as np

from .sensors import DATA,write_json
from .unified_data import BASE,ART
from .unified_grid import corrected_camera_grid,ground_plane,corridor,X,Y


def prepare(folder):
    if not (folder/"plane.npy").is_file():
        planes=np.stack([ground_plane(lidar) for lidar in np.load(folder/"lidar.npy",mmap_mode="r")])
        np.save(folder/"plane.pending.npy",planes)
        (folder/"plane.pending.npy").replace(folder/"plane.npy")
    if (folder/"nav.npy").exists():return
    with np.load(folder/"labels.npz") as data:
        raster=np.stack([corridor(points,float(s)) for points,s in zip(data["nav"],data["shift"])])
    pending=folder/"nav.pending.npy"
    np.save(pending,raster)
    pending.replace(folder/"nav.npy")


def preview():
    folder=ART/"alignment"
    folder.mkdir(exist_ok=True,parents=True)
    entries=json.loads((BASE/"train-original.json").read_bytes())
    examples={}
    for name in entries:
        src=BASE/name
        scenario=src.name.split("-")[0]
        if scenario not in ("highway","transition","left","right","braking") or scenario in examples:continue
        meta=json.loads((src/"complete.json").read_bytes())
        with np.load(src/"labels.npz") as lab:
            i=len(lab["v"])//2
            ranges=lab["rgb"][i]
            nav=corridor(lab["nav"][i],float(lab["shift"][i]))/255.
            with (DATA/meta["rgb_source"]).open("rb") as f:
                f.seek(int(ranges[0])); raw=f.read(int(ranges[1]-ranges[0]))
            image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            scan=np.load(src/"lidar.npy",mmap_mode="r")[i]
            grid,mask=corrected_camera_grid(ground_plane(scan))
            u=(grid[...,0]+1)*320-.5
            v=(grid[...,1]+1)*180-.5
            ipm=cv2.remap(image,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            if ipm[~mask].any():raise ValueError("RGB outside footprint")
            lidar=np.load(src/"lidar.npy",mmap_mode="r")[i]/255.
            overlay=ipm.astype(float)*.8
            overlay[:,:,1]+=nav*125
            overlay[:,:,2]+=np.minimum(1.,2.*lidar[0])*95
            overlay[:,:,0]+=np.maximum(lidar[1],lidar[2])*100
            top=np.clip(overlay,0,255).astype(np.uint8)[::-1,::-1].transpose(1,0,2).copy()
            top=cv2.resize(top,(420,420),interpolation=cv2.INTER_NEAREST)
            canvas=np.full((460,1090,3),25,np.uint8)
            canvas[:420,:420]=top;canvas[30:390,435:1075]=image
            cv2.putText(canvas,"Navigation: green   LiDAR: red / blue   IPM: RGB",(10,448),cv2.FONT_HERSHEY_SIMPLEX,.62,(235,235,235),1,cv2.LINE_AA)
            if not cv2.imwrite(str(folder/(scenario+".png")),canvas):raise IOError(scenario)
            examples[scenario]=dict(episode=name,index=i,frame=int(lab["frame"][i]))
    if len(examples)!=5:raise ValueError("Five distinct examples required")
    write_json(folder/"frames.json",examples)


if __name__=="__main__":
    preview()
    for path in sorted(BASE.rglob("complete.json")):prepare(path.parent)
