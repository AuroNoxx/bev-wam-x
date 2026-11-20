"""Check planar camera IPM against physical road-marking geometry at the five preview poses."""
import json
import carla
import numpy as np
from .sensors import ROOT,DATA,write_json
from .unified_grid import ground_plane
from .unified_data import BASE,ART


def main():
    records=json.loads((ART/"alignment/frames.json").read_bytes())
    maps={}
    result={}
    for name,rec in records.items():
        folder=BASE/rec["episode"]
        meta=json.loads((folder/"complete.json").read_bytes())
        old=(DATA/meta["rgb_source"]).parent
        with np.load(old/"stream.npz") as src:
            idx=np.flatnonzero(src["frame"]==rec["frame"]).item()
            pose=src["poses"][idx]
        town=meta["town"]
        if town not in maps:
            raw=ROOT/".cache/v6-dual-system/carla-client/0.9.15/Carla/Maps/OpenDrive"/(town+".xodr")
            maps[town]=carla.Map(town,raw.read_text(encoding="utf8"))
        wp=maps[town].get_waypoint(carla.Location(*pose[:3,3]))
        plane=ground_plane(np.load(folder/"lidar.npy",mmap_mode="r")[rec["index"]])
        marks=[]
        for _ in range(48):
            for sign in (-1,1):
                tr=wp.transform
                pt=tr.location+tr.get_right_vector()*(sign*wp.lane_width/2)
                local=(np.array([pt.x,pt.y,pt.z])-pose[:3,3])@pose[:3,:3]
                x,y,z=local
                if not 5<x<45:continue
                focal=320./np.tan(np.deg2rad(40.))
                u=320+focal*y/(x-1.3)
                v=180+focal*(2-z)/(x-1.3)
                if not (0<=u<640 and 0<=v<360):continue
                # A real marking pixel comes from z. The fixed inverse assumes z=0.
                # Intersect the camera ray with the independent LiDAR plane.
                ray=np.array([x-1.3,-y,z-2.])
                scale=(plane[0]*1.3+plane[2]-2.)/(ray[2]-plane[0]*ray[0]-plane[1]*ray[1])
                planar_x=1.3+scale*ray[0]
                planar_y=-scale*ray[1]
                marks.append([x,y,z,abs(planar_x-x),abs(planar_y-y)/.4])
            choices=wp.next(1.)
            if not choices:break
            wp=min(choices,key=lambda w:abs(w.transform.rotation.yaw-wp.transform.rotation.yaw))
        a=np.asarray(marks)
        if not len(a):raise ValueError(name)
        err=a[:,3:].max(1)
        result[name]=dict(ground_plane=plane.tolist(),samples=len(a),max_error_cells=float(err.max()),p95_error_cells=float(np.percentile(err,95)),
                          within_one_cell_fraction=float((err<=1).mean()),worst_sample_xy_z_dxcell_dycell=a[err.argmax()].tolist())
    out=dict(passed=all(r["max_error_cells"]<=1. for r in result.values()),metric="max per-axis BEV cell error, known road-boundary ground points projected through current-LiDAR ground-plane IPM",examples=result)
    write_json(ART/"alignment/geometry-check.json",out)
    print(json.dumps(out))

if __name__=="__main__":main()
