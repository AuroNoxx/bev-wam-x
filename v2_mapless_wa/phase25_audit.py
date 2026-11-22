"""Compare frozen Town12 val boundary labels against live semantic LiDAR at the same poses."""
import argparse
import json
from pathlib import Path
import queue

import carla
import numpy as np

from .collect import DT, SEM, read
from .geometry import local, pixels, world_at
from .labels import Surfaces, map_vectors
from .phase2_data import manifest
from .phase2_live import server
from .render_probe import lidar_blueprint


HARD_TAGS = ('Poles','Walls','Fences','GuardRail','TrafficSigns')
SAMPLES = (25, 180, 340, 500, 650, 735, 860, 985, 1120, 1370, 1630, 1890)


def transform(pose, elevation):
    r=pose[:3,:3]; t=pose[:3,3]+r[:,2]*elevation
    yaw=float(np.degrees(np.arctan2(r[1,0],r[0,0])))
    pitch=float(np.degrees(np.arcsin(np.clip(r[2,0],-1,1))))
    roll=float(np.degrees(np.arctan2(-r[2,1],r[2,2])))
    return carla.Transform(carla.Location(*t.tolist()),carla.Rotation(pitch=pitch,yaw=yaw,roll=roll))


def boxes(world):
    centers=[]
    for kind in HARD_TAGS:
        for item in world.get_level_bbs(getattr(carla.CityObjectLabel,kind)):
            centers.append((item.location.x,item.location.y,item.location.z))
    return np.asarray(centers,dtype=float).reshape(-1,3)


def targets(road, pose, driven, points, base_mask):
    uv=np.rint(pixels(points[(points[:,2]>.15)&(points[:,2]<3),:2])).astype(int)
    near=uv[((uv>=0)&(uv<128)).all(1)]
    mask=base_mask.copy()
    mask[2,near[:,1],near[:,0]]=1
    r,c=np.nonzero(mask[2])
    static=np.column_stack((80-(r+.5)*.75,19.2-(c+.5)*.30))
    _,valid=map_vectors(road,pose,driven,static,mask)
    return valid[[3,4]].sum(1).tolist(),mask[2].astype(bool)


def read_example(dataset,row):
    folder=dataset/'episodes'/row['episode']/f'{row["frame"]:07d}'
    with np.load(folder/'index.npz') as index, np.load(folder/'labels.npz') as labels:
        pose=index['world_from_ego']
        gt=labels['path_gt']
        driven=np.vstack((pose[:3,3],world_at(np.column_stack((gt,np.zeros(len(gt)))),pose)))
        return pose,driven,labels['point_valid'][[3,4]].sum(1).tolist(),labels['masks'][2]


def run(args):
    all_rows,_=manifest(args.dataset)
    rows=[r for r in all_rows if r['map']=='Town12' and r['split']=='val']
    args.out.mkdir(parents=True,exist_ok=False)
    data=[]
    with server(args.carla_root.resolve(),args.out,args.port) as (client,_):
        world=client.load_world('Town12');road=world.get_map();spectator=world.get_spectator()
        surfaces=Surfaces(road)
        cfg=world.get_settings();cfg.synchronous_mode=True;cfg.fixed_delta_seconds=DT;cfg.spectator_as_ego=True
        world.apply_settings(cfg)
        tags={int(getattr(carla.CityObjectLabel,k)):k for k in HARD_TAGS}
        tail,driven,*_=read_example(args.dataset,rows[-1]); tail=tail.copy();tail[:3,3]=driven[-1]
        spectator.set_transform(transform(tail,8))
        for _ in range(50):world.tick(45)
        terminal=boxes(world)
        blueprint=lidar_blueprint(world.get_blueprint_library(),True)
        for k in SAMPLES:
            row=rows[k];pose,driven,stored,old_mask=read_example(args.dataset,row)
            spectator.set_transform(transform(pose,8))
            for _ in range(35):world.tick(45)
            sensor=world.spawn_actor(blueprint,transform(pose,2.2))
            q=queue.Queue();sensor.listen(q.put)
            try:
                for _ in range(6):
                    frame=world.tick(45); measurement=read(q,frame)
                sem=np.frombuffer(measurement.raw_data,SEM)
                points=sem['xyz'].copy()
                points[:,1]*=-1;points[:,2]+=2.2
                region=(points[:,0]>=0)&(points[:,0]<=60)&(abs(points[:,1])<=19.2)&(points[:,2]>.15)&(points[:,2]<3)
                near=region&np.isin(sem['tag'],list(tags))
                tag_ids,tag_counts=np.unique(sem['tag'][region&(points[:,1]>1.8)],return_counts=True)
                base=surfaces.draw(pose)
                curb_only=targets(road,pose,driven,np.empty((0,3)),base)[0]
                live, mask=targets(road,pose,driven,points[near],base)
                box_now=targets(road,pose,driven,local(boxes(world),pose),base)[0]
                box_terminal=targets(road,pose,driven,local(terminal,pose),base)[0]
                overlap=int((mask & old_mask.astype(bool)).sum())
                w=road.get_waypoint(carla.Location(*pose[:3,3].tolist()),lane_type=carla.LaneType.Driving)
                record=dict(index=k,frame=row['frame'],road=w.road_id,lane=w.lane_id,stored_points=stored,
                            curb_only_points=curb_only,semantic_lidar_points=live,near_hard_returns=int(near.sum()),
                            lidar_mask_overlap_cells=overlap,lidar_hard_cells=int(mask.sum()),
                            current_box_center_points=box_now,terminal_box_center_points=box_terminal,
                            types={tag:int(((sem['tag']==v)&near).sum()) for v,tag in tags.items()},
                            left_roi_all_semantic_tags={int(tag):int(n) for tag,n in zip(tag_ids,tag_counts)})
                data.append(record);print(json.dumps(record),flush=True)
                (args.out/'boundary-audit.json').write_text(json.dumps(data,indent=2))
            finally:
                sensor.stop();sensor.destroy()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--carla-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--port',type=int,default=2460)
    run(p.parse_args())
