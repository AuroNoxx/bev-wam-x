"""Collect missing native-grid LiDAR and matched five-time actor boxes on registered routes."""
import argparse
import json
import random
from collections import deque
import carla
import numpy as np

from .collect_fast_data import Expert,Road,neighbor,path_from,spawn
from .collect_spatial import forks
from .diagnose_gt_controller import Reference
from .sensors import DATA,DT,Sensors,MissingFrame,local_client,encode_rgb,write_json
from .unified_data import BASE,route_polyline
from .unified_grid import HORIZONS,lidar_raster,occupancy


def record(world,path,alternative,meta,split,name,quota,speed,case,branch_arc=0.):
    out=BASE/"supplement"/name
    out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").is_file():return str(out.relative_to(BASE))
    road=Road(path)
    driver=Expert(road)
    ref=Reference(path)
    alt=Reference(alternative) if alternative else None
    actors=[]
    rig=sensor=None
    hits=[]
    physics=[]
    pending=deque()
    observations={}
    try:
        ego=spawn(world,path[0],"vehicle.tesla.model3")
        actors.append(ego)
        lead=spawn(world,path[24],"vehicle.audi.tt")
        actors.append(lead)
        follower=Expert(Road(path[24:]))
        for a in actors:a.apply_control(carla.VehicleControl(brake=1))
        for _ in range(18):world.tick()
        rig=Sensors(world,ego)
        sensor=world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),carla.Transform(),attach_to=ego)
        sensor.listen(lambda e:hits.append(e.frame))
        for _ in range(12):world.tick()
        for a in actors:
            fwd=a.get_transform().get_forward_vector()
            a.apply_control(carla.VehicleControl())
            a.set_target_velocity(fwd*(speed/3.6))
        fork=case in ("fork-left","fork-right")
        for tick in range(560 if fork else 405):
            lead_speed=50 if case=="congestion" else speed
            lead.apply_control(follower.control(lead,lead_speed))
            tr=ego.get_transform()
            loc=lead.get_location()
            ahead,right,_,_ = np.asarray(tr.get_inverse_matrix())@[loc.x,loc.y,loc.z,1.]
            v=ego.get_velocity().length()
            target=speed
            if ahead>0 and abs(right)<2.2:
                target=min(target,max(0.,lead.get_velocity().length()*3.6+1.3*(ahead-9.-1.4*v)))
            u=np.clip((tick-35)*DT/3.,0.,1.)
            shift=0.
            if case in ("left","right"):
                shift=(1 if case=="left" else -1)*path[road.index].lane_width*u*u*(3-2*u)
            ego.apply_control(driver.control(ego,target,shift))
            f=world.tick()
            pose=np.asarray(ego.get_transform().get_matrix())
            arc,lat,k=ref.project(pose)
            nav=route_polyline(ref,pose,arc)
            other=route_polyline(alt,pose,alt.project(pose)[0]) if alt else nav
            vertices=np.array([[p.x,p.y,p.z] for p in lead.bounding_box.get_world_vertices(lead.get_transform())])
            cmd=(1 if case=="left" else 2) if case in ("left","right") and 35<=tick<95 else 0
            locked=(1 if case=="left" else -1)*path[road.index].lane_width if case in ("left","right") and tick>=35 else 0.
            physics.append((f,pose,vertices,ego.get_velocity().length(),nav,other,k,cmd,locked,arc-branch_arc))
            pending.append((tick,f))
            if len(pending)>3:
                i,idx=pending.popleft()
                try:rgb,scan=rig.read(idx)
                except MissingFrame:continue
                observations[i]=(encode_rgb(rgb),lidar_raster(scan))
            if hits:break
            if fork and len(physics)>85 and physics[-82][-1]>40:break
        stop=min(hits,default=float("inf"))
        eligible=[i for i in observations if i+80<len(physics) and physics[i+80][0]<stop and
                  physics[i+80][0]-physics[i][0]==80 and (not fork or -80<=physics[i][-1]<=40)]
        if len(eligible)<min(15,quota):raise ValueError(f"{name} only {len(eligible)} valid frames")
        selected=np.asarray(eligible)[np.linspace(0,len(eligible)-1,min(quota,len(eligible)),dtype=int)]
        result={k:[] for k in ("rgb","nav","alt","v","xy","curve","cmd","shift","occ","station","frame")}
        grids=[]
        with (out/"rgb.bin").open("wb") as images:
            for i in selected:
                f,pose,_,v,nav,other,k,cmd,locked,station=physics[i]
                jpg,grid=observations[i]
                begin=images.tell()
                images.write(jpg)
                grids.append(grid)
                positions=np.stack([physics[i+2*t][1][:3,3] for t in range(1,41)])
                traj=((positions-pose[:3,3])@pose[:3,:3])[:,:2]*[1.,-1.]
                occ=np.stack([occupancy([physics[i+round(t/DT)][2]],pose) for t in HORIZONS])
                for key,value in zip(result,((begin,images.tell()),nav,other,v,traj,k,cmd,locked,np.packbits(occ,axis=-1),station,f)):
                    result[key].append(value)
        np.save(out/"lidar.npy",grids)
        np.savez_compressed(out/"labels.npz",**{k:np.asarray(v) for k,v in result.items()})
        write_json(out/"complete.json",dict(rgb_source=str((out/"rgb.bin").relative_to(DATA)),n=len(selected),
                   occupancy_valid=True,group=meta["group"],split=split,fork=fork,case=case,town=meta["town"],
                   route_start=path[0].id,source="CARLA 0.9.15 synchronous same-frame boxes",station_m=branch_arc))
        print(split,name,len(selected),flush=True)
        return str(out.relative_to(BASE))
    finally:
        if sensor:sensor.stop();sensor.destroy()
        if rig:rig.close()
        for actor in reversed(actors):actor.destroy()


def main(port,town_only=None):
    client=local_client(port)
    if not client.get_server_version().startswith("0.9.15"):raise ValueError("CARLA 0.9.15 required")
    original=json.loads((DATA/"manifest-corrective.json").read_bytes())
    spatial=DATA/"spatial-navigation"
    desc=json.loads((spatial/"manifest.json").read_bytes())
    output={"train":[],"validation":[]}
    for town in ((town_only,) if town_only else ("Town04","Town06")):
        world=client.get_world()
        if not world.get_map().name.endswith(town):world=client.load_world(town)
        mp=world.get_map()
        before=world.get_settings(); cfg=world.get_settings()
        cfg.synchronous_mode=True;cfg.fixed_delta_seconds=DT;cfg.no_rendering_mode=False
        world.apply_settings(cfg)
        try:
            for (road,lane),(approach,branches) in forks(mp):
                group=f"{town}-fork-{road}-{lane}"
                name=group+"-left"
                if name not in desc["episodes"]+desc["holdout"]:continue
                split="train" if name in desc["episodes"] else "validation"
                paths=[approach+path_from(b,360) for b in branches]
                station=Reference(approach).arc[-1]
                for i,side in enumerate(("left","right")):
                    result=record(world,paths[i],paths[1-i],dict(group=group,town=town),split,group+"-"+side,80,48.,"fork-"+side,station)
                    output[split].append(result)
            # Follow the original primary split for additional traffic supervision.
            for split in output:
                seen=set()
                for ep in sorted({e for e,_ in original[split]}):
                    meta=json.loads((DATA/ep/"complete.json").read_bytes())
                    if meta["town"]!=town or meta["scene"] not in ("left","right","highway","braking","transition"):continue
                    case={"braking":"congestion","transition":"curve"}.get(meta["scene"],meta["scene"])
                    if case in seen:continue
                    seen.add(case)
                    path=path_from(mp.get_waypoint_xodr(*meta["route"][0]))
                    name=f"{town}-{case}-{meta['seed']}"
                    result=record(world,path,None,dict(group=ep,town=town),split,name,200,
                                  56. if case=="congestion" else 90. if case=="highway" else 58.,case)
                    output[split].append(result)
        finally:world.apply_settings(before)
    for split in output:
        episodes=[]
        for file in sorted((BASE/"supplement").glob("*/complete.json")):
            if json.loads(file.read_bytes())["split"]==split:episodes.append(str(file.parent.relative_to(BASE)))
        write_json(BASE/(split+"-supplement.json"),episodes)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--port",type=int,default=2240)
    p.add_argument("--town",choices=["Town04","Town06"])
    args=p.parse_args()
    main(args.port,args.town)
