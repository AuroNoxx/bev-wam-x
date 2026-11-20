"""Targeted CARLA recordings of alternative highway branches and rightmost-lane recovery."""
import argparse
import json
import math
from collections import deque

import carla
import numpy as np

from .collect_fast_data import Expert, Road, neighbor, path_from, spawn, wrap, label_episode
from .derive_lane_supervision import lane_target
from .diagnose_gt_controller import Reference
from .navigation import anchors
from .sensors import DATA, CAL_ID, DT, MissingFrame, Sensors, bev, encode_rgb, local_client, state, write_json

BASE=DATA/"spatial-navigation"


def forks(map_):
    candidates={}
    for wp in map_.generate_waypoints(4):
        if wp.is_junction or not (neighbor(wp,True) or neighbor(wp,False)):continue
        choices=wp.next(4.)
        if len(choices)<2:continue
        branches=[]
        for branch in choices:
            route=path_from(branch,320)
            if len(route)<120:continue
            tf=wp.transform
            end=route[20].transform
            angle=abs(wrap(end.rotation.yaw-tf.rotation.yaw))
            if angle>70:continue
            dx=end.location.x-tf.location.x
            dy=end.location.y-tf.location.y
            right=tf.get_right_vector()
            branches.append((dx*right.x+dy*right.y,branch))
        branches.sort(key=lambda pair:pair[0])
        if len(branches)<2 or branches[-1][0]-branches[0][0]<1.:continue
        before=[wp]
        for _ in range(50):
            prev=before[-1].previous(2.)
            if not prev:break
            before.append(min(prev,key=lambda w:abs(wrap(w.transform.rotation.yaw-before[-1].transform.rotation.yaw))))
        if len(before)<51:continue
        approach=list(reversed(before))
        candidates[(wp.road_id,wp.lane_id)]=(approach,[branches[0][1],branches[-1][1]])
    return sorted(candidates.items())


def right_routes(map_,blocked):
    result=[]
    for wp in sorted(map_.generate_waypoints(28),key=lambda w:w.id):
        if wp.is_junction or wp.road_id in blocked or wp.road_id in [p[0].road_id for p in result]:continue
        right=neighbor(wp,False)
        if not right or neighbor(right,False):continue
        path=path_from(wp,720)
        if len(path)<180:continue
        segment=path[:108:3]
        if any(w.road_id in blocked or not neighbor(w,False) or neighbor(neighbor(w,False),False)
               or abs(wrap(w.transform.rotation.yaw-wp.transform.rotation.yaw))>24 for w in segment):continue
        result.append(path)
        if len(result)==3:break
    return result


def record(world,map_,folder,path,case,branch_station,quota,town,group,split,cruise):
    if (folder/"complete.json").exists():return
    folder.mkdir(parents=True,exist_ok=True)
    road=Road(path)
    expert=Expert(road)
    ref=Reference(path)
    ego=rig=collision=None
    hits=[]; rows=[]; pictures=[]; grids=[]; history=deque(maxlen=7); pending=deque()
    is_right=case=="right-recovery"
    try:
        ego=spawn(world,path[0],"vehicle.tesla.model3")
        ego.apply_control(carla.VehicleControl(brake=1))
        for _ in range(18):world.tick()
        rig=Sensors(world,ego)
        collision=world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),carla.Transform(),attach_to=ego)
        collision.listen(lambda e:hits.append(e.frame))
        for _ in range(12):world.tick()
        ego.apply_control(carla.VehicleControl())
        fwd=ego.get_transform().get_forward_vector()
        ego.set_target_velocity(carla.Vector3D(fwd.x*cruise/3.6,fwd.y*cruise/3.6,fwd.z*cruise/3.6))
        for tick in range(450):
            u=min(1.,max(0.,(tick-34)*DT/2.8)) if is_right else 0.
            shift=-path[road.index].lane_width*u*u*(3.-2.*u)
            ego.apply_control(expert.control(ego,cruise,shift))
            frame=world.tick()
            tr=ego.get_transform()
            pose=np.array(tr.get_matrix())
            lateral=road.locate(tr)
            lane,wp,_=lane_target(map_,pose)
            nav=anchors(ref,pose)
            arc=ref.project(pose)[0]
            row=dict(frame=frame,pose=pose,state=state(ego),target=cruise,boxes=[],offset=lateral,
                     disturb=False,recovery=False,curvature=road.curvature(),lane=lane,nav=nav,arc=arc,
                     road=wp.road_id,lane_id=wp.lane_id)
            pending.append(row)
            if len(pending)<4:continue
            row=pending.popleft()
            try:rgb,scan=rig.read(row["frame"])
            except MissingFrame:
                history.clear()
                continue
            history.append((scan,row["pose"]))
            pictures.append(encode_rgb(rgb))
            grids.append(bev(scan,history[0][0],row["pose"],history[0][1]))
            rows.append(row)
            if hits:break
            if not is_right and row["arc"]>=branch_station+45 and len(rows)>82 and rows[-82]["arc"]>=branch_station+42:break
            if is_right and len(rows)>=280:break
        if hits:raise ValueError(f"Expert collision at {folder.name}: {hits}")
        labels=label_episode(rows,"right" if is_right else "highway",len(rows))
        frames=labels["frame"]
        if is_right:
            eligible=np.flatnonzero((labels["state"][:,0]*3.6>=100)&(labels["state"][:,0]*3.6<=120))
            if not np.any(labels["intent"][eligible]==2) or not any(rows[frames[i]]["offset"]<-2.8 for i in eligible):
                raise ValueError("Rightmost recovery not observed")
        else:
            station=np.array([rows[f]["arc"]-branch_station for f in frames])
            eligible=np.flatnonzero((station>=-80)&(station<=40))
            if len(eligible)<2 or station[eligible[0]]>-76 or station[eligible[-1]]<36:
                raise ValueError(f"Incomplete fork coverage {folder}: {station[eligible][:1]}..{station[eligible][-1:]}, end={rows[-1]['arc']}, gaps={sum(np.diff([r['frame'] for r in rows])!=1)}")
            expected=path[51].road_id
            if not any(rows[frames[i]]["road"]==expected for i in eligible):raise ValueError("Expert missed assigned branch")
        if len(eligible)<quota:raise ValueError(f"Only {len(eligible)} eligible rows for {quota} at {folder}")
        keep=eligible[np.linspace(0,len(eligible)-1,quota,dtype=int)]
        labels={k:v[keep] for k,v in labels.items()}
        if not is_right:labels["intent"][:]=0
        for name in ("lane","nav","arc","road","lane_id"):
            labels[name]=np.array([rows[f][name] for f in labels["frame"]])
        np.savez_compressed(folder/"labels.npz",**labels)
        offsets=[0]
        with (folder/"rgb.bin").open("wb") as images,(folder/"bev.bin").open("wb") as bevfile:
            for jpg,grid in zip(pictures,grids):
                images.write(jpg);offsets.append(images.tell());bevfile.write(grid.tobytes())
        np.savez(folder/"stream.npz",rgb=offsets,poses=[r["pose"] for r in rows],frame=[r["frame"] for r in rows])
        write_json(folder/"complete.json",dict(town=town,group=group,case=case,count=quota,split=split,
                   calibration=CAL_ID,route_start=path[0].id,branch_station=branch_station,
                   speed_range_kmh=(labels["state"][:,0].min()*3.6,labels["state"][:,0].max()*3.6),
                   roads=sorted({r["road"] for r in rows}),branch_road=path[51].road_id if not is_right else None))
        print(folder.name,case,quota,split,flush=True)
    finally:
        if rig:rig.close()
        if collision:collision.stop();collision.destroy()
        if ego:ego.destroy()


def collect(port,selected_town=None):
    BASE.mkdir(exist_ok=True)
    client=local_client(port)
    if not client.get_server_version().startswith("0.9.15"):raise ValueError("CARLA version mismatch")
    manifest=json.loads((DATA/"manifest-corrective.json").read_bytes())
    val=[json.loads((DATA/e/"complete.json").read_bytes()) for e in {e for e,i in manifest["validation"]}]
    for town in ([selected_town] if selected_town else ("Town04","Town06")):
        world=client.get_world()
        if not world.get_map().name.endswith(town):world=client.load_world(town)
        map_=world.get_map()
        blocked={w[0] for m in val if m["town"]==town for w in m["route"]}
        original=world.get_settings();cfg=world.get_settings()
        cfg.synchronous_mode=True;cfg.fixed_delta_seconds=DT;world.apply_settings(cfg)
        try:
            forks_here=forks(map_)
            if not forks_here:raise ValueError("No highway fork found")
            # A group stays together even when a repeated pass travels farther than the first.
            def partition():
                episodes=[(p.parent,json.loads(p.read_bytes())) for p in BASE.glob(town+"-*/complete.json")]
                reserved={m["group"] for _,m in episodes if m["split"]=="holdout" or set(m["roads"])&blocked}
                return episodes,reserved

            available=[]
            for (road,lane),(approach,branches) in forks_here:
                group=f"{town}-fork-{road}-{lane}"
                paths=[approach+path_from(b,360) for b in branches]
                protected=bool({w.road_id for path in paths for w in path[:130]} & blocked) or (town=="Town04" and road==50)
                station=float(Reference(approach).arc[-1])
                for case,path in zip(("left","right"),paths):
                    record(world,map_,BASE/(group+"-"+case),path,case,station,30,town,group,
                           "holdout" if protected else "train",48.)
                if not protected:available.append((group,paths,station))
            episodes,reserved=partition()
            available=[item for item in available if item[0] not in reserved]
            for case_id,case in enumerate(("left","right")):
                count=sum(m["count"] for folder,m in episodes if m["case"]==case and not folder.name.endswith("-54") and m["group"] not in reserved)
                need=300-count
                for j,(group,paths,station) in enumerate(available):
                    quota=need//len(available)+int(j<need%len(available))
                    if quota<=0:continue
                    record(world,map_,BASE/(group+"-"+case+"-54"),paths[case_id],case,station,
                           quota,town,group,"train",54.)
            routes=right_routes(map_,blocked)
            if not routes:raise ValueError(f"No rightmost highway stretch in {town}")
            for j in range(5):
                path=routes[j%len(routes)]
                group=f"{town}-edge-{path[0].id}"
                record(world,map_,BASE/(group+f"-{j}"),path,"right-recovery",0,80,town,group,"train",108.+2*j)
            episodes,reserved=partition()
            eligible=[path for path in routes if f"{town}-edge-{path[0].id}" not in reserved]
            for j in range(12):
                count=sum(m["count"] for _,m in episodes if m["case"]=="right-recovery" and m["group"] not in reserved)
                if count>=400:break
                path=eligible[j%len(eligible)]
                group=f"{town}-edge-{path[0].id}"
                record(world,map_,BASE/(group+f"-extra-{j}"),path,"right-recovery",0,min(80,400-count),town,group,"train",109.+2*(j%5))
                episodes,reserved=partition()
            selection=[path.name for path,m in episodes if m["group"] not in reserved]
            write_json(BASE/(town+"-train.json"),selection)
        finally:world.apply_settings(original)
    if not all((BASE/(t+"-train.json")).exists() for t in ("Town04","Town06")):return
    names=sum([json.loads((BASE/(t+"-train.json")).read_bytes()) for t in ("Town04","Town06")],[])
    meta=[json.loads((BASE/n/"complete.json").read_bytes()) for n in names]
    counts={k:sum(m["count"] for m in meta if m["case"]==k) for k in ("left","right","right-recovery")}
    if list(counts.values())!=[600,600,800]:raise ValueError(counts)
    remaining=sorted(p.parent.name for p in BASE.glob("*/complete.json") if p.parent.name not in names)
    write_json(BASE/"manifest.json",dict(counts=counts,episodes=sorted(names),holdout=remaining))
    print(counts,flush=True)
