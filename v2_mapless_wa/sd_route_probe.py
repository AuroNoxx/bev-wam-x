"""Read-only corpus direction audit and a fixed-pose Town04 navigation comparison."""
import argparse
import json
from pathlib import Path
import queue

import carla
import cv2
import numpy as np

from .collect import DT, matrix, read
from .geometry import local, world_at
from .navigation import DirectedRoute, gps_drift, waypoint_forward
from .phase2_live import RoadSurvey, reference, server
from .resolution_probe import camera_image, project


def classify(road, points, directions):
    states, dot = [], []
    for point,axis in zip(points,directions):
        w = road.get_waypoint(carla.Location(*point.tolist()),project_to_road=False,lane_type=carla.LaneType.Driving)
        if w is None:
            states.append('outside_driving'); dot.append(None)
        else:
            similarity=float(waypoint_forward(w)@axis)
            states.append('opposite' if similarity<-.5 else 'same' if similarity>.5 else 'other_direction')
            dot.append(similarity)
    return states,dot


def heading(points):
    delta = np.gradient(points,axis=0)
    return delta/np.linalg.norm(delta,axis=1,keepdims=True).clip(1e-8)


def pick_pose(road,old,directed,rows):
    coords=np.array([r['ego_xyz'] for r in rows],float)
    directions=heading(coords)
    candidates=[]
    for i in range(33,min(len(rows)-4,590),3):
        r=rows[i]
        if r['speed_kmh']<12 or r['lateral_error_m']>.85 or r.get('command',0):
            continue
        x,y,z=directions[i]
        pose=carla.Transform(carla.Location(*coords[i].tolist()),carla.Rotation(
            yaw=float(np.degrees(np.arctan2(y,x))),pitch=float(np.degrees(np.arctan2(z,np.hypot(x,y))))))
        t=matrix(pose)
        p,_=directed.match(t)
        prior,_=old.match(t[:2,3],r['nav_station_m'])
        nav=old.anchors(prior,t,i*DT)
        xyz=world_at(np.c_[np.vstack(([0.,0.],nav)),np.zeros(4)],t)
        s=np.arange(0.,61.,2.)
        prior_line=np.column_stack([np.interp(s,[0,20,40,60],column) for column in xyz.T])
        updated=directed.positions(p,s,gps_drift(.72,i*DT))
        axis=heading(updated)
        prior_line[:,2] = updated[:,2]
        a,angle_a=classify(road,prior_line,axis)
        b,angle_b=classify(road,updated,axis)
        if 'same' not in b or b.count('same')<len(b)-1:
            continue
        key=(a.count('opposite'),a.count('outside_driving'),-abs(i-100))
        candidates.append((key,i,pose,prior_line,updated,a,b,angle_a,angle_b))
    if not candidates:
        raise RuntimeError('No comparable fixed-pose snapshot from the Town04 trace')
    return max(candidates,key=lambda x:x[0])


def town04(world, args):
    road=world.get_map()
    spec=json.loads(args.plan.read_text())[0]
    old=RoadSurvey(spec['xy']); directed=DirectedRoute(reference(road,old,spec))
    rows=[json.loads(line) for line in args.trace.read_text().splitlines()]
    key,i,pose,before,after,prior,now,prior_dot,new_dot=pick_pose(road,old,directed,rows)
    settings=world.get_settings(); settings.synchronous_mode=True; settings.fixed_delta_seconds=DT
    world.apply_settings(settings); world.set_weather(carla.WeatherParameters.ClearNoon)
    lib=world.get_blueprint_library(); vehicle=lib.find('vehicle.lincoln.mkz')
    start=carla.Transform(pose.location+carla.Location(z=.35),pose.rotation)
    ego=world.spawn_actor(vehicle,start); ego.set_simulate_physics(False); ego.set_transform(pose)
    eq=queue.Queue(); camera=None
    mount=carla.Transform(carla.Location(x=-8.2,z=4.1),carla.Rotation(pitch=-11.5))
    try:
        bp=lib.find('sensor.camera.rgb')
        for name,value in dict(image_size_x=1280,image_size_y=720,fov=92,enable_postprocess_effects='True').items():
            bp.set_attribute(name,str(value))
        camera=world.spawn_actor(bp,mount,attach_to=ego); camera.listen(eq.put)
        for _ in range(38):
            tick=world.tick(45); observation=read(eq,tick)
        native=camera_image(observation,(1280,720))
        panes=[]
        for title,points,verdict in [('BEFORE',before,prior),('AFTER',after,now)]:
            img=native.copy()
            uv,_=project(points,matrix(observation.transform),92,(1280,720))
            uv=np.rint(uv).astype(np.int32)
            cv2.polylines(img,[uv[2:]],False,(24,24,24),9,cv2.LINE_AA)
            cv2.polylines(img,[uv[2:]],False,(255,219,76),5,cv2.LINE_AA)
            for p,typ in zip(uv[[10,20,30]],np.array(verdict)[[10,20,30]]):
                cv2.circle(img,tuple(p),8,(57,78,255) if typ!='same' else (255,219,76),-1,cv2.LINE_AA)
            audit=f'{title}  |  same {verdict.count("same")}/31   opposite {verdict.count("opposite")}   off-road {verdict.count("outside_driving")}'
            cv2.rectangle(img,(16,12),(860,61),(25,24,22),-1)
            cv2.putText(img,audit,(28,46),cv2.FONT_HERSHEY_SIMPLEX,.82,(247,249,253),2,cv2.LINE_AA)
            panes.append(cv2.resize(img,(880,495),interpolation=cv2.INTER_AREA))
        if not cv2.imwrite(str(args.out/'town04-sd-before-after.png'),np.hstack(panes)):
            raise IOError(args.out)
    finally:
        if camera is not None:
            camera.stop(); camera.destroy()
        ego.destroy()
    result=dict(archived_frame=rows[i]['frame'],trace_index=i,pose=matrix(pose).tolist(),
                counts_before={s:prior.count(s) for s in sorted(set(prior))},
                counts_after={s:now.count(s) for s in sorted(set(now))},
                before_anchor_direction=[prior[k] for k in (10,20,30)],
                after_anchor_direction=[now[k] for k in (10,20,30)],
                before_dot=prior_dot,after_dot=new_dot)
    (args.out/'town04-sd-proof.json').write_text(json.dumps(result,indent=2))
    print('Town04',json.dumps(result),flush=True)


def audit_map(road, town, args, rows, lateral, expert, nav):
    result=[]
    episodes=sorted({r['episode'] for r in rows if r['map']==town})
    for ep in episodes:
        ids=np.array([i for i,r in enumerate(rows) if r['episode']==ep])
        high=(np.abs(lateral[ids])>=8).any(1)
        for group, subset in [('tail',ids[high]),('control',ids[~high])]:
            count=min(len(subset),36 if group=='tail' else 12)
            if count==0:
                continue
            for i in subset[np.linspace(0,len(subset)-1,count).astype(int)]:
                row=rows[i]
                folder=args.dataset/'episodes'/ep/f'{row["frame"]:07d}'
                with np.load(folder/'index.npz') as idx:
                    pose=idx['world_from_ego']
                with np.load(folder/'labels.npz') as lab:
                    path=lab['path_gt']
                arc=np.arange(1,41)*1.5
                near=np.array([20.,40.,60.])
                off=lambda qs:np.column_stack([np.interp(qs,arc,col) for col in path.T])
                tangent=off(np.minimum(near+3,60))-off(near-3)
                axis=np.c_[tangent,np.zeros(3)]*[1,-1,1]@pose[:3,:3].T
                axis/=np.linalg.norm(axis,axis=1,keepdims=True)
                world_nav=world_at(np.c_[nav[i],np.zeros(3)],pose)
                world_truth=world_at(np.c_[expert[i],np.zeros(3)],pose)
                status,cosines=classify(road,world_nav,axis)
                true_status,_=classify(road,world_truth,axis)
                result.append(dict(map=town,episode=ep,frame=row['frame'],group=group,
                                   lateral=lateral[i].tolist(),status=status,cosine=cosines,expert_status=true_status))
    return result


def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    with np.load(args.offsets) as audit:
        lateral,expert=audit['lateral'],audit['expert']
    rows=[json.loads(line) for line in (args.dataset/'manifest.jsonl').read_text().splitlines()]
    nav=np.load(args.nav,mmap_mode='r')
    collected=[]
    with server(args.carla_root.resolve(),args.out,args.port) as (client,_):
        town04(client.load_world('Town04_Opt'),args)
        for town in ('Town12','Town10HD_Opt'):
            road=client.load_world(town).get_map()
            collected.extend(audit_map(road,town,args,rows,lateral,expert,nav))
            (args.out/'sd-direction-sample.json').write_text(json.dumps(collected,indent=2))
            print(town,'direction audit',len([r for r in collected if r['map']==town]),'frames',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('out','carla_root','plan','trace','dataset','offsets','nav'):
        parser.add_argument('--'+field.replace('_','-'),type=Path,required=True)
    parser.add_argument('--port',type=int,default=2460)
    run(parser.parse_args())
