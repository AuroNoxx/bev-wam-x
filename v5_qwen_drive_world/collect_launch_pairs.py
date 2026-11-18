"""Collect paired six-view, stopped-ego CARLA episodes with a shared following expert."""
import json
import faulthandler
import math
import time
import queue
from pathlib import Path
import carla
import numpy as np
from PIL import Image
from .collect_highway import CAMERAS, box
from .collect_lead_pairs import speed
from .build_real_manifest import ROOT, VIEWS, actor as make_actor
from .real_data_geometry import labels
from .mine_launch_pairs import DATASET

TOWNS = ('Town01','Town04')
N = 8


def follow_lane(vehicle, target, roadmap):
    pose = vehicle.get_transform()
    wp = roadmap.get_waypoint(pose.location)
    nxt = wp.next(6)
    yaw = wp.transform.rotation.yaw
    if nxt:
        loc = nxt[0].transform.location
        yaw = math.degrees(math.atan2(loc.y-pose.location.y,loc.x-pose.location.x))
    steer = float(np.clip(((yaw-pose.rotation.yaw+180)%360-180)/45,-.65,.65))
    error = target-speed(vehicle)
    vehicle.apply_control(carla.VehicleControl(steer=steer,
        throttle=min(.75,max(0.,.48*error+.1)) if target else 0.,
        brake=min(1.,max(0.,-.5*error)) if target else 1.))


def branch(world, roadmap, start, ahead, folder, moving, k, fast=False):
    query = 26 if fast else 20
    folder.mkdir(parents=True,exist_ok=True)
    lib=world.get_blueprint_library()
    actors=[]; sensors=[]; collisions=[]
    queues={}; mounts={}; calibration={}; rows=[]; captured={}
    try:
        for name,wp in [('vehicle.tesla.model3',start),('vehicle.lincoln.mkz_2020',ahead)]:
            pose=wp.transform
            pose.location.z+=.35
            bp=lib.find(name)
            bp.set_attribute('color','75,90,110')
            vehicle=world.try_spawn_actor(bp,pose)
            if not vehicle:raise ValueError('occupied vehicle spawn')
            actors.append(vehicle)
        ego,lead=actors
        for name,(yaw,w,h,fov) in CAMERAS.items():
            bp=lib.find('sensor.camera.rgb')
            for key,v in dict(image_size_x=w,image_size_y=h,fov=fov,sensor_tick=.1).items():bp.set_attribute(key,str(v))
            pose=carla.Transform(carla.Location(x=1.3,z=2),carla.Rotation(yaw=yaw))
            cam=world.spawn_actor(bp,pose,attach_to=ego);sensors.append(cam)
            queues[name]=queue.Queue()
            cam.listen(queues[name].put)
            mounts[name]=np.array(pose.get_matrix())
            f=w/(2*math.tan(math.radians(fov)/2))
            calibration[name]=dict(width=w,height=h,ego_from_sensor=pose.get_matrix(),
                                   intrinsics=[[f,0,w/2],[0,f,h/2],[0,0,1]])
        col=world.spawn_actor(lib.find('sensor.other.collision'),carla.Transform(),attach_to=ego)
        actors.append(col);col.listen(lambda e:collisions.append(e.frame))
        for _ in range(12):
            follow_lane(ego,0,roadmap);follow_lane(lead,0,roadmap);world.tick()
            for q in queues.values():
                while not q.empty():q.get()
        initial=ego.get_location().distance(lead.get_location())
        nav=start.next(30)[0].transform.location
        live=False
        for i in range(query+51):
            gap=ego.get_location().distance(lead.get_location())
            live=live or gap-initial>(10. if fast else 3.3)
            ref=min(8.,max(0.,speed(lead)+.65*(gap-initial))) if live else 0.
            follow_lane(ego,ref,roadmap)
            follow_lane(lead, (5.5+k%4) if moving and i>=(k%5 if fast else 8+k%4) else 0., roadmap)
            world.tick()
            snap=world.get_snapshot()
            eb,lb=box(ego,True),box(lead)
            rows.append(dict(index=i,timestamp=snap.timestamp.elapsed_seconds,speed=eb['speed'],
                             target_speed_kmh=ref*3.6,nav_world=[nav.x,nav.y,nav.z],images={n:'' for n in CAMERAS},
                             bounding_boxes=[eb,lb]))
            if i<=query+6:
                time.sleep(.035)
                for name,q in queues.items():
                    while not q.empty():
                        im=q.get()
                        t=round((im.timestamp-rows[0]['timestamp'])/.1)
                        if t not in ([query-15,query-10,query-5,query] if name=='CAM_FRONT' else [query]) or t>i:continue
                        es=rows[t]['bounding_boxes'][0]
                        rel=np.array(es['world2ego'])@np.array(im.transform.get_matrix())
                        err=float(np.linalg.norm(rel[:3,3]-mounts[name][:3,3]))
                        angle=float(np.degrees(np.arccos(np.clip((np.trace(rel[:3,:3].T@mounts[name][:3,:3])-1)/2,-1,1))))
                        if abs(im.timestamp-rows[t]['timestamp'])>.005 or err>.002 or angle>.1:continue
                        path=folder/name/f'{t:05d}.jpg'
                        path.parent.mkdir(exist_ok=True)
                        rgb=np.frombuffer(im.raw_data,np.uint8).reshape(im.height,im.width,4)[:,:,2::-1]
                        if rgb.std()<3:raise ValueError('constant image')
                        Image.fromarray(rgb).save(path,quality=95)
                        captured[name,t]=dict(path=str(path.relative_to(ROOT)),time=im.timestamp,
                            mount_error_m=err,mount_error_deg=angle,world2cam=im.transform.get_inverse_matrix(),
                            intrinsic=calibration[name]['intrinsics'])
                if i==query+6:
                    for s in sensors:s.stop();s.destroy()
                    sensors.clear()
                    if len(captured)!=9:raise ValueError(f'exposure alignment: {len(captured)}/9')
        if collisions:raise ValueError('collision')
        name=folder.parent.name
        actor=make_actor(rows,query,name,'highway')
        actor['camera_calibration']=calibration
        cameras=[captured['CAM_FRONT',t] for t in (query-10,query-5,query)]
        for name_cam,view in zip(CAMERAS,VIEWS):
            times=[query-15,query-10,query-5,query] if name_cam=='CAM_FRONT' else [query]
            actor['images'][view]=[captured[name_cam,t]['path'] for t in times]
        truth=labels(rows,query,'highway')
        distance=float(np.linalg.norm(truth['trajectory'],axis=1).max())
        transform=np.array(rows[query]['bounding_boxes'][0]['world2ego'])
        past,cur=[np.array(rows[t]['bounding_boxes'][1]['location']) for t in (query-10,query)]
        delta=float(transform[0,:3]@(cur-past))
        if (actor['target_speed_kmh']!=0 or actor['speed_kmh']>3 or
            (moving and (distance<3 or delta<=.3)) or (not moving and (distance>=.5 or abs(delta)>.15))):
            raise ValueError(f'group labels: {distance}, {delta}, {actor["target_speed_kmh"]}')
        (folder/'actor.json').write_text(json.dumps(actor))
        np.savez_compressed(folder/'labels.npz',trajectory=truth['trajectory'],controls=truth['controls'])
        (folder/'physics.json').write_text(json.dumps([dict(time=r['timestamp'],ego=r['bounding_boxes'][0]['world_from_ego'],
                 lead=r['bounding_boxes'][1]['world_from_ego'],expert_speed_kmh=r['target_speed_kmh']) for r in rows]))
        return dict(route=name,branch='launch' if moving else 'hold',index=query,source='carla',lead_dx=delta,
                    distance=distance,actor_path=str((folder/'actor.json').relative_to(ROOT)),
                    label_path=str((folder/'labels.npz').relative_to(ROOT)),paths=[c['path'] for c in cameras],
                    cameras=cameras,ego=actor['world_from_ego'],exposures=[dict(time=c['time'],mount_error_m=c['mount_error_m'],
                    mount_error_deg=c['mount_error_deg']) for c in captured.values()])
    finally:
        for a in list(reversed(sensors))+list(reversed(actors)):
            if a.type_id.startswith('sensor.'):a.stop()
            a.destroy()
        world.tick()


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--fast',action='store_true')
    fast=parser.parse_args().fast
    root=DATASET/('carla-fast' if fast else 'carla')
    client=carla.Client('127.0.0.1',2175);client.set_timeout(40)
    result=[]
    for town in (('Town05',) if fast else TOWNS):
        world=client.get_world()
        if world.get_map().name.split('/')[-1]!=town:world=client.load_world(town)
        original=world.get_settings(); settings=world.get_settings()
        settings.synchronous_mode=True;settings.fixed_delta_seconds=.1;world.apply_settings(settings)
        try:
            roadmap=world.get_map()
            saved=[json.loads(path.read_text()) for path in sorted(root.glob(f'{town}-*/pair.json'))]
            result.extend(r for pair in saved for r in pair)
            accepted={int(pair[0]['route'].split('-r')[1].split('-l')[0]) for pair in saved}
            protected={int(r['route'].split('-road')[1].split('-lane')[0])
                       for r in json.loads((DATASET.parent/'lead-pairs-aligned/manifest.json').read_text())['windows']
                       if r['route'].startswith(town+'-')} if fast else set()
            candidates=sorted(roadmap.generate_waypoints(24),key=lambda w:(w.road_id,w.s,w.lane_id))
            for wp in candidates:
                k=len(accepted)
                if k==N:break
                if wp.road_id in accepted or wp.road_id in protected or wp.is_junction or abs(wp.transform.rotation.pitch)>4:continue
                ahead=wp.next([8,10,12,14][k%4])
                if len(ahead)!=1 or ahead[0].road_id!=wp.road_id or ahead[0].is_junction:continue
                tail=wp.next(42)
                if len(tail)!=1 or tail[0].is_junction or tail[0].road_id!=wp.road_id:continue
                if abs((wp.transform.rotation.yaw-tail[0].transform.rotation.yaw+180)%360-180)>6:continue
                folder=root/f'{town}-r{wp.road_id}-l{wp.lane_id}-s{wp.s:.0f}'
                folder.mkdir(parents=True,exist_ok=True)
                path=folder/'pair.json'
                if not path.exists():
                    world.set_weather([carla.WeatherParameters.ClearNoon,carla.WeatherParameters.CloudyNoon][k%2])
                    try:
                        pair=[branch(world,roadmap,wp,ahead[0],folder/b,b=='launch',k,fast) for b in ('hold','launch')]
                    except ValueError as e:
                        print(folder.name,str(e),flush=True)
                        continue
                    path.write_text(json.dumps(pair,indent=2))
                result.extend(json.loads(path.read_text()));accepted.add(wp.road_id)
                print(town,len(accepted),'/',N,flush=True)
            print(town,'accepted',len(accepted),flush=True)
        finally:world.apply_settings(original)
    (DATASET/('carla-fast.json' if fast else 'carla.json')).write_text(json.dumps(result,indent=2))
    if not result:raise ValueError('No usable pairs')


if __name__=='__main__':
    faulthandler.dump_traceback_later(60,repeat=True)
    main()
