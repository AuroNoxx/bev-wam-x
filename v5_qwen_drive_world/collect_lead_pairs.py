"""Collect matched stopping/departing lead-car branches on unseen CARLA roads."""
import json
import math
import queue
import time
from pathlib import Path

import carla
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/v5-qwen-drive-world/lead-pair-evaluation.json').read_text())
OUT=ROOT/'data/v5-qwen-drive-world/lead-pairs-aligned'


def sites(roadmap,reverse=False):
    accepted=[]
    for wp in sorted(roadmap.generate_waypoints(12),key=lambda p:(p.road_id,p.lane_id,p.s),reverse=reverse):
        if wp.is_junction or abs(wp.transform.rotation.pitch)>4:
            continue
        nxt=wp.next(15)
        if len(nxt)!=1 or nxt[0].is_junction or nxt[0].road_id!=wp.road_id:
            continue
        bend=(wp.transform.rotation.yaw-nxt[0].transform.rotation.yaw+180)%360-180
        if abs(bend)>4 or any(p.road_id==wp.road_id or p.transform.location.distance(wp.transform.location)<60 for p in accepted):
            continue
        accepted.append(wp)
        yield wp,nxt[0]


def speed(actor):
    v=actor.get_velocity()
    return math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)


def drive(actor,target):
    error=target-speed(actor)
    actor.apply_control(carla.VehicleControl(throttle=min(.60,max(0.,.44*error+.12)) if target else 0.,
                                             brake=min(1.,max(0.,-.45*error)) if target else 1.))


def branch(world,points,name,moving,folder,*,ego_cruise=2.10,brake_rate=.23,lead_cruise=1.9,release_tick=31):
    lib=world.get_blueprint_library()
    actors=[]
    camera=None
    events=[]
    try:
        for bp,wp in zip(('vehicle.tesla.model3','vehicle.lincoln.mkz_2020'),points):
            trans=wp.transform
            trans.location.z+=.35
            vehicle_bp=lib.find(bp)
            if vehicle_bp.has_attribute('color'):
                vehicle_bp.set_attribute('color','75,90,110')
            vehicle=world.try_spawn_actor(vehicle_bp,trans)
            if vehicle is None:
                raise ValueError('spawn occupied')
            actors.append(vehicle)
        ego,lead=actors
        cb=lib.find('sensor.camera.rgb')
        for k,v in [('image_size_x',CFG['width']),('image_size_y',CFG['height']),('fov',CFG['fov'])]:
            cb.set_attribute(k,str(v))
        cb.set_attribute('sensor_tick',str(CFG['dt_s']))
        camera=world.spawn_actor(cb,carla.Transform(carla.Location(x=.8,z=1.6)),attach_to=ego)
        col=world.spawn_actor(lib.find('sensor.other.collision'),carla.Transform(),attach_to=ego)
        actors.extend([camera,col])
        col.listen(lambda ev:events.append(ev.frame))
        images=queue.Queue()
        camera.listen(images.put)
        world.tick()
        drive(ego,0);drive(lead,0)
        for _ in range(12):
            world.tick()
            while not images.empty(): images.get()
        poses=[];captured={}
        q=CFG['query_tick']
        wanted={q-15,q-10,q-5,q}
        def receive(im):
            idx=round((im.timestamp-poses[0]['time'])/CFG['dt_s'])
            if idx not in wanted or idx>=len(poses): return
            rec=poses[idx]
            if abs(im.timestamp-rec['time'])>.005 or abs(im.frame-rec['frame'])>1: return
            mount=np.linalg.solve(rec['ego'],im.transform.get_matrix())
            err=float(np.linalg.norm(mount[:3,3]-[.8,0,1.6]))
            rot=float(np.rad2deg(np.arccos(np.clip((np.trace(mount[:3,:3])-1)/2,-1,1))))
            if err<=CFG['max_mount_error_m'] and rot<=CFG['max_mount_rotation_deg']:
                if idx not in captured or err<captured[idx][1]:
                    captured[idx]=(im,err,rot)
        for i in range(q+6):
            time.sleep(CFG['dt_s'])
            drive(ego,max(.4,ego_cruise-brake_rate*max(0,i-36)))
            drive(lead,lead_cruise if moving and i>=release_tick else 0.)
            f=world.tick()
            snap=world.get_snapshot()
            assert f==snap.frame
            es=snap.find(ego.id);ls=snap.find(lead.id)
            poses.append(dict(frame=f,time=snap.timestamp.elapsed_seconds,ego=es.get_transform().get_matrix(),
                              lead=ls.get_transform().get_matrix(),speed=speed(ego)))
            try:
                im=images.get(timeout=.12)
                receive(im)
            except queue.Empty:
                pass
            # Wait for this exposure or the previous one, allowing for rendering latency.
            while not images.empty():
                receive(images.get())
        for _ in range(24):
            if len(captured)==4: break
            world.tick()
            try:
                im=images.get(timeout=.15)
                receive(im)
            except queue.Empty:
                continue
            while not images.empty():
                receive(images.get())
        if len(captured)!=4:
            raise ValueError(f'missing exposure ticks {sorted(wanted-set(captured))}')
        if events: raise ValueError('ego collision')
        folder.mkdir(parents=True,exist_ok=False)
        intr=CFG['width']/2/math.tan(math.radians(CFG['fov']/2))
        observations=[]
        for idx,(im,mount_error,mount_rotation) in sorted(captured.items()):
            rec=poses[idx]
            rgb=np.frombuffer(im.raw_data,np.uint8).reshape(im.height,im.width,4)[:,:,2::-1]
            if rgb.std()<3: raise ValueError('constant RGB')
            file=folder/f'{idx:03}.jpg'
            Image.fromarray(rgb).save(file,quality=95)
            observations.append(dict(path=str(file.relative_to(ROOT)),frame=im.frame,time=im.timestamp,physics_frame=rec['frame'],
                physics_time=rec['time'],mount_error_m=mount_error,mount_rotation_deg=mount_rotation,
                camera=dict(world2cam=im.transform.get_inverse_matrix(),intrinsic=[[intr,0,im.width/2],[0,intr,im.height/2],[0,0,1]],
                            image_size_x=im.width,image_size_y=im.height)))
        cur=poses[q];prev=poses[q-10];T=np.linalg.inv(cur['ego'])
        def xy(rec,key):
            return (T@np.array(rec[key]))[:2,3]
        delta=xy(cur,'lead')-xy(prev,'lead')
        gap=xy(cur,'lead')
        actor=dict(world_from_ego=cur['ego'],observations=observations)
        (folder/'actor.json').write_text(json.dumps(actor))
        (folder/'physics.json').write_text(json.dumps(poses))
        return dict(route=name,branch='depart' if moving else 'hold',target=float(delta[0]),
                    gap=float(gap[0]),lateral=float(gap[1]),speed_kmh=cur['speed']*3.6,
                    ego_approach_m=float(-xy(prev,'ego')[0]),actor=str((folder/'actor.json').relative_to(ROOT)))
    finally:
        for item in reversed(actors):
            if item.type_id.startswith('sensor.'): item.stop()
            item.destroy()
        world.tick()


def main():
    OUT.mkdir(exist_ok=True)
    client=carla.Client('127.0.0.1',2175);client.set_timeout(90)
    pairs=[];rejected=[]
    for town in CFG['maps']:
        saved=[json.loads(p.read_text()) for p in sorted(OUT.glob(f'{town}-*/pair.json'))]
        valid=[r for r in saved if r['valid']]
        if len(valid)==CFG['pairs_per_map']:
            for r in valid: pairs.extend(r['rows'])
            rejected.extend(p.parent.name for p in OUT.glob(f'{town}-*/pair.json') if not json.loads(p.read_text())['valid'])
            continue
        world=client.get_world()
        if world.get_map().name.split('/')[-1]!=town:
            world=client.load_world(town)
        original=world.get_settings();settings=world.get_settings()
        settings.synchronous_mode=True;settings.fixed_delta_seconds=CFG['dt_s']
        world.apply_settings(settings)
        try:
            count=0
            accepted_roads=set()
            accepted_positions=[]
            roadmap=world.get_map()
            candidates=[(wp,ld,rev) for rev in (False,True) for wp,ld in sites(roadmap,rev)]
            for wp,lead,reverse in candidates:
                if wp.road_id in accepted_roads or any(wp.transform.location.distance(loc)<60 for loc in accepted_positions):
                    continue
                name=f'{town}-road{wp.road_id}-lane{wp.lane_id}'+(f'-s{wp.s:.0f}' if reverse else '')
                folder=OUT/name
                meta=folder/'pair.json'
                previous=json.loads(meta.read_text()) if meta.exists() else {}
                retry=previous.get('error','').startswith(('missing exposure','inconsistent within-window')) and len(list(folder.glob('attempt-*')))<4
                if not meta.exists() or retry:
                    # A fresh attempt isolates incomplete sensor output after an interruption.
                    folder.mkdir(exist_ok=True)
                    attempt=folder/f'attempt-{len(list(folder.glob("attempt-*")))+1}'
                    attempt.mkdir()
                    world.set_weather([carla.WeatherParameters.ClearNoon,carla.WeatherParameters.CloudyNoon][count%2])
                    try:
                        rows=[branch(world,(wp,lead),name,mode,attempt/('depart' if mode else 'hold')) for mode in (False,True)]
                        a,b=rows
                        valid=(abs(a['target'])<=CFG['stationary_lead_max_abs_m'] and b['target']>CFG['decision_threshold_m']
                               and all(r['speed_kmh']<=CFG['max_current_ego_kmh'] and r['ego_approach_m']>=CFG['min_ego_approach_last_1s_m']
                                       and 3<r['gap']<35 and abs(r['lateral'])<2 for r in rows)
                               and abs(a['gap']-b['gap'])<=CFG['max_pair_gap_difference_m'])
                        result=dict(valid=valid,rows=rows)
                    except ValueError as e:
                        result=dict(valid=False,error=str(e))
                    meta.write_text(json.dumps(result,indent=2))
                result=json.loads(meta.read_text())
                if result['valid']:
                    pairs.extend(result['rows']);count+=1
                    accepted_roads.add(wp.road_id)
                    accepted_positions.append(wp.transform.location)
                else: rejected.append(name)
                print(town,name,'accepted',result['valid'],'pairs',count,flush=True)
                if count==CFG['pairs_per_map']: break
            if count!=CFG['pairs_per_map']:raise RuntimeError(f'{town} only collected {count} valid pairs')
        finally:
            world.apply_settings(original)
    (OUT/'manifest.json').write_text(json.dumps(dict(config=CFG,windows=pairs,rejected=rejected),indent=2))


if __name__=='__main__':
    main()
