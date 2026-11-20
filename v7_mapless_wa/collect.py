"""Bounded UE5 pilot acquisition. Run as python -m v7_mapless_wa.collect."""
import argparse,json,math,queue,time
from pathlib import Path
import carla,cv2,numpy as np
from .geometry import CAMS,NAMES,projection_tables,raster
from .scenarios import routes

DT=.05
N=25
WEATHER=('ClearNoon','ClearSunset','CloudyNoon','WetCloudyNoon')
SEM=np.dtype([('xyz','<f4',(3,)),('cos','<f4'),('obj','<u4'),('tag','<u4')])

def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')

def matrix(t):
    return np.asarray(t.get_matrix(),dtype=np.float64)

def read(q,frame):
    while True:
        item=q.get(timeout=12)
        if item.frame==frame:return item
        if item.frame>frame:raise RuntimeError(f'sensor skipped {frame}: {item.frame}')

def box(actor,pose):
    pts=actor.bounding_box.get_world_vertices(pose)
    a=np.array([[p.x,p.y,p.z] for p in pts])
    return a[[0,4,6,2]] if np.ptp(a[[0,4,6,2],2])<.5 else a[np.argsort(a[:,2])[:4]]

def episode(world,tm,road_map,selection,kind,weather,seed,out):
    rng=np.random.default_rng(seed)
    i,wp,route,kappa,pole,fork,branch=selection
    lib=world.get_blueprint_library(); actors=[];sensors=[];qs=[];trace=[]; sem_points=[]
    folder=out/kind;folder.mkdir()
    frames=[]; times=[]; arc=0.; prev=None;first=None
    world.set_weather(getattr(carla.WeatherParameters,weather))
    try:
        tf=wp.transform;tf.location.z+=.38
        perturb=.42 if kind=='straight' else 0.
        right=tf.get_right_vector();tf.location.x+=right.x*perturb;tf.location.y+=right.y*perturb
        ego=world.spawn_actor(lib.find('vehicle.lincoln.mkz'),tf); actors.append(ego)
        candidates=road_map.get_spawn_points();rng.shuffle(candidates)
        candidates=sorted(candidates,key=lambda s:not 24<s.location.distance(tf.location)<110)
        for other in candidates:
            if len(actors)>=10:break
            if other.location.distance(tf.location)<23:continue
            a=world.try_spawn_actor(lib.find(rng.choice(['vehicle.mini.cooper','vehicle.nissan.patrol','vehicle.dodge.charger'])),other)
            if a:
                actors.append(a);a.set_autopilot(True,tm.get_port())
        ego.set_autopilot(True,tm.get_port());tm.auto_lane_change(ego,False)
        tm.set_desired_speed(ego,32);tm.set_path(ego,[w.transform.location for w in route[2::2]])
        cam_tf=[]
        def sensor(bp,t):
            a=world.spawn_actor(bp,t,attach_to=ego);sensors.append(a)
            q=queue.Queue();qs.append(q);a.listen(q.put)
        for x,y,z,yaw,pitch,fov in CAMS:
            bp=lib.find('sensor.camera.rgb')
            for k,v in dict(image_size_x=384,image_size_y=224,fov=fov,sensor_tick=0).items():bp.set_attribute(k,str(v))
            t=carla.Transform(carla.Location(x,y,z),carla.Rotation(pitch=pitch,yaw=yaw))
            cam_tf.append(matrix(t));sensor(bp,t)
        tables,valid=projection_tables(cam_tf)
        if not (out/'calibration.npz').exists():np.savez_compressed(out/'calibration.npz',uv=tables,visible=valid,extrinsics=cam_tf)
        for model in ('sensor.lidar.ray_cast','sensor.lidar.ray_cast_semantic'):
            bp=lib.find(model)
            for k,v in dict(range=88,channels=64,rotation_frequency=20,points_per_second=640000,upper_fov=12,lower_fov=-30,sensor_tick=0).items():bp.set_attribute(k,str(v))
            if bp.has_attribute('dropoff_general_rate'):bp.set_attribute('dropoff_general_rate','0')
            sensor(bp,carla.Transform(carla.Location(z=2.2)))
        np.savez_compressed(folder/'survey.npz',route=np.array([[w.road_id,w.lane_id,w.s] for w in route]))
        begin=time.perf_counter()
        for tick in range(2200):
            t0=time.perf_counter();fid=world.tick(); snap=world.get_snapshot()
            bundle=[read(q,fid) for q in qs]; t=ego.get_transform(); pose=matrix(t)
            if snap.frame!=fid:raise RuntimeError('snapshot mismatch')
            if prev is not None:arc+=np.linalg.norm(pose[:2,3]-prev)
            prev=pose[:2,3].copy()
            vel=ego.get_velocity();v=math.hypot(vel.x,vel.y)
            trace.append((fid,pose,[box(a,a.get_transform()) for a in actors[1:]],arc))
            yaw_delta=abs((t.rotation.yaw-wp.transform.rotation.yaw+180)%360-180)
            at_scene=(kind not in ('curve','fork') or
                      (kind=='curve' and yaw_delta>=12.) or
                      (kind=='fork' and road_map.get_waypoint(t.location).is_junction))
            if first is None and tick>=6 and v>1.6 and at_scene:first=tick
            keep=first is not None and first<=tick<first+N
            if keep or (first is not None and tick%4==0):
                sem=np.frombuffer(bundle[4].raw_data,SEM)
                tags=[int(getattr(carla.CityObjectLabel,k)) for k in ('Poles','Walls','Fences','GuardRail','TrafficSigns','RoadLines','Sidewalks')]
                use=np.isin(sem['tag'],tags)
                loc=sem['xyz'][use].astype(np.float64);loc[:,2]+=2.2
                xyz_world=loc@pose[:3,:3].T+pose[:3,3]
                sem_points.append((fid,xyz_world.astype(np.float32),sem['tag'][use].copy()))
            if keep:
                frame_dir=folder/f'{fid:07d}'; frame_dir.mkdir()
                for name,img in zip(NAMES,bundle[:3]):
                    bgr=np.frombuffer(img.raw_data,np.uint8).reshape(224,384,4)[:,:,:3]
                    if not cv2.imwrite(str(frame_dir/(name+'.png')),bgr):raise IOError(name)
                raw=np.frombuffer(bundle[3].raw_data,np.float32).reshape(-1,4).copy()
                raw[:,1]*=-1;raw[:,2]+=2.2
                np.savez_compressed(frame_dir/'lidar.npz',bev=raster(raw))
                np.savez_compressed(frame_dir/'audit-scan.npz',returns=raw)
                frames.append(fid);times.append(time.perf_counter()-t0)
            if len(frames)==N and fid-trace[first+N-1][0]>=80 and arc-trace[first+N-1][3]>=61:
                break
        if len(frames)!=N or arc-trace[first+N-1][3]<60:raise RuntimeError(f'{kind}: insufficient driven future {arc:.1f} m, frames={len(frames)}')
        np.savez_compressed(folder/'teacher.npz',fid=[a[0] for a in trace],pose=[a[1] for a in trace],
                            boxes=np.asarray([a[2] for a in trace],np.float32),actor_ids=[a.id for a in actors[1:]],
                            sem_fid=[a[0] for a in sem_points],sem_count=[len(a[1]) for a in sem_points],
                            sem_xyz=np.concatenate([a[1] for a in sem_points]),sem_tag=np.concatenate([a[2] for a in sem_points]))
        meta=dict(scenario=kind,map=road_map.name,weather=weather,seed=seed,branch=branch,route_start=i,
                  planned_peak_kappa_first_46m=kappa,nearest_pole_xy_m=pole,first_choices_m=fork,
                  npc_count=len(actors)-1,perturb_m=perturb,frames=frames,dt=DT,elapsed=time.perf_counter()-begin,
                  retained_tick_s=times,driven_m=arc,sim_ticks=len(trace),episode_group=f'{road_map.name}/{wp.road_id}/{wp.lane_id}/{i}',
                  drift_phase=float(rng.uniform(-np.pi,np.pi)))
        write(folder/'episode.json',meta);print(kind,'OK',len(frames),round(arc),len(trace),flush=True)
    finally:
        for a in sensors:
            if a.is_listening:a.stop()
        for a in reversed(sensors):a.destroy()
        for a in reversed(actors):a.destroy()

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=2460);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--weather-seed',type=int,default=1)
    p.add_argument('--only',nargs='+',choices=('straight','curve','fork','pole'));args=p.parse_args()
    out=args.out;out.mkdir(exist_ok=True,parents=True)
    c=carla.Client('127.0.0.1',args.port);c.set_timeout(45)
    if c.get_server_version()!='0.10.0':raise RuntimeError(c.get_server_version())
    world=c.get_world();road_map=world.get_map(); orig=world.get_settings();tm=c.get_trafficmanager(args.port+5)
    if not road_map.name.endswith('Town10HD_Opt'):raise ValueError('This pilot uses Town10HD_Opt survey coordinates')
    old_weather=world.get_weather()
    try:
        cfg=world.get_settings();cfg.synchronous_mode=True;cfg.fixed_delta_seconds=DT;cfg.no_rendering_mode=False;world.apply_settings(cfg)
        tm.set_synchronous_mode(True);tm.set_random_device_seed(714)
        poles=[b.location for b in world.get_level_bbs(carla.CityObjectLabel.Poles) if min(b.extent.x,b.extent.y)<.8]
        selected=routes(road_map,np.array([[p.x,p.y,p.z] for p in poles]))
        (out/'map.xodr').write_text(road_map.to_opendrive(),encoding='utf8')
        write(out/'sim.json',dict(server=c.get_server_version(),client=c.get_client_version(),maps=c.get_available_maps(),quality='Epic',
                                  tags={k:int(getattr(carla.CityObjectLabel,k)) for k in ('Poles','Walls','Fences','GuardRail','TrafficSigns','RoadLines','Sidewalks')}))
        weather_order=np.random.default_rng(args.weather_seed).permutation(WEATHER)
        for idx,(kind,s) in enumerate(selected.items()):
            if args.only and kind not in args.only:continue
            if (out/kind/'episode.json').exists():continue
            print('starting',kind,s[0],s[3:6],flush=True)
            episode(world,tm,road_map,s,kind,str(weather_order[idx]),715+idx,out)
    finally:
        tm.set_synchronous_mode(False);world.apply_settings(orig);world.set_weather(old_weather)

if __name__=='__main__':main()
