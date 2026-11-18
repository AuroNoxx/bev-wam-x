"""Isolated CARLA 0.9.15 closed-loop diagnostic for the preserved v2 best.

Uses official Bench2Drive XML route geometry, not its ScenarioRunner scoring.
OCC is explicitly privileged simulator truth. --occ-source upstream uses the
stock online generator; sparse-proxy retains the earlier diagnostic adapter.
"""
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import carla
import cv2
from native_v2_carla_contract import (native_points_grid,history_payload,state20,
                                     trajectory_to_world,TrajectoryController,carla_grid_centers,vehicle_geometry,merge_semantic_scans)

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/CARLA-0.9.15/WindowsNoEditor/PythonAPI/carla'))
from agents.navigation.global_route_planner import GlobalRoutePlanner
LIDAR_DTYPE=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('cos','<f4'),('object','<u4'),('tag','<u4')])


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def vec(v):return [v.x,v.y,v.z]


def packet(q,frame):
    while True:
        p=q.get(timeout=15)
        if p.frame==frame:return p
        if p.frame>frame:raise RuntimeError('Sensor frame is in the future')


def route_geometry(world,route_id):
    path=ROOT/'runtime/Bench2Drive-0.0.4/leaderboard/data/bench2drive220.xml'
    row=next(r for r in ET.parse(path).getroot().findall('route') if r.get('id')==route_id)
    if row.get('town') not in world.get_map().name:raise ValueError('Wrong map for route')
    anchors=[carla.Location(**{k:float(v) for k,v in p.attrib.items()}) for p in row.find('waypoints')]
    planner=GlobalRoutePlanner(world.get_map(),1.)
    route=planner.trace_route(anchors[0],anchors[-1])
    xyz=np.array([vec(wp.transform.location) for wp,cmd in route])
    anchor_error=float(max(min(np.linalg.norm(xyz-np.array(vec(p)),axis=1)) for p in anchors))
    if anchor_error>4.:raise RuntimeError('Planner shortcuts XML route: '+str(anchor_error))
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
    return route,xyz,arc,{'xml':str(path.relative_to(ROOT)),'xml_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'route_id':route_id,'town':row.get('town'),'source_scenario_names':[s.get('type') for s in row.findall('.//scenario')],
                         'max_xml_anchor_to_dense_route_m':anchor_error,'route_xyz':xyz.tolist(),
                         'route_commands':[int(cmd.value) for wp,cmd in route],
                         'route_length_m':float(arc[-1]),'official_scenario_runner':False}


def oracle_grid(world,ego,scan,static_scans):
    raw=np.frombuffer(scan.raw_data,LIDAR_DTYPE)
    points=np.column_stack([raw['x'],raw['y'],raw['z'],np.ones(len(raw))])
    world_points=(points@np.asarray(scan.transform.get_matrix()).T)[:,:3]
    dynamic=np.isin(raw['tag'],[12,13,14,15,16,17,18,19,21])
    static_scans.append((world_points[~dynamic],raw['tag'][~dynamic].copy()))
    points=np.concatenate([p for p,t in static_scans]+[world_points[dynamic]])
    tags=np.concatenate([t for p,t in static_scans]+[raw['tag'][dynamic]])
    pose=np.asarray(ego.get_transform().get_matrix())
    local=(np.column_stack([points,np.ones(len(points))])@np.linalg.inv(pose).T)[:,:3]
    grid=native_points_grid(local,tags)
    # Explicit privileged current actor boxes, including occluded actors. No future.
    centers=carla_grid_centers();world_centers=centers@pose.T
    filled=[]
    for actor in list(world.get_actors().filter('vehicle.*'))+list(world.get_actors().filter('walker.pedestrian.*')):
        if actor.id==ego.id or actor.get_location().distance(ego.get_location())>40:continue
        bb=actor.bounding_box
        box2world=np.asarray(actor.get_transform().get_matrix())@np.asarray(carla.Transform(bb.location,bb.rotation).get_matrix())
        q=world_centers@np.linalg.inv(box2world).T
        inside=(np.abs(q[:,:3])<=np.array(vec(bb.extent))+1e-5).all(1)
        label=5 if actor.type_id.startswith('walker.') else 4
        grid.reshape(-1)[inside]=label
        filled.append({'actor_id':actor.id,'type':actor.type_id,'voxels':int(inside.sum())})
    return grid,pose,{'scan_frame':scan.frame,'scan_timestamp':scan.timestamp,'points':len(raw),
                      'static_scan_count':len(static_scans),'oracle_filled_actors':filled,
                      'voxel_counts':{str(k):int(v) for k,v in zip(*np.unique(grid,return_counts=True))}}


def upstream_grid(world,ego,scan,worker,responses,folder,reset):
    raw=np.frombuffer(scan.raw_data,dtype='<f4').reshape(-1,6).copy()
    rawpath=folder/f'raw-{scan.frame}.npz';np.savez_compressed(rawpath,raw=raw)
    boxes=[]
    for actor in list(world.get_actors().filter('vehicle.*'))+list(world.get_actors().filter('walker.pedestrian.*')):
        if actor.get_location().distance(ego.get_location())>85:continue
        tr=actor.get_transform();center=tr.transform(actor.bounding_box.location)
        box={'id':actor.id,'class':'ego_vehicle' if actor.id==ego.id else 'vehicle' if actor.type_id.startswith('vehicle.') else 'walker',
             'location':vec(tr.location),'center':vec(center),'extent':vec(actor.bounding_box.extent),
             'rotation':[tr.rotation.pitch,tr.rotation.roll,tr.rotation.yaw],'semantic_tags':list(actor.semantic_tags),
             'world2ego' if actor.id==ego.id else 'world2vehicle':tr.get_inverse_matrix()}
        boxes.append(box)
    gridfile=folder/f'grid-{scan.frame}.npy'
    req={'operation':'upstream_occ','raw_file':str(rawpath.resolve()),'grid_file':str(gridfile.resolve()),
         'reset':reset,'boxes':boxes,'world2lidar':scan.transform.get_inverse_matrix()}
    (folder/f'raw-{scan.frame}.json').write_text(json.dumps({'frame':scan.frame,'timestamp':scan.timestamp,
        'source_scan_frames':scan.source_scan_frames,'source_scan_timestamps':scan.source_scan_timestamps,**req}))
    worker.stdin.write(json.dumps(req)+'\n');worker.stdin.flush();result=json.loads(responses.get(timeout=10))
    if 'error' in result:raise RuntimeError(result['error'])
    grid=np.load(gridfile,allow_pickle=False)
    return grid,np.asarray(ego.get_transform().get_matrix()),{'scan_frame':scan.frame,'scan_timestamp':scan.timestamp,
        'points':len(raw),'generation_seconds':result['generation_seconds'],'generator':result['generator'],
        'voxel_counts':result['native_counts'],'source':'stock causal B2D occ_from_carla_tick; simulator semantic/instance/pose/bbox truth; no extra full-box filling',
        'source_scan_frames':scan.source_scan_frames,'source_scan_timestamps':scan.source_scan_timestamps,
        'azimuth_occupied_span_degrees':scan.azimuth_span}


def scene_run(client,worker,responses,args,name,route_id,mode):
    out=Path(args.output)/name;out.mkdir(exist_ok=False)
    world=client.get_world();old=world.get_settings();weather=world.get_weather();actors=[];sensors=[];video=None
    report={'scenario':name,'mode':mode,'status':'preparing','simulator':client.get_server_version(),
            'map':world.get_map().name,'max_simulated_seconds':args.seconds,'takeovers':[],
            'input_source':'CARLA semantic LiDAR with simulator truth labels + privileged actor OBB fill + up to 1s past static fusion',
            'not_learned_perception':True,'upstream_occ_exact_reproduction':False,
            'quality_vetoes':[],'basic_protections':['finite/shape validation','10s inference wall timeout','legal CARLA control bounds'],
            'initial_speed_override':False,'initial_motion':'rest, brake held during real observation warm-up only'}
    report['occ_source_mode']=args.occ_source
    if args.occ_source=='upstream':
        report['input_source']='stock Bench2Drive causal OccOnlineCARLA: semantic LiDAR simulator truth, instance IDs, ego/object poses and boxes; no learned perception'
        report['upstream_occ_exact_reproduction']='generator body exact; released offline dataset generation causality/version not certified'
    steps=[];plans=[];hits=[];started=time.perf_counter();last_plan=None;active=None;plan_id=0;route_index=0
    try:
        if old.synchronous_mode:raise RuntimeError('Another synchronous simulator owner is active')
        if world.get_actors().filter('vehicle.*') or world.get_actors().filter('sensor.*'):
            raise RuntimeError('Foreign actors present; refusing to disturb world')
        settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05
        settings.no_rendering_mode=args.no_camera
        settings.substepping=True;settings.max_substep_delta_time=.01;settings.max_substeps=5
        world.apply_settings(settings);world.set_weather(carla.WeatherParameters.ClearNoon)
        route,xyz,arc,route_info=route_geometry(world,route_id);report['route']=route_info
        bp=world.get_blueprint_library();tr=route[0][0].transform;tr.location.z+=.4
        ego=world.spawn_actor(bp.find('vehicle.lincoln.mkz_2020'),tr);actors.append(ego)
        ego.apply_control(carla.VehicleControl(brake=1.))
        physical=ego.get_physics_control()
        # In synchronous mode actor snapshot transform is identity before its first
        # world tick. Use the acknowledged spawn transform, not that stale cache.
        wheelbase,geometry=vehicle_geometry([vec(w.position) for w in physical.wheels],
                                            [w.max_steer_angle for w in physical.wheels],np.asarray(tr.get_matrix()))
        wheel_limit=math.radians(max(w.max_steer_angle for w in physical.wheels))
        report['vehicle_geometry']=geometry
        report['ego_bounding_box']={'location':vec(ego.bounding_box.location),'extent':vec(ego.bounding_box.extent)}
        curve=[(p.x,p.y) for p in physical.steering_curve]
        controller=TrajectoryController(wheelbase,wheel_limit,curve)
        report['controller']={'name':'pure pursuit + signed-forward-velocity PI','wheelbase_m':wheelbase,
                              'wheel_limit_rad':wheel_limit,'steering_curve_kmh':curve,'kp':.2,'ki':.1,
                              'dt_s':.05,'lookahead':'max(2m,0.7*speed_mps)','target_speed':'raw signed model vx at current trajectory time; no minimum, upper cap or trajectory quality rejection'}
        obstacle=None;walker=None
        if name.startswith('obstacle'):
            j=int(np.searchsorted(arc,22.));tr2=route[j][0].transform;tr2.location.z+=.4
            obstacle=world.spawn_actor(bp.find('vehicle.audi.a2'),tr2);actors.append(obstacle)
            obstacle.apply_control(carla.VehicleControl(brake=1.,hand_brake=True))
            report['traffic']={'stationary_obstacle_id':obstacle.id,'route_s_m':float(arc[j])}
        if name.startswith('dynamic'):
            # Fixed-time independent pedestrian crosses at route s=15m from right to left.
            j=int(np.searchsorted(arc,15.));wtr=route[j][0].transform;right=wtr.get_right_vector()
            spawn=carla.Transform(wtr.location+carla.Location(x=right.x*5,y=right.y*5,z=1.))
            walker=world.spawn_actor(bp.filter('walker.pedestrian.*')[0],spawn);actors.append(walker)
            report['traffic']={'pedestrian_id':walker.id,'route_s_m':float(arc[j]),'schedule':'walk right->left at1.3m/s from simulation t=12s for8s; independent of ego; timing fixed after initial development run',
                               'direction_world':[-right.x,-right.y,0.]}
        qs={}
        specs={'lidar':('sensor.lidar.ray_cast_semantic',{'range':'85','channels':'64','points_per_second':'600000','rotation_frequency':'10','upper_fov':'10','lower_fov':'-30','sensor_tick':'0.0'},carla.Transform(carla.Location(x=-.39,z=1.84))),
               'imu':('sensor.other.imu',{'sensor_tick':'0.0','noise_gyro_stddev_x':'0','noise_gyro_stddev_y':'0','noise_gyro_stddev_z':'0'},carla.Transform(carla.Location(x=-1.4))),
               'camera':('sensor.camera.rgb',{'image_size_x':'640','image_size_y':'360','fov':'90','sensor_tick':'0.0'},carla.Transform(carla.Location(x=-7,z=4),carla.Rotation(pitch=-18)))}
        if args.no_camera:del specs['camera']
        for key,(kind,attrs,transform) in specs.items():
            sensorbp=bp.find(kind)
            for k,v in attrs.items():sensorbp.set_attribute(k,v)
            sensor=world.spawn_actor(sensorbp,transform,attach_to=ego);actors.append(sensor);sensors.append(sensor)
            qs[key]=queue.Queue();sensor.listen(qs[key].put)
        report['sensors']={k:{'type':v[0],'attributes':v[1],'mount':str(v[2])} for k,v in specs.items()}
        col=world.spawn_actor(bp.find('sensor.other.collision'),carla.Transform(),attach_to=ego);actors.append(col);sensors.append(col)
        def collision(ev):
            other=ev.other_actor;loc=ego.get_location()
            hits.append({'frame':ev.frame,'timestamp':ev.timestamp,'ego_world_location':vec(loc),
                         'other_id':other.id,'other_type':other.type_id,'other_world_location':vec(other.get_location()),
                         'normal_impulse':vec(ev.normal_impulse),'location_note':'actor locations at callback, CARLA CollisionEvent has no contact point'})
        col.listen(collision)
        if not args.no_camera:video=cv2.VideoWriter(str(out/'drive.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10.,(640,360))
        report['video_opened']=bool(video and video.isOpened());report['no_rendering_mode']=args.no_camera
        history=deque(maxlen=7);static_scans=deque(maxlen=10)
        warmup_ticks=40;total_ticks=warmup_ticks+round(args.seconds/.05)
        sim_start=None;log= (out/'steps.jsonl').open('w',encoding='utf-8')
        planlog=(out/'plans.jsonl').open('w',encoding='utf-8');inputdir=out/'inputs';inputdir.mkdir()
        rawdir=out/'raw-occ';rawdir.mkdir()
        last_control={'throttle':0.,'brake':1.,'steer':0.};last_detail={'warmup':True};origin=None;lastloc=None;distance=0.;previous_scan=None
        for tick in range(total_ticks):
            if time.perf_counter()-started>args.wall_timeout:raise TimeoutError('Run wall timeout')
            sim_t=max(0,(tick-warmup_ticks)*.05)
            if walker:
                direction=report['traffic']['direction_world']
                walker.apply_control(carla.WalkerControl(direction=carla.Vector3D(*direction),speed=1.3 if 12<=sim_t<20 else 0.))
            frame=world.tick(20);snap=world.get_snapshot();timestamp=snap.timestamp.elapsed_seconds
            tr=ego.get_transform();pose=np.asarray(tr.get_matrix())
            speed=float(np.dot(vec(ego.get_velocity()),vec(tr.get_forward_vector())))
            loc=np.asarray(vec(tr.location));imu=packet(qs['imu'],frame)
            current_scan=packet(qs['lidar'],frame)
            if tick==warmup_ticks:sim_start=timestamp;origin=loc.copy();lastloc=loc.copy()
            active_phase=tick>=warmup_ticks
            if active_phase:distance+=float(np.linalg.norm(loc-lastloc));lastloc=loc.copy()
            end=min(len(xyz),route_index+50)
            route_index=route_index+int(np.argmin(np.linalg.norm(xyz[route_index:end]-loc,axis=1)))
            route_error=float(np.linalg.norm(xyz[route_index,:2]-loc[:2]))
            if tick%2==1:
                if previous_scan is None:raise RuntimeError('Missing previous half scan')
                if abs(current_scan.timestamp-previous_scan.timestamp-.05)>.002:raise RuntimeError('LiDAR scan cadence mismatch')
                raw=merge_semantic_scans(np.frombuffer(previous_scan.raw_data,'<f4').reshape(-1,6),
                    np.frombuffer(current_scan.raw_data,'<f4').reshape(-1,6),
                    np.asarray(previous_scan.transform.get_matrix()),np.asarray(current_scan.transform.get_matrix()))
                angles=np.sort(np.mod(np.arctan2(raw[:,1],raw[:,0]),2*np.pi))
                azimuth_span=float(360-np.degrees(np.max(np.diff(np.r_[angles,angles[0]+2*np.pi]))))
                scan=SimpleNamespace(raw_data=raw.tobytes(),frame=current_scan.frame,timestamp=current_scan.timestamp,
                    transform=current_scan.transform,source_scan_frames=[previous_scan.frame,current_scan.frame],
                    source_scan_timestamps=[previous_scan.timestamp,current_scan.timestamp],azimuth_span=azimuth_span)
                if video:
                    cam=packet(qs['camera'],frame)
                    image=np.frombuffer(cam.raw_data,np.uint8).reshape(cam.height,cam.width,4)[:,:,:3].copy()
                    cv2.putText(image,f'{name} {mode} t={sim_t:.1f}s v={speed:.2f}m/s', (10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
                    if video.isOpened():video.write(image)
                if args.occ_source=='upstream':
                    grid,pose,occ_meta=upstream_grid(world,ego,scan,worker,responses,rawdir,not history)
                else:grid,pose,occ_meta=oracle_grid(world,ego,scan,static_scans)
                history.append({'grid':grid,'pose':pose,'timestamp':scan.timestamp,'frame':frame})
                if active_phase and (tick-warmup_ticks)%4==1:
                    history_array,history_meta=history_payload(list(history))
                    ni=int(np.searchsorted(arc,arc[route_index]+5.));fi=int(np.searchsorted(arc,arc[route_index]+30.))
                    ni=min(ni,len(route)-1);fi=min(fi,len(route)-1)
                    commands=[int(route[ni][1].value),int(route[fi][1].value)]
                    state=state20(speed,vec(imu.accelerometer),imu.gyroscope.z,pose,xyz[ni],xyz[fi],commands)
                    file=inputdir/f'{plan_id:05d}.npz';np.savez_compressed(file,history=history_array,state=state)
                    begin=time.perf_counter()
                    if mode=='model':
                        worker.stdin.write(json.dumps({'input':str(file.resolve()),'seed':1907+plan_id})+'\n');worker.stdin.flush()
                        prediction=json.loads(responses.get(timeout=10))
                        if 'error' in prediction:raise RuntimeError(prediction['error'])
                    else:
                        # Controller calibration ONLY: privileged route at fixed4m/s. Never mixed into model runs.
                        ss=np.minimum(arc[route_index]+np.arange(1,26)*.2*4,arc[-1])
                        wp=np.stack([np.interp(ss,arc,xyz[:,k]) for k in range(3)],1)
                        local=np.column_stack([wp,np.ones(25)])@np.linalg.inv(pose).T
                        prediction={'xy_flu_m':np.stack([local[:,0],-local[:,1]],1).tolist(),
                                    'velocity_flu_mps':np.tile([4.,0.],(25,1)).tolist(),'inference_seconds':0.,'controller_calibration_only':True}
                    prediction['ipc_wall_seconds']=time.perf_counter()-begin
                    path=trajectory_to_world(prediction['xy_flu_m'],pose)
                    velocity=np.asarray(prediction['velocity_flu_mps'])
                    if velocity.shape!=(25,2) or not np.isfinite(velocity).all():raise ValueError('Invalid model velocity')
                    active={'path':path,'velocity':velocity,'timestamp':timestamp,'id':plan_id}
                    row={'id':plan_id,'frame':frame,'timestamp':timestamp,'sim_t':timestamp-sim_start,'ego_pose':pose.tolist(),
                         'state20':state.tolist(),'history':history_meta,'occ':occ_meta,'input_file':str(file.relative_to(out)),
                         'input_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'navigation_commands':commands,
                         'navigation_near_world':xyz[ni].tolist(),'navigation_far_world':xyz[fi].tolist(),
                         'route_s_m':float(arc[route_index]),'route_error_m':route_error,
                         'prediction':prediction,'path_world':path.tolist()}
                    plans.append(row);planlog.write(json.dumps(row)+'\n');planlog.flush();plan_id+=1
            if active_phase:
                if active is None:
                    # First active observation arrives at +0.05s; explicitly recorded warm-up hold.
                    last_control={'throttle':0.,'brake':1.,'steer':0.};last_detail={'awaiting_first_plan':True}
                else:
                    last_control,last_detail=controller.control(active['path'],active['velocity'],pose,speed,timestamp-active['timestamp'])
                    if not all(math.isfinite(x) for x in last_control.values()):raise ValueError('Nonfinite controls')
            ego.apply_control(carla.VehicleControl(**last_control))
            gap=None
            if obstacle:
                gap=float(obstacle.get_location().distance(tr.location)-ego.bounding_box.extent.x-obstacle.bounding_box.extent.x)
            traffic=[{'id':a.id,'type':a.type_id,'location':vec(a.get_location()),'velocity':vec(a.get_velocity())} for a in actors if a.type_id.startswith(('walker.','vehicle.')) and a.id!=ego.id]
            row={'frame':frame,'timestamp':timestamp,'sim_t':None if sim_start is None else timestamp-sim_start,
                 'phase':'active' if active_phase else 'warmup','ego_pose':pose.tolist(),'location':loc.tolist(),
                 'velocity_world':vec(ego.get_velocity()),'speed_mps':speed,'acceleration_world':vec(ego.get_acceleration()),
                 'gyro_carla_rad_s':imu.gyroscope.z,'route_index':route_index,'route_s_m':float(arc[route_index]),
                 'route_error_m':route_error,'plan_id':None if active is None else active['id'],
                 'control_applied_for_next_tick':last_control,'controller':last_detail,'takeover_reason':None,
                 'obstacle_surface_gap_m':gap,'traffic_truth_evaluation':traffic}
            steps.append(row);log.write(json.dumps(row)+'\n');log.flush()
            previous_scan=current_scan
            if active_phase and (tick-warmup_ticks)%100==0:
                print(json.dumps({'scenario':name,'sim_t':sim_t,'speed':speed,'route_s':float(arc[route_index]),'collisions':len(hits),'plans':len(plans)}),flush=True)
            if active_phase and arc[-1]-arc[route_index]<2 and route_error<3:
                report['termination']='route_completed';break
        report.update(status='completed_actual_closed_loop',termination=report.get('termination','simulation_time_limit'),
                      simulated_seconds=timestamp-sim_start if sim_start else 0.,distance_travelled_m=distance,
                      displacement_m=float(np.linalg.norm(loc-origin)) if origin is not None else 0.,
                      route_completion_fraction=float(arc[route_index]/arc[-1]),planning_calls=len(plans),
                      active_frames=sum(r['phase']=='active' for r in steps),collision_events=hits,
                      max_speed_mps=max((r['speed_mps'] for r in steps if r['phase']=='active'),default=0.),
                      min_obstacle_gap_m=min((r['obstacle_surface_gap_m'] for r in steps if r['obstacle_surface_gap_m'] is not None),default=None),
                      synchronous_latency_note='world waits for inference, so this does not establish realtime closed-loop capacity')
        log.close();planlog.close()
    except Exception as e:
        report.update(status='blocked_or_failed',error=repr(e),traceback=traceback.format_exc(),collision_events=hits,
                      planning_calls=len(plans),active_frames=sum(r['phase']=='active' for r in steps))
        if 'ego' in locals():
            try:ego.apply_control(carla.VehicleControl(brake=1.));report['takeovers'].append({'reason':'program_exception:'+repr(e),'frame':world.get_snapshot().frame,'action':'brake1 then terminate; no further simulation tick'})
            except Exception as ee:report['takeover_error']=repr(ee)
        print(traceback.format_exc(),flush=True)
    finally:
        if video:video.release()
        cleanup=[]
        for s in sensors:
            try:s.stop()
            except Exception as e:cleanup.append(repr(e))
        for actor in reversed(actors):
            try:actor.destroy()
            except Exception as e:cleanup.append(repr(e))
        try:world.apply_settings(old);world.set_weather(weather)
        except Exception as e:cleanup.append(repr(e))
        report['cleanup_errors']=cleanup;report['wall_seconds']=time.perf_counter()-started
        write(out/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=2117);p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=float,default=30.);p.add_argument('--wall-timeout',type=float,default=600.)
    p.add_argument('--calibration-only',action='store_true');p.add_argument('--no-camera',action='store_true')
    p.add_argument('--occ-source',choices=['sparse-proxy','upstream'],default='sparse-proxy');a=p.parse_args()
    if a.calibration_only and a.occ_source=='upstream':raise ValueError('Upstream OCC requires inference worker; calibration uses sparse-proxy')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    c=carla.Client('127.0.0.1',a.port);c.set_timeout(60)
    if c.get_server_version()!='0.9.15':raise RuntimeError('Requires CARLA0.9.15')
    world=c.get_world()
    if a.no_camera:
        settings=world.get_settings();settings.no_rendering_mode=True;world.apply_settings(settings)
    if not world.get_map().name.endswith('Town03'):world=c.load_world('Town03',False)
    worker=None;responses=queue.Queue()
    try:
        if not a.calibration_only:
            stderr=(out/'worker.stderr.log').open('w')
            worker=subprocess.Popen([str(ROOT/'.venv-train/Scripts/python.exe'),'-u',str(ROOT/'scripts/native_v2_carla_worker.py'),'--output',str(out.resolve())],
                                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,cwd=ROOT)
            def read():
                for line in worker.stdout:responses.put(line)
            threading.Thread(target=read,daemon=True).start()
            ready=json.loads(responses.get(timeout=120));assert ready.get('ready'),ready
            print('MODEL_READY '+json.dumps({k:ready[k] for k in ['epoch_one_based','global_step','warmup_seconds']}),flush=True)
        cases=[('turn','26990'),('obstacle','26456'),('dynamic','27515')]
        if a.calibration_only:cases=[('turn-controller-calibration','26990')]
        reports=[]
        for name,route_id in cases:
            reports.append(scene_run(c,worker,responses,a,name,route_id,'calibration' if a.calibration_only else 'model'))
            write(out/'suite-summary.json',reports)
    finally:
        if worker:
            try:worker.stdin.write('{"close":true}\n');worker.stdin.flush();worker.wait(timeout=15)
            except Exception:worker.terminate()


if __name__=='__main__':main()
