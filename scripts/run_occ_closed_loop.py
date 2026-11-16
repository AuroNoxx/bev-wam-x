"""Visible autonomous stage-one OCC loop; sensor-only NPC observation.

Environment setup and evaluation own CARLA actor handles. The pure planner in
occ_classical_core receives only occupancy, ego odometry and permitted map data.
No ego autopilot, learned models, chat commands, NPC IDs/velocities or future data.
"""
import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import math
from pathlib import Path
import queue
import random
import threading
import time
import traceback

import carla
import numpy as np
from occ_classical_core import voxelize, OccTracker, LanePlanner, Controller

ROOT=Path(__file__).resolve().parents[1]
DT=.05
DTYPE=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('cos','<f4'),('object','<u4'),('tag','<u4')])


def write(path,value):
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    for _ in range(50):
        try:tmp.replace(path);return
        except PermissionError:time.sleep(.002)
    raise RuntimeError('Could not publish '+str(path))


def steering_scale(curve,speed):
    return float(np.interp(speed,[p.x for p in curve],[p.y for p in curve]))


def local_route(map_,ego_transform,road,lane,length=65):
    wp=map_.get_waypoint(ego_transform.location)
    inverse=np.array(ego_transform.get_inverse_matrix());yaw0=math.radians(ego_transform.rotation.yaw)
    result=[]
    for ds in np.arange(0,length+1,1.):
        p=map_.get_waypoint_xodr(road,lane,wp.s+ds)
        if p is None:break
        loc=p.transform.location;q=inverse@np.array([loc.x,loc.y,loc.z,1.])
        result.append([q[0],-q[1],q[2],yaw0-math.radians(p.transform.rotation.yaw)])
    if len(result)<6:raise RuntimeError('Insufficient permitted route ahead')
    return np.array(result),wp


def sensor_occ(packet,ego_transform):
    a=np.frombuffer(packet.raw_data,dtype=DTYPE)
    # Only xyz/tag fields cross the perception boundary. object IDs are never read.
    local=np.column_stack([a['x'],a['y'],a['z'],np.ones(len(a))])
    sensor2ego=np.array(ego_transform.get_inverse_matrix())@np.array(packet.transform.get_matrix())
    xyz=(local@sensor2ego.T)[:,:3];xyz[:,1]*=-1
    origin=sensor2ego[:3,3].copy();origin[1]*=-1
    return voxelize(xyz,a['tag'].copy(),origin,packet.frame,packet.timestamp)


def wait_scan(packets,frame,timeout=3.):
    deadline=time.monotonic()+timeout;latest=None
    while time.monotonic()<deadline:
        try:p=packets.get(timeout=max(.001,deadline-time.monotonic()))
        except queue.Empty:break
        latest=p
        if p.frame>=frame:return p
    raise RuntimeError(f'No same-frame LiDAR packet for {frame}; latest={None if latest is None else latest.frame}')


def evaluation_only(snapshot, npc_ids, ego_transform, tracks):
    """Isolated logging sink. Its return value is NEVER passed to planning/control."""
    inverse=np.array(ego_transform.get_inverse_matrix());truth=[]
    for actor_id in npc_ids:
        a=snapshot.find(actor_id)
        if a is None:continue
        loc=a.get_transform().location;v=a.get_velocity()
        center=inverse@np.array([loc.x,loc.y,loc.z,1.]);velocity=inverse[:3,:3]@np.array([v.x,v.y,v.z])
        truth.append({'actor_id':actor_id,'center':[float(center[0]),float(-center[1]),float(center[2])],
                      'velocity':[float(velocity[0]),float(-velocity[1])]})
    matches=[]
    for tr in tracks:
        if not truth:continue
        nearest=min(truth,key=lambda a:np.linalg.norm(np.array(a['center'][:2])-np.array(tr['center'][:2])))
        distance=float(np.linalg.norm(np.array(nearest['center'][:2])-np.array(tr['center'][:2])))
        if distance<5:
            matches.append({'track_id':tr['id'],'hits':tr['hits'],'actor_id':nearest['actor_id'],
                            'surface_centroid_to_actor_origin_m':distance,
                            'velocity_error_mps':float(np.linalg.norm(np.array(tr['velocity'])-nearest['velocity']))})
    return {'frame':snapshot.frame,'timestamp':snapshot.timestamp.elapsed_seconds,'truth_for_evaluation_only':truth,'matches':matches}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--seconds',type=float,default=120)
    parser.add_argument('--speed-kmh',type=float,default=30)
    parser.add_argument('--port',type=int,default=2100)
    parser.add_argument('--http-port',type=int,default=8792)
    parser.add_argument('--traffic',type=int,default=3)
    parser.add_argument('--run-label',default='stage1')
    args=parser.parse_args()
    if not 0<args.seconds<=600 or not 0<args.speed_kmh<=40:raise ValueError('Bounded first-stage run: <=600s, <=40km/h')
    folder=ROOT/'logs/occ-closed-loop'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+args.run_label)
    folder.mkdir(parents=True);(folder/'occ').mkdir();(folder/'frames').mkdir()
    write(folder.parent/'current.json',{'folder':str(folder),'http_port':args.http_port})
    protected={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'scripts/occ_classical_core.py',Path(__file__)]}
    write(folder/'source-sha256.json',protected)
    for name in ['occ_classical_core.py','run_occ_closed_loop.py']:(folder/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    stop=threading.Event();lock=threading.Lock();latest={'status':'starting','folder':str(folder)}
    def publish(data):
        nonlocal latest
        with lock:latest=data
        write(folder/'status.json',{k:v for k,v in data.items() if k not in ['voxels','route','tracks','trajectory']})
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def do_GET(self):
            if self.path=='/api/status':
                with lock:body=json.dumps(latest).encode()
                typ='application/json'
            elif self.path in ['/','/index.html']:
                body=(ROOT/'viewer/occ-closed-loop.html').read_bytes();typ='text/html; charset=utf-8'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',typ);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_POST(self):
            if self.path!='/stop':self.send_error(404);return
            origin=self.headers.get('Origin','');host=self.headers.get('Host','')
            if host!=f'127.0.0.1:{args.http_port}' or (origin and origin!=f'http://{host}'):self.send_error(403);return
            stop.set();self.send_response(202);self.end_headers()
    server=ThreadingHTTPServer(('127.0.0.1',args.http_port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    client=carla.Client('127.0.0.1',args.port);client.set_timeout(90)
    world=client.get_world();actors=[];sensors=[];tm=None;original=None;ego=None
    collisions=[];markings=[];rows=[];plan_times=[];frame_count=0;simtime=0.;startwall=time.monotonic()
    report={'status':'starting','mode':'continuous_sensor_OCC_classical_control','ego_autopilot':False,
            'planner_observation':['sensor_OCC_xyz_semantic','ego_odometry','navigation_lane_map'],
            'npc_actor_state_in_planner':False,'sensor_object_ids_used':False,'future_information_used':False,
            'learned_models_loaded':False,'speed_cap_kmh':args.speed_kmh,'requested_seconds':args.seconds,
            'limitations':['Simulation semantic LiDAR categories; not learned real-world perception.',
                           'Ray-cleared column support is sparse visibility evidence, not full-volume free certification.',
                           'Stage-one lane-keeping prototype; no lane changes or overtakes.',
                           'Controller parameters provisional; no high-speed or general safety qualification.']}
    publish({'status':'loading_map','folder':str(folder)})
    try:
        if world.get_actors().filter('vehicle.*'):raise RuntimeError('Existing vehicles: refusing to take over another session')
        if not world.get_map().name.endswith('/Town12'):world=client.load_world('Town12')
        original=world.get_settings();settings=world.get_settings()
        settings.synchronous_mode=True;settings.fixed_delta_seconds=DT
        settings.substepping=True;settings.max_substep_delta_time=.01;settings.max_substeps=10
        world.apply_settings(settings);world.set_weather(carla.WeatherParameters.ClearNoon)
        map_=world.get_map();lib=world.get_blueprint_library();road,lane=388,-2
        bp=lib.find('vehicle.tesla.model3');bp.set_attribute('role_name','hero');bp.set_attribute('color','245,245,245')
        tr=map_.get_waypoint_xodr(road,lane,80).transform;tr.location.z+=.4
        ego=world.spawn_actor(bp,tr);actors.append(ego);ego.set_autopilot(False);ego.apply_control(carla.VehicleControl(brake=1))
        world.get_spectator().set_transform(carla.Transform(tr.location-tr.get_forward_vector()*10+carla.Location(z=6),carla.Rotation(pitch=-18,yaw=tr.rotation.yaw)))
        rng=random.Random(20261115);traffic=[];private=[]
        for i,(nl,ns) in enumerate([(-2,120),(-1,105),(-3,155),(-1,220)][:args.traffic]):
            nb=lib.find('vehicle.tesla.model3');nb.set_attribute('role_name',f'occ_stage1_npc_{i}');nb.set_attribute('color','40,100,210')
            nt=map_.get_waypoint_xodr(road,nl,ns).transform;nt.location.z+=.4
            npc=world.spawn_actor(nb,nt);actors.append(npc);traffic.append(npc);npc.apply_control(carla.VehicleControl(brake=1))
            private.append({'id':npc.id,'initial_lane':nl,'initial_s':ns,'desired_speed_kmh':rng.uniform(21,24) if i==0 else rng.uniform(31,39)})
        write(folder/'environment-setup-audit.json',{'seed':20261115,'vehicles':private,'policy':'TM autonomous, no event schedule; never passed to planner'})
        publish({'status':'settling_vehicles','folder':str(folder)})
        print('SETTLING',str(folder),flush=True)
        for _ in range(20):world.tick(90)
        physics=ego.get_physics_control();et=ego.get_transform();forward=et.get_forward_vector()
        wheels=sorted((wheel.location-et.location).dot(forward) for wheel in physics.wheels)
        rear=(wheels[0]+wheels[1])/2;wheelbase=(wheels[2]+wheels[3])/2-rear
        angle=math.radians(max(wheel.max_steer_angle for wheel in physics.wheels))
        if not 2<wheelbase<4:raise RuntimeError('Unverified wheel geometry '+str(wheels))
        report['geometry']={'wheelbase_m':wheelbase,'rear_axle_x_m':rear,'max_steer_angle_rad':angle}
        control=Controller(wheelbase,rear,angle);planner=LanePlanner(args.speed_kmh/3.6,ego.bounding_box.extent.x,ego.bounding_box.extent.y)
        tracker=OccTracker();packets=queue.Queue(maxsize=8);images=queue.Queue(maxsize=4)
        lidar=lib.find('sensor.lidar.ray_cast_semantic')
        for k,v in {'channels':'48','range':'90','points_per_second':'160000','rotation_frequency':'20','sensor_tick':'0','upper_fov':'10','lower_fov':'-30'}.items():lidar.set_attribute(k,v)
        sensor=world.spawn_actor(lidar,carla.Transform(carla.Location(z=2.5)),attach_to=ego);actors.append(sensor);sensors.append(sensor)
        def on_scan(p):
            try:packets.put_nowait(p)
            except queue.Full:
                try:packets.get_nowait();packets.put_nowait(p)
                except queue.Empty:pass
        sensor.listen(on_scan)
        collision=world.spawn_actor(lib.find('sensor.other.collision'),carla.Transform(),attach_to=ego);actors.append(collision);sensors.append(collision)
        collision.listen(lambda e:collisions.append({'frame':e.frame,'other':e.other_actor.type_id}))
        marking=world.spawn_actor(lib.find('sensor.other.lane_invasion'),carla.Transform(),attach_to=ego);actors.append(marking);sensors.append(marking)
        marking.listen(lambda e:markings.append({'frame':e.frame,'types':[str(x.type) for x in e.crossed_lane_markings]}))
        camera=lib.find('sensor.camera.rgb')
        for k,v in {'image_size_x':'640','image_size_y':'360','sensor_tick':'1.0'}.items():camera.set_attribute(k,v)
        camera=world.spawn_actor(camera,carla.Transform(carla.Location(x=-10,z=6),carla.Rotation(pitch=-18)),attach_to=ego);actors.append(camera);sensors.append(camera)
        def on_image(im):
            try:images.put_nowait(im)
            except queue.Full:pass
        camera.listen(on_image)
        tm=client.get_trafficmanager(8102);tm.set_synchronous_mode(True);tm.set_random_device_seed(20261115)
        for npc,setup in zip(traffic,private):
            tm.set_desired_speed(npc,setup['desired_speed_kmh']);tm.auto_lane_change(npc,True)
            npc.apply_control(carla.VehicleControl());npc.set_autopilot(True,8102)
        frame=world.tick(30);packet=wait_scan(packets,frame)
        plan=None;observed=None;tracks=[];start_s=map_.get_waypoint(ego.get_location()).s
        print('READY',json.dumps({'folder':str(folder),'url':f'http://127.0.0.1:{args.http_port}','geometry':report['geometry']}),flush=True)
        runtime_start=time.monotonic()
        with (folder/'telemetry.jsonl').open('w',encoding='utf-8') as log, (folder/'evaluation-only.jsonl').open('w',encoding='utf-8') as eval_log:
            while simtime<args.seconds and not stop.is_set() and not (folder/'stop.requested').exists():
                wall=time.perf_counter();snapshot=world.get_snapshot();et=ego.get_transform();ev=ego.get_velocity()
                speed=math.sqrt(ev.x**2+ev.y**2+ev.z**2);state={'speed':speed}
                route,wp=local_route(map_,et,road,lane)
                if frame_count%4==0:
                    packet=wait_scan(packets,frame) if frame_count else packet
                    begin=time.perf_counter();observed=sensor_occ(packet,et)
                    tracks=tracker.update(observed,np.array([et.location.x,-et.location.y]),-math.radians(et.rotation.yaw))
                    plan=planner.plan(observed,tracks,state,route,snapshot.timestamp.elapsed_seconds)
                    plan_times.append((time.perf_counter()-begin)*1000)
                    eval_log.write(json.dumps(evaluation_only(snapshot,[a.id for a in traffic],et,tracks))+'\n')
                    if len(route)<60:
                        plan.update(target_speed=min(plan['target_speed'],math.sqrt(2*3.*max(0.,len(route)-12))),reason='bounded_route_end')
                    if frame_count%20==0:
                        np.savez_compressed(folder/'occ'/f'{frame:08d}.npz',xyz=observed.xyz,tags=observed.tags,
                                            cleared=np.packbits(observed.cleared),shape=observed.cleared.shape,
                                            world2ego=np.array(et.get_inverse_matrix()),sensor_frame=frame,timestamp=observed.timestamp)
                    # Display only, never fed back into the planner.
                    for a,b in zip(plan['trajectory'][:-1],plan['trajectory'][1:]):
                        aa=et.transform(carla.Location(x=a[0],y=-a[1],z=.3));bb=et.transform(carla.Location(x=b[0],y=-b[1],z=.3))
                        world.debug.draw_line(aa,bb,thickness=.06,color=carla.Color(20,240,120),life_time=.25)
                # A lost observation must not keep an old acceleration command alive.
                if snapshot.timestamp.elapsed_seconds-observed.timestamp>.35:
                    plan={**plan,'target_speed':0.,'emergency':True,'reason':'sensor_stale'}
                u=control.update(state,route,plan,DT,steering_scale(physics.steering_curve,speed*3.6))
                ego.apply_control(carla.VehicleControl(throttle=u['throttle'],brake=u['brake'],steer=u['steer']))
                frame=world.tick(30);frame_count+=1;simtime=frame_count*DT
                et=ego.get_transform();ewp=map_.get_waypoint(et.location);lane_wp=map_.get_waypoint_xodr(road,lane,ewp.s)
                if lane_wp is None:raise RuntimeError('Exited inspected route')
                diff=et.location-lane_wp.transform.location;right=lane_wp.transform.get_right_vector()
                lateral=diff.dot(right)
                row={'frame':frame,'simulation_s':simtime,'sensor_frame':observed.frame,'speed_kmh':speed*3.6,
                     'target_speed_kmh':plan['target_speed']*3.6,'reason':plan['reason'],'occ_gap_m':plan['gap_m'],
                     'occ_lead_speed_mps':plan['lead_speed_mps'],'visible_support_m':plan['visible_support_m'],
                     'tracked_objects':len(tracks),'occupied_voxels':len(observed.xyz),'rays':observed.ray_count,
                     'ego_s':ewp.s,'lateral_error_m':lateral,'control':u,'planning_ms':plan_times[-1],
                     'collision_count':len(collisions),'lane_markings':len(markings),'ego_autopilot':False}
                rows.append(row);log.write(json.dumps(row)+'\n')
                if frame_count%4==0:
                    log.flush();eval_log.flush();data={**row,'status':'running','folder':str(folder),
                                      'voxels':observed.xyz[::max(1,len(observed.xyz)//2200)].round(2).tolist(),
                                      'route':route[:,:2].round(2).tolist(),'tracks':tracks,'trajectory':plan['trajectory']}
                    publish(data)
                tr=ego.get_transform()
                world.get_spectator().set_transform(carla.Transform(tr.location-tr.get_forward_vector()*11+carla.Location(z=6),carla.Rotation(pitch=-18,yaw=tr.rotation.yaw)))
                while not images.empty():
                    im=images.get_nowait();im.save_to_disk(str(folder/'frames'/f'{im.frame:08d}.png'))
                if collisions:raise RuntimeError('Collision detected; stopped')
                if abs(lateral)+ego.bounding_box.extent.y>lane_wp.lane_width/2+.1:raise RuntimeError('Ego footprint left assigned lane')
                if frame_count%100==0:print('PROGRESS',json.dumps(row),flush=True)
                time.sleep(max(0,DT-(time.perf_counter()-wall)))
        ego.apply_control(carla.VehicleControl(brake=.8))
        for _ in range(40):world.tick(15)
        report.update(status='completed',simulated_seconds=simtime,frames=frame_count,
                      completion_reason='duration_reached' if simtime>=args.seconds else 'user_stop',
                      loop_wall_seconds=time.monotonic()-runtime_start,
                      realtime_factor=simtime/max(.001,time.monotonic()-runtime_start),
                      travelled_road_m=float(rows[-1]['ego_s']-start_s) if rows else 0.,
                      maximum_speed_kmh=max((r['speed_kmh'] for r in rows),default=0),
                      mean_speed_kmh=float(np.mean([r['speed_kmh'] for r in rows])) if rows else 0.,
                      max_abs_lateral_error_m=max((abs(r['lateral_error_m']) for r in rows),default=0),
                      occ_follow_frames=sum(r['reason']=='occ_follow' for r in rows),
                      planning_ms_p50_p95_max=np.percentile(plan_times,[50,95,100]).tolist() if plan_times else [])
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc(),simulated_seconds=simtime)
        if ego:
            try:ego.apply_control(carla.VehicleControl(brake=1))
            except Exception:pass
        print(traceback.format_exc(),flush=True)
    finally:
        report.update(collisions=collisions,lane_markings=markings,wall_seconds=time.monotonic()-startwall)
        write(folder/'report.json',report);publish({**report,'folder':str(folder)})
        for sensor in sensors:
            try:sensor.stop()
            except Exception:pass
        for actor in reversed(actors):
            try:actor.destroy()
            except Exception:pass
        if tm:
            try:tm.set_synchronous_mode(False)
            except Exception:pass
        if original:
            try:world.apply_settings(original)
            except Exception:pass
        print('FINISHED',json.dumps(report),flush=True)
        server.shutdown()
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
