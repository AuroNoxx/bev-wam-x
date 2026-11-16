"""Local CARLA semantic-hit occupancy viewer. No training, RGB input or predictions."""
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
from collections import deque
import argparse, json, math, queue, threading, time, traceback
import carla
import numpy as np
import ctypes
from ctypes import wintypes
import psutil
from spectator_input import MouseLook

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'logs/occ-live'
DTYPE=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('cos','<f4'),('object','<u4'),('tag','<u4')])
STOP=threading.Event()
LOCK=threading.Lock()
COMMANDS=queue.Queue(maxsize=16)
PACKETS=deque(maxlen=16)
PUBLISH=queue.Queue(maxsize=1)
STATE={'status':'connecting','camera_mode':'follow','source':'semantic_lidar_surface_occupancy','prediction_available':False}
ENCODED=b'{"status":"connecting"}'

def camera_keys():
    """Let CARLA own its native mouse/WASD input; only release/re-enable follow."""
    user=ctypes.WinDLL('user32')
    user.GetForegroundWindow.restype=wintypes.HWND
    user.GetWindowThreadProcessId.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.DWORD)]
    pids=set()
    for p in psutil.process_iter(['pid','name','exe']):
        if p.info['name']=='CarlaUnreal-Win64-Shipping.exe' and p.info['exe'] and Path(p.info['exe']).is_relative_to(ROOT/'runtime'):
            pids.add(p.pid)
    def poll():
        # Read short press edges as well as held keys; a GUI key press can be shorter than one tick.
        f2=user.GetAsyncKeyState(0x71)
        f1=user.GetAsyncKeyState(0x70)
        pid=wintypes.DWORD();user.GetWindowThreadProcessId(user.GetForegroundWindow(),ctypes.byref(pid))
        if pid.value not in pids:return None
        if f2&0x8001:return 'follow'  # F2
        if f1&0x8001:return 'free'  # optional detached flight; LMB always orbits in follow mode
        return None
    return poll,pids

def speed(actor):
    v=actor.get_velocity(); return math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)

def serialise_worker():
    global ENCODED
    while not STOP.is_set():
        try: job=PUBLISH.get(timeout=.5)
        except queue.Empty: continue
        try:
            started=time.perf_counter()
            scans,ego_inverse,origin,voxel,meta=job
            points=[]; tags=[]
            for packet in scans:
                a=np.frombuffer(packet.raw_data,dtype=DTYPE)
                local=np.column_stack((a['x'],a['y'],a['z'],np.ones(len(a))))
                world=local @ np.asarray(packet.transform.get_matrix()).T
                ego=world @ ego_inverse.T
                valid=(ego[:,0]>-50)&(ego[:,0]<100)&(np.abs(ego[:,1])<45)&(ego[:,2]>-4)&(ego[:,2]<10)
                q=world[valid,:3]-origin
                q[:,1]*=-1
                points.append(q); tags.append(a['tag'][valid])
            if points:
                xyz=np.concatenate(points); labels=np.concatenate(tags)
                indices=np.floor(xyz/voxel).astype(np.int32)
                # One occupied voxel per cell; prefer dynamic semantics if hits overlap.
                dynamic=np.isin(labels,[12,13,14,15,16,17,18,19])
                order=np.argsort(~dynamic,kind='stable')
                _,first=np.unique(indices[order],axis=0,return_index=True)
                selected=order[first]
                centers=(indices[selected]+.5)*voxel
                meta['voxels']=np.column_stack((np.round(centers,2),labels[selected])).tolist()
            else: meta['voxels']=[]
            meta['voxel_count']=len(meta['voxels'])
            meta['voxel_size']=voxel
            meta['processing_ms']=round((time.perf_counter()-started)*1000,2)
            meta['point_packets']=len(scans)
            meta['source']='semantic_lidar_surface_occupancy'
            meta['prediction_available']=False
            meta['unknown_policy']='Unobserved volume is unknown; no verified-free volume is inferred.'
            encoded=json.dumps(meta,separators=(',',':')).encode()
            with LOCK:
                STATE.update({k:v for k,v in meta.items() if k not in ['voxels','lanes','actors']})
                ENCODED=encoded
        except Exception:
            traceback.print_exc()

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs): super().__init__(*args,directory=str(ROOT/'viewer'),**kwargs)
    def log_message(self,*args): pass
    def do_GET(self):
        route=urlparse(self.path).path
        if route in ['/api/frame','/api/status']:
            with LOCK:
                body=ENCODED if route=='/api/frame' else json.dumps(STATE).encode()
            self.send_response(200); self.send_header('Content-Type','application/json')
            self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            try: self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError): pass
        else: super().do_GET()
    def do_POST(self):
        origin=self.headers.get('Origin','')
        host=self.headers.get('Host','')
        if host not in [f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'] or (origin and origin!=f'http://{host}'):
            self.send_error(403); return
        if self.path!='/api/control': self.send_error(404); return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=1024: raise ValueError('Invalid request length')
            cmd=json.loads(self.rfile.read(length))
            if cmd.get('action') not in ['follow','free','stop','pause_display','resume_display']:raise ValueError('Unknown action')
            COMMANDS.put_nowait(cmd['action'])
            self.send_response(202);self.send_header('Content-Length','0');self.end_headers()
        except (ValueError,queue.Full): self.send_error(400)

def view_xyz(location,origin): return [location.x-origin[0],-(location.y-origin[1]),location.z-origin[2]]

def run(args):
    client=carla.Client('127.0.0.1',args.port);client.set_timeout(90)
    world=client.get_world()
    if not world.get_map().name.endswith('/'+args.map): world=client.load_world(args.map)
    original=world.get_settings(); spawned=[]; sensor=None; tm=None
    report={'server_version':client.get_server_version(),'map':world.get_map().name,'requested_sim_hz':args.sim_hz}
    origin=None; last_publish=0; window=deque(maxlen=120); camera='follow'; show=True; count=0
    poll_camera,carla_pids=camera_keys()
    mouse=MouseLook(carla_pids);orbit_yaw=0.;orbit_pitch=0.;last_camera=time.monotonic()
    try:
        settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=1/args.sim_hz
        settings.substepping=True;settings.max_substep_delta_time=1/120;settings.max_substeps=8
        world.apply_settings(settings);world.set_weather(carla.WeatherParameters.ClearNoon)
        tm=client.get_trafficmanager(args.tm_port);tm.set_synchronous_mode(True);tm.set_random_device_seed(14)
        bps=world.get_blueprint_library();bp=bps.find('vehicle.dodge.charger')
        bp.set_attribute('role_name','hero')
        map_=world.get_map();spawns=map_.get_spawn_points()
        for transform in spawns:
            wp=map_.get_waypoint(transform.location)
            neighbour=wp.get_left_lane() or wp.get_right_lane()
            if wp.is_junction or not neighbour or neighbour.lane_type!=carla.LaneType.Driving: continue
            ego=world.try_spawn_actor(bp,transform)
            if ego: break
        else: raise RuntimeError('No multi-lane spawn available')
        spawned.append(ego);origin=np.array([transform.location.x,transform.location.y,transform.location.z])
        ego.set_autopilot(True,args.tm_port);tm.vehicle_percentage_speed_difference(ego,45)
        # Demonstration traffic; it is CARLA autopilot, never the learned policy.
        traffic_bp=bps.find('vehicle.dodge.charger');traffic_bp.set_attribute('role_name','occ_demo_traffic')
        base=map_.get_waypoint(transform.location)
        candidates=[]
        for distance in [25,55,90,140,200]:
            for wp in base.next(distance):
                candidates.append(wp)
                other=wp.get_left_lane()
                if other and other.lane_type==carla.LaneType.Driving and other.lane_id*wp.lane_id>0:candidates.append(other)
        for wp in candidates[:args.traffic]:
            t=wp.transform;t.location.z+=.5
            actor=world.try_spawn_actor(traffic_bp,t)
            if actor:
                spawned.append(actor);actor.set_autopilot(True,args.tm_port)
                tm.vehicle_percentage_speed_difference(actor,40)
        lidar=bps.find('sensor.lidar.ray_cast_semantic')
        for key,value in {'channels':'48','range':'110','points_per_second':'180000','rotation_frequency':'10','sensor_tick':'0','upper_fov':'12','lower_fov':'-30'}.items():lidar.set_attribute(key,value)
        sensor=world.spawn_actor(lidar,carla.Transform(carla.Location(z=2.5)),attach_to=ego)
        spawned.append(sensor)
        def on_scan(packet):
            with LOCK:PACKETS.append(packet)
        sensor.listen(on_scan)
        # Cache actual lane-center samples for context, not a predicted driving trajectory.
        lane_points=map_.generate_waypoints(3)
        lane_xyz=np.array([[w.transform.location.x,w.transform.location.y,w.transform.location.z+.08] for w in lane_points])
        start=time.monotonic();last_status=0
        with (LOG/'metrics.jsonl').open('w',encoding='utf-8') as log:
            while not STOP.is_set():
                before=time.monotonic()
                while not COMMANDS.empty():
                    cmd=COMMANDS.get_nowait()
                    if cmd in ['follow','free']:camera=cmd
                    elif cmd=='stop':STOP.set()
                    elif cmd=='pause_display':show=False
                    elif cmd=='resume_display':show=True
                if STOP.is_set():break
                camera=poll_camera() or camera
                # Town12 streams the hero's tile on the first tick; allow initial loading.
                frame=world.tick(90 if count==0 else 20);count+=1
                t=ego.get_transform(); control=ego.get_control()
                held,mouse_dx,mouse_dy=mouse.sample()
                now_camera=time.monotonic();camera_dt=min(.1,now_camera-last_camera);last_camera=now_camera
                if camera=='follow':
                    if held:
                        orbit_yaw=(orbit_yaw+mouse_dx*.18+180)%360-180
                        orbit_pitch=max(-49,min(24,orbit_pitch-mouse_dy*.14))
                    else:
                        decay=math.exp(-camera_dt/0.16)
                        orbit_yaw*=decay;orbit_pitch*=decay
                        if abs(orbit_yaw)<.05:orbit_yaw=0.
                        if abs(orbit_pitch)<.05:orbit_pitch=0.
                    yaw=t.rotation.yaw+orbit_yaw;pitch=-16+orbit_pitch
                    yr,pr=math.radians(yaw),math.radians(pitch)
                    # Orbit about the vehicle; ego translation and heading are followed continuously.
                    pivot=t.location+carla.Location(z=1.5)
                    loc=pivot-carla.Location(x=11*math.cos(pr)*math.cos(yr),y=11*math.cos(pr)*math.sin(yr),z=11*math.sin(pr))
                    world.get_spectator().set_transform(carla.Transform(loc,carla.Rotation(pitch=pitch,yaw=yaw)))
                tick_done=time.monotonic();window.append(tick_done)
                hz=(len(window)-1)/(window[-1]-window[0]) if len(window)>1 else 0
                meta={'status':'running','frame':frame,'sim_time':world.get_snapshot().timestamp.elapsed_seconds,
                      'camera_mode':camera,'display_enabled':show,'speed_kmh':round(speed(ego)*3.6,1),
                      'camera_dragging':held,'camera_orbit_yaw':round(orbit_yaw,2),'camera_orbit_pitch':round(orbit_pitch,2),
                      'throttle':round(control.throttle,3),'brake':round(control.brake,3),'steer':round(control.steer,3),
                      'sim_hz':round(hz,1),'target_sim_hz':args.sim_hz,'tick_ms':round((tick_done-before)*1000,2),
                      'ego':view_xyz(t.location,origin),'yaw':-math.radians(t.rotation.yaw),'map':args.map,
                      'ego_extent':[ego.bounding_box.extent.x,ego.bounding_box.extent.y,ego.bounding_box.extent.z],
                      'controller':'CARLA Traffic Manager (untrained demo)','elapsed_s':round(tick_done-start,1)}
                with LOCK:STATE.update(meta)
                if show and tick_done-last_publish>=1/args.occ_hz:
                    last_publish=tick_done
                    with LOCK:scans=list(PACKETS)
                    # Assemble one full 10 Hz revolution in world coordinates; each scan carries its acquisition transform.
                    scans=[s for s in scans if s.timestamp>=meta['sim_time']-.12]
                    meta['scan_age_s']=round(meta['sim_time']-scans[-1].timestamp,3) if scans else None
                    meta['scan_span_s']=round(scans[-1].timestamp-scans[0].timestamp,3) if len(scans)>1 else 0
                    close=np.linalg.norm(lane_xyz[:,:2]-np.array([t.location.x,t.location.y]),axis=1)<90
                    lanes=lane_xyz[close]-origin;lanes[:,1]*=-1;meta['lanes']=np.round(lanes,2).tolist()
                    meta['actors']=[]
                    for actor in spawned:
                        if actor.type_id.startswith('vehicle.') and actor.id!=ego.id:
                            at=actor.get_transform();extent=actor.bounding_box.extent
                            meta['actors'].append({'p':view_xyz(at.location,origin),'yaw':-math.radians(at.rotation.yaw),'size':[extent.x*2,extent.y*2,extent.z*2]})
                    if PUBLISH.full():
                        try:PUBLISH.get_nowait()
                        except queue.Empty:pass
                    PUBLISH.put_nowait((scans,np.asarray(t.get_inverse_matrix()),origin,args.voxel,meta))
                if tick_done-last_status>1:
                    last_status=tick_done
                    with LOCK:status=dict(STATE)
                    log.write(json.dumps(status)+'\n');log.flush()
                    (LOG/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
                if args.seconds and tick_done-start>=args.seconds:break
                time.sleep(max(0,1/args.sim_hz-(time.monotonic()-before)))
        report.update(status='stopped',ticks=count,wall_seconds=time.monotonic()-start)
    except Exception as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        STOP.set()
        mouse.close()
        if sensor:
            try:sensor.stop()
            except Exception:pass
        for actor in reversed(spawned):
            try:actor.destroy()
            except Exception:pass
        if tm:
            try:tm.set_synchronous_mode(False)
            except Exception:pass
        try:world.apply_settings(original)
        except Exception:pass
        with LOCK:STATE.update(status=report['status'],error=report.get('error'))
        (LOG/'last-run.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=2100);parser.add_argument('--tm-port',type=int,default=8100)
    parser.add_argument('--http-port',type=int,default=8765);parser.add_argument('--map',default='Town12')
    parser.add_argument('--sim-hz',type=int,choices=[30,60],default=30)
    parser.add_argument('--occ-hz',type=int,choices=[5,10],default=5)
    parser.add_argument('--voxel',type=float,default=.6);parser.add_argument('--traffic',type=int,default=6)
    parser.add_argument('--seconds',type=int,default=0)
    args=parser.parse_args();LOG.mkdir(exist_ok=True,parents=True)
    (LOG/'status.json').write_text(json.dumps(STATE,indent=2),encoding='utf-8')
    server=ThreadingHTTPServer(('127.0.0.1',args.http_port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    threading.Thread(target=serialise_worker,daemon=True).start()
    print(f'VIEWER http://127.0.0.1:{args.http_port}',flush=True)
    try:run(args)
    finally:STOP.set();server.shutdown()

if __name__=='__main__':main()
