"""Highway OCC-only autonomous runtime with isolated random Traffic Manager.

No learned planner, privileged NPC observation or live NPC parameter updates.
Environment truth is written to an evaluation-only file AFTER the ego plan.
"""
import argparse
from datetime import datetime
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

import carla
import numpy as np
from highway_occ_core import make_occ,TrafficTracker,HighwayPlanner,HighwayController
from run_occ_closed_loop import DTYPE,write,wait_scan,steering_scale

ROOT=Path(__file__).resolve().parents[1]
DT=.05


class Traffic:
    def __init__(self,folder):
        self.errors=(folder/'traffic.stderr.log').open('w')
        self.process=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/highway_traffic_worker.py'),str(folder)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.errors,text=True,bufsize=1,creationflags=subprocess.CREATE_NO_WINDOW)
        self.messages=queue.Queue()
        def reader():
            for line in self.process.stdout:
                try:self.messages.put(json.loads(line))
                except json.JSONDecodeError:pass
        threading.Thread(target=reader,daemon=True).start()
    def receive(self,timeout=30):
        result=self.messages.get(timeout=timeout)
        if result.get('status')=='error':raise RuntimeError(result['error'])
        return result
    def send(self,op,timeout=30):
        self.process.stdin.write(json.dumps({'op':op})+'\n');self.process.stdin.flush()
        return self.receive(timeout)
    def close(self):
        try:self.send('stop',3);self.process.wait(timeout=12)
        except Exception:
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except Exception:self.process.kill()
        self.errors.close()


def cache_lanes(m,road,lanes):
    cache={}
    for lane in lanes:
        samples=[]
        for s in range(0,2500):
            wp=m.get_waypoint_xodr(road,lane,float(s))
            if wp is None:break
            t=wp.transform;p=t.location
            samples.append([p.x,p.y,p.z,math.radians(t.rotation.yaw),s])
        cache[lane]=np.array(samples)
    return cache


def local_lanes(cache,tr,station):
    inverse=np.array(tr.get_inverse_matrix());yaw=math.radians(tr.rotation.yaw);out={}
    for lane,data in cache.items():
        selected=data[(data[:,4]>=max(0,station-110))&(data[:,4]<=station+210)]
        if len(selected)<10:continue
        local=np.c_[selected[:,:3],np.ones(len(selected))]@inverse.T
        out[lane]=np.c_[local[:,0],-local[:,1],local[:,2],np.unwrap(yaw-selected[:,3]),selected[:,4]-station]
    return out


def perception(packet,tr):
    a=np.frombuffer(packet.raw_data,dtype=DTYPE)
    # Sensor object IDs deliberately excluded.
    transform=np.array(tr.get_inverse_matrix())@np.array(packet.transform.get_matrix())
    points=(np.c_[a['x'],a['y'],a['z'],np.ones(len(a))]@transform.T)[:,:3];points[:,1]*=-1
    origin=transform[:3,3].copy();origin[1]*=-1
    return make_occ(points,a['tag'].copy(),origin,packet.frame,packet.timestamp)


def evaluation_snapshot(world,snapshot,ego,npc_ids,m,tracks):
    """Logging sink; result never feeds back into planner or controller."""
    gt=[];et=ego.get_transform();inv=np.array(et.get_inverse_matrix())
    for actor_id in npc_ids:
        a=snapshot.find(actor_id)
        if a is None:continue
        tr=a.get_transform();v=a.get_velocity();wp=m.get_waypoint(tr.location)
        local=inv@np.array([tr.location.x,tr.location.y,tr.location.z,1.])
        lv=inv[:3,:3]@np.array([v.x,v.y,v.z])
        gt.append({'id':actor_id,'road':wp.road_id,'lane':wp.lane_id,'s':wp.s,
                   'speed_mps':math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z),
                   'local':[float(local[0]),float(-local[1])],'velocity':[float(lv[0]),float(-lv[1])]})
    matches=[]
    for t in tracks:
        if not gt:continue
        nearest=min(gt,key=lambda a:np.linalg.norm(np.array(a['local'])-[t['x'],t['y']]))
        d=float(np.linalg.norm(np.array(nearest['local'])-[t['x'],t['y']]))
        if d<6:matches.append({'track_id':t['id'],'actor_id':nearest['id'],'hits':t['hits'],'centroid_distance_m':d,
                               'velocity_error_mps':float(np.linalg.norm(np.array(nearest['velocity'])-[t['vx'],t['vy']]))})
    return {'frame':snapshot.frame,'timestamp':snapshot.timestamp.elapsed_seconds,'npcs':gt,'matches':matches}


def footprint_inside(m,tr,box,road,lanes):
    for x in (-box.extent.x,box.extent.x):
        for y in (-box.extent.y,box.extent.y):
            point=tr.transform(box.location+carla.Location(x=x,y=y));near=m.get_waypoint(point)
            found=False
            for lane in lanes:
                wp=m.get_waypoint_xodr(road,lane,near.s)
                if wp is None:continue
                delta=point-wp.transform.location
                if abs(delta.dot(wp.transform.get_right_vector()))<=wp.lane_width/2+.03 and abs(delta.dot(wp.transform.get_forward_vector()))<1.5:
                    found=True;break
            if not found:return False
    return True


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--seconds',type=float,default=45);p.add_argument('--speed',type=float,default=60)
    p.add_argument('--traffic',type=int,default=10);p.add_argument('--seed',type=int,default=101)
    p.add_argument('--profile',choices=['low','mixed','fast','dense'],default='mixed')
    p.add_argument('--label',default='development');p.add_argument('--road',type=int,default=1157)
    p.add_argument('--no-pace',action='store_true');p.add_argument('--initial-speed',type=float,default=0)
    args=p.parse_args()
    if not 0<=args.speed<=120 or not 0<=args.initial_speed<=args.speed or not 5<=args.seconds<=120:raise ValueError('Outside experiment bounds')
    folder=ROOT/'logs/highway-occ'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+args.label)
    folder.mkdir(parents=True);(folder/'frames').mkdir();(folder/'occ').mkdir()
    cfg={'rpc_port':2100,'tm_port':8123,'road':args.road,'lanes':[-4,-3,-2],'ego_lane':-3,'ego_start_s':150,
         'traffic_count':args.traffic,'traffic_seed':args.seed,'traffic_profile':args.profile,'cruise_kmh':args.speed,
         'duration_s':args.seconds,'initial_speed_kmh':args.initial_speed,'phase':args.label}
    write(folder/'episode-config.json',cfg);write(folder.parent/'current.json',{'folder':str(folder)})
    files=['highway_occ_core.py','run_highway_occ.py','highway_traffic_worker.py']
    hashes={f:hashlib.sha256((ROOT/'scripts'/f).read_bytes()).hexdigest() for f in files}
    write(folder/'code-hashes.json',hashes)
    for f in files:(folder/f).write_bytes((ROOT/'scripts'/f).read_bytes())
    report={'status':'starting','config':cfg,'code_hashes':hashes,'ego_autopilot':False,'NPC_state_input':False,
            'NPC_runtime_parameter_changes':0,'sensor_object_ids_used':False,'training':False,'heldout_dataset_used':False,
            'limitations':['Simulation semantic LiDAR, not real sensor perception.',
                           'Sparse visibility support is not full-volume free-space certification.',
                           'Finite random traffic runs do not certify unrestricted 0-120km/h driving.']}
    write(folder/'status.json',{'status':'starting'});print('START',str(folder),flush=True)
    c=carla.Client('127.0.0.1',2100);c.set_timeout(25)
    world=None;original=None;ego=None;traffic=None;actors=[];sensors=[]
    collision=[];markings=[];telemetry=[];planning=[];sim=0.;startwall=time.monotonic()
    try:
        world=c.get_world()
        if world.get_actors().filter('vehicle.*'):raise RuntimeError('Another vehicle session exists; no takeover')
        if not world.get_map().name.endswith('/Town12'):c.set_timeout(120);world=c.load_world('Town12');c.set_timeout(25)
        original=world.get_settings();settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=DT
        settings.substepping=True;settings.max_substep_delta_time=.01;settings.max_substeps=10;world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters.ClearNoon);m=world.get_map();lib=world.get_blueprint_library()
        lane_cache=cache_lanes(m,cfg['road'],cfg['lanes']);road_end=min(a[-1,4] for a in lane_cache.values())
        bp=lib.find('vehicle.tesla.model3');bp.set_attribute('role_name','hero');bp.set_attribute('color','245,245,245')
        tr=m.get_waypoint_xodr(cfg['road'],cfg['ego_lane'],cfg['ego_start_s']).transform;tr.location.z+=.4
        ego=world.spawn_actor(bp,tr);actors.append(ego);ego.set_autopilot(False);ego.apply_control(carla.VehicleControl(brake=1))
        world.get_spectator().set_transform(carla.Transform(tr.location-tr.get_forward_vector()*12+carla.Location(z=7),carla.Rotation(pitch=-18,yaw=tr.rotation.yaw)))
        traffic=Traffic(folder);spawned=traffic.receive();report['NPCs_spawned']=spawned['count']
        write(folder/'status.json',{'status':'settling','NPCs':spawned['count']});print('SETTLING',spawned,flush=True)
        for _ in range(25):world.tick(12)
        et=ego.get_transform();physics=ego.get_physics_control();forward=et.get_forward_vector()
        wheels=sorted((w.location-et.location).dot(forward) for w in physics.wheels)
        rear=(wheels[0]+wheels[1])/2;wheelbase=(wheels[2]+wheels[3])/2-rear
        maxangle=math.radians(max(w.max_steer_angle for w in physics.wheels))
        if not 2<wheelbase<4:raise ValueError('Vehicle geometry not validated')
        controller=HighwayController(wheelbase,rear,maxangle);planner=HighwayPlanner(args.speed/3.6,cfg['ego_lane']);tracker=TrafficTracker()
        report['geometry']={'wheelbase':wheelbase,'rear_x':rear,'steer_angle':maxangle}
        packets=queue.Queue(maxsize=8);images=queue.Queue(maxsize=4)
        lidar=lib.find('sensor.lidar.ray_cast_semantic')
        for key,value in {'channels':'96','range':'220','points_per_second':'360000','rotation_frequency':'20','sensor_tick':'0','upper_fov':'8','lower_fov':'-25'}.items():lidar.set_attribute(key,value)
        sensor=world.spawn_actor(lidar,carla.Transform(carla.Location(z=2.5)),attach_to=ego);actors.append(sensor);sensors.append(sensor)
        def on_scan(scan):
            try:packets.put_nowait(scan)
            except queue.Full:
                try:packets.get_nowait();packets.put_nowait(scan)
                except queue.Empty:pass
        sensor.listen(on_scan)
        cs=world.spawn_actor(lib.find('sensor.other.collision'),carla.Transform(),attach_to=ego);actors.append(cs);sensors.append(cs)
        cs.listen(lambda e:collision.append({'frame':e.frame,'other_id':e.other_actor.id,'other_type':e.other_actor.type_id}))
        ls=world.spawn_actor(lib.find('sensor.other.lane_invasion'),carla.Transform(),attach_to=ego);actors.append(ls);sensors.append(ls)
        ls.listen(lambda e:markings.append({'frame':e.frame,'types':[str(v.type) for v in e.crossed_lane_markings]}))
        cb=lib.find('sensor.camera.rgb')
        for k,v in {'image_size_x':'640','image_size_y':'360','sensor_tick':'1'}.items():cb.set_attribute(k,v)
        cam=world.spawn_actor(cb,carla.Transform(carla.Location(x=-12,z=7),carla.Rotation(pitch=-18)),attach_to=ego);actors.append(cam);sensors.append(cam)
        def on_image(im):
            try:images.put_nowait(im)
            except queue.Full:pass
        cam.listen(on_image)
        npc_ids=[a.id for a in world.get_actors().filter('vehicle.*') if a.id!=ego.id] # evaluation sink only
        traffic.send('start')
        if args.initial_speed:
            ego.apply_control(carla.VehicleControl());ego.set_target_velocity(et.get_forward_vector()*(args.initial_speed/3.6));controller.reference=args.initial_speed/3.6
        frame=traffic.send('tick')['frame'];packet=wait_scan(packets,frame)
        tick=0;plan=None;occ=None;tracks=[];loopstart=time.monotonic();last_station=cfg['ego_start_s']
        print('READY',str(folder),flush=True)
        with (folder/'telemetry.jsonl').open('w') as log,(folder/'evaluation-only.jsonl').open('w') as evallog:
            while sim<args.seconds and not (folder/'stop.requested').exists():
                wall=time.perf_counter();snapshot=world.get_snapshot();tr=ego.get_transform();v=ego.get_velocity();wp=m.get_waypoint(tr.location)
                speed=math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)
                if wp.road_id!=cfg['road'] or wp.lane_id not in cfg['lanes']:raise RuntimeError('Left configured driving corridor')
                delta=tr.location-wp.transform.location;lat=delta.dot(wp.transform.get_right_vector())
                state={'speed':speed,'lane':wp.lane_id,'s':wp.s,'lateral_error':lat}
                lanes=local_lanes(lane_cache,tr,wp.s)
                if tick%4==0:
                    packet=wait_scan(packets,frame) if tick else packet;begin=time.perf_counter()
                    occ=perception(packet,tr);tracks=tracker.update(occ,np.array([tr.location.x,-tr.location.y]),-math.radians(tr.rotation.yaw))
                    permissions={}
                    for ln in cfg['lanes']:
                        q=m.get_waypoint_xodr(cfg['road'],ln,wp.s)
                        if q is None:continue
                        for to in (ln-1,ln+1):
                            mark=q.left_lane_marking if to>ln else q.right_lane_marking
                            flag=carla.LaneChange.Left if to>ln else carla.LaneChange.Right
                            permissions[(ln,to)]=bool(int(q.lane_change)&int(flag)) and 'Solid' not in str(mark.type)
                    plan=planner.plan(occ,tracks,state,lanes,permissions,snapshot.timestamp.elapsed_seconds)
                    # End-of-route slowdown is navigation information, not future actors.
                    remaining=road_end-wp.s
                    if remaining<max(160.,speed*5.):
                        plan['target_speed']=min(plan['target_speed'],math.sqrt(8*max(0,remaining-25.)));plan['reason']='route_end'
                    planning.append((time.perf_counter()-begin)*1000)
                    evallog.write(json.dumps(evaluation_snapshot(world,snapshot,ego,npc_ids,m,tracks))+'\n')
                    if tick%20==0:
                        np.savez_compressed(folder/'occ'/f'{frame:08d}.npz',xyz=occ.xyz,tags=occ.tags,cleared=np.packbits(occ.cleared),shape=occ.cleared.shape,world2ego=np.array(tr.get_inverse_matrix()),timestamp=occ.timestamp)
                    for a,b in zip(plan['trajectory'][::2][:-1],plan['trajectory'][::2][1:]):
                        world.debug.draw_line(tr.transform(carla.Location(x=a[0],y=-a[1],z=.3)),tr.transform(carla.Location(x=b[0],y=-b[1],z=.3)),thickness=.05,color=carla.Color(20,240,100),life_time=.25)
                else:
                    # Re-express the selected road path in the CURRENT ego frame.
                    if planner.maneuver:
                        move=planner.maneuver
                        plan['reference']=planner.blend(lanes[move['from']],lanes[move['to']],wp.s,move['start_s'],move['length'],move.get('start_y',0.))
                    else:plan['reference']=lanes[plan['target_lane']]
                if snapshot.timestamp.elapsed_seconds-occ.timestamp>.35:plan.update(target_speed=0.,emergency=True,reason='sensor_stale')
                u=controller.update(speed,plan,DT,steering_scale(physics.steering_curve,speed*3.6))
                ego.apply_control(carla.VehicleControl(throttle=u['throttle'],brake=u['brake'],steer=u['steer']))
                frame=traffic.send('tick')['frame'];tick+=1;sim=tick*DT
                row={'frame':frame,'simulation_s':sim,'speed_kmh':speed*3.6,'target_speed_kmh':plan['target_speed']*3.6,
                     's':wp.s,'lane':wp.lane_id,'target_lane':plan['target_lane'],'lateral_error_m':lat,'reason':plan['reason'],
                     'control':u,'tracks':[{k:t[k] for k in ['id','x','y','vx','vy','hits','age_s','observed']} for t in tracks],
                     'visibility':plan['visibility'],'lane_traffic':plan['lane_traffic'],'events':plan['events'] if tick%4==1 else [],
                     'planning_ms':planning[-1],'occupied_voxels':len(occ.xyz),'ray_count':occ.ray_count,
                     'sensor_frame':occ.frame,'collisions':len(collision),'marking_events':len(markings)}
                telemetry.append(row);log.write(json.dumps(row)+'\n');last_station=wp.s
                if tick%4==0:
                    log.flush();evallog.flush();write(folder/'status.json',{k:v for k,v in row.items() if k not in ['tracks','control']})
                nowtr=ego.get_transform();world.get_spectator().set_transform(carla.Transform(nowtr.location-nowtr.get_forward_vector()*13+carla.Location(z=7),carla.Rotation(pitch=-18,yaw=nowtr.rotation.yaw)))
                while not images.empty():
                    im=images.get_nowait();im.save_to_disk(str(folder/'frames'/f'{im.frame:08d}.png'))
                if collision:raise RuntimeError('Ego collision detected')
                if not footprint_inside(m,nowtr,ego.bounding_box,cfg['road'],cfg['lanes']):raise RuntimeError('Ego footprint outside legal lane union')
                if any('Solid' in t for e in markings for t in e['types']):raise RuntimeError('Solid marking crossed')
                if tick%100==0:print('PROGRESS',json.dumps({k:row[k] for k in ['simulation_s','speed_kmh','target_speed_kmh','s','lane','target_lane','reason','planning_ms','visibility']}),flush=True)
                if not args.no_pace:time.sleep(max(0,DT-(time.perf_counter()-wall)))
            report.update(status='completed',completion_reason='duration_reached' if sim>=args.seconds else 'user_stop',
                          loop_wall_seconds=time.monotonic()-loopstart,realtime_factor=sim/max(.001,time.monotonic()-loopstart))
            # Controlled end, included separately from the driving evaluation interval.
            brake_start=ego.get_location();brake_initial=ego.get_velocity();braking=[]
            ego.apply_control(carla.VehicleControl(brake=.9))
            for brake_tick in range(200):
                traffic.send('tick');vel=ego.get_velocity()
                brake_speed=math.sqrt(vel.x**2+vel.y**2+vel.z**2)
                braking.append({'time_s':(brake_tick+1)*DT,'speed_mps':brake_speed,'distance_m':ego.get_location().distance(brake_start)})
                if brake_speed<.1:break
            report['end_brake_measurement']={'initial_speed_mps':math.sqrt(brake_initial.x**2+brake_initial.y**2+brake_initial.z**2),'samples':braking,
                'note':'Ego-only brake command after scored interval; not NPC manipulation or traffic response evidence.'}
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        print(traceback.format_exc(),flush=True)
        if ego:
            try:ego.apply_control(carla.VehicleControl(brake=1))
            except Exception:pass
    finally:
        report.update(simulated_seconds=sim,frames=len(telemetry),collisions=collision,lane_markings=markings,
                      wall_seconds=time.monotonic()-startwall,
                      distance_m=(telemetry[-1]['s']-cfg['ego_start_s']) if telemetry else 0,
                      max_speed_kmh=max((r['speed_kmh'] for r in telemetry),default=0),
                      planning_ms_p50_p95_max=np.percentile(planning,[50,95,100]).tolist() if planning else [],
                      events=planner.events if 'planner' in locals() else [])
        write(folder/'report.json',report);write(folder/'status.json',report)
        # Short RPC timeout prevents a closed CARLA window leaving minutes of cleanup.
        c.set_timeout(2)
        for s in sensors:
            try:s.stop()
            except Exception:pass
        if traffic:traffic.close()
        for a in reversed(actors):
            try:a.destroy()
            except Exception:pass
        if world and original:
            try:world.apply_settings(original)
            except Exception:pass
        print('DONE',json.dumps({k:report[k] for k in ['status','simulated_seconds','distance_m','max_speed_kmh','collisions','events']}),flush=True)
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
