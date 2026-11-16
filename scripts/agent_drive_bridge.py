"""Step-and-observe CARLA bridge: agent chooses speed/lane; ego never uses TM.

No autonomous maneuver policy or schedule exists here. The simulation is paused
between externally supplied commands. Map/actor ground truth is privileged input.
Experimental pure pursuit + speed PI executes a continuous lane reference.
"""
import argparse
import hashlib
import json
import math
import queue
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import carla
from probe_highway_traffic_expert import state, gaps, inside_driving_union, change_allowed

ROOT=Path(__file__).resolve().parents[1]
POINTER=ROOT/'logs/agent-drive/current.json'
DT=.05


def steering_scale(curve, speed_kmh):
    points=sorted({float(p.x):float(p.y) for p in curve}.items())
    if not points:raise ValueError('Missing steering curve')
    if speed_kmh<=points[0][0]:return points[0][1]
    for (x0,y0),(x1,y1) in zip(points,points[1:]):
        if speed_kmh<=x1:return y0+(y1-y0)*(speed_kmh-x0)/(x1-x0)
    return points[-1][1]


def read_json(path):
    for attempt in range(100):
        try:return json.loads(path.read_text())
        except PermissionError:
            if attempt==99:raise
            time.sleep(.01)


def atomic(path,value):
    temp=path.with_suffix('.pending')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    for attempt in range(50):
        try:temp.replace(path);break
        except PermissionError:
            if attempt==49:raise
            time.sleep(.002)


def send(args):
    folder=Path(json.loads(POINTER.read_text())['folder'])
    current=read_json(folder/'observation.json')
    command={'id':current['command_id']+1,'lane':args.lane,'speed_kmh':args.speed,
             'seconds':args.seconds,'reason':args.reason,'close':args.close}
    atomic(folder/'command.json',command)
    start=time.monotonic()
    while time.monotonic()-start<60:
        result=read_json(folder/'observation.json')
        if result['command_id']>=command['id'] or result.get('status')=='failed':
            print(json.dumps(result),flush=True);return
        time.sleep(.1)
    raise RuntimeError('Command response timed out; inspect session, do not advance another client.')


def run(scenario_name=None):
    folder=ROOT/'logs/agent-drive'/datetime.now().strftime('%Y%m%d-%H%M%S')
    folder.mkdir(parents=True);(folder/'frames').mkdir();(folder/'clock').mkdir()
    scenario=None
    if scenario_name:
        scenario=json.loads((ROOT/'configs/agent_complex_scenarios.json').read_text())['cases'][scenario_name]
        atomic(folder/'scenario.json',dict(name=scenario_name,**scenario))
    road_id=(scenario or {}).get('road_id',1157)
    allowed_lanes=tuple((scenario or {}).get('lanes',[-2,-3,-4]))
    max_speed=(scenario or {}).get('max_speed_kmh',80)
    independent=(scenario or {}).get('independent_traffic',False)
    traffic_process=None
    def traffic_wait(filename):
        deadline=time.monotonic()+120
        while not (folder/filename).exists():
            if (folder/'traffic-error.json').exists():raise RuntimeError((folder/'traffic-error.json').read_text())
            if traffic_process.poll() is not None:raise RuntimeError('Independent traffic exited before ready')
            if time.monotonic()>deadline:raise RuntimeError('Independent traffic handshake timeout')
            time.sleep(.1)
        return read_json(folder/filename)
    (folder/'helper-source.py').write_bytes((ROOT/'scripts/probe_highway_traffic_expert.py').read_bytes())
    (folder/'bridge-source.py').write_bytes(Path(__file__).read_bytes())
    atomic(folder/'source-sha256.json',{'bridge':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    atomic(POINTER,{'folder':str(folder)})
    c=carla.Client('127.0.0.1',2100);c.set_timeout(120)
    w=c.get_world()
    if w.get_actors().filter('vehicle.*'):raise RuntimeError('Existing vehicles; refusing to take over')
    if not w.get_map().name.endswith('/Town12'): w=c.load_world('Town12')
    m=w.get_map();original=w.get_settings();actors=[];sensors=[];cars={};tm=None
    command_id=0;simtime=0;rows=[];decisions=[];collisions=[];interventions=[]
    applied_events=[];markings=[]
    speed_ref=0.;accel_ref=0.;integral=0.;last_steer=0.;lane=(scenario or {}).get('ego_lane',-3);plan=None
    report={'mode':'agent_decisions_synchronous_pauses','ego_autopilot':False,'status':'starting',
            'scenario':scenario_name or 'baseline',
            'limitations':['Privileged map and actor state, not observed OCC','Experimental tracking controller, not calibrated expert','Agent decisions are not real-time','No Flow/world model loaded']}
    try:
        settings=w.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=DT;w.apply_settings(settings)
        lib=w.get_blueprint_library()
        placements=scenario['vehicles'] if scenario else [('ego',-3,150,0),('lead',-3,210,45),('left_front',-2,300,80),('left_rear',-2,30,50),('right_car',-4,220,55)]
        for name,l,s,speed in placements:
            bp=lib.find('vehicle.tesla.model3');bp.set_attribute('role_name','hero' if name=='ego' else name)
            bp.set_attribute('color','255,255,255' if name=='ego' else ('200,30,30' if name=='lead' else '30,70,200'))
            tr=m.get_waypoint_xodr(road_id,l,s).transform;tr.location.z+=.4
            car=w.spawn_actor(bp,tr);actors.append(car);cars[name]=car;car.apply_control(carla.VehicleControl(brake=1))
            if name=='ego':w.get_spectator().set_transform(carla.Transform(tr.location+carla.Location(z=10),carla.Rotation(pitch=-25,yaw=tr.rotation.yaw)))
        for _ in range(25):w.tick(90)
        ego=cars['ego'];ego.set_autopilot(False)
        physics=ego.get_physics_control()
        vehicle_transform=ego.get_transform()
        forward=vehicle_transform.get_forward_vector()
        # UE5 wheel.location is world-space metres (verified against actor pose).
        wheel_x=sorted((wheel.location-vehicle_transform.location).dot(forward) for wheel in physics.wheels)
        wheelbase=((wheel_x[-1]+wheel_x[-2])-(wheel_x[0]+wheel_x[1]))/2
        rear_x=(wheel_x[0]+wheel_x[1])/2
        max_angle=math.radians(max(wheel.max_steer_angle for wheel in physics.wheels))
        if not 2<wheelbase<4 or not .2<max_angle<1.5:raise RuntimeError('Unexpected vehicle geometry')
        report['vehicle_geometry']={'wheel_actor_x_m':wheel_x,'wheelbase_m':wheelbase,'rear_axle_x_m':rear_x,'max_steer_angle_rad':max_angle}
        if independent:
            worker=ROOT/'scripts/independent_background_traffic.py'
            (folder/'background-source.py').write_bytes(worker.read_bytes())
            traffic_process=subprocess.Popen([sys.executable,str(worker),str(folder)],creationflags=subprocess.CREATE_NO_WINDOW,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            identities=traffic_wait('traffic-spawned.json')['actors']
            for _ in range(25):w.tick(60)
            for name,actor_id in identities.items():
                cars[name]=w.get_actor(actor_id)
                if cars[name] is None:raise RuntimeError('Independent NPC missing after settlement')
            atomic(folder/'traffic-start.json',{'start':True})
            traffic_wait('traffic-active.json')
            report['traffic_policy']='External setup process; CARLA TM autonomous following/lane changes; no NPC event schedule'
            report['observation_contract']='Current map/actor state within 180m; privileged state proxy, no occlusion; no NPC targets, routes or seed supplied'
        else:
            tm=c.get_trafficmanager(8100);tm.set_synchronous_mode(True);tm.set_random_device_seed(20261115)
            for name,_,_,speed in placements:
                if name=='ego':continue
                tm.auto_lane_change(cars[name],False);tm.set_desired_speed(cars[name],speed);tm.distance_to_leading_vehicle(cars[name],15);cars[name].set_autopilot(True,8100)
        if (scenario or {}).get('initialize_velocity',False):
            for name,_,_,speed in placements:
                cars[name].apply_control(carla.VehicleControl())
                cars[name].set_target_velocity(cars[name].get_transform().get_forward_vector()*(speed/3.6))
            speed_ref=placements[0][3]/3.6
            report['initial_velocity_setup']='Once before recorded motion; no later velocity override'
        report['steering_curve']=[{'speed':v.x,'scale':v.y} for v in physics.steering_curve]
        for name,car in cars.items():
            sensor=w.spawn_actor(lib.find('sensor.other.collision'),carla.Transform(),attach_to=car);actors.append(sensor);sensors.append(sensor)
            sensor.listen(lambda e,n=name:collisions.append({'frame':e.frame,'vehicle':n,'other':e.other_actor.type_id}))
        sensor=w.spawn_actor(lib.find('sensor.other.lane_invasion'),carla.Transform(),attach_to=ego);actors.append(sensor);sensors.append(sensor)
        sensor.listen(lambda e:markings.append({'frame':e.frame,'markings':[str(a.type) for a in e.crossed_lane_markings]}))
        images=queue.Queue(maxsize=8)
        bp=lib.find('sensor.camera.rgb');bp.set_attribute('image_size_x','640');bp.set_attribute('image_size_y','360');bp.set_attribute('sensor_tick','.5')
        camera=w.spawn_actor(bp,carla.Transform(carla.Location(x=-10,z=6),carla.Rotation(pitch=-18)),attach_to=ego);actors.append(camera);sensors.append(camera)
        def on_image(im):
            try:images.put_nowait(im)
            except queue.Full:pass
        camera.listen(on_image)
        rules=json.loads((ROOT/'configs/highway_tm_expert_probe.json').read_text())['decision_rules']
        clock_sequence=0
        def advance_world():
            nonlocal clock_sequence
            if not independent:return w.tick(30)
            clock_sequence+=1
            atomic(folder/'clock'/f'request-{clock_sequence:06d}.json',{'id':clock_sequence})
            deadline=time.monotonic()+45
            while time.monotonic()<deadline:
                if traffic_process.poll() is not None:raise RuntimeError('Traffic clock process exited')
                done=folder/'clock'/f'done-{clock_sequence:06d}.json'
                if done.exists():
                    result=read_json(done)
                    if result['id']==clock_sequence and w.get_snapshot().frame>=result['frame']:return result['frame']
                time.sleep(.001)
            raise RuntimeError('Independent traffic clock timeout')
        def observed_states():
            ss={name:state(car,m) for name,car in cars.items()}
            if independent:
                e=ss['ego']
                ss={n:o for n,o in ss.items() if n=='ego' or math.hypot(o['x']-e['x'],o['y']-e['y'])<=180}
            return ss
        def observe(status='paused',error=None):
            ss=observed_states();e=ss['ego']
            result={'command_id':command_id,'status':status,'simulation_s':round(simtime,2),
                    'ego':{'speed_kmh':round(e['speed']*3.6,2),'lane':e['lane'],'s':round(e['s'],2),'lateral_error_m':round(e['lateral_error'],3)},
                    'target_lane':lane,'lane_change_active':plan is not None,
                    'lanes':{str(l):[{k:(round(v,2) if isinstance(v,float) else v) for k,v in g.items()} for g in gaps(ss,l)] for l in allowed_lanes},
                    'other_speeds_kmh':{name:round(v['speed']*3.6,2) for name,v in ss.items() if name!='ego'},
                    'collision_count':len(collisions),'guard_interventions':len(interventions),'ego_autopilot':False,'folder':str(folder)}
            result['scenario']=scenario_name or 'baseline'
            result['lane_marking_events']=len(markings)
            if not independent:result['observed_background_events']=applied_events
            else:
                result['visible_tracks']={n:{'lane':o['lane'],'lateral_error_m':round(o['lateral_error'],3),'yaw_deg':round(o['yaw'],2)} for n,o in ss.items() if n!='ego'}
            if error:result['error']=error
            atomic(folder/'observation.json',result)
            return result
        observe();print(json.dumps({'ready':str(folder),'geometry':report['vehicle_geometry']}),flush=True)
        idle=time.monotonic()
        with (folder/'telemetry.jsonl').open('w',encoding='utf-8') as log:
            while True:
                if time.monotonic()-idle>240:raise RuntimeError('No agent command for 240 seconds; ended paused session')
                cp=folder/'command.json'
                if not cp.exists():time.sleep(.1);continue
                cmd=read_json(cp)
                if cmd['id']<=command_id:time.sleep(.1);continue
                command_id=cmd['id'];idle=time.monotonic()
                if cmd.get('close'):
                    observe('closed');break
                if not 0<=cmd['speed_kmh']<=max_speed or not 0<cmd['seconds']<=3 or cmd['lane'] not in allowed_lanes:
                    observe(error='Rejected out-of-bounds command');continue
                if scenario and simtime+cmd['seconds']>scenario['max_sim_seconds']+.01:
                    observe(error='Command exceeds bounded scenario time; close or shorten');continue
                ss=observed_states();e=ss['ego'];wp=m.get_waypoint(ego.get_location())
                if cmd['lane']!=lane:
                    if plan is not None or abs(cmd['lane']-lane)!=1:
                        observe(error='Finish current adjacent-lane transition first');continue
                    ok,why,checks=change_allowed(ss,wp,cmd['lane'],rules)
                    if not ok:observe(error='Lane change guard: '+why);continue
                    plan={'from':lane,'to':cmd['lane'],'start_s':e['s'],'length':max(45,e['speed']*5)};lane=cmd['lane']
                decisions.append(dict(cmd,simulation_s=simtime,observed_ego=e,observed_actors=ss,
                                      observed_gaps={str(l):gaps(ss,l) for l in allowed_lanes},ego_autopilot=False))
                atomic(folder/'decisions.json',decisions)
                start=time.monotonic()
                for step in range(round(cmd['seconds']/DT)):
                    if scenario:
                        for index,event in enumerate(scenario['events']):
                            if simtime+1e-6>=event['time'] and not any(a['index']==index for a in applied_events):
                                if event['actor']=='ego':raise RuntimeError('Scenario may not command ego')
                                tm.set_desired_speed(cars[event['actor']],event['speed_kmh'])
                                applied_events.append(dict(event,index=index,applied_at=simtime))
                                atomic(folder/'background-events.json',applied_events)
                    ss=observed_states();e=ss['ego'];tr=ego.get_transform()
                    target_v=cmd['speed_kmh']/3.6
                    emergency=False
                    # Only an emergency backstop; nominal following/changes are
                    # chosen externally by the agent, never by this bridge.
                    for g in gaps(ss,e['lane']):
                        if g['side']=='front' and (g['gap']<5 or (g['ttc'] is not None and g['ttc']<2)):
                            emergency=True
                    if emergency:
                        target_v=0;interventions.append({'time':simtime,'reason':'gap_or_ttc_emergency'})
                    want_a=max(-2,min(1.5,(target_v-speed_ref)/DT))
                    accel_ref+=max(-DT,min(DT,want_a-accel_ref))
                    if abs(target_v-speed_ref)<abs(accel_ref*DT):speed_ref=target_v;accel_ref=0
                    else:speed_ref=max(0,speed_ref+accel_ref*DT)
                    error=speed_ref-e['speed'];integral=max(-3,min(3,integral+error*DT))
                    effort=.10+.17*accel_ref+.22*error+.04*integral
                    throttle=max(0,min(.65,effort));brake=max(0,min(.4,-effort))
                    if target_v==0 and e['speed']<.3:throttle=0;brake=.3;integral=0
                    if emergency:throttle=0;brake=.6
                    lookahead=max(6,min(22,e['speed']*.8+5))
                    target_s=e['s']+lookahead
                    destination=m.get_waypoint_xodr(road_id,lane,target_s)
                    if destination is None:raise RuntimeError('Route end')
                    target=destination.transform.location
                    if plan:
                        u=max(0,min(1,(target_s-plan['start_s'])/plan['length']))
                        blend=10*u**3-15*u**4+6*u**5
                        origin=m.get_waypoint_xodr(road_id,plan['from'],target_s).transform.location
                        target=origin+(target-origin)*blend
                        if e['s']>=plan['start_s']+plan['length'] and e['lane']==lane and abs(e['lateral_error'])<.35:plan=None
                    rear=tr.transform(carla.Location(x=rear_x))
                    delta=target-rear;yaw=math.radians(tr.rotation.yaw)
                    local_y=-math.sin(yaw)*delta.x+math.cos(yaw)*delta.y
                    steer=math.atan2(2*wheelbase*local_y,max(1,delta.x*delta.x+delta.y*delta.y))/max_angle
                    if (scenario or {}).get('compensate_steering_curve',False):
                        steer/=max(.05,steering_scale(physics.steering_curve,e['speed']*3.6))
                    steer=max(last_steer-.6*DT,min(last_steer+.6*DT,steer));steer=max(-.5,min(.5,steer));last_steer=steer
                    ego.apply_control(carla.VehicleControl(throttle=throttle,brake=brake,steer=steer))
                    frame=advance_world();simtime+=DT
                    tr=ego.get_transform();w.get_spectator().set_transform(carla.Transform(tr.location-tr.get_forward_vector()*11+carla.Location(z=7),carla.Rotation(pitch=-20,yaw=tr.rotation.yaw)))
                    row={'frame':frame,'simulation_s':simtime,'command_id':command_id,'actors':{n:state(car,m) for n,car in cars.items()},'speed_reference_mps':speed_ref,'target_lane':lane,'control':{'throttle':throttle,'brake':brake,'steer':steer},'ego_autopilot':False,'emergency_guard':emergency}
                    row['lane_change_active']=plan is not None
                    if independent:
                        row['background_applied_controls']={}
                        for name,car in cars.items():
                            if name=='ego':continue
                            control=car.get_control()
                            row['background_applied_controls'][name]={'throttle':control.throttle,'brake':control.brake,'steer':control.steer}
                    rows.append(row);log.write(json.dumps(row)+'\n')
                    while not images.empty():
                        im=images.get_nowait();im.save_to_disk(str(folder/'frames'/f'{im.frame:08d}.png'))
                    if collisions:raise RuntimeError('Collision detected; diagnostic stopped')
                    for x in (-ego.bounding_box.extent.x,ego.bounding_box.extent.x):
                        for y in (-ego.bounding_box.extent.y,ego.bounding_box.extent.y):
                            if not inside_driving_union(m,tr.transform(ego.bounding_box.location+carla.Location(x=x,y=y)),road_id,allowed_lanes):raise RuntimeError('Footprint outside inspected driving corridor')
                    if row['actors']['ego']['s']>(scenario['route_end_s'] if scenario else 1600):raise RuntimeError('Reached bounded route end')
                    time.sleep(max(0,start+(step+1)*DT-time.monotonic()))
                log.flush();observe();idle=time.monotonic()
        report.update(status='completed',simulated_seconds=simtime,decisions=len(decisions),collisions=collisions,guard_interventions=interventions,
                      lane_marking_events=markings,background_events=applied_events,
                      maximum_speed_kmh=max((r['actors']['ego']['speed']*3.6 for r in rows),default=0),
                      final_state=state(ego,m),lanes_visited=sorted(set(r['actors']['ego']['lane'] for r in rows)))
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc(),simulated_seconds=simtime,collisions=collisions,guard_interventions=interventions,lane_marking_events=markings)
        atomic(folder/'observation.json',{'command_id':command_id,'status':'failed','error':repr(error),'folder':str(folder)})
        raise
    finally:
        for sensor in sensors:
            try:sensor.stop()
            except Exception:pass
        if traffic_process is not None:
            atomic(folder/'traffic-stop.json',{'stop':True})
            try:traffic_process.wait(timeout=30)
            except subprocess.TimeoutExpired:report['traffic_cleanup_error']='Traffic worker did not exit in 30s'
        for actor in reversed(actors):
            try:actor.destroy()
            except Exception:pass
        if tm:tm.set_synchronous_mode(False)
        w.apply_settings(original);atomic(folder/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--send',action='store_true');p.add_argument('--lane',type=int,default=-3)
    p.add_argument('--speed',type=float,default=0);p.add_argument('--seconds',type=float,default=2)
    p.add_argument('--reason',default='');p.add_argument('--close',action='store_true')
    p.add_argument('--scenario',choices=list(json.loads((ROOT/'configs/agent_complex_scenarios.json').read_text())['cases']));args=p.parse_args()
    if args.send:send(args)
    else:run(args.scenario)
