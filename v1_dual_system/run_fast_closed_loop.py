"""20 Hz CARLA keyboard pilot. Use the CARLA interpreter; PyTorch runs in its own worker."""
import argparse
import ctypes
import json
import math
import os
from multiprocessing import shared_memory
import random
import subprocess
import time

import carla
import cv2
import numpy as np

from .collect_fast_data import Expert, Road, choose_path, neighbor, path_from, spawn, wrap
from .legacy_controller import Controller, LaneCrossing, current_plan, lidar_clearance, path_gap
from .curve_controller import CurveController,lane_pose_now
from .diagnose_gt_controller import gentle_curve_segment, in_gentle_arc
from .sensors import CAL_ID, DT, LOG, ROOT, Sensors, bev, local_client, model_rgb, state, write_json

TITLE="v1 fast driver"
from .mouse_controls import MouseControls
from .navigation import RouteEnd, anchors, FILM_STATIONS
from .diagnose_gt_controller import Reference


def receive(worker):
    for line in worker.stdout:
        if line.startswith("V1 "):
            return json.loads(line[3:])
    raise RuntimeError(f"GPU worker exited ({worker.poll()}); see worker.err in the run folder")


def straight_segment(map_,seed,both_sides=False,signal_locations=None):
    starts=[w for w in map_.generate_waypoints(25) if not w.is_junction and (neighbor(w,True) or neighbor(w,False))]
    if both_sides:starts=[w for w in starts if neighbor(w,True) and neighbor(w,False)]
    random.Random(seed).shuffle(starts)
    candidates=[]
    for start in starts:
        heading=start.transform.rotation.yaw
        w=start
        length=0
        while length<1000:
            options=w.next(2)
            if not options:break
            # Follow precisely the same continuation as path_from, including a straight road
            # crossing an OpenDRIVE junction boundary. Stop as soon as the road bends.
            w=min(options,key=lambda p:(abs(wrap(p.transform.rotation.yaw-w.transform.rotation.yaw)),p.id))
            if abs(wrap(w.transform.rotation.yaw-heading))>=1.:break
            if both_sides and not (neighbor(w,True) and neighbor(w,False)):break
            length+=2
        if length:candidates.append((length,start))
    for length,start in sorted(candidates,key=lambda entry:-entry[0]):
        route=path_from(start)
        if signal_locations is None:return route,length
        route.extend(path_from(route[-1],distance=1400)[1:])
        xy=np.array([[p.transform.location.x,p.transform.location.y] for p in route])
        if len(signal_locations):
            distance=np.linalg.norm(xy[:,None]-signal_locations,axis=2).min(axis=1)
            near=np.flatnonzero(distance<130.)
            if len(near):route=route[:near[0]]
        if min(length,2*(len(route)-1))>=500:
            return route,min(length,2*(len(route)-1))
    raise ValueError("No suitable straight segment found")


def pilot_window_open():
    try:
        return cv2.getWindowProperty(TITLE,cv2.WND_PROP_VISIBLE)>=1
    except cv2.error:
        return False


def window_click(panel,action):
    """Post a real Win32 click to this pilot's HighGUI image child without moving the cursor."""
    from ctypes import wintypes
    ui=ctypes.windll.user32
    ui.FindWindowW.argtypes=[wintypes.LPCWSTR,wintypes.LPCWSTR]
    ui.FindWindowW.restype=wintypes.HWND
    ui.FindWindowExW.argtypes=[wintypes.HWND,wintypes.HWND,wintypes.LPCWSTR,wintypes.LPCWSTR]
    ui.FindWindowExW.restype=wintypes.HWND
    ui.GetWindowThreadProcessId.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.DWORD)]
    ui.PostMessageW.argtypes=[wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM]
    hwnd=ui.FindWindowW(None,TITLE)
    child=ui.FindWindowExW(hwnd,None,"HighGUI class",None) if hwnd else None
    pid=wintypes.DWORD()
    ui.GetWindowThreadProcessId(child,ctypes.byref(pid))
    if not child or pid.value!=os.getpid():raise RuntimeError("Pilot image window unavailable")
    rect=wintypes.RECT()
    ui.GetClientRect.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.RECT)]
    if not ui.GetClientRect(child,ctypes.byref(rect)):raise OSError("Cannot locate pilot controls")
    x,y,w,h=panel.bounds[action]
    width,height=panel.image_size
    pos=(int((x+w/2)*rect.right/width) | (int((y+h/2)*rect.bottom/height)<<16))
    for message,flags in ((0x201,1),(0x202,0)):
        if not ui.PostMessageW(child,message,flags,pos):raise OSError("Mouse delivery failed")


class Keyboard:
    def __init__(self):
        self.target=0
        self.intent=0
        self.until=0.
        self.heartbeat=time.monotonic()

    def poll(self,key,sim_time):
        now=time.monotonic()
        if now-self.heartbeat>3 or sim_time>=self.until:self.intent=0
        self.heartbeat=now
        if key in ("a","d"):
            self.intent=1 if key=="a" else 2
            self.until=sim_time+3.
        if key=="w":self.target=min(120,self.target+10)
        if key=="s":self.target=max(0,self.target-10)
        if key=="k":self.intent,self.until=0,0.


def run(args):
    weights=ROOT/"artifacts/v1-dual-system"
    city=args.simulator_version=="0.10.0" if args.simulator_version else args.town=="Town10HD_Opt"
    if args.unified and (args.plan_seconds!=4 or (city and not args.interactive) or args.blue_cruise):
        raise ValueError("Unified Stage 2 requires CARLA 0.9.15 and the four-second plan")
    if args.blue_cruise:
        args.film=True
        if args.plan_seconds!=4:raise ValueError("Blue cruise requires four-second green preview")
    if args.film:args.spatial=True
    if args.unified:weights=weights/"unified-ipm"
    elif args.spatial:
        args.navigation=True
        weights=weights/("fork-film" if args.film else "spatial")
    elif args.navigation:weights=weights/'navigation'
    elif args.dagger:weights=weights/"dagger"
    elif args.road_roi:weights=weights/"road-roi"
    elif args.lane_supervised:weights=weights/"lane-supervised"
    elif args.corrective:weights=weights/"corrective"
    if args.model_directory:
        if not args.unified:raise ValueError("A model bundle requires the unified driver")
        weights=ROOT/args.model_directory
    ckpt=weights/"best.ckpt"
    if not ckpt.is_file(): raise FileNotFoundError(ckpt)
    if args.road_roi and not args.dagger:
        from .road_gate import check_gate
        check_gate(ckpt)
    folder=LOG/("closed-loop-"+time.strftime("%Y%m%d-%H%M%S")+"-"+args.scenario)
    folder.mkdir(parents=True,exist_ok=False)
    client=local_client(args.port)
    if args.unified and not city and (client.get_server_version()!=client.get_client_version() or
                         not client.get_server_version().startswith("0.9.15")):
        raise RuntimeError("Unified evaluation requires matching CARLA 0.9.15 client and server")
    if city and (not client.get_server_version().startswith("0.10.0") or
                 client.get_server_version()!=client.get_client_version()):
        raise RuntimeError(f"Town10 requires matching 0.10.0 builds: {client.get_server_version()} / {client.get_client_version()}")
    world=client.get_world()
    if not world.get_map().name.endswith("/"+args.town):world=client.load_world(args.town)
    map_=world.get_map()
    straight_length=None
    single_key=args.scenario in ("lane-left","lane-right")
    keyboard_lane=args.scenario=="lane" or single_key
    if city or args.scenario=="city":
        from .town10_demo import city_route,add_traffic
        route=city_route(map_,args.seed,long_demo=args.unified)
    elif args.scenario=="dagger":
        from .dagger_collect import collection_route
        route,blocked=collection_route(map_,args.town,args.seed)
    elif args.scenario=="comfort":
        from .diagnose_gt_controller import comfort_curve_segment,in_comfort_arc
        route,straight_length=comfort_curve_segment(map_)
    elif args.scenario=="gentle":route,straight_length=gentle_curve_segment(map_)
    elif args.scenario=="fork":
        if args.town!="Town04":raise ValueError("Registered fork requires Town04")
        branch_id=1162 if args.fork_right else 1174
        wp=map_.get_waypoint_xodr(50,6,130.)
        route=[wp]
        while wp.road_id==50 and len(route)<100:
            options=wp.next(2.)
            wp=next((w for w in options if w.road_id==branch_id),options[0])
            route.append(wp)
        if wp.road_id!=branch_id:raise ValueError(f"Missing registered branch {branch_id}")
        route.extend(path_from(wp)[1:])
        straight_length=290.
    elif args.scenario=="traffic-right-fork":
        from .collect_spatial import forks
        if args.town!="Town04":raise ValueError("The curved right fork is registered in Town04")
        approach,branches=dict(forks(map_))[(39,-4)]
        if branches[-1].road_id!=1191:raise ValueError("Registered right branch changed")
        route=approach+path_from(branches[-1])
    elif args.scenario in ("traffic","follow50","gap-accel","gap-yield"):
        signals=None
        if args.scenario=="traffic":
            lights=list(world.get_actors().filter("traffic.traffic_light*"))
            locations=[light.get_location() for light in lights]
            locations.extend(wp.transform.location for light in lights for wp in light.get_affected_lane_waypoints())
            signals=np.array([(p.x,p.y) for p in locations]).reshape(-1,2)
        route,initial_straight=straight_segment(map_,args.seed,both_sides=True,signal_locations=signals)
        if args.scenario=="traffic":
            print(f"{args.town} highway: {initial_straight} m straight with adjacent lanes, road {route[0].road_id}, {2*(len(route)-1)} m checked against {len(lights)} signals (130 m exclusion)",flush=True)
    elif args.scenario=="straight":route,straight_length=straight_segment(map_,args.seed)
    elif keyboard_lane:route,straight_length=straight_segment(map_,args.seed,both_sides=True)
    else:route=choose_path(map_,random.Random(args.seed),"left",args.scenario=="curve")
    use_route_as_planned=args.scenario=="traffic" or (args.scenario=="city" and args.unified)
    road=Road(route if use_route_as_planned else route + path_from(route[-1],distance=1400)[1:])
    navigator=Reference(road.path) if args.navigation or args.unified else None
    if city and args.unified:
        print(f"{args.town}: {navigator.arc[-1]:.0f} m planned on connected driving lanes",flush=True)
    if args.unified:
        from .controller import CorridorTraffic,LaneRequest,UnifiedController
        from .unified_grid import lidar_raster,ground_plane,corridor
        from .unified_data import route_polyline
        contacts=CorridorTraffic();request=LaneRequest()
    handoff=LaneCrossing()
    panel=MouseControls() if args.interactive or args.keyboard_test else None
    old=world.get_settings()
    cfg=world.get_settings()
    cfg.synchronous_mode,cfg.fixed_delta_seconds,cfg.no_rendering_mode=True,DT,False
    actors,rows,hits=[],[],[]
    rig=collision=worker=slot=recorder=chase=None
    evidence=None
    highway_traffic=args.scenario in ("traffic","traffic-right-fork")
    incident_recording=args.record_chase and highway_traffic
    traffic_log=None
    recording_started=False
    manager=None
    city_traffic=[]
    rollout_finished=False
    end_reason="duration"
    try:
        world.apply_settings(cfg)
        if city and args.interactive:
            daylight=carla.WeatherParameters.ClearNoon
            daylight.sun_altitude_angle=32.
            daylight.sun_azimuth_angle=(route[0].transform.rotation.yaw+65.)%360.
            world.set_weather(daylight)
        if incident_recording:
            client.start_recorder(str(folder/"world.rec"),True)
            recording_started=True
            traffic_log=(folder/"traffic.jsonl").open("w",buffering=1)
        vehicle=sorted(bp.id for bp in world.get_blueprint_library().filter("vehicle.*lincoln*"))[0] if city else "vehicle.tesla.model3"
        ego=spawn(world,route[0],vehicle)
        actors.append(ego)
        if args.unified:unified_control=UnifiedController(ego.type_id)
        if city or highway_traffic or args.scenario=="city":
            from .town10_demo import add_traffic
            manager=client.get_trafficmanager(args.port+5)
            manager.set_synchronous_mode(True)
            traffic_seed=args.traffic_seed if args.traffic_seed is not None else random.SystemRandom().randrange(2**31)
            manager.set_random_device_seed(traffic_seed)
            city_traffic=add_traffic(world,map_,route,manager,actors,traffic_seed,highway=highway_traffic,highway_cruise=args.scenario=="traffic",count=args.npc_count)
            print(f"Traffic Manager: {len(city_traffic)} NPC vehicles, seed {traffic_seed}",flush=True)
            if incident_recording:
                write_json(folder/"traffic-setup.json",dict(seed=traffic_seed,route_seed=args.seed,
                           checkpoint=str(ckpt),plan_seconds=args.plan_seconds,ego=ego.id,
                           actors={a.id:a.type_id for a in actors}))
            for _ in range(12):world.tick()
        if args.scenario=="dagger":
            from .dagger_collect import DaggerRecorder
            world.tick()
            others=[a.type_id for a in world.get_actors() if a.id!=ego.id and
                    a.type_id.startswith(("vehicle.","walker.pedestrian."))]
            if others:raise RuntimeError(f"DAgger scene contains other traffic: {others}")
            recorder=DaggerRecorder(args.town,route,blocked,map_,args.seed,args.collect_frames,
                                    "93f9184ff76908f77ce89dc016ce4ff6f35064bd68630c53c085ce0ede4a6541")
        width=ego.bounding_box.extent.y*2
        dims=[ego.bounding_box.extent.x*2,width]
        physics=ego.get_physics_control()
        wheels=physics.wheels
        # Wheel positions may be rotated in world space.
        if city:
            front,rear=wheels[0].location,wheels[2].location
            wheelbase=math.hypot(front.x-rear.x,front.y-rear.y)
            nominal_mkz=wheelbase==0 and vehicle=="vehicle.lincoln.mkz"
            if nominal_mkz:
                # The 0.10.0 Chaos RPC sends zero wheel locations for this sedan.
                wheelbase=2.85
        else:
            wheelbase=math.hypot(wheels[0].position.x-wheels[2].position.x,wheels[0].position.y-wheels[2].position.y)/100
        angle=math.radians(wheels[0].max_steer_angle)
        steer_curve=np.array([(point.x,point.y) for point in physics.steering_curve])
        if not 1.5<wheelbase<4 or not .25<angle<1.6:raise ValueError((wheelbase,angle))
        npc=expert=None
        traffic_fixture=args.scenario in ("follow50","gap-accel","gap-yield")
        if args.scenario in ("braking","blind") or traffic_fixture:
            wp=route[18] if args.scenario in ("braking","follow50") else neighbor(route[0],True)
            if wp is None:raise RuntimeError("No adjacent driving lane for blind-spot test")
            if args.scenario in ("blind","gap-yield"):
                previous=wp.previous(5. if args.scenario=="blind" else 9.)
                if not previous:raise RuntimeError("No rear approach for blind-spot test")
                wp=previous[0]
            npc=spawn(world,wp,"vehicle.audi.tt")
            actors.append(npc)
            expert=Expert(Road(path_from(wp)))
        ego.apply_control(carla.VehicleControl(brake=1))
        collision=world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),carla.Transform(),attach_to=ego)
        def record_collision(event):
            other=event.other_actor
            contact=(event.frame,other.type_id)
            # Persist the contact before the simulation loop can start tearing down actors.
            with (folder/"collisions.jsonl").open("a",encoding="utf-8") as log:
                log.write(json.dumps(dict(frame=event.frame,actor_id=other.id,actor_type=contact[1]))+"\n")
            hits.append(contact)
        collision.listen(record_collision)
        rig=Sensors(world,ego)
        if args.crash_evidence:
            from .crash_evidence import CrashEvidence
            evidence=CrashEvidence(world,ego,folder,ckpt)
        if panel and not args.chase_view:
            cv2.namedWindow(TITLE,cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(TITLE,panel.click)
        if args.chase_view:
            from .chase_view import ChaseView
            chase=ChaseView(world,ego,folder,args.record_chase,incident_recording)
            cv2.namedWindow(TITLE,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(TITLE,1280,720)
            if panel:cv2.setMouseCallback(TITLE,panel.click)
            loading=np.zeros((chase.size[1],chase.size[0],3),np.uint8)
            cv2.putText(loading,"Starting chase camera...",(32,65),cv2.FONT_HERSHEY_SIMPLEX,.8,(230,230,230),1,cv2.LINE_AA)
            cv2.imshow(TITLE,panel.draw(loading,0) if panel else loading)
            cv2.waitKey(1)
        for _ in range(12):world.tick()
        channels=3 if args.unified else 6
        size=channels*360*640+(9 if args.unified else 3)*128*128
        slot=shared_memory.SharedMemory(create=True,size=size)
        rgb_buf=np.ndarray((channels,360,640),np.uint8,slot.buf)
        bev_buf=np.ndarray((3,128,128),np.uint8,slot.buf,offset=rgb_buf.nbytes)
        if args.unified:
            nav_buf=np.ndarray((128,128),np.uint8,slot.buf,offset=rgb_buf.nbytes+bev_buf.nbytes)
            occ_buf=np.ndarray((5,128,128),np.uint8,slot.buf,offset=rgb_buf.nbytes+bev_buf.nbytes+nav_buf.nbytes)
        with (folder/"worker.err").open("w") as err:
            cache=ROOT/".cache/v1-dual-system"
            cache.mkdir(parents=True,exist_ok=True)
            env=os.environ.copy()
            env.update({k:str(cache) for k in ("TEMP","TMP","MIOPEN_USER_DB_PATH","MIOPEN_CUSTOM_CACHE_DIR")})
            worker=subprocess.Popen([str(ROOT/".venv-train/Scripts/python.exe"),"-u","-m","v1_dual_system.unified_inference" if args.unified else "v1_dual_system.inference",
                                     slot.name,str(ckpt)]+(["--chase-view"] if chase else []),cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                    stderr=err,text=True,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
            hello=receive(worker)
            monitor=subprocess.Popen(["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass","-File",
                str(ROOT/"v1_dual_system/watch_gpu.ps1"),str(hello["pid"]),str(folder/"gpu.jsonl")],
                creationflags=subprocess.CREATE_NO_WINDOW)
            ready=receive(worker)
            if not ready["ready"]:raise RuntimeError("GPU worker startup")
            frames={}
            poses={}
            last_used=-1
            unplanned=0
            first=None
            lane_keys=[]
            target_road=None
            target_width=0.
            brake_at=None
            real_warmup=None
            keys=Keyboard()
            if city and args.unified and args.interactive:keys.target=40
            if traffic_fixture:
                keys.target=100
                ego.apply_control(carla.VehicleControl())
                ego.set_target_velocity(ego.get_transform().get_forward_vector()*(50/3.6))
                npc_start=68. if args.scenario=="gap-yield" else 50.
                npc.set_target_velocity(npc.get_transform().get_forward_vector()*(npc_start/3.6))
            if args.unified and args.scenario=="braking":
                keys.target=100
                for vehicle_at_speed in (ego,npc):
                    vehicle_at_speed.apply_control(carla.VehicleControl())
                    forward=vehicle_at_speed.get_transform().get_forward_vector()
                    vehicle_at_speed.set_target_velocity(forward*(55./3.6))
            if highway_traffic and (args.interactive or args.scenario=="traffic-right-fork"):keys.target=40
            if args.rolling_start:
                if args.scenario not in ("straight","comfort","traffic"):raise ValueError("Rolling start requires a registered highway segment")
                forward=ego.get_transform().get_forward_vector()
                ego.apply_control(carla.VehicleControl())
                start_kmh=80 if args.scenario=="traffic" else 120
                ego.set_target_velocity(forward*(start_kmh/3.6))
                keys.target=100 if args.scenario=="traffic" else 120
            if args.initial_target is not None:keys.target=args.initial_target
            control=None
            if not args.unified:
                control=CurveController(spatial=args.spatial) if (args.navigation or args.lane_supervised or args.road_roi or args.dagger) and not args.baseline_controller else Controller()
            last=None
            deadline=time.perf_counter()
            def pace():
                nonlocal deadline
                deadline+=DT
                left=deadline-time.perf_counter()
                if left>0:time.sleep(left)
                elif left < -DT:deadline=time.perf_counter()
            with (folder/"frames.jsonl").open("w") as log:
                for tick in range(round(args.seconds/DT)):
                    if args.stop_file and os.path.isfile(args.stop_file):
                        end_reason="launcher_stop"
                        break
                    start=time.perf_counter()
                    if npc:
                        speed=ego.get_velocity().length()*3.6 if args.scenario=="blind" else 75
                        if traffic_fixture:
                            speed=95. if args.scenario=="follow50" and tick*DT>=22. else npc_start
                        if args.unified and args.scenario=="braking":speed=55.
                        npc_cmd=expert.control(npc,speed)
                        if args.scenario=="braking":
                            loc=npc.get_location()
                            forward,side,_=(np.asarray(ego.get_transform().get_inverse_matrix())@[loc.x,loc.y,loc.z,1])[:3]
                            if (brake_at is None and (not args.unified or tick*DT>=1.) and 12<forward<45 and abs(side)<2.
                                and ego.get_velocity().length()>14 and npc.get_velocity().length()>11):
                                brake_at=tick*DT
                            if brake_at is not None:npc_cmd=carla.VehicleControl(brake=1)
                        npc.apply_control(npc_cmd)
                    frame=world.tick()
                    tr=ego.get_transform()
                    if traffic_log:
                        snapshot=world.get_snapshot()
                        local=np.array(tr.get_inverse_matrix())
                        observed=[]
                        for actor in actors[1:]:
                            other=snapshot.find(actor.id)
                            if other is None:continue
                            loc=other.get_transform().location
                            x,y,z,_=local@[loc.x,loc.y,loc.z,1.]
                            observed.append((actor.id,x,y,z,other.get_velocity().length()*3.6))
                        traffic_log.write(json.dumps(dict(frame=snapshot.frame,sim_s=tick*DT,
                            vehicles_id_forward_right_up_kmh=observed))+"\n")

                    if first is None:first=frame
                    poses[frame]=(np.array(tr.get_matrix()),state(ego))
                    cutoff=max(first,frame-15)
                    for store in (frames,poses):
                        for old_frame in list(store):
                            if old_frame<cutoff:del store[old_frame]
                    for observed,rgb,scan in rig.ready_pairs(cutoff):
                        frames[observed]=(model_rgb(rgb),scan)
                    available=[f for f in frames if f>last_used and f in poses and (args.unified or (f-6 in frames and f-6 in poses))]
                    if not available:
                        if rows:unplanned+=1
                        pace()
                        continue
                    f=last_used=max(available)
                    rgb,scan=frames[f]
                    pose,ego_state=poses[f]
                    if args.unified:
                        rgb_buf[:]=rgb.transpose(2,0,1)
                        bev_buf[:]=lidar_raster(scan)
                        station,route_offset,_=navigator.project(pose)
                        if navigator.arc[-1]-station<65:
                            end_reason="navigation_route_complete";break
                        poly=route_polyline(navigator,pose,station)
                        plane=ground_plane(bev_buf).tolist()
                    else:
                        old_rgb,old_scan=frames[f-6]
                        rgb_buf[:3]=old_rgb.transpose(2,0,1)
                        rgb_buf[3:]=rgb.transpose(2,0,1)
                        bev_buf[:]=bev(scan,old_scan,pose,poses[f-6][0])
                    try:
                        nav_input=dict(nav_anchors=anchors(navigator,pose,FILM_STATIONS if args.film else (30.,60.)).tolist()) if navigator and not args.unified else {}
                    except RouteEnd:
                        end_reason="navigation_route_complete"
                        break
                    if real_warmup is None:
                        if args.unified:
                            nav_buf[:]=corridor(poly,request.target_lane_shift)
                            packet=dict(state=ego_state[:1],intent=0,plane=plane)
                        else:
                            packet=dict(state=ego_state,target=0,intent=0,blind=False,dimensions=dims,**nav_input)
                        worker.stdin.write(json.dumps(packet)+"\n")
                        worker.stdin.flush()
                        real_warmup=receive(worker)["inference_ms"]
                        deadline=time.perf_counter()
                        continue
                    key=""
                    extra_key=""
                    scheduled=""
                    target_error=target_road.locate(tr) if target_road is not None else None
                    if not args.interactive:
                        setpoint=40 if args.scenario=="traffic-right-fork" else 100 if traffic_fixture else 50 if city else 80 if keyboard_lane else 120
                        if keys.target<setpoint:scheduled="w"
                        previous=rows[-1]["sim_s"] if rows else -1.
                        events={"blind":[(1.3,"a"),(2.5,"a"),(4.,"a")]}
                        if args.scenario in ("gap-accel","gap-yield"):events[args.scenario]=[(1.,"a")]
                        for when,action in events.get(args.scenario,[]):
                            if previous<when<=tick*DT:scheduled=action
                        if keyboard_lane and len(lane_keys)<(1 if single_key else 2) and ego_state[0]*3.6>=50:
                            left=not lane_keys and args.scenario!="lane-right"
                            eligible=tick*DT>=(8. if not lane_keys else lane_keys[0]+9.)
                            lane=map_.get_waypoint(tr.location)
                            if eligible and not lane.is_junction and neighbor(lane,left):
                                scheduled="a" if left else "d"
                                lane_keys.append(tick*DT)
                                sibling=neighbor(lane,left)
                                target_width=sibling.lane_width
                                target_road=Road(path_from(sibling,distance=800))
                                target_error=target_road.locate(tr)
                        if (args.release_at_crossing and not args.navigation and keys.intent and not scheduled and
                                target_error is not None and abs(target_error)<=target_width/2):
                            scheduled="k"
                    if args.interactive or args.keyboard_test:
                        if not chase:
                            view=rgb[:,:,::-1].copy()
                            caption=f"{ego_state[0]*3.6:.1f} km/h   target {keys.target}   {('KEEP','LEFT','RIGHT')[keys.intent]}"
                            if rows and rows[-1]["veto"]:caption+="   VETO"
                            cv2.putText(view,caption,(14,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),4,cv2.LINE_AA)
                            cv2.putText(view,caption,(14,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
                            cv2.imshow(TITLE,panel.draw(view,keys.target,bool(rows and rows[-1]["veto"])) if panel else view)
                        if args.keyboard_test and scheduled and scheduled!="k":window_click(panel,scheduled)
                        code=cv2.waitKey(1)&255
                        if code==27 or not pilot_window_open():
                            end_reason="keyboard_exit"
                            break
                        key="k" if args.keyboard_test and scheduled=="k" else panel.poll()
                        if args.keyboard_test and scheduled and key!=scheduled:
                            raise RuntimeError(f"Keyboard receipt mismatch: sent {scheduled}, received {key!r}")
                        if args.keyboard_test and key and not scheduled:
                            extra_key,key=key,""
                    else:
                        key=scheduled
                    keys.poll(key,tick*DT)
                    if args.navigation and key in ("a","d"):keys.until=math.inf
                    requested_intent=keys.intent
                    if args.unified:
                        width_lane=road.path[navigator.source_index[navigator.index]].lane_width
                        sign=1 if key=="a" else -1 if key=="d" else 0
                        if sign and not request.pending and neighbor(map_.get_waypoint(tr.location),sign>0) is None:sign=0
                        control_dt=(frame-rows[-1]["frame"])*DT if rows else DT
                        traffic=contacts.observe(scan,poly,request.target_lane_shift,width_lane,ego_state[0],control_dt,plane)
                        crossed_before=request.crossed and request.mode=="EXECUTE"
                        ceiling=(120.5 if keys.target==120 else keys.target)/3.6
                        shift,cmd,ceiling=request.update(sign,tick*DT,control_dt,route_offset,width_lane,ego_state[0],ceiling,traffic)
                        nav_buf[:]=corridor(poly,shift)
                        requested_intent=(1 if request.pending>0 else 2 if request.pending<0 else 0)
                        blind=request.pending!=0 and request.mode!="EXECUTE"
                        packet=dict(state=ego_state[:1],intent=cmd,plane=plane)
                    else:
                        blind,_=lidar_clearance(scan,requested_intent,width)
                        packet=dict(state=ego_state,target=keys.target,intent=requested_intent,
                                    blind=blind,dimensions=dims,sim_s=tick*DT,**nav_input)
                    worker.stdin.write(json.dumps(packet)+"\n")
                    worker.stdin.flush()
                    reply=receive(worker)
                    # Occupancy veto already used its original 1/2/3-second timestamps.
                    reply["xy"]=reply["xy"][:10*args.plan_seconds]
                    switched=bool(args.unified and request.mode=="EXECUTE" and request.crossed and not crossed_before)
                    if args.unified:reply["veto"]=blind
                    if reply.get("handoff") or (args.navigation and not args.spatial and handoff.update(reply["intent"],reply["lane_pose"][0])):
                        keys.intent=0
                        keys.until=0.
                        switched=True
                    blue_keep=(args.blue_keep or args.navigation) and reply["intent"]==0
                    if args.blue_keep:reply["blue_keep"]=blue_keep
                    if args.blue_cruise:reply["blue_keep"]=blue_keep
                    elif args.navigation:reply["fused_keep"]=blue_keep
                    if chase:
                        layers=None
                        if args.unified:
                            from .chase_view import unified_layers
                            layers=unified_layers(rgb,bev_buf,nav_buf,occ_buf,plane)
                        view=chase.update(f,pose,reply,ego_state[0]*3.6,layers)
                        if view is not None:
                            displayed=panel.draw(view,keys.target,reply["veto"]) if panel else view
                            cv2.imshow(TITLE,displayed)
                            if layers is not None and chase.shown==20:
                                cv2.imwrite(str(folder/"chase-bev.jpg"),displayed)
                        if not (args.interactive or args.keyboard_test) and cv2.waitKey(1)&255==27:
                            end_reason="keyboard_exit";break
                    if args.capture_commands and key in ("a","d"):
                        # Physics is paused between ticks. The observation still precedes
                        # application of this frame's requested steering and acceleration.
                        from .command_probe import save_observation
                        save_observation(folder,key,frame,f,map_,pose,rgb_buf,bev_buf,ego_state,
                                         keys.target,reply)
                    if reply["veto"] and not args.unified:
                        # A rejected keypress expires here. Reconsider only a new driver command.
                        keys.intent=0
                        keys.until=0.
                    xy=current_plan(reply["xy"],pose,poses[frame][0],(frame-f)*DT,extrapolate=args.plan_seconds==4)
                    control_dt=(frame-rows[-1]["frame"])*DT if rows else DT
                    speed=ego.get_velocity().length()
                    scaled_angle=angle*np.interp(speed*3.6,steer_curve[:,0],steer_curve[:,1])
                    fork_input={}
                    if args.film and blue_keep and not args.blue_cruise:
                        nx,ny=nav_input["nav_anchors"][:2]
                        current=np.linalg.solve(poses[frame][0],pose)@np.array([nx,-ny,0.,1.])
                        fork_input["nav30"]=(current[0],-current[1])
                    if args.unified:
                        lead=traffic[0,1]
                        if request.mode=="EXECUTE":
                            target_lead=traffic[request.pending,1]
                            if target_lead and (lead is None or target_lead.x<lead.x):lead=target_lead
                        throttle,brake,steer,diagnostic=unified_control.step(xy,poses[frame][0],speed,ceiling,lead,
                            occ_buf,dims,wheelbase,scaled_angle,control_dt,sensor_plan=reply["xy"],
                            scan=scan,plane=plane,nav=nav_buf,poly=poly,shift=shift,sensor_pose=pose,keep=cmd==0 and request.mode!="EXECUTE")
                        xy=unified_control.path
                        gap=max(0.,lead.x-dims[0]/2) if lead else 120.
                        if gap < 1 and not (folder/"near-contact.npz").exists():
                            np.savez_compressed(folder/"near-contact.npz",scan=scan,polyline=poly,
                                shift=request.target_lane_shift,width=width_lane,frame=f)
                        reply["risk"]=diagnostic["urgent"]
                    else:
                        gap=path_gap(scan,reply["xy"],width)
                        throttle,brake,steer=control.step(xy,ego.get_velocity().length(),keys.target,gap,
                                                      reply["risk"],wheelbase,scaled_angle,control_dt,**fork_input,
                                                      **({"lane_pose":lane_pose_now(reply["lane_pose"],pose,poses[frame][0]),"fused":args.navigation and not args.blue_cruise,"handoff":switched,
                                                         "blue_cruise":args.blue_cruise} if blue_keep else {}))
                    ego.apply_control(carla.VehicleControl(throttle=throttle,brake=brake,steer=steer))
                    wp=map_.get_waypoint(tr.location)
                    center=wp.transform.location
                    right=wp.transform.get_right_vector()
                    lateral=(tr.location.x-center.x)*right.x+(tr.location.y-center.y)*right.y
                    real_npc=None
                    if npc:
                        loc=npc.get_location()
                        ahead,side,_=(np.asarray(tr.get_inverse_matrix())@[loc.x,loc.y,loc.z,1])[:3]
                        real_npc=dict(ahead_m=float(ahead),left_m=float(-side),kmh=npc.get_velocity().length()*3.6)
                    elapsed=time.perf_counter()-start
                    row=dict(frame=frame,sensor_frame=f,source_age_ms=(frame-f)*50,sim_s=tick*DT,
                             kmh=ego_state[0]*3.6,target=keys.target,key=key,intent=requested_intent,
                             scheduled_key=scheduled,extra_test_window_key=extra_key,
                             executed=reply["intent"],veto=reply["veto"],blind=blind,risk=reply["risk"],
                             footprint_score=reply["footprint_score"],requested_score=reply["requested_footprint_score"],
                             plan_1s_xy=xy[9].tolist(),plan_points=len(xy),
                             gap=gap,steer=steer,throttle=throttle,brake=brake,lateral_m=lateral,
                             road=wp.road_id,lane=wp.lane_id,route_offset=road.locate(tr),npc=real_npc,
                             curvature=road.curvature(),yaw_rate=ego_state[2],
                             world_xy=[tr.location.x,tr.location.y],heading_deg=tr.rotation.yaw,
                             inference_ms=reply["inference_ms"],work_ms=elapsed*1000,
                             peak_allocated_mib=reply["peak_allocated_mib"],peak_reserved_mib=reply["peak_reserved_mib"],
                             period_ms=(start-last)*1000 if last is not None else None)
                    row[f"plan_{args.plan_seconds}s_xy"]=xy[-1].tolist()
                    rows.append(row)
                    if chase:row["chase_frames_shown"]=chase.shown
                    if len(rows)==1 and chase:
                        print(f"Driving session: {folder.name}",flush=True)
                    if args.scenario=="fork":
                        physical=map_.get_waypoint(tr.location,project_to_road=False,lane_type=carla.LaneType.Driving)
                        row["physical_road"]=physical.road_id if physical is not None else None
                    if args.unified:
                        row["map_curvature"]=row["curvature"]
                        row.update(steering_source="unified",maneuver=request.mode,pending=request.pending,
                                   target_lane_shift=request.target_lane_shift,commanded_shift=shift,
                                   route_width_m=width_lane,physical_lane_width_m=wp.lane_width,
                                   visual_handoff=switched,**diagnostic)
                    if args.blue_cruise:
                        row.update(steering_source="blue" if blue_keep else "green",blue_weight=float(blue_keep))
                    elif args.film:row.update(fork_delta_m=control.fork_delta,blue_weight=control.blue_weight)
                    if args.navigation:row.update(nav_anchors=nav_input["nav_anchors"],visual_handoff=switched)
                    if args.blue_keep or args.navigation:
                        row.update(blue_keep=blue_keep,lane_pose=reply["lane_pose"])
                    if isinstance(control,CurveController):
                        row["curve_cap_kmh"]=control.curve_limit_kmh
                    if target_error is not None:
                        row.update(target_lane_error_m=target_error,target_lane_width_m=target_width)
                    if key in ("a","d"):
                        row["key_plan_xy_flu_m"]=[reply["xy"][i] for i in (4,9,19,39) if i<len(reply["xy"])]
                        print(json.dumps({"key":key,"sensor_frame":f,"executed":reply["intent"],
                                          ("traj_xy_0p5_1_2s_m" if args.plan_seconds==2 else "traj_xy_0p5_1_2_4s_m"):row["key_plan_xy_flu_m"]}),flush=True)
                    if args.scenario=="gentle":
                        row["in_gentle_arc"]=in_gentle_arc(road.path[road.index])
                        row["on_designated_arc"]=in_gentle_arc(wp)
                    if args.scenario=="comfort":
                        row["in_gentle_arc"]=in_comfort_arc(road.path[road.index])
                        row["on_designated_arc"]=in_comfort_arc(wp)
                    log.write(json.dumps(row)+"\n")
                    if evidence:
                        evidence.record(row,rgb_buf,bev_buf,nav_buf,plane,reply["xy"],pose,poses[frame][0],poly,scan)
                    if incident_recording or tick%20==0:log.flush()
                    last=start
                    if hits:
                        end_reason="collision";break
                    if recorder and recorder.add(f,pose,rgb_buf,bev_buf,ego_state,keys.target):
                        end_reason="collection_complete";break
                    if straight_length is not None and road.index*2>=straight_length-10:
                        end_reason=args.scenario+"_segment_complete"
                        break
                    pace()
            rollout_finished=True
    finally:
        if rollout_finished:
            write_json(folder/"termination.json",dict(end=end_reason,collision_events=hits))
        if evidence:evidence.finish(hits)
        if recording_started:client.stop_recorder()
        if traffic_log:traffic_log.close()
        if recorder:recorder.close(rollout_finished)
        if worker:
            if worker.poll() is None:
                try:worker.stdin.write("null\n");worker.stdin.flush();worker.wait(timeout=10)
                except (BrokenPipeError,subprocess.TimeoutExpired):worker.terminate();worker.wait()
        if slot:slot.close();slot.unlink()
        if chase:chase.close()
        if rig:rig.close()
        if collision:collision.stop();collision.destroy()
        if manager:
            # Unregister this run's NPCs while the TM is still synchronized.
            # Its asynchronous worker must not race destruction of live vehicles.
            for npc in actors[1:1+len(city_traffic)]:
                npc.set_autopilot(False,manager.get_port())
            world.tick()
        for a in reversed(actors):a.destroy()
        if manager:manager.set_synchronous_mode(False)
        world.apply_settings(old)
        if args.interactive or args.keyboard_test or chase:cv2.destroyAllWindows()
    if end_reason=="launcher_stop" and (not rows or (chase and not chase.shown)):return
    if chase and not chase.shown:raise RuntimeError("No synchronized third-person frames")
    if not rows:raise RuntimeError("No closed-loop commands applied")
    periods=[r["period_ms"] for r in rows if r["period_ms"] is not None]
    speeds=np.array([r["kmh"] for r in rows])
    source="mouse buttons" if args.interactive else "targeted Win32 mouse clicks" if args.keyboard_test else "scripted key replay"
    metrics=dict(calibration=CAL_ID,source=source,checkpoint=str(ckpt),
                 scenario=args.scenario,town=args.town,seed=args.seed,route_start=int(route[0].id),
                 controller="UnifiedController" if args.unified else type(control).__name__,unified=args.unified,spatial=args.spatial,film=args.film,rolling_start=args.rolling_start,navigation=args.navigation,hip_graph=ready.get("graph",False),blue_keep=args.blue_keep,blue_cruise=args.blue_cruise,release_at_crossing=args.release_at_crossing,
                 verified_straight_length_m=straight_length if args.scenario=="straight" or single_key else None,
                 frames=len(rows),end=end_reason,collision_events=hits,
                 mean_kmh=float(speeds.mean()),max_kmh=float(speeds.max()),frames_119_121=int(sum((speeds>=119)&(speeds<=121))),
                 lateral_p95_m=float(np.percentile(np.abs([r["lateral_m"] for r in rows]),95)),
                 wall_hz=1000/np.mean(periods) if periods else None,
                 inference_p95_ms=float(np.percentile([r["inference_ms"] for r in rows],95)),
                 deadline_misses=sum(r["work_ms"]>50 for r in rows),veto_frames=sum(r["veto"] for r in rows),
                 missed_sensor_ticks=unplanned,
                 startup_s=ready["startup_s"],
                 stationary_first_sensor_inference_ms=real_warmup,
                 peak_reserved_mib=max(r["peak_reserved_mib"] for r in rows))
    points=np.array([r["world_xy"] for r in rows])
    if highway_traffic or args.scenario=="city":
        metrics.update(traffic_vehicles=city_traffic,traffic_seed=traffic_seed,rendered_frames=chase.shown if chase else 0)
    if city:
        metrics.update(server=client.get_server_version(),client=client.get_client_version(),vehicle=vehicle,
                       wheelbase_m=wheelbase,wheelbase_source="nominal MKZ" if nominal_mkz else "wheel locations",
                       traffic_vehicles=city_traffic,traffic_seed=traffic_seed,
                       rendered_frames=chase.shown if chase else 0)
    metrics["distance_m"]=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
    metrics["collisions_per_km"]=len(hits)/max(.000001,metrics["distance_m"]/1000)
    metrics["road_departure_max_m"]=max(abs(r["route_offset"]) for r in rows)
    metrics["inference_max_ms"]=max(r["inference_ms"] for r in rows)
    metrics["simulated_s"]=rows[-1]["sim_s"]-rows[0]["sim_s"]
    high=[118<=r["kmh"]<=122 and abs(r["lateral_m"])<.5 for r in rows]
    longest=streak=0
    for i,good in enumerate(high):
        if i and rows[i]["frame"]!=rows[i-1]["frame"]+1:streak=0
        streak=streak+1 if good else 0
        longest=max(longest,streak)
    metrics["high_speed_centered_s"]=sum(high)*DT
    metrics["continuous_high_speed_centered_s"]=longest*DT
    if args.scenario in ("cruise","straight","curve","gentle","comfort"):
        route_errors=np.abs([r["route_offset"] for r in rows])
        high_speed=(speeds>=118)&(speeds<=122)
        qualifying=route_errors[high_speed]
        longest=streak=0
        for i,aligned in enumerate(high_speed & (route_errors<.5)):
            if i and rows[i]["frame"]!=rows[i-1]["frame"]+1:streak=0
            streak=streak+1 if aligned else 0
            longest=max(longest,streak)
        metrics["designated_lane"]=dict(
            all_p95_m=float(np.percentile(route_errors,95)),
            high_speed_frames=len(qualifying),
            high_speed_p95_m=float(np.percentile(qualifying,95)) if len(qualifying) else None,
            high_speed_max_m=float(qualifying.max()) if len(qualifying) else None,
            longest_high_speed_centered_s=longest*DT)
    if args.scenario=="fork":
        gate=map_.get_waypoint_xodr(50,6,.01).transform
        origin=[gate.location.x,gate.location.y]
        direction=gate.get_forward_vector()
        progress=(points-origin)@np.array([direction.x,direction.y])
        eligible=[r for r,s in zip(rows,progress) if 0<=s<=40]
        wrong_ids={1161,1162,1174}-{branch_id}
        metrics["fork_window"]=dict(target_road=branch_id,frames=len(eligible),complete=bool(progress.max()>=40),
            target_frames=sum(r["physical_road"]==branch_id for r in eligible),
            wrong_frames=sum(r["physical_road"] in wrong_ids for r in eligible))
        metrics["designated_branch_frames"]=sum(r["physical_road"]==branch_id for r in rows)
        metrics["off_driving_lane_frames"]=sum(r["physical_road"] is None for r in rows)
        metrics["wrong_branch_frames"]=sum(r["physical_road"] in wrong_ids for r in rows)
    if (args.navigation or args.unified) and keyboard_lane:
        events=[i for i,r in enumerate(rows) if r["key"] in ("a","d")]
        metrics["lane_handoffs"]=[]
        for j,start in enumerate(events):
            window=rows[start:events[j+1] if j+1<len(events) else len(rows)]
            crossing=next((i for i,r in enumerate(window) if r["visual_handoff"]),None)
            actual=next((i for i,r in enumerate(window) if "target_lane_error_m" in r and
                         abs(r["target_lane_error_m"])<=.5*r["target_lane_width_m"]),None)
            latency=None
            if crossing is not None:
                streak=0
                for i in range(max(crossing,actual or 0)+1,len(window)):
                    r=window[i]
                    if window[i-1]["frame"]+1!=r["frame"]:streak=0
                    good=r["executed"]==0 and abs(r.get("target_lane_error_m",float("inf")))<.5
                    streak=streak+1 if good else 0
                    if streak>=20:
                        latency=window[i-19]["sim_s"]-window[crossing]["sim_s"]
                        break
            metrics["lane_handoffs"].append(dict(direction=window[0]["key"],crossing_s=(
                window[crossing]["sim_s"] if crossing is not None else None),seconds_to_stable_center=latency,
                physical_crossing_s=window[actual]["sim_s"] if actual is not None else None,
                from_physical_crossing_s=(window[crossing]["sim_s"]+latency-window[actual]["sim_s"]
                    if actual is not None and latency is not None else None),
                within_2p5s=(actual is not None and latency is not None and
                    window[crossing]["sim_s"]+latency-window[actual]["sim_s"]<=2.5)))
    if args.scenario in ("gentle","comfort"):
        arc=[r for r in rows if r["in_gentle_arc"]]
        driven=[r for r in rows if r["on_designated_arc"]]
        high_arc=[r for r in arc if 118<=r["kmh"]<=122]
        metrics["gentle_arc"]=dict(frames=len(arc),high_speed_frames=len(high_arc),
            physical_frames=len(driven),physical_high_speed_frames=sum(118<=r["kmh"]<=122 for r in driven),
            max_kmh=max((r["kmh"] for r in arc),default=None),
            high_speed_p95_m=float(np.percentile([abs(r["route_offset"]) for r in high_arc],95)) if high_arc else None)
    metrics["control_period_p95_ms"]=float(np.percentile(periods,95)) if periods else None
    metrics["median_sensor_age_ms"]=float(np.median([r["source_age_ms"] for r in rows]))
    for shape,lo,hi in [("straight",0,.001),("curve",.001,float("inf"))]:
        section=[r for r in rows if lo<=r["curvature"]<hi and r["kmh"]>=80 and not r["intent"]]
        metrics[shape+"_above_80"]=dict(frames=len(section),lateral_rms_m=(
            float(np.sqrt(np.mean(np.square([r["lateral_m"] for r in section])))) if section else None))
    if args.scenario=="braking":
        before=[r for r in rows if brake_at is not None and brake_at-.5<=r["sim_s"]<brake_at]
        after=[r for r in rows if brake_at is not None and brake_at<=r["sim_s"]<=brake_at+6]
        metrics["lead_brake_event"]=dict(
            triggered_at_s=brake_at,
            ego_approach_kmh=before[-1]["kmh"] if before else None,
            lead_approach_kmh=before[-1]["npc"]["kmh"] if before else None,
            initial_center_gap_m=before[-1]["npc"]["ahead_m"] if before else None,
            min_forward_center_gap_m=min((r["npc"]["ahead_m"] for r in after),default=None),
            ego_min_next_6s_kmh=min((r["kmh"] for r in after),default=None),
            lead_min_next_6s_kmh=min((r["npc"]["kmh"] for r in after),default=None),
            max_abs_lane_error_next_6s_m=max((abs(r["lateral_m"]) for r in after),default=None),
            max_ego_brake=max((r["brake"] for r in after),default=None))
    metrics["key_events"]=[]
    for i,r in enumerate(rows):
        if r["key"] not in ("a","d"):continue
        future=rows[i:i+121]
        observed=future[-1]["route_offset"]-r["route_offset"]
        metrics["key_events"].append(dict(key=r["key"],sim_s=r["sim_s"],travel_left_m=observed,
            entry_kmh=r["kmh"],target_kmh=r["target"],
            veto_frames=sum(s["veto"] for s in future),from_lane=r["lane"],to_lane=future[-1]["lane"],
            npc_at_command=r["npc"]))
    monitor.wait(timeout=20)
    if (folder/"gpu.jsonl").exists():
        samples=[json.loads(s) for s in (folder/"gpu.jsonl").read_text(encoding="utf-8-sig").splitlines()]
        samples=[s for s in samples if s["instances"]]
        if samples:
            for category in ("dedicated","shared"):
                metrics["wddm_peak_"+category+"_mib"]=max(s[category+"_bytes"] for s in samples)/2**20
    write_json(folder/"summary.json",metrics)
    print(json.dumps(metrics),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--town",default="Town04")
    p.add_argument("--simulator-version",choices=("0.9.15","0.10.0"))
    p.add_argument("--model-directory")
    p.add_argument("--stop-file")
    p.add_argument("--npc-count",type=int,choices=range(41))
    p.add_argument("--initial-target",type=int,choices=range(121))
    p.add_argument("--port",type=int,default=2176)
    p.add_argument("--traffic-seed",type=int)
    p.add_argument("--seed",type=int,default=30961)
    p.add_argument("--plan-seconds",type=int,choices=(2,4),default=4)
    p.add_argument("--seconds",type=float,default=65)
    p.add_argument("--scenario",choices=("cruise","straight","curve","gentle","comfort","fork","lane","lane-left","lane-right","braking","blind","dagger","city","traffic","traffic-right-fork","follow50","gap-accel","gap-yield"),default="cruise")
    p.add_argument("--collect-frames",type=int,default=350)
    p.add_argument("--baseline-controller",action="store_true",help="Replay the pre-preview controller")
    p.add_argument("--corrective",action="store_true")
    p.add_argument("--lane-supervised",action="store_true")
    p.add_argument("--road-roi",action="store_true")
    p.add_argument("--dagger",action="store_true")
    p.add_argument("--release-at-crossing",action="store_true",help="Test operator sends KEEP on entering the target lane")
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--interactive",action="store_true")
    mode.add_argument("--keyboard-test",action="store_true")
    p.add_argument("--capture-commands",action="store_true",help="Save A/D observations and reference lanes for offline comparison")
    p.add_argument("--chase-view",action="store_true")
    p.add_argument("--record-chase",action="store_true")
    p.add_argument("--rolling-start",action="store_true")
    p.add_argument("--fork-right",action="store_true")
    p.add_argument("--unified",action="store_true")
    p.add_argument("--crash-evidence",action="store_true")
    p.add_argument("--film",action="store_true")
    p.add_argument("--spatial",action="store_true")
    p.add_argument("--navigation",action="store_true")
    p.add_argument("--blue-cruise",action="store_true",help="Blue tangent steers KEEP; accepted lane changes track the green plan until visual crossing")
    p.add_argument("--blue-keep",action="store_true",help="Steer KEEP from the RGB lane head")
    run(p.parse_args())
