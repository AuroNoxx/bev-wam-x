"""Evaluate the frozen sensor-only driver in CARLA. HD waypoints stay on the evaluator side."""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import queue
import socket
import subprocess
import time

import carla
import cv2
import numpy as np
import torch

from .collect import DT, check_versions, matrix, read
from .geometry import CAMS, raster
from .navigation import directed_step, gps_drift
from .phase2_control import SpatialController
from .phase2_demo_lane import TargetLaneMemory, X as lane_stations, lane_curve, measure_route_lane_offset, shift_route
from .phase2_infer import Inference, NativeCDriver, Stage26Driver, fold_for_inference, load_model
from .phase2_model import NATIVE_C, configure_sdpa, route_inst_vector
from .render_probe import WINDOW_FLAGS, bgr, camera_blueprint, lidar_blueprint, locked_sizes, locked_tier
from .scenarios import same_direction, xyz
from .sd_policy import RoadNavigator, route_vertices
from .survey_sd import main as survey
from v1_dual_system.curve_controller import curvature_preview


def _active_maneuver(navigator):
    inst_dist, inst_kind = navigator.instruction()
    if inst_kind == 'follow' and navigator.events and navigator.station is not None:
        recent = [
            (navigator.station - s, k)
            for s, k in navigator.events
            if k != 'follow' and 0.0 <= navigator.station - s <= 110.0
        ]
        if recent:
            return 0.0, min(recent)[1]
    return inst_dist, inst_kind


@contextmanager
def server(carla_root,out,port,offscreen=False):
    try:
        with socket.create_connection(('127.0.0.1',port),timeout=.3):
            pass
    except OSError:
        pass
    else:
        raise RuntimeError(f'CARLA RPC port {port} is already occupied')
    exe = carla_root/'CarlaUnreal/Binaries/Win64/CarlaUnreal-Win64-Shipping.exe'
    window = (*WINDOW_FLAGS, '-RenderOffScreen') if offscreen else WINDOW_FLAGS
    flags = [*window,'-dx12','-nosound',f'-carla-rpc-port={port}',f'-quality-level={locked_tier()}',
             '-stdout','-FullStdOutLogOutput']
    with (out/'server.log').open('wb') as log:
        process = subprocess.Popen([str(exe),*flags],cwd=carla_root,stdout=log,stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline = time.monotonic()+140
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'CARLA exited {process.returncode}')
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=1):
                        pass
                    break
                except OSError:
                    if time.monotonic()>=deadline:
                        raise
                    time.sleep(.3)
            client = carla.Client('127.0.0.1',port)
            client.set_timeout(150)
            versions = check_versions(client)
            (out/'session.json').write_text(json.dumps(dict(server_pid=process.pid, model_pid=os.getpid(),
                client=versions[0], server=versions[1], engine='UE 5.8',flags=flags,sim_dt=DT),indent=2))
            yield client, process.pid
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(20)
            (out/'shutdown.txt').write_text(f'CARLA PID {process.pid} stopped; exit {process.returncode}\n')


@contextmanager
def gpu_monitor(out, sim_pid):
    script = r'''
$ErrorActionPreference='Stop'
while ($true) {
  $mem=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory)
  $adapter=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory)
  $rows=@($mem | Where-Object {$_.Name -match '^pid_(SIM_PID|MODEL_PID)_'})
  $groups=@($rows | Group-Object { $_.Name -replace '^pid_[0-9]+_','' })
  $data=foreach($group in $groups) {
    $sim=($group.Group | Where-Object Name -match '^pid_SIM_PID_' | Measure-Object DedicatedUsage -Sum).Sum
    $net=($group.Group | Where-Object Name -match '^pid_MODEL_PID_' | Measure-Object DedicatedUsage -Sum).Sum
    $physical=($adapter | Where-Object Name -eq $group.Name | Measure-Object DedicatedUsage -Sum).Sum
    [ordered]@{adapter=$group.Name;ue_bytes=[long]$sim;model_bytes=[long]$net;card_bytes=[long]$physical}
  }
  [ordered]@{unix=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.;adapters=@($data)} | ConvertTo-Json -Compress -Depth 4
  Start-Sleep -Milliseconds 350
}
'''.replace('SIM_PID',str(sim_pid)).replace('MODEL_PID',str(os.getpid()))
    with (out/'gpu.jsonl').open('wb') as log, (out/'gpu-errors.log').open('wb') as errors:
        proc = subprocess.Popen(['pwsh','-NoProfile','-NonInteractive','-Command',script],stdout=log,stderr=errors,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            yield
        finally:
            proc.terminate()
            proc.wait(10)


class RoadSurvey:
    """Reference-route matching for scoring and reproducing historical navigation probes."""
    def __init__(self, vertices):
        self.vertices = np.asarray(vertices,dtype=float)
        self.delta = np.diff(self.vertices,axis=0)
        self.length = np.linalg.norm(self.delta,axis=1)
        if self.vertices.shape[1:] != (2,) or np.any(self.length<.1) or not np.isfinite(self.vertices).all():
            raise ValueError('Invalid independent SD survey')
        self.arc = np.r_[0.,self.length.cumsum()]

    def match(self, xy, previous, ahead=95.):
        indices = np.flatnonzero((self.arc[:-1]<=previous+ahead)&(self.arc[1:]>=previous-12))
        delta = xy-self.vertices[indices]
        t = np.clip((delta*self.delta[indices]).sum(1)/self.length[indices]**2,0,1)
        errors = np.linalg.norm(delta-t[:,None]*self.delta[indices],axis=1)
        n = errors.argmin()
        index = indices[n]
        return float(self.arc[index]+t[n]*self.length[index]),float(errors[n])

    def anchors(self, progress, pose, seconds):
        station = np.minimum(progress+np.array([20.,40.,60.]),self.arc[-1])
        i = np.searchsorted(self.arc[1:],station)
        xy = np.column_stack([np.interp(station,self.arc,column) for column in self.vertices.T])
        xy += gps_drift(.72,seconds)*self.delta[i,::-1]*[1,-1]/self.length[i,None]
        world = np.column_stack((xy,np.full(3,pose[2,3])))
        return ((world-pose[:3,3])@pose[:3,:3]*[1,-1,1])[:,:2].astype(np.float32)


def reference(road,sd,spec):
    """Road topology for spawn and score only. It never supplies a control or navigation input."""
    start_xy = np.array(spec.get('spawn_xy',sd.vertices[0]),float)
    direction = sd.delta[0]/sd.length[0]
    guides = []
    for wp in road.generate_waypoints(5):
        p = xyz(wp)
        d = np.linalg.norm(p[:2]-start_xy)
        if d>45 or wp.is_junction or wp.lane_type!=carla.LaneType.Driving:
            continue
        forward = wp.transform.get_forward_vector()
        if np.dot(direction,[forward.x,forward.y])<.84:
            continue
        if ('spawn_lane' in spec and wp.lane_id!=spec['spawn_lane']) or (
                'spawn_road' in spec and wp.road_id!=spec['spawn_road']):
            continue
        guides.append((d,wp))
    targets = set(spec.get('expected_roads',()))
    best = None
    for error, start in sorted(guides,key=lambda item:item[0])[:40]:
        queue_ = [(float(error),sd.match(xyz(start)[:2],0.)[0],[start])]
        for _ in range(int(sd.arc[-1]/3)+25):
            next_queue = {}
            for cost, station, path in queue_:
                for wp in path[-1].next(3.):
                    if not directed_step(path[-1],wp):
                        continue
                    progress, residual = sd.match(xyz(wp)[:2],station,45.)
                    if residual>55 or progress<station-4:
                        continue
                    new_cost = .92*cost + residual + 2*max(0.,station+1.3-progress)
                    new_path = [*path,wp]
                    meets_target = not targets or any(p.road_id in targets for p in new_path)
                    rank = progress - .36*new_cost + (65 if meets_target else 0)
                    if best is None or rank>best[0]:
                        best = (rank,new_path,meets_target,progress)
                    if meets_target and progress>sd.arc[-1]-24:
                        return new_path
                    key = wp.road_id,wp.lane_id,round(wp.s/3)
                    if key not in next_queue or new_cost<next_queue[key][0]:
                        next_queue[key] = (new_cost,max(station,progress),new_path)
            if not next_queue:
                break
            queue_ = sorted(next_queue.values(),key=lambda p:p[0]-.20*p[1])[:12]
        if best and best[2] and best[3]>sd.arc[-1]*.85:
            break
    if best is None or not best[2] or best[3]<min(160,sd.arc[-1]*.65):
        raise RuntimeError(f'No legal scoring reference for {spec["name"]}')
    return best[1]


def score_path(ref, poses, actual_roads, speeds, half_width, half_length, spec):
    xyz_ref = np.stack([xyz(w) for w in ref]); xy = xyz_ref[:,:2]; delta = np.diff(xy,axis=0)
    length = np.linalg.norm(delta,axis=1).clip(1e-5); arc = np.r_[0.,length.cumsum()]
    curvature_s,kappa = curvature_preview(xy)
    lateral, cross, tight_curve, position = [],[],[],[]
    station = 0.
    for pose in poses:
        idx = np.flatnonzero((arc[:-1]<=station+65)&(arc[1:]>=station-20))
        offset = pose[:2,3]-xy[idx]
        f = np.clip((offset*delta[idx]).sum(1)/length[idx]**2,0,1)
        residual = offset - f[:,None]*delta[idx]
        squared = (residual**2).sum(1)+(xyz_ref[idx,2]-pose[2,3])**2
        j = squared.argmin(); i = idx[j]
        left = np.array([delta[i,1],-delta[i,0]])/length[i]
        distance = abs(float(residual[j]@left))
        width = half_width*abs(float(pose[:2,1]@left))+half_length*abs(float(pose[:2,0]@left))
        progress = arc[i]+f[j]*length[i]; station = max(station,progress)
        tight = len(kappa) and abs(np.interp(progress,curvature_s,kappa))>.025
        cross.append(bool(tight and distance+width>ref[i].lane_width/2))
        tight_curve.append(bool(tight))
        lateral.append(distance);position.append(float(progress))
    moving = np.array(speeds)>1.
    close = moving & (np.asarray(lateral)<.35)
    longest = 0; run = 0
    for hit in close:
        run = run+1 if hit else 0; longest = max(longest,run)
    forks = []
    explicit = spec.get('expected_roads',())
    if explicit:
        at = np.flatnonzero(np.isin([w.road_id for w in ref],explicit))
        if len(at):
            forks.append((max(0.,arc[at[0]]-45),set(explicit)))
    elif spec.get('count_fork'):
        for i,wp in enumerate(ref[:-8]):
            if len(wp.next(3.))>1 and (not forks or arc[i]>forks[-1][0]+55):
                forks.append((max(0.,arc[i]-15),{ref[i+8].road_id}))
    attempted = [(s,roads) for s,roads in forks if max(position,default=0)>=s]
    hits = sum(bool(roads.intersection(actual_roads)) for _,roads in attempted)
    use = np.asarray(lateral)[moving]
    return dict(lateral_p95_m=float(np.percentile(use,95)) if use.size else None,
                lateral_all_ticks_p95_m=float(np.percentile(lateral,95)),
                moving_frames=int(moving.sum()),centered_longest_s=longest*DT,
                tight_curve_frames=sum(tight_curve),tight_curve_over_line_frames=sum(cross),forks_expected=len(forks),
                forks_reached=len(attempted),forks_hit=hits,reference_progress_m=max(position,default=0),
                reference_length_m=float(arc[-1]),lateral_m=lateral,progress_m=position)


def actor_observation(images,scan,sd_anchors,command=0):
    raw = np.frombuffer(scan.raw_data,np.float32).reshape(-1,4).copy()
    raw[:,1]*=-1;raw[:,2]+=2.2
    views=[torch.from_numpy(cv2.cvtColor(np.frombuffer(im.raw_data,np.uint8).reshape(im.height,im.width,4),
                                         cv2.COLOR_BGRA2RGB)).permute(2,0,1)[None] for im in images]
    return (*views,torch.from_numpy(raster(raw))[None],torch.from_numpy(np.ascontiguousarray(sd_anchors))[None],
            torch.from_numpy(np.eye(3,dtype=np.uint8)[command])[None]),raw


def score_requested_lane(road, positions, direction, sd, end_reason):
    """Score the physical adjacent lane through road-ID changes, after the drive."""
    origin = road.get_waypoint(carla.Location(*positions[0]),project_to_road=False)
    side = 'left' if direction==1 else 'right'
    mark = carla.LaneChange.Left if direction==1 else carla.LaneChange.Right
    target = same_direction(origin,side) if origin and int(origin.lane_change)&int(mark) else None
    if target is None:
        return dict(eligible=False, held_until_end=False)
    chain = [target]
    distance = np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()
    station = sd.match(positions[0,:2],0.,ahead=sd.arc[-1])[0]
    ended = False
    for _ in range(int((distance+30)/1.5)+1):
        choices = chain[-1].next(1.5)
        if not choices:
            ended = True
            break
        nxt = min(choices,key=lambda w:sd.match(xyz(w)[:2],station,20.)[1])
        station = sd.match(xyz(nxt)[:2],station,20.)[0]
        chain.append(nxt)
    centers = np.stack([xyz(w) for w in chain])
    key = lambda w:(w.road_id,w.section_id,w.lane_id)
    keys = [key(w) for w in chain]
    matched, errors = [], []
    index = 0
    for pos in positions:
        lo, hi = max(0,index-4), min(len(chain),index+12)
        index = lo+int(np.linalg.norm(centers[lo:hi]-pos,axis=1).argmin())
        wp = chain[index]
        f = wp.transform.get_forward_vector()
        if ended and index==len(chain)-1 and (pos-centers[index])@[f.x,f.y,f.z]>1.5:
            end_reason = 'target_lane_ended'
            break
        actual = road.get_waypoint(carla.Location(*pos),project_to_road=False)
        matched.append(bool(actual and key(actual) in keys[max(0,index-2):index+3]))
        r = wp.transform.get_right_vector()
        errors.append(abs((pos-centers[index])@[r.x,r.y,r.z]))
    entered = next((i for i,hit in enumerate(matched) if hit),None)
    after = matched[entered:] if entered is not None else []
    lost = next((i for i,hit in enumerate(after) if not hit),None)
    return dict(eligible=True, initial_target=list(keys[0]), entered_after_ticks=entered,
                ticks_after_entry=len(after), outside_after_entry=after.count(False),
                first_exit_after_entry_ticks=lost, end_reason=end_reason,
                target_lateral_p95_m=float(np.percentile(errors[entered:],95)) if after else None,
                held_until_end=bool(after and lost is None and end_reason in
                                    ('reached_end','next_request','target_lane_ended')))


def run_route(client,world,spec,engine,port,out,demo=None):
    road = world.get_map()
    if road.name.rsplit('/',1)[-1]!=spec['town'] or spec['town'] not in ('Town04_Opt','Town12'):
        raise ValueError(f'Unexpected evaluation world {road.name}')
    vertices = route_vertices(spec)
    sd = RoadSurvey(vertices)
    ref = reference(road,sd,spec)
    tm = client.get_trafficmanager(port+5)
    old = world.get_settings(); settings = world.get_settings()
    settings.synchronous_mode=True;settings.fixed_delta_seconds=DT;settings.no_rendering_mode=False
    world.apply_settings(settings)
    tm.set_synchronous_mode(True);tm.set_random_device_seed(421)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    library = world.get_blueprint_library()
    bp = library.find('vehicle.lincoln.mkz');bp.set_attribute('role_name','hero')
    tf = ref[0].transform;tf.location.z+=.38
    actors,sensors,queues,events,trace = [],[],[],[],[]

    def attach(blueprint,mount,callback=None):
        sensor = world.spawn_actor(blueprint,mount,attach_to=ego)
        sensors.append(sensor)
        if callback is None:
            q = queue.Queue(); queues.append(q);callback=q.put
        sensor.listen(callback)

    try:
        georef=spec['georef']
        world.get_spectator().set_transform(carla.Transform(tf.location+carla.Location(z=12),carla.Rotation(pitch=-50)))
        for _ in range(20):world.tick(45)
        ego = world.spawn_actor(bp,tf);actors.append(ego)
        box = ego.bounding_box
        half_width,half_length = box.extent.y,box.extent.x
        max_steer = max(w.max_steer_angle for w in ego.get_physics_control().wheels)
        controller = SpatialController(spec.get('cruise_kmh',46),2.85,half_width,math.radians(max_steer),
                                       box.location.x+half_length,box.location.x-half_length,box.location.z+box.extent.z)
        rng = np.random.default_rng(714)
        if spec.get('npc_along_route'):
            exclude = ('truck', 'bus', 'firetruck', 'ambulance', 'carlacola', 'cybertruck', 'sprinter', 'fusorosa', 't2')
            car_bps = [
                b for b in library.filter('vehicle.*')
                if b.has_attribute('number_of_wheels') and int(b.get_attribute('number_of_wheels')) == 4
                and not any(tag in b.id.lower() for tag in exclude)
            ] or [library.find('vehicle.mini.cooper')]
            npc_target = int(spec.get('npc', 0))
            candidates = []
            # ref waypoints are spaced ~3 m apart; sample along the first lap when multi-lap
            laps = max(1, int(spec.get('laps') or 1))
            lap_len = max(60, len(ref) // laps)
            step = max(14, lap_len // max(npc_target + 2, 1))
            for idx in range(18, max(19, lap_len - 25), step):
                base_wp = ref[idx]
                lane_opts = [base_wp]
                for side in ('left', 'right'):
                    adj = same_direction(base_wp, side)
                    if adj is not None:
                        lane_opts.append(adj)
                rng.shuffle(lane_opts)
                candidates.extend(lane_opts[:2])
            for wp in candidates:
                if len(actors) > npc_target:
                    break
                if wp.transform.location.distance(tf.location) < 42.0:
                    continue
                if any(wp.transform.location.distance(a.get_transform().location) < 22.0 for a in actors):
                    continue
                spawn_tf = carla.Transform(
                    carla.Location(wp.transform.location.x, wp.transform.location.y, wp.transform.location.z + 0.45),
                    wp.transform.rotation,
                )
                bp_npc = car_bps[int(rng.integers(0, len(car_bps)))]
                npc = world.try_spawn_actor(bp_npc, spawn_tf)
                if npc is not None:
                    actors.append(npc)
                    npc.set_autopilot(True, tm.get_port())
                    tm.ignore_lights_percentage(npc, 100.0)
                    tm.auto_lane_change(npc, True)
                    tm.vehicle_percentage_speed_difference(npc, float(rng.uniform(-15.0, 12.0)))
        else:
            locations = road.get_spawn_points();rng.shuffle(locations)
            for t in locations:
                if len(actors)>spec.get('npc',0):break
                if not 36<t.location.distance(tf.location)<180:continue
                npc = world.try_spawn_actor(library.find('vehicle.mini.cooper'),t)
                if npc is not None:
                    actors.append(npc);npc.set_autopilot(True,tm.get_port())
        for (x,y,z,yaw,pitch,fov),size in zip(CAMS,locked_sizes()):
            mount=carla.Transform(carla.Location(x,y,z),carla.Rotation(yaw=yaw,pitch=pitch))
            attach(camera_blueprint(library,fov,size),mount)
        attach(lidar_blueprint(library,False),carla.Transform(carla.Location(z=2.2)))
        for kind in ('gnss','imu'):
            attach(library.find('sensor.other.'+kind),carla.Transform())
        navigator=RoadNavigator(vertices,georef,.72,spec.get('events',()),spec.get('sd_accuracy'))
        if demo is not None:
            attach(*demo.chase_camera(library))
        attach(library.find('sensor.other.collision'),carla.Transform(),lambda e:events.append(
            dict(frame=e.frame,actor=e.other_actor.id,kind=e.other_actor.type_id,
                 impulse_magnitude=e.normal_impulse.length())))
        ego.apply_control(carla.VehicleControl(brake=1.))
        for _ in range(32):
            frame=world.tick(45)
            for q in queues:read(q,frame)
        warm_events=events.copy();events.clear()
        poses,roads,speeds=[],[],[]
        progress=0.; last_progress=progress; stagnant=0
        capture_start=time.time(); reason='tick_budget'; last_frame=None
        right_request_tick=None
        requested_lane=None
        target_lane_run=target_lane_longest=0
        lane_memory = TargetLaneMemory() if demo is not None or spec.get('hold_target_lane',True) else None
        if spec.get('acknowledge_request'):
            from .phase2_demo_lane import AdjacentLaneRequest
            request = AdjacentLaneRequest()
        scripted = {int(t):int(k) for t,k in spec.get('intent_requests',[])}
        if scripted and (not spec.get('acknowledge_request') or
                         set(scripted.values())-{1,2} or spec.get('test_right_on_bend')):
            raise ValueError('Scheduled lane requests require left/right commands and acknowledgement')
        expires = 0
        speed_requests = dict(spec.get('speed_requests_kmh',[]))
        is_stage26 = bool(getattr(engine, 'is_stage26', False))
        prev_map_tokens = torch.zeros((1, 7, 256), dtype=torch.float32)
        prev_compass = None
        prev_speed = None
        has_prev_flag = False
        align_cruise = bool(demo is not None or spec.get('align_cruise_nav', False))
        cruise_sd_offset = None
        for i in range(spec.get('max_ticks',1200)):
            if demo is not None and not demo.advance(i):
                if not trace:
                    return None
                reason='demo_restart' if demo.restart else 'demo_closed'
                break
            start=time.perf_counter();frame=world.tick(45);snap=world.get_snapshot()
            bundle=[read(q,frame) for q in queues]
            if snap.frame!=frame or last_frame is not None and frame!=last_frame+1:
                raise RuntimeError(f'Frame mismatch on {spec["name"]}, frame {frame}')
            last_frame=frame
            sim_done=time.perf_counter()
            sim_tick_ms=(sim_done-start)*1000
            ego_snap=snap.find(ego.id);pose=matrix(ego_snap.get_transform())
            gnss,imu=bundle[4:6]
            try:
                nav=navigator.update(gnss.latitude,gnss.longitude,imu.compass,i*DT)
            except ValueError:
                ego.apply_control(carla.VehicleControl(brake=1.))
                if not trace:raise
                reason='no_forward_sd_road'
                break
            progress,nav_error=navigator.station,navigator.error
            command = demo.command if demo is not None else 0
            if scripted:
                if i in scripted:
                    request.begin(scripted[i])
                    expires = i+70
                if i<expires:
                    command = request.direction
            if spec.get('test_right_on_bend') and i>=20:
                preview=navigator.points([5.,15.,25.])
                heading=np.arctan2(np.diff(preview[:,1]),np.diff(preview[:,0]))
                curvature=abs(np.diff(np.unwrap(heading))[0])/10.
                trigger = (i==spec['request_tick'] if 'request_tick' in spec else .001<=curvature<.018)
                if right_request_tick is None and trigger:
                    right_request_tick=i
                    if spec.get('acknowledge_request'):
                        request.begin(2)
                if right_request_tick is not None and i-right_request_tick<70 and (
                        not spec.get('acknowledge_request') or request.direction):
                    command=2
            user_command = command
            new_request = i in scripted or i==right_request_tick or (demo is not None and demo.intent_tick==i)
            coarse = navigator.points(np.arange(0.,61.,2.))
            near_nav_maneuver = bool(
                navigator.station is not None
                and any(k != 'follow' and -110.0 <= (s - navigator.station) <= 130.0 for s, k in navigator.events)
            )
            route = None
            if lane_memory is not None and lane_memory.world is not None:
                route = lane_memory.guidance(coarse,navigator)
                nav = route[[10,20,30]]
                cruise_sd_offset = None
            elif align_cruise and cruise_sd_offset is not None:
                if near_nav_maneuver:
                    cruise_sd_offset *= 0.85
                    if abs(cruise_sd_offset) < 0.05:
                        cruise_sd_offset = 0.0
                route = shift_route(coarse, cruise_sd_offset)
                nav = route[[10,20,30]]
            elif lane_memory is not None:
                route = lane_memory.guidance(coarse,navigator)
                nav = route[[10,20,30]]
            if lane_memory is not None and not user_command:
                command = lane_memory.command
            if lane_memory is not None and lane_memory.rejected and not new_request:
                command = 0
            vel=ego_snap.get_velocity(); v=math.hypot(vel.x,vel.y)
            args,raw=actor_observation(bundle[:3],bundle[3],nav,command)
            stage26_extra = ()
            if is_stage26:
                inst_dist, inst_kind = _active_maneuver(navigator)
                route_inst_t = torch.from_numpy(route_inst_vector(inst_kind))[None]
                route_dist_t = torch.tensor([float(inst_dist)], dtype=torch.float32)
                if has_prev_flag and prev_compass is not None and prev_speed is not None:
                    dyaw = ((prev_compass - float(imu.compass) + math.pi) % (2.0 * math.pi)) - math.pi
                    ds = 0.5 * (prev_speed + v) * DT
                    dx = -ds * math.cos(0.5 * dyaw)
                    dy = -ds * math.sin(0.5 * dyaw)
                    ego_delta_t = torch.tensor([[dx, dy, dyaw]], dtype=torch.float32)
                    has_prev_t = torch.tensor([1.0], dtype=torch.float32)
                else:
                    ego_delta_t = torch.zeros((1, 3), dtype=torch.float32)
                    has_prev_t = torch.zeros((1,), dtype=torch.float32)
                stage26_extra = (route_inst_t, route_dist_t, prev_map_tokens, ego_delta_t, has_prev_t)
            obs_ms=(time.perf_counter()-sim_done)*1000
            inference_start=time.perf_counter(); output=engine.run_cpu(*args, *stage26_extra)
            path=output['path'][0].float().cpu().numpy().copy(); occ=output['occ'][0].float().cpu().numpy().copy()
            inference_ms=(time.perf_counter()-inference_start)*1000
            ctrl_start=time.perf_counter(); extra_inference_ms=0.
            demo_trace = {}
            if demo is not None or scripted or spec.get('test_right_on_bend'):
                demo_trace.update(requested_intent=user_command,new_request=new_request,command=command)
            confirmed = False
            if demo is not None:
                confirmed = demo.observe_command(output)
                demo_trace['planner']='model_with_lane_feedback'
                controller.cruise=demo.target_kmh/3.6
            elif spec.get('acknowledge_request') and user_command:
                confirmed = request.observe(output)
            if lane_memory is not None:
                lane_memory.update(output,navigator,path,user_command,confirmed,command,raw,v,new_request)
                if new_request:
                    selected = lane_memory.guidance(coarse,navigator)
                    corrected = selected[[10,20,30]]
                    applied_command = 0 if lane_memory.rejected else command
                    if applied_command!=command or not np.array_equal(corrected,nav):
                        command = applied_command
                        route, nav = selected, corrected
                        signal = torch.zeros_like(args[-1])
                        signal[0,command] = 1
                        begin = time.perf_counter()
                        output = engine.run_cpu(*args[:-2],torch.from_numpy(nav)[None],signal,*stage26_extra)
                        path = output['path'][0].float().cpu().numpy().copy()
                        occ = output['occ'][0].float().cpu().numpy().copy()
                        dt_inf = (time.perf_counter()-begin)*1000
                        inference_ms += dt_inf
                        extra_inference_ms += dt_inf
                        demo_trace['command'] = command
                demo_trace.update(target_status=lane_memory.status, hold_command=0 if user_command else command)
                if spec.get('audit_lane_request'):
                    demo_trace.update(target_match_error_m=lane_memory.match_error,
                                      target_near_xy=((lane_memory.world-navigator.gnss_xy)@navigator.axes)[::4].tolist()
                                      if lane_memory.world is not None else None)
                if demo is not None and lane_memory.status:
                    demo.lane_status = lane_memory.status
            if is_stage26:
                prev_map_tokens = output['map_tokens'].detach().float().cpu().clone()
                prev_compass = float(imu.compass)
                prev_speed = float(v)
                has_prev_flag = True
            if i in speed_requests:
                controller.cruise = float(speed_requests[i])/3.6
            target = lane_memory.local_target(navigator) if lane_memory is not None else None
            control_lane = target
            if lane_memory is not None and target is None and (not lane_memory.status or lane_memory.rejected or lane_memory.status=='Target lost / Keep lane'):
                if output['exists'][0,0].item()>0:
                    sensed = lane_curve(output['lines'][0,0].float().cpu().numpy())
                    if sensed is not None:
                        c_vec = coarse[12]-coarse[3]
                        if np.hypot(c_vec[0],c_vec[1])>1.0:
                            sensed_heading = math.atan2(float(sensed[16]-sensed[4]),18.0)
                            route_heading = math.atan2(float(c_vec[1]),float(c_vec[0]))
                            heading_div = abs((route_heading-sensed_heading+math.pi)%(2*math.pi)-math.pi)
                        else:
                            heading_div = 0.0
                        path_ref = np.interp(path[4:12,0],lane_stations,sensed)
                        path_div = float(np.mean(np.abs(path[4:12,1]-path_ref)))
                        if align_cruise and lane_memory.world is None and not near_nav_maneuver and heading_div<=math.radians(10.0):
                            meas_off = measure_route_lane_offset(coarse, np.column_stack((lane_stations[:17],sensed[:17])))
                            if meas_off is not None:
                                cruise_sd_offset = meas_off if cruise_sd_offset is None else 0.80*cruise_sd_offset + 0.20*meas_off
                        if not near_nav_maneuver and heading_div<=math.radians(8.0) and path_div<=1.25:
                            control_lane = np.column_stack((lane_stations[:17],sensed[:17]))
            model_path = path
            changing = target is not None and not lane_memory.acquired
            path,limit,offset = controller.lane_tracking(model_path,control_lane,changing,speed=v)
            if lane_memory is not None and lane_memory.status and lane_memory.status!='Target lost / Keep lane' and not lane_memory.rejected and (target is None or lane_memory.missing>=4):
                horizon = 0. if target is None else max(0.,target[-1,0]-4.)
                limit = min(limit,math.sqrt(2.*2.5*horizon))
            if lane_memory is not None and lane_memory.status=='Selected lane blocked':
                limit = 0.
            throttle,brake,steer,safety=controller.step(path,occ,raw,v,speed_limit=limit,
                                                       lane_tracking=offset is not None,
                                                       route_preview=coarse)
            if lane_memory is not None:
                demo_trace.update(lane_offset_m=offset,lane_limit_kmh=3.6*limit,
                                  nav_input=nav.tolist(),sd_route_input=coarse[[10,20,30]].tolist(),
                                  path_correction_m=float(np.max(np.abs(path[:12,1]-model_path[:12,1]))),
                                  fork_yield=bool(control_lane is None and target is None))
            disturbance = spec.get('evaluation_steer_pulse')
            if disturbance and demo is None:
                tick,span,amplitude = disturbance
                delta = 0.
                if tick<=i<tick+span and not (safety.get('safety_brake') or safety.get('safety_steer')):
                    delta = amplitude*math.sin(math.pi*(i-tick+1)/(span+1))
                    steer = float(np.clip(steer+delta,-1.,1.))
                demo_trace['evaluation_steer_delta'] = delta
            ego.apply_control(carla.VehicleControl(throttle=throttle,brake=brake,steer=steer))
            control_end=time.perf_counter()
            control_ms=(control_end-ctrl_start)*1000-extra_inference_ms
            agent_ms=obs_ms+inference_ms+control_ms
            loop_ms=(control_end-start)*1000
            if demo is not None:
                demo.present(bundle[:3],bundle[6],args[3][0,5].numpy(),raw,output,nav,path,occ,
                             v*3.6,throttle,brake,safety,progress,
                             route=route,target=target,steer=steer)
            # Ground truth is read after control and is used only for scoring.
            waypoint=road.get_waypoint(ego_snap.get_transform().location)
            if spec.get('test_right_on_bend'):
                if i==right_request_tick:
                    adjacent=same_direction(waypoint,'right')
                    if adjacent is not None and int(waypoint.lane_change)&int(carla.LaneChange.Right):
                        requested_lane=(adjacent.road_id,adjacent.lane_id)
                actual=road.get_waypoint(ego_snap.get_transform().location,project_to_road=False)
                matched=(requested_lane is not None and actual is not None and not actual.is_junction
                         and (actual.road_id,actual.lane_id)==requested_lane)
                target_lane_run=target_lane_run+1 if matched else 0
                target_lane_longest=max(target_lane_longest,target_lane_run)
                demo_trace.update(command=command,lane_id=waypoint.lane_id)
            if spec.get('audit_lane_request'):
                actual = road.get_waypoint(ego_snap.get_transform().location,project_to_road=False)
                observations = []
                for x,y in nav:
                    p = pose[:3,3]+pose[:3,0]*x-pose[:3,1]*y
                    w = road.get_waypoint(carla.Location(*p),project_to_road=False)
                    if w is None or abs(w.transform.location.z-p[2])>4:
                        observations.append('off')
                    else:
                        f = w.transform.get_forward_vector()
                        dot = pose[:2,0]@[f.x,f.y]
                        observations.append('forward' if dot>.5 else 'opposing' if dot<-.5 else 'cross')
                demo_trace.update(nav_class=observations, lane_id=waypoint.lane_id,
                                  own_edges=output['lines'][0,1:3,:4].float().cpu().tolist(),
                                  own_center=output['lines'][0,0,:4].float().cpu().tolist(),
                                  on_driving_lane=actual is not None)
            if demo is not None:
                on_lane=road.get_waypoint(ego_snap.get_transform().location,project_to_road=False)
                demo_trace.update(command=command,target_kmh=demo.target_kmh,lane_id=waypoint.lane_id,
                                  on_driving_lane=on_lane is not None)
            poses.append(pose);roads.append(waypoint.road_id);speeds.append(v)
            end_tick=time.perf_counter()
            audit_ms=(end_tick-control_end)*1000
            total_wall_ms=(end_tick-start)*1000
            trace.append(dict(frame=frame,road=waypoint.road_id,ego_xyz=pose[:3,3].tolist(),speed_kmh=v*3.6,
                              nav_station_m=progress,nav_distance_m=nav_error,
                              sim_tick_ms=sim_tick_ms,obs_ms=obs_ms,inference_ms=inference_ms,
                              control_ms=control_ms,agent_ms=agent_ms,loop_ms=loop_ms,
                              audit_ms=audit_ms,total_wall_ms=total_wall_ms,
                              throttle=throttle,brake=brake,steer=steer,**safety,**demo_trace))
            if (i+1)%100==0:print(spec['name'],i+1,'ticks',round(progress,1),'m',round(v*3.6,1),'km/h',flush=True)
            stagnant=stagnant+1 if progress<last_progress+.045 and v<.55 else 0
            last_progress=progress
            if events:
                reason='collision';break
            if stagnant>=(600 if demo is not None else 200):
                reason='stalled_10s';break
            if nav_error>33:
                reason='deviated_from_sd_corridor';break
            if progress>=navigator.arc[-1]-45:
                reason='reached_end';break
        capture_end=time.time()
        metrics=score_path(ref,poses,roads,speeds,half_width,half_length,spec)
        collisions=[]
        for event in events:
            if not any(x['actor']==event['actor'] and abs(x['frame']-event['frame'])<=10 for x in collisions):
                collisions.append(event)
        metrics.update(map=road.name,route=spec['name'],reason=reason,frames=len(trace),npc_count=len(actors)-1,
                       sim_seconds=len(trace)*DT,wall_seconds=capture_end-capture_start,
                       loop_p95_ms=float(np.percentile([t['loop_ms'] for t in trace],95)),
                       sim_tick_p95_ms=float(np.percentile([t['sim_tick_ms'] for t in trace],95)),
                       obs_p95_ms=float(np.percentile([t['obs_ms'] for t in trace],95)),
                       control_p95_ms=float(np.percentile([t['control_ms'] for t in trace],95)),
                       agent_p95_ms=float(np.percentile([t['agent_ms'] for t in trace],95)),
                       audit_p95_ms=float(np.percentile([t['audit_ms'] for t in trace],95)),
                       total_wall_p95_ms=float(np.percentile([t['total_wall_ms'] for t in trace],95)),
                       wall_fps=len(trace)/(capture_end-capture_start),
                       inference_p95_ms=float(np.percentile([t['inference_ms'] for t in trace],95)),
                       collision_count=len(collisions),collisions=collisions,warmup_collisions=warm_events,
                       capture_start=capture_start,capture_end=capture_end,
                       max_speed_kmh=3.6*max(speeds),lidar_brake_frames=sum(r.get('safety_brake',False) for r in trace),
                       occupancy_risk_frames=sum(r.get('occupancy_risk',False) for r in trace))
        lateral=metrics.pop('lateral_m'); ref_progress=metrics.pop('progress_m')
        requests = [(j,r['requested_intent']) for j,r in enumerate(trace) if r.get('requested_intent')
                    and (r.get('new_request') or j==0 or r['requested_intent']!=trace[j-1].get('requested_intent'))]
        if requests:
            positions = np.stack([p[:3,3] for p in poses])
            scores = []
            for n,(j,direction) in enumerate(requests):
                end = requests[n+1][0] if n+1<len(requests) else len(trace)
                result = score_requested_lane(road,positions[j:end],direction,sd,
                                              'next_request' if end<len(trace) else reason)
                scores.append(dict(request_tick=j,direction=direction,**result))
            metrics['target_retention'] = scores
        if spec.get('test_right_on_bend'):
            metrics.update(right_request_tick=right_request_tick,right_target_lane=requested_lane,
                           longest_in_requested_lane_s=target_lane_longest*DT,
                           right_change_held_one_second=target_lane_longest>=20)
        for row,error,station in zip(trace,lateral,ref_progress):
            row['lateral_error_m']=error;row['reference_station_m']=station
            for key in ('lidar_gap_m','left_clearance_m','right_clearance_m'):
                if key in row and not math.isfinite(row[key]):row[key]=None
        (out/f'{spec["name"]}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in trace))
        (out/f'{spec["name"]}.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
        print(json.dumps(metrics),flush=True)
        return metrics
    finally:
        for s in reversed(sensors):s.stop();s.destroy()
        for a in reversed(actors):a.destroy()
        tm.set_synchronous_mode(False);world.apply_settings(old)


def summarize_vram(out,metrics):
    rows=[]
    for line in (out/'gpu.jsonl').read_text(encoding='utf-8-sig').splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    driving=[]
    for m in metrics:
        points=[]
        for row in rows:
            if m['capture_start']<=row['unix']<=m['capture_end']:
                physical=max(row['adapters'],key=lambda a:a['ue_bytes']+a['model_bytes'],default=None)
                if physical is not None and physical['ue_bytes'] and physical['model_bytes']:points.append(physical)
        if points:
            m['vram'] = dict(samples=len(points),ue_peak_gib=max(a['ue_bytes'] for a in points)/2**30,
                             model_peak_gib=max(a['model_bytes'] for a in points)/2**30,
                             joint_peak_gib=max(a['ue_bytes']+a['model_bytes'] for a in points)/2**30,
                             card_peak_gib=max(a['card_bytes'] for a in points)/2**30)
            driving.extend(points)
        else:m['vram']=None
    (out/'closed_loop.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    joint=[]
    all_adapters=[]
    for row in rows:
        for a in row['adapters']:
            all_adapters.append(a)
            if a['ue_bytes'] and a['model_bytes']:joint.append(a)
    result=dict(samples=len(joint),
                joint_peak_gib=max((a['ue_bytes']+a['model_bytes'] for a in joint),default=0)/2**30,
                card_peak_gib=max((a['card_bytes'] for a in joint),default=0)/2**30,
                driving_joint_peak_gib=max((a['ue_bytes']+a['model_bytes'] for a in driving),default=0)/2**30,
                driving_card_peak_gib=max((a['card_bytes'] for a in driving),default=0)/2**30,
                ue_only_load_peak_gib=max((a['ue_bytes'] for a in all_adapters),default=0)/2**30,
                card_peak_all_samples_gib=max((a['card_bytes'] for a in all_adapters),default=0)/2**30)
    (out/'session-vram.json').write_text(json.dumps(result,indent=2))
    return result


def main(args,demo=None):
    args.out.mkdir(parents=True,exist_ok=False)
    if args.survey:
        # Descriptive aerial views only, in the evaluation output tree. No training artifact.
        with server(args.carla_root,args.out,args.port) as (client,_):
            world=client.load_world('Town04_Opt');road=world.get_map();points=np.stack([xyz(w) for w in road.generate_waypoints(10)])
            center=(points.min(0)+points.max(0))/2;center[2]=np.median(points[:,2]); altitude=float(max(np.ptp(points[:,0]),np.ptp(points[:,1]))*.56)
            (args.out/'bounds.json').write_text(json.dumps(dict(lo=points.min(0).tolist(),hi=points.max(0).tolist())))
            survey(args.port,args.out,[center],altitude,1800,100,'Town04_Opt',evaluation=True)
        return
    if not all((args.weights,args.plan,args.latency)):
        raise ValueError('Closed loop requires frozen weights, independent SD routes and deployment benchmark')
    routes=json.loads(args.plan.read_text())
    atlas_file=getattr(args,'sd_atlas',None)
    if atlas_file:
        atlas=json.loads(atlas_file.read_text())
        for spec in routes:
            mission=spec.get('navigation_mission',spec['name'])
            candidates=[m for m in atlas[spec['town']]['missions'] if m['name']==mission]
            if len(candidates)!=1:raise ValueError(f'Missing independent SD mission {mission}')
            nav=candidates[0]
            spec.update({k:nav[k] for k in ('xy','georef','events','sd_accuracy','source')})
            spec['laps']=nav.get('laps',1)
    for spec in routes:
        if spec.get('source')!='nadir-road-area-width-v2' or not spec.get('sd_accuracy') or not spec.get('georef'):
            raise ValueError(f'{spec["name"]} needs its independent road survey and width bounds')
    latency=json.loads(args.latency.read_text());model=load_model(args.weights)
    if latency['folded_batch_norm']:
        fold_for_inference(model)
    sizes=locked_sizes()
    if tuple(sizes)!=NATIVE_C:
        raise ValueError(f'Unexpected closed-loop camera sizes {sizes}')
    engine=None
    results=[]
    with server(args.carla_root,args.out,args.port,offscreen=demo is not None) as (client,pid):
        with gpu_monitor(args.out,pid):
            current=None
            for spec in routes:
                if spec['town'] not in ('Town12','Town04_Opt'):raise ValueError('Invalid closed-loop map')
                if current!=spec['town']:
                    world=client.load_world(spec['town']);current=spec['town']
                if engine is None:
                    if not world.get_settings().synchronous_mode:
                        for _ in range(8):
                            world.wait_for_tick(10.0)
                    time.sleep(1.5)
                    configure_sdpa('aotriton');torch.cuda.tunable.tuning_enable(False);torch.set_num_threads(3)
                    is_stage26 = bool(getattr(model, 'temporal', False) or getattr(model, 'nav_rich', False))
                    if is_stage26:
                        gpu_model = Stage26Driver(model).eval().cuda().to(memory_format=torch.channels_last)
                        example = (
                            *(torch.zeros((1, 3, h, w), dtype=torch.uint8) for w, h in sizes),
                            torch.zeros((1, 6, 128, 128), dtype=torch.float16),
                            torch.zeros((1, 3, 2), dtype=torch.float32),
                            torch.tensor([[1, 0, 0]], dtype=torch.uint8),
                            torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.float32),
                            torch.tensor([-1.0], dtype=torch.float32),
                            torch.zeros((1, 7, 256), dtype=torch.float32),
                            torch.zeros((1, 3), dtype=torch.float32),
                            torch.zeros((1,), dtype=torch.float32),
                        )
                    else:
                        gpu_model = NativeCDriver(model).eval().cuda().to(memory_format=torch.channels_last)
                        example = (
                            *(torch.zeros((1, 3, h, w), dtype=torch.uint8) for w, h in sizes),
                            torch.zeros((1, 6, 128, 128), dtype=torch.float16),
                            torch.zeros((1, 3, 2)),
                            torch.tensor([[1, 0, 0]], dtype=torch.uint8),
                        )
                    engine = Inference(gpu_model, example, graph=latency['selected'] == 'hip_graph')
                    engine.is_stage26 = is_stage26
                if demo is None:
                    results.append(run_route(client,world,spec,engine,args.port,args.out))
                    continue
                while not demo.closed:
                    out=args.out/f'run-{len(results)+1:02d}'
                    out.mkdir(exist_ok=True)
                    demo.start_route(spec)
                    result=run_route(client,world,spec,engine,args.port,out,demo)
                    if result is not None:
                        results.append(result)
                    demo.wait_for_restart(result)
    memory=summarize_vram(args.out,results)
    if any(spec.get('joint_budget_gib') is not None for spec in routes):
        budget=min(spec['joint_budget_gib'] for spec in routes if 'joint_budget_gib' in spec)
        if not memory['samples'] or memory['joint_peak_gib']>budget:
            raise RuntimeError(f'Closed-loop VRAM gate {budget} GiB: {memory}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--carla-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--port',type=int,default=2460);p.add_argument('--survey',action='store_true')
    p.add_argument('--weights',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--latency',type=Path)
    p.add_argument('--sd-atlas',type=Path)
    a=p.parse_args();a.carla_root=a.carla_root.resolve();main(a)
