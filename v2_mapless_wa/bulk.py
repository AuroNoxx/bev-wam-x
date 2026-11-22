"""Record continuous, resumable episodes after the rendering and calibration gates."""
import argparse
import copy
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
import msvcrt
import os
from pathlib import Path
import queue
import time

import carla
import cv2
import numpy as np
import psutil
from scipy.spatial import cKDTree

from .collect import COLLECTION_MAPS, DT, SEM, box, check_versions, matrix, read
from .geometry import CAMS, NAMES, local, pixels, polygon, raster
from .labels import Surfaces, map_vectors
from .frames import load_observation
from .clean_driver import CollectionDriver
from .sd_policy import RoadNavigator, fix_values, route_vertices
from .navigation import DirectedRoute, directed_step, gps_drift
from .render_probe import WINDOW_FLAGS, bgr, camera_blueprint, lidar_blueprint, locked_tier, locked_sizes
from .scenarios import same_direction, xyz

N = 2000
FUTURE_TICKS = (10, 20, 40, 60, 80)
FILES = (*[name + '.png' for name in NAMES], 'lidar.npz', 'input.npz', 'labels.npz', 'index.npz')
WEATHER = ('ClearNoon', 'ClearSunset', 'CloudyNoon', 'WetCloudyNoon')


class Mission:
    """A directed road sketch from the nadir survey, independent of OpenDRIVE."""

    def __init__(self, spec):
        if spec.get('source') not in ('nadir-road-area','nadir-road-area-width-v2'):
            raise ValueError('An independently surveyed ordinary road route is required')
        self.georef=spec['georef'];self.events=spec['events'];self.accuracy=spec.get('sd_accuracy')
        self.spec=spec
        self.name = spec['name']
        self.split = spec['split']
        self.speed = spec['speed_kmh']
        self.required_roads = set(spec.get('required_roads', ()))
        self.planned_lane_changes = tuple(spec.get('planned_lane_changes', ()))
        self.xy = route_vertices(spec)
        self.delta = np.diff(self.xy, axis=0)
        self.length = np.linalg.norm(self.delta, axis=1)
        if not np.isfinite(self.xy).all() or np.any(self.length < .1):
            raise ValueError(f'{self.name}: invalid road sketch')
        self.station = np.r_[0, self.length.cumsum()]

    def match(self, xy, last, ahead=75):
        indices = np.flatnonzero((self.station[:-1] <= last + ahead) & (self.station[1:] >= last - 15))
        delta = xy - self.xy[indices]
        fraction = np.clip((delta * self.delta[indices]).sum(1) / self.length[indices] ** 2, 0, 1)
        dist = np.linalg.norm(delta - fraction[:, None] * self.delta[indices], axis=1)
        nearest = dist.argmin()
        i = indices[nearest]
        return self.station[i] + fraction[nearest] * self.length[i], float(dist[nearest])

def allowed(xy, split, bounds):
    if bounds is None:
        return split == 'train'
    low, high = bounds
    if split == 'val':
        return bool(np.all(xy >= low) and np.all(xy <= high))
    return bool(np.any(xy < low - 150) or np.any(xy > high + 150))


def expert_route(road, mission, bounds, sampled=None):
    """Match an expert route to the independent mission without exposing lane data to inputs."""
    a = mission.xy[0]
    direction = mission.delta[0] / mission.length[0]
    sampled = road.generate_waypoints(10) if sampled is None else sampled
    starts = []
    for wp in sampled:
        p = xyz(wp)[:2]
        if np.linalg.norm(p-a)>130:
            continue
        v = wp.transform.get_forward_vector()
        if v.x * direction[0] + v.y * direction[1] < .8 or wp.is_junction or wp.lane_type!=carla.LaneType.Driving:
            continue
        station, error = mission.match(p, 0, 110)
        if allowed(p, mission.split, bounds) and error < 55:
            side = mission.planned_lane_changes[0] if mission.planned_lane_changes else None
            priority = 12 if side and same_direction(wp,side) is None else 0
            starts.append((error + .35 * station + priority, wp, station))
    starts.sort(key=lambda x: x[0])
    best = (0, [])
    for _, start, progress in starts[:12]:
        path = [start]
        distance = 0.
        for _ in range(1200):
            if mission.station[-1] - progress < 14:
                break
            opts = []
            for wp in path[-1].next(5):
                if not directed_step(path[-1],wp):
                    continue
                p = xyz(wp)[:2]
                if not allowed(p, mission.split, bounds):
                    continue
                station, error = mission.match(p, progress, 36)
                if station < progress - 5 or error > 63:
                    continue
                forward = wp.transform.get_forward_vector()
                probe = p + 12 * np.array([forward.x, forward.y])
                next_station, next_dist = mission.match(probe, station, 40)
                score = error + next_dist*.75 + max(0., progress+2-next_station)*3
                opts.append((score, wp, station))
            if not opts:
                break
            _, wp, station = min(opts, key=lambda x: x[0])
            distance += np.linalg.norm(xyz(wp)[:2] - xyz(path[-1])[:2])
            path.append(wp)
            progress = max(station, progress)
            if distance >= mission.speed / 3.6 * (N*DT + 10) + 110:
                return path
        if distance > best[0]:
            best = distance, path
    minimum = max(240, mission.speed / 3.6 * N * DT * .76)
    if best[0] < minimum:
        beam_distance, beam_path = search_branches(mission, bounds, starts)
        if beam_distance > best[0]:
            best = beam_distance, beam_path
    if best[0] < minimum:
        end = xyz(best[1][-1])[:2].tolist() if best[1] else None
        raise RuntimeError(f'{mission.name}: expert path from surveyed roads only {best[0]:.1f} m, last {end}')
    return best[1]


def search_branches(mission, bounds, starts):
    """Retain distinct ramps at splits where the road sketch cannot resolve lane choice yet."""
    frontier = [(score, station, 0., [wp]) for score, wp, station in starts[:50]]
    longest = (0., [])
    desired = mission.speed / 3.6 * (N*DT + 10) + 110
    for _ in range(900):
        next_states = {}
        for score, progress, distance, path in frontier:
            previous = xyz(path[-1])[:2]
            for wp in path[-1].next(5):
                if not directed_step(path[-1],wp):
                    continue
                p = xyz(wp)[:2]
                if not allowed(p, mission.split, bounds):
                    continue
                station, error = mission.match(p, progress, 42)
                if error > 75 or station < progress-6 or mission.station[-1]-max(progress,station) < 12:
                    continue
                step = np.linalg.norm(p-previous)
                cost = .93*score + error + max(0., 2.5-(station-progress))*3
                new_path = path + [wp]
                new_distance = distance+step
                key = (wp.road_id, wp.lane_id, round(wp.s/5))
                rank = cost - max(progress,station)*.20
                if key not in next_states or rank < next_states[key][0]:
                    next_states[key] = (rank, cost, max(station,progress), new_distance, new_path)
                if new_distance > longest[0]:
                    longest = new_distance, new_path
                if new_distance >= desired:
                    return new_distance, new_path
        if not next_states:
            break
        frontier = [s[1:] for s in sorted(next_states.values(), key=lambda s:s[0])[:32]]
    return longest


def future_geometry(poses):
    path = poses[:, :3, 3]
    step = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    arc = np.r_[0., np.cumsum(step)]
    keep = np.r_[True, step > 1e-5]
    return path, arc, keep


def save_inputs(folder, images, scan):
    folder.mkdir()
    for name, image in zip(NAMES, images):
        ok = cv2.imwrite(str(folder / (name + '.png')), image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise IOError(folder / (name + '.png'))
    np.savez_compressed(folder / 'lidar.npz', bev=raster(scan))


def label_frame(folder, i, entry, pose, road, surfaces, static, path, arc, keep, vehicles, nav, command):
    fid = entry['frame']
    local_path = arc[i] + np.arange(1, 41) * 1.5
    if local_path[-1] > arc[-1] or i + FUTURE_TICKS[-1] >= len(vehicles):
        raise RuntimeError(f'{fid}: missing measured future')
    target_xyz = np.column_stack([np.interp(local_path, arc[keep], axis[keep]) for axis in path.T])
    expert = local(target_xyz, pose)[:, :2].astype(np.float32)
    anchor_s = np.array([20.,40.,60.])
    arc_m = np.arange(1,41)*1.5
    def at(stations):
        return np.column_stack([np.interp(stations,arc_m,axis) for axis in expert.T])
    tangent = at(np.minimum(anchor_s+3,60))-at(anchor_s-3)
    left = tangent[:,::-1]*[-1,1]/np.linalg.norm(tangent,axis=1,keepdims=True)
    entry['nav_lateral_m']=((nav-at(anchor_s))*left).sum(1).tolist()
    anchor_z=np.interp(arc[i]+anchor_s,arc[keep],path[keep,2])
    before=np.stack([np.interp(arc[i]+anchor_s-2,arc[keep],v[keep]) for v in path.T],axis=1)
    after=np.stack([np.interp(arc[i]+anchor_s,arc[keep],v[keep]) for v in path.T],axis=1)
    axes=after-before
    axes/=np.linalg.norm(axes,axis=1,keepdims=True).clip(1e-6)
    def direction(xy):
        result=[]
        for (x,y),z,axis in zip(xy,anchor_z,axes):
            w=road.get_waypoint(carla.Location(float(x),float(y),float(z)),project_to_road=False)
            if w is None or abs(w.transform.location.z-z)>3.5:result.append('outside')
            else:
                v=w.transform.get_forward_vector()
                dot=np.array([v.x,v.y,v.z])@axis
                result.append('same' if dot>.5 else 'opposite' if dot<-.5 else 'crossing')
        return result
    entry['sd_road_direction']=direction(entry.pop('_atlas_xy'))
    relative=np.column_stack((nav[:,0],-nav[:,1],np.zeros(3)))
    entry['ego_anchor_direction']=direction((relative@pose[:3,:3].T+pose[:3,3])[:,:2])
    occ = np.zeros((5, 128, 128), np.uint8)
    for grid, dt in zip(occ, FUTURE_TICKS):
        for corners in vehicles[i+dt]:
            pts = local(corners, pose)
            center = pts.mean(0)
            if not (-20 < center[0] < 86 and abs(center[1]) < 25 and abs(center[2]) < 3):
                continue
            polygon(grid, cv2.convexHull(pts[:, :2].astype(np.float32)).reshape(-1, 2))
    mask = surfaces.draw(pose)
    # Semantic returns and OpenDRIVE provenance are supervision only.
    mask[2] |= static
    r,c=np.nonzero(static)
    xy=np.column_stack((80-(r+.5)*.75,19.2-(c+.5)*.30))
    sources=np.zeros((2,20),np.uint8)
    lines,valid=map_vectors(road,pose,np.vstack((pose[:3,3],target_xyz)),xy,mask,hard_sources=sources)
    np.savez_compressed(folder / 'input.npz', sd_nav=nav, cmd_one_hot=np.eye(3, dtype=np.uint8)[command])
    np.savez_compressed(folder / 'labels.npz', map_lines=lines, exists=valid.any(1), point_valid=valid,
                        masks=mask, path_gt=expert, p_occ=occ, hard_sources=sources)
    yaw = math.degrees(math.atan2(pose[1, 0], pose[0, 0]))
    np.savez_compressed(folder / 'index.npz', frame=np.int64(fid), pose_xyzyaw=np.r_[pose[:3, 3], yaw],
                        episode_tick=i, world_from_ego=pose, sd_drift_m=entry['sd_drift_m'], sensor_frames=entry['sensor_frames'])


def samples(trace, scenario, policy_version='v1'):
    poses = np.stack([t['pose'] for t in trace])
    path, arc, keep = future_geometry(poses)
    yaw = np.unwrap(np.arctan2(poses[:, 1, 0], poses[:, 0, 0]))
    stations = np.minimum(arc + 8, arc[-1])
    turn = abs(np.interp(stations, arc[keep], yaw[keep])-yaw) / np.maximum(stations - arc, .1)
    changed = {}
    for i in range(1, N-10):
        before, now = trace[i-1]['waypoint'], trace[i]['waypoint']
        if before.road_id != now.road_id or before.is_junction or now.is_junction or before.lane_id == now.lane_id:
            continue
        future = trace[i:i+10]
        if any(f['waypoint'].road_id != now.road_id or f['waypoint'].lane_id != now.lane_id for f in future):
            continue
        for side in ('left','right'):
            adjacent = same_direction(before, side)
            if adjacent is not None and adjacent.lane_id == now.lane_id:
                for k in range(max(0, i-12), min(N, i+10)):
                    changed[k] = side
    # Retain the measured maneuver around a confirmed, stable adjacent-lane crossing.
    # Expand the crossing window only where commanded and physically moving sideways.
    actual={}
    for i in range(1,N-10):
        before,now=trace[i-1]['waypoint'],trace[i]['waypoint']
        if before.road_id!=now.road_id or before.lane_id==now.lane_id or before.is_junction or now.is_junction:continue
        if not all(t['waypoint'].road_id==now.road_id and t['waypoint'].lane_id==now.lane_id for t in trace[i:i+10]):continue
        side=changed.get(i)
        if side is None:continue
        cmd=1 if side=='left' else 2
        if not any(t['tactical']==cmd for t in trace[max(0,i-10):i+7]):continue
        start=i-1
        while start>0 and i-start<100 and trace[start]['tactical']==cmd and trace[start]['waypoint'].road_id==before.road_id and trace[start]['waypoint'].lane_id==before.lane_id:start-=1
        end=i
        while end<N and end-i<100 and trace[end]['tactical']==cmd and trace[end]['waypoint'].road_id==now.road_id and trace[end]['waypoint'].lane_id==now.lane_id:end+=1
        if start+1>=i or end<=i+1:continue
        for j in range(start+1,end):
            actual[j]=side
    changed=actual
    offset=np.array([t['lane_offset'] for t in trace])
    recovery=set()
    requested=np.array([t['recovery'] for t in trace[:N]],bool)
    for chunk in np.split(np.flatnonzero(requested),np.flatnonzero(np.diff(np.flatnonzero(requested))!=1)+1):
        if len(chunk)<4:continue
        first,last=chunk[0],chunk[-1]
        window=trace[first:last+1]
        if policy_version=='v1':
            if (abs(offset[first])>=.20 and abs(offset[last])<.18 and np.max(abs(offset[chunk]))<=.5
                    and all((t['waypoint'].road_id,t['waypoint'].lane_id)==(window[0]['waypoint'].road_id,window[0]['waypoint'].lane_id) for t in window)):
                recovery.update(chunk.tolist())
        else:
            if any(t['waypoint'].is_junction for t in window):continue
            end_wp=trace[last]['waypoint']
            errs=[]
            for k in chunk:
                wk=trace[k]['waypoint']
                if (wk.road_id,wk.lane_id)==(end_wp.road_id,end_wp.lane_id):
                    errs.append(offset[k])
                elif wk.road_id==end_wp.road_id:
                    cand=next((c for s in ('left','right') if (c:=same_direction(wk,s)) is not None and c.lane_id==end_wp.lane_id),None)
                    if cand is None:
                        errs.append(np.nan)
                    else:
                        d=trace[k]['pose'][:2,3]-xyz(cand)[:2]
                        rv=cand.transform.get_right_vector()
                        errs.append(float(d@[rv.x,rv.y]))
                elif wk.lane_id==end_wp.lane_id:
                    errs.append(offset[k])
                else:
                    errs.append(np.nan)
            errs=np.asarray(errs,float)
            if (np.isfinite(errs).all() and abs(errs[0])>=.35 and abs(offset[last])<.22
                    and np.max(np.abs(errs))<=2.3 and abs(errs[0])-abs(errs[-1])>=.35):
                recovery.update(chunk.tolist())
    return poses, path, arc, keep, turn, changed, recovery


def lane_change_available(waypoint, side, vehicles):
    mark = carla.LaneChange.Left if side == 'left' else carla.LaneChange.Right
    if waypoint.is_junction or not int(waypoint.lane_change) & int(mark):
        return False
    target = same_direction(waypoint, side)
    if target is None or target.is_junction:
        return False
    for lane in (waypoint,target):
        ahead = lane.next(45)
        if len(ahead) != 1 or ahead[0].is_junction or ahead[0].road_id != lane.road_id:
            return False
    center = xyz(target)
    heading = target.transform.get_forward_vector()
    forward = np.array([heading.x, heading.y])
    for corners in vehicles:
        offset = corners.mean(0) - center
        along = offset[:2] @ forward
        cross = offset[:2] @ [-forward[1], forward[0]]
        if -30 < along < 35 and abs(cross) < 2.5 and abs(offset[2]) < 3:
            return False
    return True


class LaneTransition:
    """Follow a smooth, measured lane transition before handing back to Traffic Manager."""

    def __init__(self, waypoint, side, tick):
        adjacent = same_direction(waypoint, side)
        lengths = np.arange(0., 46., 2.)
        source = np.array([xyz(waypoint)]+[xyz(waypoint.next(float(s))[0]) for s in lengths[1:]])
        target = np.array([xyz(adjacent)]+[xyz(adjacent.next(float(s))[0]) for s in lengths[1:]])
        self.station = lengths
        self.source = source
        self.target = target
        self.start_tick = tick
        self.steer = 0.

    def control(self, snapshot, pose):
        station = self.station[np.linalg.norm(self.source[:,:2]-pose[:2,3],axis=1).argmin()]
        look = min(station+7., self.station[-1])
        alpha = min(look/26.,1.)
        blend = alpha**3 * (10. + alpha*(-15.+6.*alpha))
        path = (1.-blend)*self.source + blend*self.target
        aim = np.array([np.interp(look,self.station,c) for c in path.T])
        f, l, _ = local(aim,pose)
        steering = float(np.clip(-math.atan2(5.8*l,max(1.,f*f+l*l))/.64,-.38,.38))
        self.steer = float(np.clip(steering,self.steer-.055,self.steer+.055))
        velocity = snapshot.get_velocity()
        speed = math.hypot(velocity.x,velocity.y)
        return carla.VehicleControl(throttle=float(np.clip(.14*(5.8-speed)+.19,0,.45)),
                                   brake=float(np.clip(.22*(speed-6.2),0,.55)),steer=self.steer)


def episode(world, tm, road, mission, bounds, route, episode_id, out, surfaces, tier, attempt):
    seed = 2913 + int(episode_id.split('_')[1])*113 + attempt
    rng = np.random.default_rng(seed)
    folder = out / 'episodes' / f'{episode_id}.partial-{seed}'
    folder.mkdir()
    weather_name = WEATHER[(int(episode_id.split("_")[1])-1)%len(WEATHER)]
    weather = getattr(carla.WeatherParameters, weather_name)
    weather.sun_altitude_angle = float(rng.uniform(6, 16) if 'Sunset' in weather_name else rng.uniform(32, 78))
    world.set_weather(weather)
    library = world.get_blueprint_library()
    ego_bp = library.find('vehicle.lincoln.mkz')
    ego_bp.set_attribute('role_name', 'hero')
    transform = route[0].transform
    initial_offset = 0.
    if initial_offset and route[0].lane_width < 2.94:
        initial_offset = 0.
    transform.location += carla.Location(z=.35) + transform.get_right_vector()*initial_offset
    world.get_spectator().set_transform(carla.Transform(transform.location + carla.Location(z=12),
                                                        carla.Rotation(pitch=-55)))
    actors, sensors, queues, trace = [], [], [], []
    scenario = dict(weather=weather_name, sun_altitude_deg=weather.sun_altitude_angle,
                    initial_offset_m=initial_offset, seed=seed, mission=mission.name)

    def attach(blueprint, mount):
        s = world.spawn_actor(blueprint, mount, attach_to=ego)
        q = queue.Queue(maxsize=8)
        sensors.append(s)
        queues.append(q)
        s.listen(q.put)

    try:
        for _ in range(12):
            world.tick(45)
        ego = world.spawn_actor(ego_bp, transform)
        actors.append(ego)
        path_tree = cKDTree(np.asarray([xyz(w)[:2] for w in route]))
        candidates = road.get_spawn_points()
        xy = np.asarray([[s.location.x, s.location.y] for s in candidates])
        near, station = path_tree.query(xy)
        selected = np.flatnonzero((near < 115) & (station > 3) & (station < 170))
        rng.shuffle(selected)
        parked = ['vehicle.mini.cooper', 'vehicle.nissan.patrol', 'vehicle.dodge.charger']
        exclusion = 62 if mission.planned_lane_changes and road.name.endswith('Town10HD_Opt') else 23
        for j in selected:
            s = candidates[j]
            if s.location.distance(transform.location) < exclusion:
                continue
            actor = world.try_spawn_actor(library.find(str(rng.choice(parked))), s)
            if actor is not None:
                actors.append(actor)
                actor.set_autopilot(True, tm.get_port())
                tm.vehicle_percentage_speed_difference(actor, float(rng.uniform(12, 36)))
            if len(actors) >= (22 if road.name.endswith('/Town12') else 14):
                break
        limit_kmh=3.6*(sum(np.linalg.norm(xyz(b)[:2]-xyz(a)[:2]) for a,b in zip(route,route[1:]))-100)/(2000*DT+6)
        policy_version=mission.spec.get('driver_policy','v1')
        driver=CollectionDriver(route,min(mission.speed,limit_kmh),seed,mission.split=='train',policy_version=policy_version)
        sizes=locked_sizes()
        hard_tags=[int(getattr(carla.CityObjectLabel,k)) for k in ('GuardRail','Walls','Poles','Fences')]
        pole_tag=int(carla.CityObjectLabel.Poles)
        for (x,y,z,yaw,pitch,fov),size in zip(CAMS,sizes):
            attach(camera_blueprint(library,fov,size),carla.Transform(carla.Location(x,y,z),carla.Rotation(yaw=yaw,pitch=pitch)))
        for semantic in (False,True):
            attach(lidar_blueprint(library,semantic),carla.Transform(carla.Location(z=2.2)))
        collisions=[]
        collision=world.spawn_actor(library.find('sensor.other.collision'),carla.Transform(),attach_to=ego)
        sensors.append(collision)
        collision.listen(lambda event:collisions.append((event.frame,event.other_actor.type_id)))
        for kind in ('gnss','imu'):
            attach(library.find('sensor.other.'+kind),carla.Transform())
        drift_phase=float(rng.uniform(-np.pi,np.pi))
        navigator=RoadNavigator(mission.xy,mission.georef,drift_phase,mission.events,mission.accuracy)
        start = time.perf_counter()
        first_frame = None
        progress = 0.
        jobs = deque()
        traveled = 0.
        last = None
        cutoff = None
        tactical = 0
        action_until = 0
        next_change = 40 if road.name.endswith('Town10HD_Opt') else 160
        requests = []
        transition = None
        with ThreadPoolExecutor(max_workers=3) as pool:
            for tick in range(N+2400):
                fid = world.tick(45)
                snapshot = world.get_snapshot()
                scans = [read(q, fid) for q in queues]
                if snapshot.frame != fid:
                    raise RuntimeError(f'World snapshot mismatch: {fid}')
                ego_state = snapshot.find(ego.id)
                tf = ego_state.get_transform()
                pose = matrix(tf)
                wp = road.get_waypoint(tf.location)
                if last is not None:
                    traveled += np.linalg.norm(last-pose[:2,3])
                last = pose[:2, 3].copy()
                if tick < 6:
                    continue
                if collisions:
                    raise RuntimeError(f'Expert collision at {collisions[-1]}')
                if first_frame is None:
                    first_frame = fid
                i = len(trace)
                if fid != first_frame + i:
                    raise RuntimeError(f'Dropped simulation tick at {fid}')
                if not allowed(pose[:2, 3], mission.split, bounds):
                    raise RuntimeError(f'{mission.name}: measured future left the geographic split')
                vehicles = []
                traffic_velocities = []
                for actor in actors[1:]:
                    snap = snapshot.find(actor.id)
                    if snap is not None:
                        vehicles.append(box(actor, snap.get_transform()))
                        speed=snap.get_velocity()
                        traffic_velocities.append([speed.x,speed.y,speed.z])
                dr = tf.location - wp.transform.location
                right = wp.transform.get_right_vector()
                # During selected highway missions introduce a vehicle approaching the
                # intended gap. Only actual observed gaps and applied controls are labeled.
                if mission.spec.get('traffic_gap_probe') and not requests and driver.excursions:
                    start_station,_,side=driver.excursions[0]
                    if start_station-30<driver.station<start_station-17:
                        adjacent=same_direction(wp,side)
                        behind=adjacent.previous(12) if adjacent else []
                        if len(behind)==1:
                            placement=behind[0].transform
                            placement.location.z+=.35
                            other=world.try_spawn_actor(library.find('vehicle.mini.cooper'),placement)
                            if other is not None:
                                actors.append(other)
                                other.set_target_velocity(placement.get_forward_vector()*(ego_state.get_velocity().length()+2.))
                                other.set_autopilot(True,tm.get_port())
                                tm.auto_lane_change(other,False)
                                tm.set_desired_speed(other,mission.speed+10)
                                requests.append(dict(frame=fid,side=side,scenario='approaching_target_lane'))
                red_light=ego.is_at_traffic_light() and ego.get_traffic_light_state() in (
                    carla.TrafficLightState.Red,carla.TrafficLightState.Yellow)
                control,tactical,recover=driver.step(pose,ego_state.get_velocity(),vehicles,traffic_velocities,red_light)
                active=ego.get_control()
                applied=[active.steer,active.throttle,active.brake]
                # CARLA rounds vehicle controls in the server RPC. Validate the
                # reported actuation on the first tick following the command.
                if trace and not np.allclose(applied,trace[-1]['requested'],atol=.0011,rtol=0):
                    raise RuntimeError(f'{fid}: control alignment {applied} vs {trace[-1]["requested"]}')
                fix=fix_values(scans[5],scans[6])
                nav=navigator.update(fix[0],fix[1],fix[3],DT*i)
                if navigator.error>20 or navigator.station+60>navigator.arc[-1] and i<N:
                    raise ValueError(f'{fid}: SD route mismatch at {navigator.station:.1f} m, distance {navigator.error:.1f} m, route end {navigator.arc[-1]:.1f}')
                ego.apply_control(control)
                sem=np.frombuffer(scans[4].raw_data,SEM)
                pt=sem['xyz']
                keep_sem=np.isin(sem['tag'],hard_tags)&(pt[:,2]>-2.05)&(pt[:,2]<.8)
                nearby_pole=bool(np.any((sem['tag']==pole_tag)&(np.hypot(pt[:,0],pt[:,1])<12)))
                uv=np.floor(pixels(pt[keep_sem,:2]*[1,-1])+.5).astype(int)
                uv=uv[((uv>=0)&(uv<128)).all(1)]
                semantic_mask=np.zeros((128,128),np.uint8)
                semantic_mask[uv[:,1],uv[:,0]]=1
                trace.append(dict(frame=fid,pose=pose,waypoint=wp,vehicles=vehicles,progress=driver.station,wait_reason=driver.wait_reason,
                                  lane_offset=dr.x*right.x+dr.y*right.y,tactical=tactical,recovery=recover,
                                  atlas_xy=nav@navigator.axes.T+navigator.xy,fix=fix,requested=[control.steer,control.throttle,control.brake],applied=applied,
                                  speed=ego_state.get_velocity().length(),nav=nav,sd_station=navigator.station,
                                  sd_drift=navigator.drift,hint=navigator.instruction(),
                                  semantic=semantic_mask,pole=nearby_pole,sensor_frames=[z.frame for z in scans]))
                if i < N:
                    images = [bgr(img,size) for img,size in zip(scans[:3],sizes)]
                    scan = np.frombuffer(scans[3].raw_data, np.float32).reshape(-1,4).copy()
                    scan[:,1] *= -1
                    scan[:,2] += 2.2
                    if len(jobs) >= 24:
                        jobs.popleft().result()
                    jobs.append(pool.submit(save_inputs, folder / f'{fid:07d}', images, scan))
                    if (i+1) % 250 == 0:
                        print(episode_id, mission.name, 'retained', i+1, round((i+1)/(time.perf_counter()-start),2), 'FPS',flush=True)
                if i == N-1:
                    cutoff = traveled
                if i >= N-1+FUTURE_TICKS[-1] and traveled-cutoff >= 61:
                    break
            for job in jobs:
                job.result()
        if cutoff is None or len(trace) < N+80 or traveled - cutoff < 60:
            raise RuntimeError(f'{episode_id}: incomplete retained frames or measured future')
        traveled_roads = {t['waypoint'].road_id for t in trace[:N]}
        if mission.required_roads - traveled_roads:
            raise RuntimeError(f'{mission.name}: did not drive required road IDs {sorted(mission.required_roads-traveled_roads)}')
        capture_s = time.perf_counter() - start
    finally:
        for s in reversed(sensors):
            s.stop()
            s.destroy()
        for a in reversed(actors):
            a.destroy()

    poses, path, arc, keep, turn, changed, recovery = samples(trace, scenario, policy_version=policy_version)
    boxes = [t['vehicles'] for t in trace]
    rows = []
    np.savez_compressed(folder/'drive-log.npz',frame=[t['frame'] for t in trace],dt=DT,
                        expert_pose=[t['pose'] for t in trace],wait_reason=[t['wait_reason'] for t in trace],fix=[t['fix'] for t in trace],speed_mps=[t['speed'] for t in trace],
                        requested=[t['requested'] for t in trace],applied=[t['applied'] for t in trace],
                        effective_frame=[t['frame']+1 for t in trace[:-1]],tactical=[t['tactical'] for t in trace],
                        route_hint=[t['hint'][1] for t in trace],hint_distance=[t['hint'][0] for t in trace])
    grade = np.divide(np.interp(np.minimum(arc+10,arc[-1]), arc[keep], path[keep,2])-path[:,2],10)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = deque()
        for i, t in enumerate(trace[:N]):
            if len(jobs) >= 24:
                jobs.popleft().result()
            pose = poses[i]
            drift=t['sd_drift']
            directed_station=t['sd_station']
            nav=t['nav']
            lane_change = changed.get(i)
            future = int(np.searchsorted(arc,arc[i]+60,'right'))
            wp = t['waypoint']
            nearby = local(np.asarray(boxes[i]).reshape(-1,3),pose).reshape(-1,4,3).mean(1)
            close_car = np.any((nearby[:,0]>1) & (nearby[:,0]<27) & (abs(nearby[:,1])<1.7) & (abs(nearby[:,2])<2))
            tags = dict(straight=bool(turn[i]<.008 and not wp.is_junction), sharp_curve=bool(turn[i]>=.035),
                        junction=bool(wp.is_junction), uphill=bool(grade[i]>.025), downhill=bool(grade[i]<-.025),
                        following=bool(close_car), roadside_pole=t['pole'])
            row = dict(map=road.name.rsplit('/',1)[-1], episode=episode_id, frame=t['frame'], split=mission.split,
                       kappa_max=float(turn[i:future].max()), lane_change=lane_change, recovery=i in recovery,
                       road_id=wp.road_id,lane_id=wp.lane_id,sd_station_m=directed_station,_atlas_xy=t['atlas_xy'],
                       sd_drift_m=float(drift),sensor_frames=t['sensor_frames'],wait_reason=t['wait_reason'], **tags)
            rows.append(row)
            jobs.append(pool.submit(label_frame,folder/f'{t["frame"]:07d}',i,row,pose,road,surfaces,t['semantic'],
                                    path,arc,keep,boxes,nav,t['tactical']))
        for job in jobs:
            job.result()
    final = out / 'episodes' / episode_id
    meta = dict(frames=rows, quality=tier, capture_s=capture_s, elapsed_s=time.perf_counter()-start,
                final_frame_future_m=float(arc[-1]-arc[N-1]), final_frame_future_ticks=len(trace)-N,
                plan_world_xy=mission.xy.tolist(), sd_navigation='independent-SD-GNSS-IMU-v1',georef=mission.georef,events=mission.events,sd_accuracy=mission.accuracy,drift_phase=drift_phase,rgb_sizes=sizes,hard_sources='1=OpenDRIVE carriageway edge, 2=semantic LiDAR',
                actor_count=len(actors),retained_road_ids=sorted(traveled_roads),
                scheduled_recoveries=driver.recovery_plans,scheduled_lane_excursions=len(driver.excursions),
                observed_collisions=collisions,
                pose_xy_bounds=[path[:N,:2].min(0).tolist(),path[:N,:2].max(0).tolist()], planned_commands=requests, **scenario)
    (folder/'episode.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    validate(folder, meta)
    folder.rename(final)
    print(episode_id, 'COMMITTED', mission.split, len(rows), flush=True)
    return rows


def pilot(out,world,tm,road,mission,bounds,path,surfaces,tier):
    global N
    pilot_dir=out/f'preflight-{time.time_ns()}'
    (pilot_dir/'episodes').mkdir(parents=True)
    retained=N
    N=48
    try:
        short=copy.copy(mission)
        short.required_roads=set()
        rows=episode(world,tm,road,short,bounds,path,'ep_0001',pilot_dir,surfaces,tier,0)
        source_points=np.zeros((2,3),int)
        for row in rows:
            folder=pilot_dir/'episodes'/row['episode']/f'{row["frame"]:07d}'
            assert row['sensor_frames']==[row['frame']]*7
            sample=load_observation(folder)
            assert [v.shape[:2][::-1] for v in sample['rgb']]==list(locked_sizes())
            with np.load(folder/'labels.npz') as labels:
                for j,src in enumerate(labels['hard_sources']):
                    source_points[j]+=np.bincount(src,minlength=3)
        if not np.all(source_points[:,1:].sum(1)>0):
            raise ValueError(f'Missing side in hard-boundary pilot: {source_points}')
        import subprocess,sys
        subprocess.run([sys.executable,'-m','v2_mapless_wa.sd_replay',str(pilot_dir/'episodes'/'ep_0001')],check=True)
        result=dict(frames=N,simulator_free_replay=True,semantic_and_xodr=source_points.tolist(),rgb=locked_sizes(),sync=True)
        (out/'pilot.json').write_text(json.dumps(result,indent=2))
        print('Pilot passed',result,flush=True)
    finally:
        N=retained


def validate(folder, meta):
    rows = meta['frames']
    frames = [r['frame'] for r in rows]
    if meta.get('sd_navigation')!='independent-SD-GNSS-IMU-v1':raise ValueError('Legacy navigation provenance')
    if len(frames) != N or not np.all(np.diff(frames)==1):
        raise ValueError(f'{folder}: invalid continuous frame count')
    if meta['final_frame_future_ticks'] < 80 or meta['final_frame_future_m'] < 60:
        raise ValueError(f'{folder}: incomplete future support')
    for frame in frames:
        for name in FILES:
            if (folder/f'{frame:07d}'/name).stat().st_size == 0:
                raise ValueError(f'{folder}/{frame:07d}/{name}: empty file')


def summarize(out, episodes, rows, bounds):
    fields = ('straight','sharp_curve','junction','uphill','downhill','following','roadside_pole','recovery')
    distributions = {}
    for town, split in sorted({(r['map'],r['split']) for r in rows}):
        subset = [r for r in rows if r['map']==town and r['split']==split]
        counts = {field:sum(bool(r[field]) for r in subset) for field in fields}
        counts.update({f'lane_change_{side}':sum(r['lane_change']==side for r in subset) for side in ('left','right')})
        distributions[f'{town}/{split}'] = dict(frames=len(subset), counts=counts,
                                                fractions={k:v/len(subset) for k,v in counts.items()})
    finished_bytes = 0
    for episode in episodes:
        first, *_, last = episode['frames']
        folder = out/'episodes'/first['episode']
        for item in (first,episode['frames'][len(episode['frames'])//2],last):
            fd = folder/f'{item["frame"]:07d}'
            sample = load_observation(fd)
            if [v.shape[:2][::-1] for v in sample['rgb']]!=list(locked_sizes()) or sample['lidar'].shape!=(6,128,128) or sample['sd_nav'].shape!=(3,2):
                raise ValueError(f'{fd}: observation shape verification failed')
            with np.load(fd/'labels.npz') as labels:
                expected = dict(map_lines=(7,20,2),exists=(7,),point_valid=(7,20),masks=(4,128,128),
                                path_gt=(40,2),p_occ=(5,128,128))
                if any(labels[k].shape != size or not np.isfinite(labels[k]).all() for k,size in expected.items()):
                    raise ValueError(f'{fd}: teacher label verification failed')
        for item in episode['frames']:
            fd = folder/f'{item["frame"]:07d}'
            with np.load(fd/'index.npz') as index:
                region = bounds if item['map']=='Town12' else None
                if int(index['frame'])!=item['frame'] or not allowed(index['pose_xyzyaw'][:2],item['split'],region):
                    raise ValueError(f'{fd}: frame or geographic split mismatch')
            finished_bytes += sum(f.stat().st_size for f in fd.iterdir())
        finished_bytes += (folder/'episode.json').stat().st_size
    summary = dict(frames=len(rows), episodes=len(episodes), distribution=distributions,
                   capture_s=sum(ep['capture_s'] for ep in episodes),
                   committed_episode_s=sum(ep['elapsed_s'] for ep in episodes),
                   finished_bytes=finished_bytes+(out/'manifest.jsonl').stat().st_size,
                   retained_frame_gaps=0,
                   recovery_requested_episodes=sum(bool(ep.get('initial_offset_m')) for ep in episodes),
                   recovery_verified_episodes=sum(any(r['recovery'] for r in ep['frames']) for ep in episodes),
                   weather=Counter(ep['weather'] for ep in episodes),
                   validation_bounds_xy=bounds.tolist() if bounds is not None else None)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


def sync_manifest(out, episodes):
    path = out/'manifest.jsonl'
    rows = [row for ep in episodes for row in ep['frames']]
    target = {(r['episode'],r['frame']):r for r in rows}
    if len(rows) != len(target):
        raise ValueError('Duplicate frame keys in committed episodes')
    current = set()
    with path.open('a+b') as f:
        f.seek(0)
        while True:
            position = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                f.truncate(position)
                break
            row = json.loads(line)
            key = row['episode'],row['frame']
            if key in current or target.get(key) != row:
                raise ValueError(f'Manifest conflicts with committed episode: {key}')
            current.add(key)
        for key, row in target.items():
            if key not in current:
                f.write(json.dumps(row,separators=(',',':')).encode()+b'\n')
        f.flush()
        os.fsync(f.fileno())
    return rows


def main(args):
    out = args.out
    gate = json.loads(args.calibration.read_text(encoding='utf-8'))
    ab = json.loads(args.ab.read_text(encoding='utf-8'))
    tier = locked_tier()
    if ab['selected'] != tier or gate['tier'] != tier or not gate['white_line_alignment']['passed']:
        raise ValueError('The locked tier does not have passing A/B and straight-road evidence')
    actual_args = psutil.Process(int((out/'server.pid').read_text())).cmdline()
    if not all(flag in actual_args for flag in (*WINDOW_FLAGS,f'-quality-level={tier}',f'-carla-rpc-port={args.port}')):
        raise ValueError('The active simulator process differs from the locked renderer')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    val_bounds = np.asarray(plan['Town12']['validation_bounds_xy'], dtype=float)
    bounds_file = out/'validation_bounds.json'
    if bounds_file.exists() and not np.array_equal(json.loads(bounds_file.read_text()),val_bounds):
        raise ValueError('The Town12 validation region is locked across resumed runs')
    if not bounds_file.exists():
        bounds_file.write_text(json.dumps(val_bounds.tolist()),encoding='utf-8')
    (out/'episodes').mkdir(exist_ok=True)
    committed = []
    for folder in sorted((out/'episodes').glob('ep_*')):
        if '.partial-' in folder.name:
            continue
        meta = json.loads((folder/'episode.json').read_text(encoding='utf-8'))
        validate(folder,meta)
        if meta['quality'] != tier or meta.get('sd_navigation')!='independent-SD-GNSS-IMU-v1':
            raise ValueError(f'{folder}: renderer differs from locked config')
        committed.append(meta)
    if committed and not args.resume:
        raise ValueError('Use --resume with committed episodes')
    rows = sync_manifest(out,committed)
    counts = Counter(r['map'] for r in rows)
    targets = {'Town12':args.target_town12,'Town10HD_Opt':args.target_town10}
    c = carla.Client('127.0.0.1',args.port)
    c.set_timeout(150)
    check_versions(c)
    for town in args.maps:
        target = targets[town]
        if counts[town] > target or (target-counts[town]) % N:
            raise ValueError(f'{town}: target is incompatible with {N}-frame segments')
        if counts[town] == target:
            continue
        data = plan[town]
        bounds = val_bounds if town=='Town12' else None
        missions = [Mission(m) for m in data['missions']]
        if town=='Town12' and not args.supplement and not any(m.split=='val' for m in missions):
            raise ValueError('A geographically isolated Town12 validation route is required')
        world = c.get_world()
        if world.get_map().name.rsplit('/',1)[-1]!=town:
            world = c.load_world(town)
        road = world.get_map()
        tm = c.get_trafficmanager(args.port+5)
        original = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode=True
        settings.fixed_delta_seconds=DT
        settings.no_rendering_mode=False
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(714)
        try:
            print('Building offline map supervision',town,flush=True)
            surfaces = Surfaces(road,indexed=True)
            surveyed_start_candidates = road.generate_waypoints(10)
            print('Map supervision ready',town,len(surfaces.centers),flush=True)
            for mission in missions:
                if counts[town] == target:
                    break
                if any(ep['mission']==mission.name for ep in committed):
                    continue
                revised=json.loads(args.plan.read_text(encoding='utf-8'))[town]['missions']
                mission=Mission(next(spec for spec in revised if spec['name']==mission.name))
                if mission.spec.get('expert_search')=='both_directions':
                    from .teacher_route import route as find_teacher
                    path=find_teacher(road,mission,bounds,surveyed_start_candidates)
                else:
                    path=expert_route(road,mission,bounds,surveyed_start_candidates)
                print('Expert route',mission.name,len(path),'waypoints',flush=True)
                ep_id = f'ep_{len(committed)+1:04d}'
                attempt = sum(1 for _ in (out/'episodes').glob(f'{ep_id}.partial-*'))
                if not committed and not (out/'pilot.json').is_file():
                    pilot(out,world,tm,road,mission,bounds,path,surfaces,tier)
                new_rows = episode(world,tm,road,mission,bounds,path,ep_id,out,surfaces,tier,attempt)
                meta = json.loads((out/'episodes'/ep_id/'episode.json').read_text(encoding='utf-8'))
                committed.append(meta)
                counts[town] += len(new_rows)
                sync_manifest(out,committed)
            if counts[town] != target:
                raise RuntimeError(f'{town}: reached {counts[town]} of {target} with available verified missions')
        finally:
            tm.set_synchronous_mode(False)
            world.apply_settings(original)
    if set(args.maps) == set(COLLECTION_MAPS) and any(counts[t]!=targets[t] for t in targets):
        raise RuntimeError(f'Capture stopped at {dict(counts)}')
    if all(counts[t]==targets[t] for t in targets):
        rows = sync_manifest(out,committed)
        if not args.supplement and not any(r['map']=='Town12' and r['split']=='val' for r in rows):
            raise RuntimeError('The completed counts do not include a Town12 validation episode')
        summarize(out,committed,rows,val_bounds)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--ab',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--port',type=int,default=2460)
    p.add_argument('--maps',choices=COLLECTION_MAPS,nargs='+',default=('Town12','Town10HD_Opt'))
    p.add_argument('--target_town12',type=int,default=12000)
    p.add_argument('--target_town10',type=int,default=8000)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--supplement',action='store_true')
    args = p.parse_args()
    with (args.out/'capture.lock').open('a+b') as lock:
        if lock.tell()==0:
            lock.write(b'\0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main(args)
