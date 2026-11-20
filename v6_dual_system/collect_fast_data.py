"""Acquire route-isolated RGB/LiDAR expert episodes in the project CARLA 0.9.15 environment."""
import argparse
import json
import math
import random
from collections import deque

import carla
import cv2
import numpy as np

from .sensors import CALIBRATION, CAL_ID, DATA, DT, MissingFrame, Sensors, bev, encode_rgb, local_client, state, write_json

SCENES = ("launch", "transition", "highway", "braking", "left", "right", "recovery")
MIX = (.10, .12, .23, .16, .12, .12, .15)
TOWNS = {"Town04": (12000, "train"), "Town06": (3000, "validation")}


def wrap(degrees):
    return (degrees + 180) % 360 - 180


def neighbor(wp, left):
    other = wp.get_left_lane() if left else wp.get_right_lane()
    if other and other.lane_type == carla.LaneType.Driving and wp.lane_id * other.lane_id > 0:
        return other


def path_from(wp, distance=1650):
    path = [wp]
    for _ in range(distance // 2):
        options = path[-1].next(2)
        if not options:
            break
        path.append(min(options, key=lambda w: (abs(wrap(w.transform.rotation.yaw-path[-1].transform.rotation.yaw)), w.id)))
    return path


class Road:
    def __init__(self, path):
        self.path = path
        self.xy = np.array([[w.transform.location.x, w.transform.location.y] for w in path])
        heading = np.deg2rad([w.transform.rotation.yaw for w in path])
        self.left = np.stack((np.sin(heading), -np.cos(heading)), axis=1)
        self.index = 0

    def locate(self, tr):
        xy = np.array([tr.location.x, tr.location.y])
        start = max(0, self.index-4)
        dx = self.xy[start:self.index+40] - xy
        self.index = start + np.argmin(np.sum(dx*dx, axis=1))
        offset = float((xy-self.xy[self.index]) @ self.left[self.index])
        return offset

    def curvature(self):
        end = min(self.index+10, len(self.path)-1)
        angle = wrap(self.path[end].transform.rotation.yaw-self.path[self.index].transform.rotation.yaw)
        return abs(math.radians(angle)) / max(2, 2*(end-self.index))


def choose_path(map_, rng, scene, curve):
    starts = [w for w in map_.generate_waypoints(18)
              if not w.is_junction and w.lane_width >= 3.0 and
              (neighbor(w, True) or neighbor(w, False))]
    rng.shuffle(starts)
    best = None
    for start in starts[:120]:
        route = path_from(start)
        if len(route) < 400:
            continue
        segment = route[:320:8]
        if any(w.is_junction for w in segment[:9]):
            continue
        if scene in ("left", "right") and not all(neighbor(w, scene=="left") for w in segment[:18]):
            continue
        if scene=="paired" and not all(neighbor(w,True) and neighbor(w,False) for w in segment[:16]):
            continue
        bends = np.array([abs(wrap(b.transform.rotation.yaw-a.transform.rotation.yaw))
                          for a, b in zip(segment, segment[1:])])
        score = np.quantile(bends, .9)
        target = 4 if curve else .2
        rank = abs(score-target) + 5*sum(w.is_junction for w in segment)
        if best is None or rank < best[0]:
            best = rank, route
        if rank < 1:
            break
    if best is None:
        raise RuntimeError(f"No eligible road for {scene}")
    return best[1]


def spawn(world, wp, model):
    tr = wp.transform
    tr.location.z += .35
    return world.spawn_actor(world.get_blueprint_library().find(model), tr)


class Expert:
    def __init__(self, road):
        self.road = road
        self.integral = 0.

    def control(self, actor, target, offset=0):
        tr = actor.get_transform()
        v = actor.get_velocity().length()
        road = self.road
        road.locate(tr)
        ahead = max(3.5, min(24, 3.5+.62*v))
        index = min(road.index + max(2, round(ahead/2)), len(road.xy)-1)
        dx, dy = road.xy[index] + road.left[index]*offset - [tr.location.x, tr.location.y]
        yaw = math.radians(tr.rotation.yaw)
        lateral = -dx*math.sin(yaw) + dy*math.cos(yaw)
        steer = math.atan2(2*2.85*lateral, max(dx*dx+dy*dy, 1)) / .70
        limit = math.sqrt(3.8 / max(road.curvature(), .0001))
        error = min(target / 3.6, limit) - v
        self.integral = float(np.clip(self.integral + error*DT, -3, 6))
        effort = .35*error + .12*self.integral + .06 + .00025*v*v
        return carla.VehicleControl(throttle=float(np.clip(effort,0,1)), brake=float(np.clip(-effort,0,1)),
                                    steer=float(np.clip(steer,-1,1)))


def occ(boxes, inverse):
    mask = np.zeros((128,128), np.uint8)
    for vertices in boxes:
        pts = vertices @ inverse[:3,:3].T + inverse[:3,3]
        if pts[:,2].min() > 3 or pts[:,2].max() < -1:
            continue
        pix = np.stack((pts[:,0]+15.5, (-pts[:,1]+19.2)/.3-.5), axis=1)
        cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(pix*256).astype(np.int32)), 1, shift=8)
    return mask


def label_episode(rows, scene, limit):
    poses = np.array([r["pose"] for r in rows])
    centers = poses[:,:3,3]
    result = {k: [] for k in ("frame","history","state","target","intent","xy","occ","offset","curvature","recovery")}
    for i in range(6, len(rows)-80):
        window = rows[i-6:i+81]
        if any(r["disturb"] for r in window) or window[-1]["frame"] - window[0]["frame"] != 86:
            continue
        if scene == "recovery" and not rows[i]["recovery"]:
            continue
        inv = np.linalg.inv(poses[i])
        future = centers[i+2:i+81:2] @ inv[:3,:3].T + inv[:3,3]
        xy = future[:,:2] * [1,-1]
        offset = rows[i+60]["offset"] - rows[i]["offset"]
        intent = 1 if offset >= 1.2 else 2 if offset <= -1.2 else 0
        event = {"frame":i,"history":i-6,"state":rows[i]["state"],"target":rows[i]["target"],
                 "intent":intent,"xy":xy,"offset":rows[i]["offset"],"curvature":rows[i]["curvature"],
                 "recovery":rows[i]["recovery"],
                 "occ":np.packbits(np.stack([occ(rows[i+d]["boxes"],inv) for d in (0,20,40,60)]), axis=-1)}
        for key in result:
            result[key].append(event[key])
    # Keep a deterministic spread over the complete episode when finishing a quota.
    ids = np.linspace(0, len(result["frame"])-1, min(len(result["frame"]),limit), dtype=int)
    return {k:np.asarray(v)[ids] for k,v in result.items()}


def episode(client, world, map_, town, split, scene, seed, limit, folder, adaptation=None):
    plan=adaptation or {}
    route_seed=plan.get("route_seed",seed)
    rng = random.Random(route_seed)
    path = choose_path(map_,rng,plan.get("route_kind",scene),route_seed%2==0)
    road = Road(path)
    expert = Expert(road)
    target = plan.get("speed",{"launch":38,"transition":76,"highway":120,"braking":98,"left":85,"right":85,"recovery":68}[scene])
    actors, others, rows = [], [], []
    rig = collision = None
    hits = []
    history = deque(maxlen=7)
    pending = deque()
    length = plan.get("ticks",740 if scene != "recovery" else 1100)
    folder.mkdir(parents=True,exist_ok=False)
    try:
        ego = spawn(world,path[0],"vehicle.tesla.model3")
        actors.append(ego)
        # Leading traffic is independently controlled along its own route.
        start_idx = 20 if scene in ("braking","left","right") else 38
        lead_wp = path[85 if plan.get("route_kind")=="paired" else start_idx]
        if scene not in ("braking","left","right") and plan.get("route_kind")!="paired":
            lead_wp = neighbor(lead_wp,True) or neighbor(lead_wp,False) or lead_wp
        lead = spawn(world,lead_wp,"vehicle.audi.tt")
        actors.append(lead)
        others.append((lead,Expert(Road(path_from(lead_wp)))))
        for a in actors:
            a.apply_control(carla.VehicleControl(brake=1))
        for _ in range(15):
            world.tick()
        rig = Sensors(world,ego)
        collision = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),carla.Transform(),attach_to=ego)
        collision.listen(lambda e: hits.append(e.frame))
        for _ in range(10):
            world.tick()
        # Store signed, calibrated raw scans and JPEGs with explicit offsets.
        rgb_index, lidar_index = [0], [0]
        disturbance = recovery = False
        next_push, push_start, release, change_start = plan.get("event_tick",100), None, None, None
        with (folder/"rgb.bin").open("wb") as images, (folder/"lidar.bin").open("wb") as scans, (folder/"bev.bin").open("wb") as grids:
            for tick in range(length):
                offset = road.locate(ego.get_transform())
                speed = ego.get_velocity().length()
                if scene=="recovery":
                    if tick >= next_push and speed>plan.get("push_min_mps",10) and not (disturbance or recovery):
                        disturbance, push_start = True, tick
                    if disturbance and abs(offset)>=.45:
                        disturbance = False
                        recovery = abs(offset)<=.9
                        release, next_push = tick, tick+140
                    if disturbance and tick-push_start>35:
                        disturbance, next_push = False, tick+60
                    if recovery and tick-release>8 and abs(offset)<.12:
                        recovery = False
                shift = 0.
                if scene in ("left","right") and tick>=plan.get("event_tick",112):
                    change_start = tick if change_start is None else change_start
                    u = min(1., (tick-change_start)*DT/3.5)
                    shift = (1 if scene=="left" else -1)*path[road.index].lane_width*(u*u*(3-2*u))
                ego_target = target
                for npc,driver in others:
                    npc_speed = 58 if scene in ("left","right") else 82
                    if plan.get("route_kind")=="paired":npc_speed=target
                    if scene=="braking":
                        npc_speed = 0 if 290<=tick<=470 else 77
                    npc_control = driver.control(npc,npc_speed)
                    if scene=="braking" and 290<=tick<=340:
                        npc_control.throttle, npc_control.brake = 0., 1.
                    npc.apply_control(npc_control)
                    tr = ego.get_transform()
                    rel = tr.get_inverse_matrix()
                    loc = npc.get_location()
                    ahead, side, _ = (np.asarray(rel) @ [loc.x,loc.y,loc.z,1])[:3]
                    if abs(side)<2.2 and 0<ahead<85:
                        gap = ahead-4.8
                        ego_target = min(ego_target, max(0,npc.get_velocity().length()*3.6 + 1.6*(gap-5-speed*1.15)))
                command = expert.control(ego,ego_target,shift)
                if tick < 16:
                    command = carla.VehicleControl(brake=1)
                if disturbance:
                    command.steer = float(np.clip(command.steer + (.12 if (next_push//50)%2 else -.12),-1,1))
                ego.apply_control(command)
                frame = world.tick()
                tr = ego.get_transform()
                pose = np.asarray(tr.get_matrix())
                boxes = [np.array([[p.x,p.y,p.z] for p in a.bounding_box.get_world_vertices(a.get_transform())]) for a,_ in others]
                actual_offset = road.locate(tr)
                pending.append(dict(frame=frame,pose=pose,state=state(ego),target=target,boxes=boxes,offset=actual_offset,
                                    disturb=disturbance,recovery=recovery,curvature=road.curvature()))
                # GPU readback trails physics. Retain the exact same-tick labels.
                if len(pending)<4:
                    continue
                row = pending.popleft()
                try:
                    rgb,scan = rig.read(row["frame"])
                except MissingFrame:
                    history.clear()
                    continue
                if not len(scan) or rgb.max()==rgb.min():
                    raise ValueError("Empty sensor payload")
                rows.append(row)
                pose = row["pose"]
                history.append((scan,pose))
                if not row["disturb"]:
                    images.write(encode_rgb(rgb))
                    scans.write(scan.tobytes())
                rgb_index.append(images.tell())
                lidar_index.append(scans.tell())
                grids.write(bev(scan,history[0][0],pose,history[0][1]).tobytes())
                if hits or road.index >= len(path)-30:
                    break
        # Cut collision tails before assigning four-second future labels.
        if hits:
            rows = [r for r in rows if r["frame"]<min(hits)]
        if len(rows)<88:
            raise RuntimeError("Episode too short for causal history and future labels")
        labels = label_episode(rows,scene,len(rows) if plan else limit)
        if plan.get("route_kind")=="paired":
            expected={"highway":0,"left":1,"right":2}[scene]
            event=plan["event_tick"]
            # Windows before and during the physical maneuver, paired by route and schedule.
            keep=(labels["intent"]==expected)&(labels["frame"]>=event-40)&(labels["frame"]<=event+64)
            labels={k:v[keep] for k,v in labels.items()}
        count = len(labels["frame"])
        np.savez_compressed(folder/"labels.npz", **labels)
        np.savez(folder/"stream.npz",rgb=np.asarray(rgb_index),lidar=np.asarray(lidar_index),
                 frame=np.array([r["frame"] for r in rows]),poses=np.array([r["pose"] for r in rows]))
        meta = dict(town=town,split=split,scene=scene,seed=seed,count=count,calibration=CAL_ID,
                    dt=DT,raw_frames=len(rgb_index)-1,collisions=hits,route_start=int(path[0].id),
                    route=[[int(w.road_id),int(w.lane_id),float(w.s)] for w in path[::8]],adaptation=adaptation)
        write_json(folder/"complete.json",meta)
        return count
    finally:
        if rig: rig.close()
        if collision:
            collision.stop()
            collision.destroy()
        for actor in reversed(actors):
            actor.destroy()


def collect(town,port):
    total,split = TOWNS[town]
    root = DATA/town
    root.mkdir(parents=True,exist_ok=True)
    write_json(DATA/"calibration.json",CALIBRATION)
    client = local_client(port)
    world = client.get_world()
    if not world.get_map().name.endswith("/"+town):
        world = client.load_world(town)
    map_ = world.get_map()
    original = world.get_settings()
    cfg = world.get_settings()
    cfg.synchronous_mode, cfg.fixed_delta_seconds, cfg.no_rendering_mode = True, DT, False
    world.apply_settings(cfg)
    try:
        for group,(scene,fraction) in enumerate(zip(SCENES,MIX)):
            quota = round(total*fraction)
            done = sum(json.loads(p.read_bytes())["count"] for p in root.glob(scene+"-*/complete.json"))
            attempt = 0
            while done<quota:
                attempt += 1
                seed = (10000 if split=="train" else 20000)+group*500+attempt
                folder = root/f"{scene}-{seed}"
                if folder.exists(): continue
                count = episode(client,world,map_,town,split,scene,seed,quota-done,folder)
                done += count
                print(town,scene,seed,count,f"{done}/{quota}",flush=True)
                if attempt>80: raise RuntimeError(f"Insufficient eligible frames for {town}/{scene}")
            if scene=="highway":
                paths = list(root.glob("highway-*/labels.npz"))
                def peak(p):
                    with np.load(p) as labels:
                        return max(labels["state"][:,0]*3.6,default=0)
                if max(map(peak,paths),default=0)<120:
                    for extra in range(3):
                        extra_seed=(18000 if split=="train" else 28000)+extra
                        folder=root/f"endpoint-{extra_seed}"
                        if not folder.exists():
                            episode(client,world,map_,town,split,scene,extra_seed,300,folder)
                        if (folder/"labels.npz").exists() and peak(folder/"labels.npz")>=120:
                            break
    finally:
        world.apply_settings(original)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--town",choices=TOWNS)
    p.add_argument("--port",type=int,default=2176)
    p.add_argument("--spatial",action="store_true")
    a=p.parse_args()
    if a.spatial:
        from .collect_spatial import collect as collect_spatial
        collect_spatial(a.port,a.town)
    elif a.town:collect(a.town,a.port)
    else:p.error("Choose --town or --spatial")
