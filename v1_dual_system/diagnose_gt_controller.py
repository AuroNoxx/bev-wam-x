"""Drive fixed CARLA routes with a delayed map-centerline plan and the deployed controller."""
import argparse
import json
import math
import random
import time

import carla
import numpy as np

from .collect_fast_data import choose_path, path_from, spawn
from .legacy_controller import Controller, current_plan
from .curve_controller import CurveController
from .sensors import DT, LOG, local_client, write_json


class Reference:
    def __init__(self, waypoints):
        locations = [point.transform.location for point in waypoints]
        xyz = np.array([[p.x, p.y, p.z] for p in locations])
        length = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
        self.source_index = np.flatnonzero(np.r_[True, length > .001])
        self.xyz = xyz[self.source_index]
        self.segments = np.diff(self.xyz[:, :2], axis=0)
        self.length = np.linalg.norm(self.segments, axis=1)
        self.arc = np.r_[0., np.cumsum(self.length)]
        self.heading = np.unwrap(np.arctan2(self.segments[:, 1], self.segments[:, 0]))
        self.index = 0

    def project(self, pose):
        start = max(0, self.index-5)
        end = min(len(self.segments), self.index+25)
        delta = pose[:2, 3] - self.xyz[start:end, :2]
        segment = self.segments[start:end]
        u = np.clip(np.sum(delta*segment, axis=1) / self.length[start:end]**2, 0, 1)
        residual = delta - u[:, None]*segment
        best = np.argmin(np.sum(residual**2, axis=1))
        self.index = start + best
        tangent = segment[best] / self.length[self.index]
        signed_left = residual[best] @ np.array([tangent[1], -tangent[0]])
        arc = self.arc[self.index] + u[best]*self.length[self.index]
        k = np.interp(arc+5, self.arc[:-1], self.heading) - np.interp(arc-5, self.arc[:-1], self.heading)
        return float(arc), float(signed_left), float(k/10)

    def plan(self, pose, arc):
        # Constant commanded along-route speed. The production PID owns all acceleration.
        s = arc + np.arange(1, 41) * .1 * (120/3.6)
        xyz = np.stack([np.interp(s, self.arc, axis) for axis in self.xyz.T], axis=1)
        local = (xyz-pose[:3, 3]) @ pose[:3, :3]
        return local[:, :2] * [1., -1.]


def stats(rows):
    if not rows:
        return {"frames": 0}
    speed, error, curvature = np.array([[r["kmh"], abs(r["offset_m"]), abs(r["curvature_1pm"])] for r in rows]).T
    return dict(frames=len(rows), seconds=len(rows)*DT, speed_min_kmh=float(speed.min()),
                speed_max_kmh=float(speed.max()), offset_p95_m=float(np.percentile(error, 95)),
                offset_max_m=float(error.max()), within_half_m=float(np.mean(error<=.5)),
                curvature_max_1pm=float(curvature.max()))


def gentle_curve_segment(road_map):
    """Town04 approach to its longest constant arc with .001 <= |k| <= 5/(120/3.6)^2."""
    if not road_map.name.endswith("/Town04") and road_map.name!="Town04":
        raise ValueError("The registered gentle curve is in Town04")
    start = road_map.get_waypoint_xodr(50, 6, 38.81130948)
    route = path_from(start)
    finish = next(i for i,w in enumerate(route) if w.road_id==23 and w.lane_id==-2 and w.s>=380.)
    return route, 2.*finish


def in_gentle_arc(waypoint):
    return waypoint.road_id==23 and waypoint.lane_id==-2 and 165.5841<=waypoint.s<=309.6862


def comfort_curve_segment(road_map):
    """Town04's longest non-junction constant arc allowing 120 at 3 m/s²."""
    if not road_map.name.endswith("Town04"):raise ValueError("Registered arc is in Town04")
    start=road_map.get_waypoint_xodr(45,-4,504.24946)
    return path_from(start),460.


def in_comfort_arc(waypoint):
    return waypoint.road_id==22 and waypoint.lane_id==-2 and 200.44<=waypoint.s<=248.71


def run(town, port, seed, seconds, preview, gentle):
    from .run_fast_closed_loop import straight_segment
    client = local_client(port)
    world = client.get_world()
    if not world.get_map().name.endswith("/"+town):
        world = client.load_world(town)
    road_map = world.get_map()
    straight_m = None
    if gentle:
        route, finish_m = gentle_curve_segment(road_map)
    elif town == "Town04":
        route, straight_m = straight_segment(road_map, seed)
        finish_m = straight_m
    else:
        route = choose_path(road_map, random.Random(seed), "left", True)
        finish_m = None
    evaluation_route = path_from(route[0], distance=3000)
    ref = Reference(evaluation_route)
    folder = LOG / ("gt-controller-"+time.strftime("%Y%m%d-%H%M%S")+"-"+town)
    folder.mkdir(parents=True)
    rows, hits = [], []
    old = world.get_settings()
    cfg = world.get_settings()
    cfg.synchronous_mode, cfg.fixed_delta_seconds = True, DT
    ego = collision = None
    reason = "duration"
    try:
        world.apply_settings(cfg)
        ego = spawn(world, route[0], "vehicle.tesla.model3")
        physics = ego.get_physics_control()
        front, rear = physics.wheels[0], physics.wheels[2]
        wheelbase = math.hypot(front.position.x-rear.position.x,front.position.y-rear.position.y)/100
        angle = math.radians(front.max_steer_angle)
        steer_curve = np.array([(p.x,p.y) for p in physics.steering_curve])
        collision = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"), carla.Transform(), attach_to=ego)
        collision.listen(lambda event: hits.append((event.frame,event.other_actor.type_id)))
        ego.apply_control(carla.VehicleControl(brake=1.))
        for _ in range(12):
            world.tick()
        previous = np.asarray(ego.get_transform().get_matrix())
        old_arc = ref.project(previous)[0]
        control = CurveController() if preview else Controller()
        for tick in range(round(seconds/DT)):
            frame = world.tick()
            pose = np.asarray(ego.get_transform().get_matrix())
            arc, offset, curvature = ref.project(pose)
            speed = ego.get_velocity().length()
            plan = current_plan(ref.plan(previous, old_arc), previous, pose, DT)
            scale = np.interp(speed*3.6, *steer_curve.T)
            throttle, brake, steer = control.step(plan, speed, 120., 120., False, wheelbase, angle*scale, DT)
            ego.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
            rows.append(dict(frame=frame, sim_s=tick*DT, kmh=speed*3.6, arc_m=arc,
                             offset_m=offset, curvature_1pm=curvature, steer=steer,
                             throttle=throttle, brake=brake, curve_limit_kmh=getattr(control,"curve_limit_kmh",120.),
                             registered_arc=in_gentle_arc(evaluation_route[ref.source_index[ref.index]]) if gentle else False))
            previous, old_arc = pose, arc
            if hits:
                reason = "collision"
                break
            if finish_m is not None and arc>=finish_m-10:
                reason = "gentle_segment_complete" if gentle else "straight_segment_complete"
                break
    finally:
        if collision:
            collision.stop()
            collision.destroy()
        if ego:
            ego.destroy()
        world.apply_settings(old)
    with (folder/"frames.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row)+"\n")
    high = [r for r in rows if 118<=r["kmh"]<=122]
    streak = longest = 0
    for row in rows:
        streak = streak+1 if 118<=row["kmh"]<=122 and abs(row["offset_m"])<.5 else 0
        longest = max(streak,longest)
    result = dict(town=town, route_start=int(route[0].id), seed=seed, end=reason, collision_events=hits,
                  controller="preview" if preview else "baseline",
                  straight_length_m=straight_m, distance_along_route_m=rows[-1]["arc_m"],
                  plan_age_s=DT, target_kmh=120, traffic="empty", world_risk=False, lidar_gap_m=120.,
                  longest_high_centered_s=longest*DT, all=stats(rows), high=stats(high),
                  below118=stats([r for r in rows if r["kmh"]<118]),
                  registered_arc=stats([r for r in rows if r["registered_arc"]]),
                  registered_arc_high=stats([r for r in high if r["registered_arc"]]),
                  high_straight=stats([r for r in high if abs(r["curvature_1pm"])<.001]),
                  high_curve=stats([r for r in high if abs(r["curvature_1pm"])>=.001]),
                  sharp_curves=stats([r for r in rows if abs(r["curvature_1pm"])>=.015]))
    write_json(folder/"summary.json",result)
    print(folder, json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--town", choices=["Town04","Town06"],required=True)
    parser.add_argument("--port", type=int, default=2176)
    parser.add_argument("--seed", type=int, default=30961)
    parser.add_argument("--seconds", type=float, default=45)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--gentle", action="store_true")
    args = parser.parse_args()
    run(args.town,args.port,args.seed,args.seconds,args.preview,args.gentle)
