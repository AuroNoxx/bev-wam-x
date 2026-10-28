"""Privileged development teacher for controlled straight-road maneuver probes.

Actor objects, velocities and lane permissions are teacher/diagnostic only.
Nothing here is a learner observation or a WAM prediction.
"""
import numpy as np
from controller.pure_pursuit import SpeedPID, pursuit_steering
from mini_wa_jepa.geometry import world_to_ego, wrap_angle
from env.lane_permissions import legal_adjacent_lane


def quintic_blend(fraction):
    t = np.clip(fraction, 0., 1.)
    return t*t*t*(10.+t*(-15.+6.*t))


def lane_clearance(lane, ego, actors):
    es, _ = lane.local_coordinates(ego.position)
    front = rear = float('inf'); min_ttc = float('inf')
    for actor in actors:
        if actor is ego:
            continue
        s, lat = lane.local_coordinates(actor.position)
        if abs(lat) > lane.width_at(np.clip(s, 0, lane.length))*.5+actor.WIDTH*.5:
            continue
        gap = s-es; half = (actor.LENGTH+ego.LENGTH)*.5
        if gap >= 0:
            clear = gap-half; front = min(front, clear)
            closing = ego.speed-actor.speed
        else:
            clear = -gap-half; rear = min(rear, clear)
            closing = actor.speed-ego.speed
        if closing > .05:
            min_ttc = min(min_ttc, clear/closing)
    return front, rear, min_ttc


class TrafficManeuverTeacher:
    def __init__(self, sim, dt=.1):
        self.sim = sim; self.dt = dt; self.elapsed = 0.
        car = sim.agent
        self.home = car.lane
        self.lanes = sim.current_map.road_network.graph[self.home.index[0]][self.home.index[1]]
        self.home_id = self.home.index[-1]; self.target_id = self.home_id
        self.phase = 'FOLLOW'; self.transition_started = 0.; self.transition_duration = 4.
        self.lead = None; self.completed = False; self.stable_steps = 0
        self.pid = SpeedPID(dt=dt)
        self.wheelbase = car.FRONT_WHEELBASE+car.REAR_WHEELBASE
        self.max_steer = np.deg2rad(car.max_steering)

    def act(self):
        car = self.sim.agent
        s, lateral = self.home.local_coordinates(car.position)
        if s > self.home.length-10:
            raise RuntimeError('Straight maneuver teacher reached its declared road scope')
        actors = list(self.sim.engine.get_objects(lambda o: getattr(o, 'metadrive_type', None) == 'VEHICLE').values())
        home_front = []
        for actor in actors:
            if actor is car:
                continue
            os, ol = self.home.local_coordinates(actor.position)
            if abs(ol) < self.home.width_at(s)*.5+actor.WIDTH*.5 and os > s:
                home_front.append((os-s-(actor.LENGTH+car.LENGTH)*.5, actor))
        closest = min(home_front, key=lambda row: row[0]) if home_front else None
        if self.phase == 'FOLLOW' and closest is not None and closest[0] < 25 and car.speed > 3:
            for lane_id in (self.home_id-1, self.home_id+1):
                if not 0 <= lane_id < len(self.lanes):
                    continue
                lane = self.lanes[lane_id]
                front, rear, ttc = lane_clearance(lane, car, actors)
                if (legal_adjacent_lane(self.home, lane) and self.home.length-s > 100 and
                        closest[1].speed < 4 and front > 40 and rear > 15 and ttc > 5):
                    self.target_id = lane_id; self.lead = closest[1]
                    self.phase = 'CHANGE_OUT'; self.transition_started = self.elapsed
                    break
        if self.phase == 'CHANGE_OUT' and self.elapsed-self.transition_started >= self.transition_duration:
            _, target_lat = self.lanes[self.target_id].local_coordinates(car.position)
            if abs(target_lat) < .35:
                self.phase = 'PASS'
        lead_gap = None
        if self.lead is not None:
            if self.lead not in actors:
                raise RuntimeError('Tracked lead removed from active simulator objects; maneuver is incomplete')
            lead_s, _ = self.home.local_coordinates(self.lead.position)
            lead_gap = s-lead_s-(car.LENGTH+self.lead.LENGTH)*.5
        if self.phase == 'PASS' and lead_gap is not None and lead_gap > 10:
            front, rear, ttc = lane_clearance(self.home, car, actors)
            if front > 20 and rear > 8 and ttc > 5:
                self.phase = 'CHANGE_BACK'; self.transition_started = self.elapsed
        target_lane = self.lanes[self.target_id]
        _, dest_lateral = self.home.local_coordinates(target_lane.position(s, 0))
        ahead = max(5., .8*car.speed)
        fraction = (self.elapsed-self.transition_started+.6)/self.transition_duration
        if self.phase == 'CHANGE_OUT':
            target_lateral = dest_lateral*quintic_blend(fraction)
        elif self.phase == 'CHANGE_BACK':
            target_lateral = dest_lateral*(1.-quintic_blend(fraction))
        elif self.phase == 'PASS':
            target_lateral = dest_lateral
        else:
            target_lateral = 0.
        heading_error = float(wrap_angle(car.heading_theta-self.home.heading_theta_at(s)))
        if self.phase == 'CHANGE_BACK' and abs(lateral) < .35 and abs(heading_error) < .08:
            self.stable_steps += 1
            if self.stable_steps >= 10:
                self.phase = 'COMPLETE'; self.completed = True
        else:
            self.stable_steps = 0
        target = self.home.position(min(s+ahead, self.home.length), target_lateral)
        local = world_to_ego(target, car.position, car.heading_theta)
        steering = pursuit_steering(local, self.wheelbase, self.max_steer)
        # Geometric comfort cap, then footprint-aware teacher-only following.
        curvature = abs(np.tan(steering*self.max_steer)/self.wheelbase)
        speed_target = min(6., np.sqrt(3./max(curvature, 1e-4)))
        min_clearance = float('inf')
        for actor in actors:
            if actor is car:
                continue
            os, ol = self.home.local_coordinates(actor.position)
            gap = os-s-(car.LENGTH+actor.LENGTH)*.5
            # Keep lead braking during a lane transition while bodies overlap.
            if abs(ol-lateral) < (car.WIDTH+actor.WIDTH)*.5+.4 and gap > -1:
                min_clearance = min(min_clearance, gap)
                if actor.speed < 0.2:
                    standstill_gap = 6.5
                    if gap <= standstill_gap:
                        target = 0.0
                    else:
                        target = max(0.0, (gap - standstill_gap) * 0.45)
                    speed_target = min(speed_target, target)
                else:
                    speed_target = min(speed_target, max(0., actor.speed+(gap-5.-1.8*car.speed)/1.8))
        action = np.array([steering, self.pid.step(speed_target, car.speed)], np.float32)
        self.elapsed += self.dt
        return action, dict(phase=self.phase, home_lateral_m=float(lateral), target_lane=int(self.target_id),
                            heading_error_rad=heading_error, target_speed_m_s=float(speed_target),
                            lead_rear_clearance_m=lead_gap, front_clearance_m=None if not np.isfinite(min_clearance) else float(min_clearance),
                            complete=self.completed)


def spawn_slow_actor(sim, lane, longitude, seed, stationary=False, dynamic_stop_resume=False, stop_step=40, resume_step=90):
    """Source-verified PGTrafficManager native spawn/policy/managed-list path."""
    from metadrive.component.vehicle.vehicle_type import TrafficDefaultVehicle
    from metadrive.policy.idm_policy import IDMPolicy

    class SlowLaneIDM(IDMPolicy):
        NORMAL_SPEED = 7.2  # km/h == 2m/s

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.is_stationary = bool(stationary)
            self.dynamic = bool(dynamic_stop_resume)
            self.stop_step = int(stop_step)
            self.resume_step = int(resume_step)
            self.step_idx = 0

        def act(self, *args, **kwargs):
            self.step_idx += 1
            if self.is_stationary:
                # Physical full brake on a straight road; target_speed=0 is not
                # valid for native IDM's speed ratio (not_zero(..., eps=0)).
                action = [0., -1.]
                self.action_info['action'] = action
                return action
            if self.dynamic:
                if self.stop_step <= self.step_idx < self.resume_step:
                    action = [0., -1.]
                    self.action_info['action'] = action
                    return action
            return super().act(*args, **kwargs)

    manager = sim.engine.traffic_manager
    config = dict(sim.engine.global_config['traffic_vehicle_config'])
    config.update(spawn_lane_index=lane.index, spawn_longitude=float(longitude), enable_reverse=False)
    actor = manager.spawn_object(TrafficDefaultVehicle, vehicle_config=config)
    manager.add_policy(actor.id, SlowLaneIDM, actor, int(seed))
    sim.engine.get_policy(actor.id).enable_lane_change = False
    manager._traffic_vehicles.append(actor)
    return actor
