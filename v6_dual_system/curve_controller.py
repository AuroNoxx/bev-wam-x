"""Curvature preview from a planner's local XY points for the existing steering and speed loops."""
import math

import numpy as np

from .legacy_controller import Controller, blue_lateral, curve_speed_cap, fused_steer


def lane_pose_now(lane_pose, observation_pose, current_pose):
    """Carry the observed lane tangent through the measured ego motion (FLU)."""
    offset,yaw=lane_pose
    motion=np.linalg.solve(observation_pose,current_pose)
    offset+=motion[0,3]*math.sin(yaw)-motion[1,3]*math.cos(yaw)
    yaw-=math.atan2(motion[1,0],motion[0,0])
    return float(offset),float(yaw)


def curvature_preview(xy):
    points = np.asarray(xy, dtype=np.float64)
    chord = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.r_[True, chord > 1e-3]]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    half_span = 4.
    centres=np.flatnonzero((arc>=half_span)&(arc<=arc[-1]-half_span))
    distance=arc[centres]
    if not len(distance):
        return distance, np.empty(0)
    # Three actual waypoints preserve a circle's radius; interpolating along its
    # chords biases peak curvature when the input spacing approaches the span.
    left=np.searchsorted(arc,distance-half_span,side="right")-1
    right=np.searchsorted(arc,distance+half_span).clip(0,len(points)-1)
    before, after=points[centres]-points[left],points[right]-points[centres]
    cross = before[:, 0]*after[:, 1] - before[:, 1]*after[:, 0]
    lengths = (np.linalg.norm(before, axis=1)*np.linalg.norm(after, axis=1)
               * np.linalg.norm(before+after, axis=1))
    return distance, 2*cross / np.maximum(lengths, 1e-6)


class CurveController(Controller):
    def __init__(self,max_lateral_accel=3.0,spatial=False):
        super().__init__()
        self.last_steer = 0.
        self.spatial=spatial
        self.filtered_lane=None
        self.recovery_left=0.
        self.at_fork=False
        self.blue_weight=.6
        self.fork_delta=None
        self.max_lateral_accel=max_lateral_accel
        self.curve_limit_kmh = 120.

    def step(self,xy,speed,target,gap,world_risk,wheelbase,max_angle,dt=.05,lane_pose=None,fused=False,handoff=False,nav30=None,blue_cruise=False):
        distance, curvature = curvature_preview(xy)
        cruise=self.spatial and target>=120.
        if cruise:target=120.5
        limit = target/3.6
        straight=False
        feedforward = 0.
        if len(curvature):
            # Budget lateral acceleration and leave enough measured distance for braking.
            if self.spatial:curvature=np.where(abs(curvature)<.0015,0.,curvature)
            allowed_sq = self.max_lateral_accel / np.maximum(abs(curvature), 1e-4)
            lead_distance = np.maximum(0.,distance-.5*speed)
            preview_cap=curve_speed_cap(xy,distance,curvature,self.max_lateral_accel,.0015 if self.spatial else 0.)
            straight=math.isinf(preview_cap)
            limit = min(limit,preview_cap,
                        math.sqrt(float(np.min(allowed_sq+2*4.*lead_distance))))
            near_k = np.interp(.25*speed, distance, curvature)
            feedforward = -.0014 * speed**2 * near_k / max_angle
        self.curve_limit_kmh = 3.6*limit
        throttle, brake, steer = super().step(xy, speed, self.curve_limit_kmh, gap,
                                               world_risk, wheelbase, max_angle, dt,
                                               cruise_reference=blue_cruise or (cruise and straight and lane_pose is not None and gap>=120. and not world_risk))
        if nav30 is None or lane_pose is None:
            self.at_fork=False
            self.blue_weight=.6
            self.fork_delta=None
        if lane_pose is None:
            self.filtered_lane=None
            self.recovery_left=0.
            steer+=feedforward
        elif fused:
            if self.spatial:
                measured=np.asarray(lane_pose)
                if handoff:self.recovery_left=2.
                if self.filtered_lane is None or handoff:self.filtered_lane=measured
                else:self.filtered_lane+=.35*(measured-self.filtered_lane)
                lane_pose=self.filtered_lane
            duration=.75 if self.recovery_left>0 else 1.5
            # Extra recovery feedback is confined to a confirmed lane handoff.
            # Taper back to steady cruise during the last half second.
            gain=1.+.8*min(1.,self.recovery_left/.5)
            if nav30 is not None:
                x,y=nav30
                blue=blue_lateral(x,max(0.,x)/max(speed,1.),lane_pose,duration,gain)
                self.fork_delta=abs(y-blue)
                if self.fork_delta>=1.2:self.at_fork=True
                elif self.fork_delta<1.:self.at_fork=False
                target_weight=.15 if self.at_fork else .6
                step=.45*dt/.25
                self.blue_weight+=float(np.clip(target_weight-self.blue_weight,-step,step))
            steer=fused_steer(xy,lane_pose,speed,wheelbase,max_angle,duration,gain,self.blue_weight)
            self.recovery_left=max(0.,self.recovery_left-dt)
        else:
            offset,yaw=lane_pose
            # Pure Pursuit on the RGB head's local center tangent. Positive CARLA steer turns right.
            ahead=float(np.clip(speed*.70,3.5,24.))
            lateral=-offset*math.cos(yaw)-ahead*math.sin(yaw)
            steer=-math.atan2(2*wheelbase*lateral,ahead**2+offset**2)/max_angle
        steer = float(np.clip(steer,-1.,1.))
        if speed>20.:
            steer = float(np.clip(steer,self.last_steer-1.2*dt,self.last_steer+1.2*dt))
        self.last_steer = steer
        return throttle, brake, steer
