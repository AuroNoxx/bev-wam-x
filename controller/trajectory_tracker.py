"""Adaptive pure pursuit + velocity-profile feedforward/PID, in SI units."""
import numpy as np
from controller.pure_pursuit import SpeedPID,pursuit_steering


class TrajectoryTracker:
    def __init__(self,car,dt_s=.1):
        self.wheelbase=float(car.FRONT_WHEELBASE+car.REAR_WHEELBASE)
        self.max_steering=np.deg2rad(car.max_steering)
        self.pid=SpeedPID(dt=dt_s)
        self.speed_limit=15.

    def act(self,trajectory,speed):
        trajectory=np.asarray(trajectory)
        if trajectory.shape!=(10,4) or not np.isfinite(trajectory).all():
            raise ValueError('Expected finite 10 x (x,y,v,heading) trajectory')
        lookahead=max(4.,.65*float(speed))
        distances=np.linalg.norm(trajectory[:,:2],axis=-1)
        eligible=np.flatnonzero((distances>=lookahead)&(trajectory[:,0]>0))
        index=int(eligible[0]) if len(eligible) else len(trajectory)-1
        target=trajectory[index,:2]
        steer=pursuit_steering(target,self.wheelbase,self.max_steering)
        target_speed=float(np.clip(trajectory[index,2],0,self.speed_limit))
        if target_speed <= 0.1 and speed <= 0.5:
            return np.array([0.0, -1.0], np.float32), dict(target_speed_m_s=target_speed,
                lookahead_m=lookahead, feedforward=0.0)
        acceleration=float(np.clip((trajectory[1,2]-trajectory[0,2])/.2,-5,5))
        feedforward=.03*acceleration
        throttle=float(np.clip(self.pid.step(target_speed,speed)+feedforward,-1,1))
        return np.array([steer,throttle],np.float32),dict(target_speed_m_s=target_speed,
            lookahead_m=lookahead,feedforward=feedforward)
