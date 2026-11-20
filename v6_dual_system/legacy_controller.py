"""Pure Pursuit in ego FLU, CARLA steering sign, longitudinal PID and LiDAR veto."""
import math

import numpy as np

from .sensors import ego_points


def curve_speed_cap(xy,distance,curvature,max_lateral_accel=3.0,deadzone=0.):
    """Comfort cap in m/s using the peak absolute curvature 1.5–2.5 s ahead."""
    if max_lateral_accel<=0:
        raise ValueError("Lateral acceleration limit must be positive")
    if not len(distance):
        return math.inf
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    lo,hi=np.interp([1.5,2.5],np.arange(1,len(arc)+1)*.1,arc)
    # Include both interval boundaries even when the spatial grid straddles them.
    sample=np.r_[lo,distance[(distance>lo)&(distance<hi)],hi]
    peak=np.max(np.abs(np.interp(sample,distance,curvature)))
    if peak<deadzone:return math.inf
    return math.sqrt(max_lateral_accel/max(float(peak),1e-4))


def blue_lateral(x,t,lane_pose,recovery_seconds,offset_gain):
    offset,yaw=lane_pose
    yaw=float(np.clip(yaw,-1.,1.))
    u=min(1.,t/recovery_seconds)
    return -x*math.tan(yaw)-offset_gain*offset*u*u*(3.-2.*u)/math.cos(yaw)


def fused_steer(xy, lane_pose, speed, wheelbase, max_angle, recovery_seconds=1.5, offset_gain=1.,blue_weight=.6):
    """Follow a 60/40 mix at the same station in the first 1.5 s of both plans."""
    near=np.vstack(([0.,0.],np.asarray(xy,dtype=float)[:15]))
    length=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(near,axis=0),axis=1))]
    ahead=min(float(np.clip(3.5+.615*speed,3.5,24.)),length[-1])
    if ahead<.05:return 0.
    x,y=(np.interp(ahead,length,axis) for axis in near.T)
    t=np.interp(ahead,length,np.arange(16)*.1)
    blue=blue_lateral(x,t,lane_pose,recovery_seconds,offset_gain)
    lateral=blue_weight*blue+(1.-blue_weight)*y
    return -math.atan2(2*wheelbase*lateral,max(.5,x*x+lateral*lateral))/max_angle


class LaneCrossing:
    """A signed offset must cross the divider before the target-centre threshold applies."""
    def __init__(self):
        self.intent=0
        self.approached=False
        self.crossed=False

    def update(self, intent, offset):
        if self.intent!=intent:
            self.intent=intent
            self.approached=self.crossed=False
        if not intent:return False
        signed=offset if intent==1 else -offset
        if signed>.8:self.approached=True
        if self.approached and signed<-.4:self.crossed=True
        return self.crossed and abs(offset)<1.2


class TimedLaneCrossing:
    """Latch a signed nearest-lane boundary transition during an executed manoeuvre."""
    def __init__(self):
        self.intent=0
        self.start=0.
        self.approached=False
        self.crossed=False

    def update(self,intent,offset,seconds):
        if intent!=self.intent:
            self.intent=intent
            self.start=seconds
            self.approached=self.crossed=False
        if not intent:return False
        signed=offset if intent==1 else -offset
        if signed>0.:self.approached=True
        if self.approached and signed<-.6:self.crossed=True
        return self.crossed and seconds-self.start>1.


def current_plan(xy,pose,current,age,extrapolate=True):
    xy=np.asarray(xy)
    times=np.arange(1,len(xy)+1)*.1
    future=times+age
    extended=np.stack([np.interp(future,times,axis) for axis in xy.T],axis=1)
    if extrapolate:
        extended+=np.maximum(0,future-times[-1])[:,None]*(xy[-1]-xy[-2])/.1
    fru=np.stack((extended[:,0],-extended[:,1],np.zeros(len(xy))),axis=1)
    align=np.linalg.solve(current,pose)
    local=fru@align[:3,:3].T+align[:3,3]
    return local[:,:2]*[1,-1]


def lidar_clearance(scan,intent,width):
    x,right,z=ego_points(scan).T
    left=-right
    above=(z>.38)&(z<2.4)
    front=above&(x>2.6)&(np.abs(left)<width*.48)
    # Require several returns so a lone LiDAR outlier cannot command a stop.
    ranges=x[front]
    gap=float(np.partition(ranges,4)[4]-2.4) if len(ranges)>=5 else 120.
    side=left if intent==1 else -left
    side_returns=above&(x>=-16)&(x<5)&(side>1.7)&(side<5.4)
    return bool(intent and np.count_nonzero(side_returns)>=8),max(0.,gap)


def path_gap(scan,xy,width):
    pts=ego_points(scan)
    x,right,z=pts.T
    xy=np.asarray(xy)
    forward=np.r_[True,np.diff(xy[:,0])>0]
    path=xy[forward]
    if not len(path):return 120.
    path=path[np.argsort(path[:,0])]
    near=(x>2.6)&(x<path[-1,0])&(z>.38)&(z<2.4)
    x,left=x[near],-right[near]
    lateral=np.interp(x,path[:,0],path[:,1],left=0.)
    ranges=np.hypot(x,left)[np.abs(left-lateral)<width*.48]
    return float(max(0.,np.partition(ranges,4)[4]-2.4)) if len(ranges)>=5 else 120.


class Controller:
    def __init__(self):
        self.integral=0.
        self.last_speed=None
        self.last_gap=None
        self.gap_age=0.
        self.closing=0.

    def step(self,xy,speed,target,gap,world_risk,wheelbase,max_angle,dt=.05,cruise_reference=False):
        xy=np.asarray(xy,dtype=np.float64)
        distance=np.linalg.norm(xy,axis=1)
        ahead=float(np.interp(speed,[0,120/3.6],[3.5,24]))
        candidates=np.flatnonzero((distance>=ahead)&(xy[:,0]>0))
        i=int(candidates[0]) if len(candidates) else int(np.argmax(distance))
        x,y=xy[i]
        if i and distance[i-1]<ahead<=distance[i]:
            blend=(ahead-distance[i-1])/max(distance[i]-distance[i-1],1e-6)
            x,y=xy[i-1]+blend*(xy[i]-xy[i-1])
        steer=-math.atan2(2*wheelbase*y,max(x*x+y*y,.5))/max_angle
        # Track the imminent plan. The command supplies the cruising ceiling.
        desired=min(target/3.6,max(0.,float(xy[9,0])))
        if cruise_reference:desired=target/3.6
        err=desired-speed
        derivative=0. if self.last_speed is None else (speed-self.last_speed)/dt
        integral=float(np.clip(self.integral+err*dt,-2.,4.))
        drive=.55*err+.12*integral-.035*derivative+.00022*desired**2
        if not ((drive>=1 and err>0) or (drive<=-1 and err<0)):
            self.integral=integral
        self.gap_age+=dt
        if self.gap_age>.2:
            self.last_gap=None
            self.closing=0.
        if gap<120.:
            if self.last_gap is not None:
                rate=(self.last_gap-gap)/self.gap_age
                if abs(rate)<=speed+8:
                    weight=1-math.exp(-self.gap_age/.15)
                    self.closing+=weight*(max(0.,rate)-self.closing)
                else:
                    self.closing=0.
            self.last_gap=gap
            self.gap_age=0.
        elif self.last_gap is not None:
            gap=max(0.,self.last_gap-self.closing*self.gap_age)
        # Both vehicles can brake. Relative-speed-only stopping distance assumes
        # the lead maintains its speed and underestimates the needed separation.
        lead_speed=max(0.,speed-self.closing)
        stopping_gap=2.+.45*speed+speed**2/(2*6.)-lead_speed**2/(2*10.)
        imminent=gap<max(2.,.35*speed) or (self.closing>.5 and gap<stopping_gap)
        self.last_speed=speed
        if imminent or world_risk:
            drive=-1.
        return float(np.clip(drive,0,1)),float(np.clip(-drive,0,1)),float(np.clip(steer,-1,1))
