"""Physics-driven collection expert. Its lane geometry is never an actor observation."""
import math

import carla
import numpy as np

from .geometry import local
from .navigation import DirectedRoute
from .scenarios import same_direction, xyz


class CollectionDriver:
    def __init__(self, waypoints, speed, seed, allow_changes, policy_version='v1'):
        self.waypoints = waypoints
        self.route = DirectedRoute(waypoints)
        self.station = 0.
        self.speed = speed / 3.6
        self.previous_steer = 0.
        self.shift = np.zeros(len(waypoints))
        self.commands = np.zeros(len(waypoints), dtype=int)
        self.recoveries = np.zeros(len(waypoints), dtype=bool)
        self.excursions=[]
        self.recovery_plans=[]
        self.recovery_events=[]
        self.cruise_windows=[]
        self.checked=set()
        self.recovery_checked=set()
        self.wait_reason=''
        self.policy_version = policy_version
        self.rng = np.random.default_rng(seed)
        if policy_version == 'v1':
            self.schedule(allow_changes)
        elif policy_version == 'v2':
            self.schedule_v2(allow_changes)
        else:
            raise ValueError(f'Unknown policy_version: {policy_version!r}')

    def schedule(self, allow_changes):
        arc = self.route.arc
        start = self.rng.uniform(38,65)
        while start + 120 < arc[-1]:
            end = start + 115
            ids = np.flatnonzero((arc >= start) & (arc < end))
            if len(ids)<4:
                break
            ways = [self.waypoints[i] for i in ids]
            headings=self.route.forward[ids,:2]
            curvature=np.linalg.norm(np.diff(headings,axis=0),axis=1)/np.diff(arc[ids])
            cruise=not any(w.is_junction for w in ways) and curvature.max()<.018
            if not cruise:
                start += 20
                continue
            t = arc[ids]-start
            self.cruise_windows.append(float(start))
            if self.rng.random()<.19:
                # Smooth excursion and return, always inside the origin lane.
                self.shift[ids]=self.rng.choice((-.42,.42))*np.sin(np.clip(t/50,0,1)*np.pi)**2
                self.recoveries[ids]=(t>=25)&(t<=50)
                self.recovery_plans.append(float(start))
                start+=85
                continue
            moved=False
            for side in self.rng.permutation(('left','right')) if allow_changes else ():
                mark=carla.LaneChange.Left if side=='left' else carla.LaneChange.Right
                back=carla.LaneChange.Right if side=='left' else carla.LaneChange.Left
                adj=[same_direction(w,side) for w in ways]
                if any(a is None or a.is_junction or not int(w.lane_change)&int(mark) or
                       not int(a.lane_change)&int(back) for w,a in zip(ways,adj)):
                    continue
                v=self.route.forward[ids,:2]
                left=v[:,::-1]*[1,-1]
                offset=((np.stack([xyz(a) for a in adj])[:,:2]-self.route.xyz[ids,:2])*left).sum(1)
                if np.any(np.abs(offset)>4.7) or np.any(np.abs(offset)<2.5):
                    continue
                def smooth(x):
                    x=np.clip(x,0.,1.)
                    return x**3*(10+x*(-15+6*x))
                self.shift[ids]=offset*(smooth(t/40)-smooth((t-65)/40))
                self.commands[ids[(t>=0)&(t<=40)]]=1 if side=='left' else 2
                self.commands[ids[(t>=65)&(t<=105)]]=2 if side=='left' else 1
                self.excursions.append((start,ids,side))
                moved=True
                break
            start+=130 if moved else 30

    def schedule_v2(self, allow_changes):
        arc = self.route.arc
        start = self.rng.uniform(38, 65)
        def smooth(x):
            x = np.clip(x, 0., 1.)
            return x**3 * (10 + x*(-15 + 6*x))
        while start + 80 < arc[-1]:
            rec_end = start + 65
            rec_ids = np.flatnonzero((arc >= start) & (arc < rec_end))
            if len(rec_ids) < 4:
                break
            rec_ways = [self.waypoints[i] for i in rec_ids]
            rec_headings = self.route.forward[rec_ids, :2]
            rec_curv = np.linalg.norm(np.diff(rec_headings, axis=0), axis=1) / np.diff(arc[rec_ids])
            if any(w.is_junction for w in rec_ways) or rec_curv.max() >= .022:
                start += 20
                continue
            self.cruise_windows.append(float(start))
            if self.rng.random() < .25:
                t = arc[rec_ids] - start
                safe_sides = []
                for side in ('left', 'right'):
                    adj = [same_direction(w, side) for w in rec_ways]
                    if all(a is not None and not a.is_junction for a in adj):
                        safe_sides.append(side)
                if safe_sides:
                    side = str(self.rng.choice(safe_sides))
                    sign = 1.0 if side == 'left' else -1.0
                    amp = float(self.rng.uniform(0.85, 1.65))
                    self.recovery_events.append((float(start), rec_ids, side))
                else:
                    sign = float(self.rng.choice((-1.0, 1.0)))
                    amp = float(self.rng.uniform(0.55, 0.80))
                out = np.where(t <= 28., np.sin(np.pi * np.clip(t, 0., 28.) / 56.)**2,
                               np.where(t <= 56., np.cos(np.pi * np.clip(t - 28., 0., 28.) / 56.)**2, 0.))
                self.shift[rec_ids] = sign * amp * out
                self.recoveries[rec_ids] = (t >= 26.) & (t <= 58.)
                self.recovery_plans.append(float(start))
                start += 80
                continue
            moved = False
            if allow_changes:
                for span, hold_end in ((240., 190.), (185., 135.)):
                    if start + span >= arc[-1]:
                        continue
                    ids = np.flatnonzero((arc >= start) & (arc < start + span))
                    if len(ids) < 6:
                        continue
                    ways = [self.waypoints[i] for i in ids]
                    headings = self.route.forward[ids, :2]
                    curvature = np.linalg.norm(np.diff(headings, axis=0), axis=1) / np.diff(arc[ids])
                    if any(w.is_junction for w in ways) or curvature.max() >= .022:
                        continue
                    t = arc[ids] - start
                    for side in self.rng.permutation(('left', 'right')):
                        mark = carla.LaneChange.Left if side == 'left' else carla.LaneChange.Right
                        back = carla.LaneChange.Right if side == 'left' else carla.LaneChange.Left
                        adj = [same_direction(w, side) for w in ways]
                        if any(a is None or a.is_junction or not int(w.lane_change) & int(mark) or
                               not int(a.lane_change) & int(back) for w, a in zip(ways, adj)):
                            continue
                        v = self.route.forward[ids, :2]
                        left = v[:, ::-1] * [1, -1]
                        offset = ((np.stack([xyz(a) for a in adj])[:, :2] - self.route.xyz[ids, :2]) * left).sum(1)
                        if np.any(np.abs(offset) > 4.7) or np.any(np.abs(offset) < 2.5):
                            continue
                        self.shift[ids] = offset * (smooth(t / 40.) - smooth((t - hold_end) / 40.))
                        self.commands[ids[(t >= 0) & (t <= 40)]] = 1 if side == 'left' else 2
                        self.commands[ids[(t >= hold_end) & (t <= hold_end + 40)]] = 2 if side == 'left' else 1
                        self.excursions.append((float(start), ids, side))
                        moved = True
                        start += span + 15.
                        break
                    if moved:
                        break
            if not moved:
                start += 30

    def step(self,pose,velocity,traffic,traffic_velocities=None,red_light=False):
        self.station,nearest=self.route.match(pose,self.station)
        if nearest>7:
            raise ValueError(f'Expert departed the directed driving corridor by {nearest:.2f} m')
        s=self.station
        speed=math.hypot(velocity.x,velocity.y)
        self.wait_reason=''
        centers=local(np.asarray(traffic).mean(1),pose) if len(traffic) else np.empty((0,3))
        for start,ids,side in self.recovery_events:
            if start-14<=s<=start+7 and start not in self.recovery_checked:
                left_sign=1 if side=='left' else -1
                occupied=np.any((centers[:,0]>-20)&(centers[:,0]<25)&
                                (centers[:,1]*left_sign>1.2)&(centers[:,1]*left_sign<5.5)&
                                (abs(centers[:,2])<2.2))
                if occupied:
                    self.shift[ids]=0
                    self.recoveries[ids]=False
                    self.recovery_checked.add(start)
        for start,ids,side in self.excursions:
            if start-14<=s<=start+7 and start not in self.checked:
                left_sign=1 if side=='left' else -1
                occupied=np.any((centers[:,0]>-30)&(centers[:,0]<25)&
                                (centers[:,1]*left_sign>1.6)&(centers[:,1]*left_sign<6.)&
                                (abs(centers[:,2])<2.2))
                if occupied:
                    self.shift[ids]=0
                    self.commands[ids[self.route.arc[ids]>start+40]]=0
                    self.checked.add(start)
        for start,ids,side in self.excursions:
            if start in self.checked and start<=s<=start+40:
                self.wait_reason='occupied_target_lane'
                break
        look=np.clip(4.0+.38*speed,4.5,10.5)
        samples=s+np.array([look,look+4,look+8,look+12])
        center=self.route.positions(s,samples-s,0)
        f=np.stack([np.interp(samples,self.route.arc,axis) for axis in self.route.forward[:,:2].T],axis=1)
        normal=f[:,::-1]*[1,-1]/np.linalg.norm(f,axis=1,keepdims=True)
        center[:,:2]+=normal*np.interp(samples,self.route.arc,self.shift)[:,None]
        points=local(center,pose)[:,:2]
        ahead,left=points[0]
        steer=np.clip(-math.atan2(5.7*left,max(1,ahead*ahead+left*left))/.66,-.96,.96)
        self.previous_steer=float(np.clip(steer,self.previous_steer-.11,self.previous_steer+.11))
        yaw=np.unwrap(np.arctan2(f[:,1],f[:,0]))
        curvature=max(.001,float(np.max(np.abs(np.diff(yaw)))/4))
        target=min(self.speed,math.sqrt(2.2/curvature))
        if len(traffic):
            lead=centers[(centers[:,0]>0)&(abs(centers[:,1])<1.9)&(abs(centers[:,2])<2.2)]
            if len(lead):
                target=min(target,max(0.,(lead[:,0].min()-6)*.55))
            if traffic_velocities is not None:
                times=np.arange(.25,3.01,.25)
                preview=self.route.positions(s,times*max(2.,speed),0)
                obstacles=np.asarray(traffic).mean(1)[:,None,:]+np.asarray(traffic_velocities)[:,None,:]*times[None,:,None]
                delta=obstacles-preview[None,:,:]
                crossing=(abs(centers[:,1])>1.9)&(centers[:,0]>-2)
                hazard=(np.linalg.norm(delta[:,:,:2],axis=2)<4.6)&(abs(delta[:,:,2])<2.2)
                if np.any(hazard[crossing]):
                    target=0.
        if red_light:
            target=0.
        index=min(int(self.route.arc.searchsorted(s)),len(self.commands)-1)
        hold=target<.1
        return (carla.VehicleControl(throttle=0. if hold else float(np.clip(.19+.19*(target-speed),0,.78)),
                                     brake=max(.4 if hold else 0.,float(np.clip(.27*(speed-target),0,.9))),steer=self.previous_steer),
                int(self.commands[index]),bool(self.recoveries[index]))
