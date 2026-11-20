"""Single-trajectory control for CARLA. Inputs are odometry, the selected route and sensors."""
from dataclasses import dataclass
import math

import numpy as np

from .sensors import ego_points
from .unified_grid import HORIZONS,X,Y


@dataclass
class Contact:
    x: float
    speed: float


class CorridorTraffic:
    """Estimate the nearest forward and rear returns and their speeds in route-normal bands."""
    def __init__(self):
        self.tracks={}

    def observe(self,scan,polyline,locked,width,v,dt,plane=None):
        point=ego_points(scan)
        x,y,z=point[:,0],-point[:,1],point[:,2]
        if plane is not None:
            z=z-(plane[0]*x+plane[1]*y+plane[2])
        valid=(z>.4)&(z<2.3)&(abs(x)<100)&((abs(x)>2.6)|(abs(y)>1.15))
        x,y=x[valid],y[valid]
        poly=np.asarray(polyline)
        # At the beginning/end of a route the sampled stations repeat its endpoint.
        # A zero tangent would project even orthogonal roadside returns onto the ego lane.
        poly=poly[np.r_[True,np.linalg.norm(np.diff(poly,axis=0),axis=1)>.001]]
        if len(poly)<2:
            raise ValueError("Cannot project traffic onto a zero-length route")
        # Use the increasing-x approach only; after a hairpin it may turn back.
        indices=np.flatnonzero(np.diff(poly[:,0]) < -.05)
        if len(indices):poly=poly[:max(2,indices[0]+1)]
        # Continue rear tangent behind the sparse navigation horizon.
        j=np.searchsorted(poly[:,0],x).clip(1,len(poly)-1)
        dx,dy=(poly[j]-poly[j-1]).T
        n=np.maximum(.01,np.hypot(dx,dy))
        lateral=((y-poly[j-1,1])*dx-(x-poly[j-1,0])*dy)/n
        traffic={}
        for lane in (0,1,-1):
            rel=x[abs(lateral-locked-lane*width)<1.55]
            sorted_x=np.sort(rel[(rel>-45.)&(rel<55.)])
            clusters=[group for group in np.split(sorted_x,np.flatnonzero(np.diff(sorted_x)>2.)+1)
                      if len(group)>=4]
            if clusters:
                nearest=min(clusters,key=lambda g:np.min(abs(g)))
                traffic[lane,"adjacent_span"]=(float(nearest[0]),float(nearest[-1]))
                traffic[lane,"single_blocker"]=all(g is nearest or g[-1]<-max(12.,v) or
                                                    g[0]>max(22.,4.5+1.5*v) for g in clusters)
            ahead = np.sort(rel[(rel >= 12.) & (rel <= 45.)])
            traffic[lane,"accel_blocked"] = bool(len(ahead) >= 4 and np.any(ahead[3:] - ahead[:-3] < 2.5))
            for side in (1,-1):
                key=(round(locked/width)+lane,side)
                signed=side*rel
                hits=np.sort(signed[signed>=0])
                # A run of neighbouring range bins suppresses isolated returns.
                front=None
                if len(hits)>=4:
                    good=np.flatnonzero(hits[3:]-hits[:-3]<2.5)
                    if len(good):front=float(hits[good[0]+1])*side
                old=self.tracks.get(key)
                if front is not None:
                    rate=v if old is None else old[1]
                    if old is not None:
                        predicted=old[0]+(old[1]-v)*dt
                        if abs(front-predicted)<max(2.,dt*30):
                            rate=np.clip(old[1]+.12*(front-predicted)/max(.05,dt),0.,55.)
                        else:rate=v
                    self.tracks[key]=(front,float(rate),0.)
                elif old is not None and old[2]+dt<=.35:
                    self.tracks[key]=(old[0]+(old[1]-v)*dt,old[1],old[2]+dt)
                else:
                    self.tracks.pop(key,None)
                track=self.tracks.get(key)
                traffic[lane,side]=Contact(*track[:2]) if track is not None else None
        return traffic


class LaneRequest:
    def __init__(self):
        self.pending=0
        self.target_lane_shift=0.
        self.proposed=0.
        self.started=0.
        self.stable=0.
        self.mode="KEEP"
        self.crossed=False
        self.accel_started=None
        self.initial_gap_deficit=0.
        self.yield_latched=False

    def update(self,click,now,dt,offset,width,v,ceiling,traffic):
        if click and self.pending and click==-self.pending:
            self.pending=0
            self.mode="KEEP"
            self.proposed=self.target_lane_shift
            self.crossed=False
        elif click and not self.pending:
            self.pending=click;self.started=now;self.stable=0.;self.crossed=False
            self.accel_started=None
            self.yield_latched=False
            self.proposed=self.target_lane_shift+click*width
        if not self.pending:return self.target_lane_shift,0,ceiling
        if self.mode!="EXECUTE" and now-self.started>=8.:
            self.pending=0;self.mode="KEEP"
            return self.target_lane_shift,0,ceiling
        front,rear=(traffic[self.pending,side] for side in (1,-1))
        lead=traffic[0,1]
        room=lead is None or lead.x>max(25.,1.7*v+4.5)
        front_ok=front is None or front.x>max(22.,4.5+1.5*v+max(v-front.speed,0)*2.2)
        rear_ok=rear is None or -rear.x>max(10.,4.5+.7*rear.speed+max(rear.speed-v,0)*2.2)
        span=traffic.get((self.pending,"adjacent_span"))
        alongside=span is not None and span[0]<=3. and span[1]>=-10.
        can_pass=room and not traffic.get((self.pending,"accel_blocked"),False)
        # An adjacent car can have returns on both sides of x=0. It occupies
        # one gap and is eligible to be passed; two distinct blockers stay WAIT.
        single=traffic.get((self.pending,"single_blocker"),False)
        if self.mode!="EXECUTE":
            if front_ok and rear_ok:self.mode="EXECUTE"
            elif not front_ok and not rear_ok and not single:self.mode="WAIT"
            elif self.yield_latched:self.mode="YIELD_GAP"
            elif rear and rear.speed>v+10/3.6 and not (single and alongside and front):self.mode="YIELD_GAP"
            elif can_pass and single and alongside:self.mode="ACCEL_GAP"
            elif front and front.x>3 and not front_ok:self.mode="YIELD_GAP"
            elif room and not traffic.get((self.pending,"accel_blocked"),False) and (front is None or front.x<=3 or front.x>45):self.mode="ACCEL_GAP"
            else:self.mode="WAIT"
        if self.mode=="EXECUTE":
            self.crossed=self.crossed or self.pending*(offset-self.target_lane_shift)>=width*.5
            self.stable=self.stable+dt if self.crossed and abs(offset-self.proposed)<.5 else 0.
            if self.stable>=.5:
                self.target_lane_shift=self.proposed;self.pending=0;self.mode="KEEP"
            cmd=(1 if self.pending>0 else 2) if self.pending and not self.crossed else 0
            return self.proposed,cmd,ceiling
        adjacent=front if front and front.x<=3 else rear
        if self.mode=="ACCEL_GAP" and self.accel_started is None:
            self.accel_started=now
            self.initial_gap_deficit=span[1] if span else adjacent.x if adjacent else 0.
        if self.mode!="EXECUTE" and self.accel_started is not None and now-self.accel_started>=2.5:
            progress_x=span[1] if span else adjacent.x if adjacent else math.inf
            if progress_x>self.initial_gap_deficit-1.:
                self.yield_latched=True
                if self.mode!="WAIT":self.mode="YIELD_GAP"
            self.accel_started=None
        if self.mode=="ACCEL_GAP" and adjacent is None:adjacent=front
        if self.mode=="ACCEL_GAP" and adjacent:
            ceiling=min(120/3.6,max(ceiling+15/3.6,adjacent.speed+22/3.6))
        if self.mode=="YIELD_GAP":
            adjacent=rear if rear and rear.speed>v+10/3.6 else front or rear
            if adjacent:ceiling=min(ceiling,max(0., min(v-1.,adjacent.speed-12/3.6)))
        return self.target_lane_shift,0,ceiling


def footprint_overlap(xy,probabilities,length,width):
    """Test the current footprint dimensions at the matching five future timesteps."""
    idx=(np.rint(np.array(HORIZONS)*10)-1).astype(int)
    pos=xy[idx];direction=xy[idx]-xy[idx-1]
    theta=np.arctan2(direction[:,1],direction[:,0])
    # Sample footprint on a grid at least as fine as occupancy resolution.
    ox,oy=np.meshgrid(np.linspace(-length/2,length/2,math.ceil(length/.4)+1),
                      np.linspace(-width/2,width/2,math.ceil(width/.4)+1))
    x=pos[:,0,None]+np.cos(theta)[:,None]*ox.ravel()-np.sin(theta)[:,None]*oy.ravel()
    y=pos[:,1,None]+np.sin(theta)[:,None]*ox.ravel()+np.cos(theta)[:,None]*oy.ravel()
    col=np.floor(x+16).astype(int);row=np.floor((y+25.6)/.4).astype(int)
    known=(col>=0)&(col<128)&(row>=0)&(row<128)
    occupied=probabilities[np.arange(5)[:,None],row.clip(0,127),col.clip(0,127)]>.5
    return (occupied&known).any(1),known.all(1)


def plan_curvature(xy,origin=(0.,0.)):
    """Signed three-point curvature over a metric baseline on the current trajectory."""
    points=np.vstack((origin,xy))
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    half_span=min(16.,arc[-1]/4.)
    if half_span<1.:
        return np.empty(0),np.empty(0)
    station=np.arange(half_span,arc[-1]-half_span+.01,1.)
    samples=[np.column_stack([np.interp(station+off,arc,vals) for vals in points.T])
             for off in (-half_span,0.,half_span)]
    before=samples[1]-samples[0]
    after=samples[2]-samples[1]
    scale=np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1)*np.linalg.norm(before+after,axis=1)
    cross=before[:,0]*after[:,1]-before[:,1]*after[:,0]
    # Distances denote the first point supported by each fit, including imminent bends.
    return station-half_span,2.*cross/np.maximum(scale,1e-6)


def transform_flu(points,motion):
    xyz=np.zeros((len(points),3))
    xyz[:,:points.shape[1]]=points
    xyz[:,1]*=-1
    xyz=xyz@motion[:3,:3].T+motion[:3,3]
    xyz[:,1]*=-1
    return xyz


def navigation_profile(poly,shift,nav,motion):
    """Follow the selected branch and locate its subpixel ridge in the transmitted Ch 6."""
    poly=poly[np.r_[True,np.linalg.norm(np.diff(poly,axis=0),axis=1)>.001]]
    tangent=np.gradient(poly,axis=0)
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-5)
    points=poly+shift*np.column_stack((-tangent[:,1],tangent[:,0]))
    current=transform_flu(points,motion)[:,:2]
    i=int(np.argmin(np.linalg.norm(current,axis=1)))
    arc,k=plan_curvature(current[i+1:],current[i])
    # For a folded road, choose its first forward branch rather than a second peak.
    end=np.flatnonzero(np.diff(points[:,0])<-.01)
    end=end[end>=i]
    front=points[max(0,i-3):end[0]+1 if len(end) else None]
    keep=np.r_[True,np.diff(front[:,0])>.01]
    front=front[keep]
    cols=np.flatnonzero((X>=max(-3.,front[0,0]))&(X<=min(35.,front[-1,0])))
    expected=np.interp(X[cols],front[:,0],front[:,1])
    weights=nav[:,cols].astype(float)**4*(abs(Y[:,None]-expected)<2.4)
    total=weights.sum(0)
    valid=total>0.
    ridge=np.column_stack((X[cols[valid]],(Y@weights[:,valid])/total[valid]))
    return transform_flu(ridge,motion)[:,:2],arc,k


def obstacle_points(scan,plane,motion,vehicle="vehicle.tesla.model3"):
    raw=ego_points(scan)
    raw[:,1]*=-1
    height=raw[:,2]-(raw[:,:2]@plane[:2]+plane[2])
    nearby=(height>=.25)&(height<=2.2)&(abs(raw[:,0])<25.)&(abs(raw[:,1])<22.)
    # Roof sensor sees the Tesla's own rear deck. Mask this measured internal
    # surface only; keep returns along the external sides and front corners.
    deck=(raw[:,0]>-2.32)&(raw[:,0]<-2.10)&(abs(raw[:,1])<.45)&(raw[:,2]>.99)&(raw[:,2]<1.25)
    if vehicle=="vehicle.lincoln.mkz":
        # Two stationary 64-beam scans of the UE5 MKZ rear deck, inside its
        # measured body envelope. Keep the external bumper and side returns.
        deck=(raw[:,0]>-2.40)&(raw[:,0]<-2.12)&(abs(raw[:,1])<.55)&(raw[:,2]>1.04)&(raw[:,2]<1.20)
    nearby&=~deck
    return transform_flu(raw[nearby],motion)[:,:2]


def swept_clearance(points,length,steer,wheelbase,angle,horizon):
    """Distances to oriented body rectangles along the commanded bicycle arc."""
    left=right=hit=math.inf
    if not len(points):return left,right,hit
    curvature=-math.tan(steer*angle)/wheelbase
    for s in np.arange(0.,horizon+.01,.25):
        yaw=curvature*s
        sn,cs=math.sin(yaw),math.cos(yaw)
        cx=s*np.sinc(yaw/np.pi)
        cy=s*np.sinc(.5*yaw/np.pi)*math.sin(.5*yaw)
        dx,dy=points[:,0]-cx,points[:,1]-cy
        forward=dx*cs+dy*sn
        side=-dx*sn+dy*cs
        distance=np.hypot(np.maximum(abs(forward)-length/2,0.),np.maximum(abs(side)-.98,0.))
        left=min(left,distance[side>=0].min(initial=math.inf))
        right=min(right,distance[side<=0].min(initial=math.inf))
        if s<=6. and np.any(distance==0.):hit=min(hit,s)
    return left,right,hit


class UnifiedController:
    def __init__(self,vehicle="vehicle.tesla.model3"):
        self.vehicle=vehicle
        self.path=None
        self.last_steer=0.
        self.corridor_guard=False
        self.last_accel=0.
        self.closing_age=0.
        self.previous_speed=None
        self.measured_accel=0.
        self.drive_trim=0.
        self.previous_effort=0.

    def step(self,xy,pose,speed,ceiling,lead,probabilities,dimensions,wheelbase,max_angle,dt=.05,sensor_plan=None,
             *,scan,plane,nav,poly,shift,sensor_pose,keep):
        xy=np.asarray(xy,dtype=np.float64)
        sensor_plan=xy if sensor_plan is None else np.asarray(sensor_plan,dtype=np.float64)
        self.path=xy
        motion=np.linalg.solve(pose,sensor_pose)
        ridge,arc,k=navigation_profile(poly,shift,nav,motion)
        l0=float(np.clip(3.5+.615*speed,3.5,24))
        near=abs(k[arc<=l0+4.])
        curvature=float(near.max()) if len(near) else 0.
        look=l0/(1+40*curvature)
        if curvature>.008:look=np.clip(look,5.5,8.)
        points=np.vstack(([0.,0.],xy))
        lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        x,y=(np.interp(look,lengths,values) for values in points.T)
        corridor_error=0.
        if keep and len(ridge)>=2:
            end=min(20.,ridge[-1,0],xy[:,0].max())
            if end>=5.:
                stations=np.linspace(5.,end,16)
                # Interpolate only the first increasing-x section of the predicted path.
                folds=np.flatnonzero(np.diff(points[:,0])<=0)
                forward=points[:folds[0]+1] if len(folds) else points
                stations=stations[stations<=forward[-1,0]]
                if len(stations):
                    corridor_error=float(np.max(abs(np.interp(stations,forward[:,0],forward[:,1])
                                         -np.interp(stations,ridge[:,0],ridge[:,1]))))
            self.corridor_guard=corridor_error>(.9 if self.corridor_guard else 1.8)
            centre=np.interp(x,ridge[:,0],ridge[:,1])
            offset=abs(np.interp(0.,ridge[:,0],ridge[:,1]))
            bias=np.clip((offset-.20)/.8,0.,.75)
            if self.corridor_guard:
                y=centre+np.clip(y-centre,-1.5,1.5)*.20
            else:y+=bias*(centre-y)
        else:self.corridor_guard=False
        desired=-math.atan2(2*wheelbase*y,max(.5,x*x+y*y))/max_angle
        steer=float(np.clip(desired,self.last_steer-2.8*dt,self.last_steer+2.8*dt))
        steer=float(np.clip(steer,-1.,1.))
        obstacles=obstacle_points(scan,np.asarray(plane),motion,self.vehicle)
        horizon=max(6.,min(10.,look))
        left,right,hit=swept_clearance(obstacles,dimensions[0],steer,wheelbase,max_angle,horizon)
        closest=min(left,right)
        if closest<.85:
            if max(left,right)>=.85:
                sign=1. if left<right else -1.  # Positive CARLA steer goes right.
                steer=sign*max(sign*steer,.34*(.85-closest)/.85)
            else:steer=0.
            final_left,final_right,final_hit=swept_clearance(obstacles,dimensions[0],steer,wheelbase,max_angle,horizon)
            closest=min(closest,final_left,final_right)
            hit=min(hit,final_hit)
        self.last_steer=steer
        # Begin braking early enough for upcoming bends. The instantaneous steering
        # target uses local curvature so a distant bend cannot shorten today's lookahead.
        speed_squared=2.2/np.maximum(abs(k),.0001)+2.*3.*np.maximum(0.,arc-l0)
        curve_cap=math.sqrt(float(speed_squared.min())) if len(k) else math.inf
        target=min(ceiling,curve_cap)
        gap=max(.1,lead.x-dimensions[0]/2) if lead else math.inf
        closing=speed-lead.speed if lead else 0.
        sstar=4.5+max(0.,1.5*speed+speed*closing/(2*math.sqrt(6.)))
        acc=2.*(1.-(speed/max(.2,target))**4-(sstar/gap)**2)
        if target<.1:acc=min(acc,-2.)
        overlap,known=footprint_overlap(sensor_plan,probabilities,*dimensions)
        ttc=gap/closing if closing>.1 else math.inf
        self.closing_age=self.closing_age+dt if closing>.1 else 0.
        near_overlap=overlap[np.asarray(HORIZONS)<2.2]
        urgent=bool(near_overlap.any() and ttc<2.2 and self.closing_age>=.15)
        a=-7. if urgent else float(np.clip(np.clip(acc,-3.,2.),self.last_accel-3.*dt,self.last_accel+3.*dt))
        previous_accel=self.last_accel
        self.last_accel=a
        # Track the IDM acceleration through the simulated gearbox and drag. Update
        # against the previous acceleration command, on the same odometry interval.
        if self.previous_speed is not None:
            observed=np.clip((speed-self.previous_speed)/max(dt,.001),-9.,5.)
            self.measured_accel+=(1.-math.exp(-dt/.3))*(observed-self.measured_accel)
            error=previous_accel-self.measured_accel
            saturated=(self.previous_effort>=3.7 and error>0.) or (self.previous_effort<=-7. and error<0.)
            if not saturated:
                self.drive_trim=float(np.clip(self.drive_trim+.45*error*dt,-2.5,3.5))
        if urgent or (speed<.2 and (gap<=4.5 or target<.1)):
            self.drive_trim=0.
        self.previous_speed=speed
        effort=a+.12+.00075*speed*speed+self.drive_trim
        self.previous_effort=effort
        throttle=float(np.clip(effort/3.7,0.,1.))
        brake=float(np.clip(-effort/7.,0.,1.))
        safety_brake=bool(closest<.45 or hit<=6.)
        if safety_brake:
            throttle=0.
            severity=max((.45-closest)/.45,1.-hit/6.)
            brake=max(brake,min(1.,.30+.70*max(0.,severity)))
            self.drive_trim=0.
            self.previous_effort=-7.*brake
        return throttle,brake,float(np.clip(steer,-1.,1.)),dict(lookahead_m=float(look),curvature=curvature,
               target_mps=target,accel=a,measured_accel=self.measured_accel,drive_trim=self.drive_trim,
               urgent=urgent or safety_brake,ttc_s=ttc if math.isfinite(ttc) else None,
               d_static_L_m=left if np.isfinite(left) else None,d_static_R_m=right if np.isfinite(right) else None,
               static_brake=safety_brake,static_hit_arc_m=hit if np.isfinite(hit) else None,
               corridor_guard=self.corridor_guard,corridor_error_m=corridor_error,
               overlap=overlap.tolist(),covered=known.tolist(),
               pursuit_point_flu=[float(x),float(y)],curve_cap_mps=curve_cap)
