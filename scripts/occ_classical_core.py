"""Sensor-OCC-only stage-one planner. No CARLA import or actor-state interface.

FLU metres, seconds. Sparse occupied voxels + ray-cleared voxels are distinct.
Column visibility is a sampling-support diagnostic, not proof of full free volume.
"""
from dataclasses import dataclass
import math
import numpy as np

VEHICLE_TAGS = (14, 15, 16, 17, 18, 19)
GROUND_TAGS = (1, 24, 27)
RES = .5
ORIGIN = np.array([-20., -25., -1.])
SHAPE = np.array([200, 100, 12])  # xyz


@dataclass
class Occupancy:
    xyz: np.ndarray
    tags: np.ndarray
    cleared: np.ndarray
    frame: int
    timestamp: float
    ray_count: int


def voxelize(points, tags, sensor_origin, frame, timestamp):
    """Carve only sampled ray cells BEFORE a hit; never fill occluded space."""
    finite = np.isfinite(points).all(1)
    points, tags = points[finite], tags[finite]
    # Remove ego-body hits geometrically; do not consume sensor object IDs.
    self_hit = (np.abs(points[:, 0]) < 2.6) & (np.abs(points[:, 1]) < 1.25) & (points[:, 2] < 2.1)
    points, tags = points[~self_hit], tags[~self_hit]
    indices = np.floor((points-ORIGIN)/RES).astype(np.int32)
    valid = ((indices >= 0) & (indices < SHAPE)).all(1)
    # Prefer vehicle tag on mixed-hit cells, without labelling unobserved cells free.
    order = np.argsort(~np.isin(tags[valid], VEHICLE_TAGS), kind='stable')
    ijk, labs = indices[valid][order], tags[valid][order]
    _, first = np.unique(ijk, axis=0, return_index=True)
    centers, labs = ORIGIN+(ijk[first]+.5)*RES, labs[first]
    cleared = np.zeros(tuple(SHAPE), bool)
    # Ray sampling <= half voxel width. Last 0.45m before hit is not cleared.
    vectors = points-sensor_origin
    lengths = np.linalg.norm(vectors, axis=1)
    use = (lengths > 1.) & (lengths < 100.)
    vectors, lengths = vectors[use], lengths[use]
    for start in range(0, len(lengths), 128):
        v, d = vectors[start:start+128], lengths[start:start+128]
        if not len(d): continue
        steps = np.arange(.25, min(100., d.max()), .25)
        mask = steps[None] < (d[:, None]-.45)
        q = sensor_origin+v[:, None]*(steps[None, :, None]/d[:, None, None])
        idx = np.floor((q-ORIGIN)/RES).astype(np.int32)
        ok = mask & ((idx >= 0) & (idx < SHAPE)).all(-1)
        ii = idx[ok]
        cleared[ii[:, 0], ii[:, 1], ii[:, 2]] = True
    if len(ijk): cleared[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = False
    return Occupancy(centers.astype(np.float32), labs, cleared, int(frame), float(timestamp), len(points))


def vehicle_components(occ):
    pts = occ.xyz[np.isin(occ.tags, VEHICLE_TAGS)]
    if not len(pts): return []
    cells = np.floor(pts[:, :2]/1.).astype(int)
    buckets = {}
    for i, key in enumerate(map(tuple, cells)): buckets.setdefault(key, []).append(i)
    unseen = set(buckets); result = []
    while unseen:
        start = unseen.pop(); stack = [start]; members = []
        while stack:
            x, y = stack.pop(); members.extend(buckets[(x,y)])
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    key=(x+dx,y+dy)
                    if key in unseen: unseen.remove(key);stack.append(key)
        if len(members) >= 4:
            q=pts[members];result.append({'center':np.median(q,axis=0),'points':q})
    return result


class OccTracker:
    """Nearest-neighbour alpha-beta tracks from OCC component centroids only."""
    def __init__(self): self.tracks={};self.next_id=1

    def update(self, occ, ego_xy, yaw_flu):
        co,si=math.cos(yaw_flu),math.sin(yaw_flu)
        rotation=np.array([[co,-si],[si,co]])
        detections=vehicle_components(occ);unmatched=set(self.tracks);result=[]
        for d in detections:
            z=d['center'][:2]@rotation.T+ego_xy
            candidate=None;best=5.
            for tid in unmatched:
                tr=self.tracks[tid];dt=occ.timestamp-tr['time']
                distance=np.linalg.norm(z-(tr['position']+tr['velocity']*dt))
                if distance<best:candidate=tid;best=distance
            if candidate is None:
                tid=self.next_id;self.next_id+=1
                tr={'position':z.copy(),'velocity':np.zeros(2),'time':occ.timestamp,'hits':1}
            else:
                tid=candidate;unmatched.remove(tid);old=self.tracks[tid];dt=max(.05,occ.timestamp-old['time'])
                predicted=old['position']+old['velocity']*dt;innovation=z-predicted
                velocity=old['velocity']+.12*innovation/dt
                norm=np.linalg.norm(velocity)
                if norm>35:velocity*=35/norm
                tr={'position':predicted+.65*innovation,'velocity':velocity,'time':occ.timestamp,'hits':old['hits']+1}
            self.tracks[tid]=tr
            result.append({'id':tid,'center':d['center'].tolist(),
                           'velocity':(tr['velocity']@rotation).tolist(),
                           'hits':tr['hits'],'observed_voxels':len(d['points']),
                           'min_x':float(d['points'][:,0].min()),'max_x':float(d['points'][:,0].max()),
                           'min_y':float(d['points'][:,1].min()),'max_y':float(d['points'][:,1].max())})
        self.tracks={k:v for k,v in self.tracks.items() if occ.timestamp-v['time']<.8}
        return result


def project(points, route):
    if not len(points):return np.empty(0),np.empty(0),np.empty(0,dtype=int)
    distances=((points[:,None,:2]-route[None,:,:2])**2).sum(-1)
    indices=distances.argmin(1);delta=points[:,:2]-route[indices,:2]
    tangent=np.stack([np.cos(route[indices,3]),np.sin(route[indices,3])],-1)
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1))]
    along=arc[indices]+(delta*tangent).sum(-1)
    lateral=tangent[:,0]*delta[:,1]-tangent[:,1]*delta[:,0]
    return along,lateral,indices


def visible_distance(occ, route):
    # Count ray-cleared evidence in the future body-height corridor.
    columns=occ.cleared[:,:,3:6].any(-1)  # 0.5..2m local z samples
    seen=[];arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1))]
    for p in route:
        normal=np.array([-math.sin(p[3]),math.cos(p[3])])
        queries=p[:2]+np.linspace(-.8,.8,5)[:,None]*normal
        ij=np.floor((queries-ORIGIN[:2])/RES).astype(int)
        valid=((ij>=0)&(ij<SHAPE[:2])).all(1)
        evidence=np.zeros(5,bool);evidence[valid]=columns[ij[valid,0],ij[valid,1]]
        seen.append(evidence.mean()>=.4)
    # The known self footprint is not an unknown obstacle; beyond it require evidence.
    reach=3.;missing=0
    for s,yes in zip(arc,seen):
        if s<3:continue
        missing=0 if yes else missing+1
        if missing>=3:break
        if yes:reach=float(s)
    return reach,seen


class LanePlanner:
    def __init__(self, cruise_mps=8.33, half_length=2.4, half_width=1.0):
        self.cruise=cruise_mps;self.half_length=half_length;self.half_width=half_width

    def plan(self, occ, tracks, state, route, now):
        v=state['speed'];visible,support=visible_distance(occ,route)
        target=self.cruise;reason='cruise';gap=None;lead_speed=None
        # Unknown horizon limits speed under a conservative assumed deceleration.
        horizon_cap=max(0.,-1.5+math.sqrt(1.5**2+2*3.*max(0,visible-5.)))
        if target>horizon_cap:target=horizon_cap;reason='visibility_limit'
        p=occ.xyz
        obstacle=(~np.isin(occ.tags,GROUND_TAGS)) & (p[:,2]>.25) & (p[:,2]<2.4)
        q=p[obstacle];tags=occ.tags[obstacle]
        along,lat,indices=project(q,route)
        corridor=(along>self.half_length+.2)&(np.abs(lat)<self.half_width+.45)
        if corridor.any():
            chosen=np.flatnonzero(corridor)[np.argmin(along[corridor])]
            gap=float(along[chosen]-self.half_length-.6)
            lead_speed=0.
            if tags[chosen] in VEHICLE_TAGS:
                best=None;distance=6.
                for tr in tracks:
                    d=np.linalg.norm(np.array(tr['center'])[:2]-q[chosen,:2])
                    if d<distance:best=tr;distance=d
                if best is not None and best['hits']>=4:
                    tangent=route[indices[chosen],3]
                    lead_speed=max(0.,float(np.dot(best['velocity'],[math.cos(tangent),math.sin(tangent)]))-1.)
            desired_gap=4.+v*1.8
            following=max(0.,lead_speed+(gap-desired_gap)/2.)
            if following<target:target=following;reason='occ_follow'
            stop_cap=math.sqrt(2*3.*max(0.,gap-3.))
            if lead_speed<.5 and target>stop_cap:target=stop_cap;reason='occ_stop'
        stale=now-occ.timestamp>.16
        emergency=stale or (gap is not None and gap<max(3.,v*.6))
        if stale:reason='sensor_stale';target=0.
        if emergency:target=0.;reason='emergency_stop'
        # Deterministic timed trajectory on the permitted reference corridor.
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1))]
        distance=0.;speed=v;traj=[]
        for i in range(25):
            acceleration=float(np.clip((target-speed)/.8,-(6. if emergency else 3.),1.4))
            new_speed=max(0.,speed+.2*acceleration);distance+=.1*(speed+new_speed);speed=new_speed
            xy=[np.interp(distance,arc,route[:,j]) for j in (0,1)]
            yaw=float(np.interp(distance,arc,np.unwrap(route[:,3])))
            traj.append([*xy,yaw,speed,(i+1)*.2])
        return {'target_speed':float(target),'reason':reason,'gap_m':gap,'lead_speed_mps':lead_speed,
                'visible_support_m':visible,'support':support,'emergency':bool(emergency),
                'trajectory':traj,'sensor_frame':occ.frame,'sensor_age_s':now-occ.timestamp}


class Controller:
    def __init__(self,wheelbase,rear_x,max_angle):
        self.wheelbase=wheelbase;self.rear_x=rear_x;self.max_angle=max_angle
        self.integral=0.;self.last_steer=0.;self.reference=0.

    def update(self, state, route, plan, dt=.05, steering_scale=1.):
        speed=state['speed'];desired=plan['target_speed']
        if plan['emergency']:self.reference=0.
        else:self.reference+=float(np.clip(desired-self.reference,-3*dt,1.4*dt))
        error=self.reference-speed
        self.integral=float(np.clip(self.integral+error*dt,-2,2))
        effort=.10+.30*error+.05*self.integral
        throttle=float(np.clip(effort,0,.55));brake=float(np.clip(-effort,0,.65))
        if desired<.1 and speed<.3:throttle=0.;brake=.4;self.integral=0.
        if plan['emergency']:throttle=0.;brake=.8;self.integral=0.
        look=max(5.,min(14.,speed*.8+4.))
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1))]
        x=np.interp(look,arc,route[:,0])-self.rear_x;y=np.interp(look,arc,route[:,1])
        # FLU positive left -> CARLA control positive right.
        steer=-math.atan2(2*self.wheelbase*y,max(1.,x*x+y*y))/self.max_angle/max(.1,steering_scale)
        steer=float(np.clip(steer,self.last_steer-dt*.8,self.last_steer+dt*.8))
        steer=float(np.clip(steer,-.5,.5));self.last_steer=steer
        return {'throttle':throttle,'brake':brake,'steer':steer,'speed_reference':self.reference}
