"""Multi-lane OCC planning. Pure numpy, no simulator or NPC-state access.

Coordinates: current ego FLU. Map lanes and ego odometry are permitted inputs.
The sensor raster is enlarged only together with a real 220m simulated LiDAR.
"""
from dataclasses import dataclass
import math
import numpy as np
from occ_classical_core import VEHICLE_TAGS, GROUND_TAGS, project

RES=.5
ORIGIN=np.array([-120.,-35.,-1.5],dtype=np.float32)
SHAPE=np.array([680,140,16])


@dataclass
class HighwayOcc:
    xyz:np.ndarray
    tags:np.ndarray
    cleared:np.ndarray
    frame:int
    timestamp:float
    ray_count:int


def make_occ(points,tags,sensor_origin,frame,timestamp):
    points=np.asarray(points,dtype=np.float32);sensor_origin=np.asarray(sensor_origin,dtype=np.float32)
    good=np.isfinite(points).all(1)&~((np.abs(points[:,0])<2.6)&(np.abs(points[:,1])<1.3)&(points[:,2]<2.1))
    points,tags=points[good],tags[good]
    ijk=np.floor((points-ORIGIN)/RES).astype(np.int32)
    valid=((ijk>=0)&(ijk<SHAPE)).all(1)
    order=np.argsort(~np.isin(tags[valid],VEHICLE_TAGS),kind='stable')
    indices,labels=ijk[valid][order],tags[valid][order]
    _,first=np.unique(indices,axis=0,return_index=True)
    centers=ORIGIN+(indices[first]+.5)*RES;labels=labels[first]
    free=np.zeros(tuple(SHAPE),bool)
    vec=points-sensor_origin;length=np.linalg.norm(vec,axis=1)
    # Distance buckets prevent short ground returns being traced to max sensor range.
    for lo,hi in [(1,10),(10,20),(20,40),(40,80),(80,140),(140,221)]:
        mask=(length>lo)&(length<=hi);vv,dd=vec[mask],length[mask]
        for start in range(0,len(dd),128):
            v,d=vv[start:start+128],dd[start:start+128]
            steps=np.arange(.25,float(d.max()),.25,dtype=np.float32)
            q=sensor_origin+v[:,None]*(steps[None,:,None]/d[:,None,None])
            idx=np.floor((q-ORIGIN)/RES).astype(np.int32)
            ok=(steps[None]<d[:,None]-.45)&((idx>=0)&(idx<SHAPE)).all(-1)
            ii=idx[ok];free[ii[:,0],ii[:,1],ii[:,2]]=True
    if len(indices):free[indices[:,0],indices[:,1],indices[:,2]]=False
    return HighwayOcc(centers.astype(np.float32),labels,free,int(frame),float(timestamp),len(points))


def components(occ):
    points=occ.xyz[np.isin(occ.tags,VEHICLE_TAGS)]
    if not len(points):return []
    bins=np.floor(points[:,:2]).astype(int);buckets={}
    for i,key in enumerate(map(tuple,bins)):buckets.setdefault(key,[]).append(i)
    unseen=set(buckets);out=[]
    while unseen:
        todo=[unseen.pop()];ids=[]
        while todo:
            x,y=todo.pop();ids.extend(buckets[(x,y)])
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    key=(x+dx,y+dy)
                    if key in unseen:unseen.remove(key);todo.append(key)
        if len(ids)>=2:
            p=points[ids];out.append({'center':np.median(p,axis=0),'points':p})
    return out


class TrafficTracker:
    def __init__(self):self.tracks={};self.next_id=1

    def update(self,occ,ego_xy,ego_yaw):
        c,s=math.cos(ego_yaw),math.sin(ego_yaw);r=np.array([[c,-s],[s,c]])
        detections=components(occ);measurements=[d['center'][:2]@r.T+ego_xy for d in detections]
        proposals=[]
        for di,z in enumerate(measurements):
            for tid,tr in self.tracks.items():
                dt=occ.timestamp-tr['time'];pred=tr['position']+tr['velocity']*dt
                error=(z-pred)@r
                longitudinal_gate=max(6.,50*dt) if tr['hits']<4 else max(4.,10*dt)
                if abs(error[0])<longitudinal_gate and abs(error[1])<2.2:
                    proposals.append((float(abs(error[0])*.3+abs(error[1])*2),di,tid))
        assigned={};used=set()
        for _,di,tid in sorted(proposals):
            if di not in assigned and tid not in used:assigned[di]=tid;used.add(tid)
        seen=set()
        for di,(d,z) in enumerate(zip(detections,measurements)):
            if di in assigned:
                tid=assigned[di];old=self.tracks[tid];dt=max(.05,occ.timestamp-old['time'])
                if old['hits']==1:velocity=(z-old['measurement'])/dt
                else:
                    innovation=z-(old['position']+old['velocity']*dt)
                    velocity=old['velocity']+.16*innovation/dt
                norm=np.linalg.norm(velocity)
                if norm>50:velocity*=50/norm
                predicted=old['position']+old['velocity']*dt
                position=predicted+.7*(z-predicted)
                hits=old['hits']+1
            else:
                tid=self.next_id;self.next_id+=1;velocity=np.zeros(2);position=z.copy();hits=1
            size=np.ptp(d['points'][:,:2],axis=0)
            self.tracks[tid]={'position':position,'measurement':z,'velocity':velocity,'time':occ.timestamp,
                             'hits':hits,'size':size,'voxels':len(d['points'])};seen.add(tid)
        self.tracks={k:v for k,v in self.tracks.items() if occ.timestamp-v['time']<.8}
        result=[]
        for tid,tr in self.tracks.items():
            age=occ.timestamp-tr['time'];position=(tr['position']+tr['velocity']*age-ego_xy)@r
            velocity=tr['velocity']@r
            result.append({'id':tid,'x':float(position[0]),'y':float(position[1]),
                           'vx':float(velocity[0]),'vy':float(velocity[1]),'hits':tr['hits'],
                           'age_s':age,'observed':tid in seen,'voxel_count':tr['voxels'],
                           'half_length':float(np.clip(tr['size'][0]/2,2.4,6.)),
                           'half_width':float(np.clip(tr['size'][1]/2,.95,1.5))})
        return result


def visibility(occ,route):
    # Sparse ray support, NOT a certified free-space volume.
    columns=occ.cleared[:,:,4:8].any(-1)
    normal=np.stack([-np.sin(route[:,3]),np.cos(route[:,3])],-1)
    points=route[:,None,:2]+normal[:,None,:]*np.linspace(-.75,.75,5)[None,:,None]
    indices=np.floor((points-ORIGIN[:2])/RES).astype(int)
    ok=((indices>=0)&(indices<SHAPE[:2])).all(-1)
    evidence=np.zeros(ok.shape,bool);ii=indices[ok];evidence[ok]=columns[ii[:,0],ii[:,1]]
    support=evidence.mean(1)>=.4
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1))]
    # route is ordered from behind ego to ahead; find current longitudinal station.
    at=int(np.argmin(np.abs(route[:,0])))
    def extent(sequence):
        reach=0.;missing=0
        for i in sequence:
            distance=abs(arc[i]-arc[at])
            if distance<4:continue
            missing=0 if support[i] else missing+1
            if missing>=4:break
            if support[i]:reach=float(distance)
        return reach
    return {'front':extent(range(at,len(route))),'rear':extent(range(at,-1,-1)),
            'support':support.tolist()}


def lane_traffic(tracks,route,ego_speed):
    if not tracks:return {'front_gap':float('inf'),'rear_gap':float('inf'),'front_speed':50.,'rear_speed':0.,'lead_id':None,'cutin':False}
    points=np.array([[t['x'],t['y']] for t in tracks]);along,lat,indices=project(points,route)
    origin=int(np.argmin(np.abs(route[:,0])));arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1))]
    along-=arc[origin]
    result={'front_gap':float('inf'),'rear_gap':float('inf'),'front_speed':50.,'rear_speed':0.,'lead_id':None,'cutin':False}
    for t,x,y,i in zip(tracks,along,lat,indices):
        heading=route[i,3];vx=t['vx']*math.cos(heading)+t['vy']*math.sin(heading)
        vy=-t['vx']*math.sin(heading)+t['vy']*math.cos(heading)
        band=1.+t['half_width']+.35
        incoming=bool(abs(y)>band and abs(y+np.clip(vy,-2.,2.)*1.5)<band)
        if abs(y)>=band and not incoming:continue
        uncertainty=1.+t['age_s']*5.
        if x>0:
            gap=x-t['half_length']-2.5-uncertainty
            if gap<result['front_gap']:
                result.update(front_gap=float(gap),front_speed=max(0.,float(vx)-1.) if t['hits']>=3 else 0.,lead_id=t['id'],cutin=incoming)
        else:
            gap=-x-t['half_length']-2.5-uncertainty
            if gap<result['rear_gap']:result.update(rear_gap=float(gap),rear_speed=max(0.,float(vx)+1.) if t['hits']>=3 else 45.)
    return result


def permitted_gap(info,speed):
    front_closing=max(0.,speed-info['front_speed']);rear_closing=max(0.,info['rear_speed']-speed)
    front_required=max(14.,speed*1.5+front_closing*2.)
    rear_required=max(14.,info['rear_speed']*1.2,rear_closing*5.)
    return info['front_gap']>front_required and info['rear_gap']>rear_required


def speed_limit(info,speed,cruise,observed_range,braking=6.):
    # Stop before the observed horizon, accounting for .5s response and 8m buffer.
    horizon=max(0.,-braking*.5+math.sqrt((braking*.5)**2+2*braking*max(0.,observed_range-8.)))
    target=min(cruise,horizon);reason='cruise' if target>=cruise-.1 else 'visibility_limit'
    gap=info['front_gap'];lead=info['front_speed']
    if math.isfinite(gap):
        desired=5.+speed*1.6
        follow=max(0.,lead+(gap-desired)/2.5)
        # Relative stopping constraint includes a braking front vehicle.
        safe=max(0.,-braking*.5+math.sqrt((braking*.5)**2+lead**2+2*braking*max(0.,gap-5.)))
        if min(follow,safe)<target:target=min(follow,safe);reason='cutin_yield' if info['cutin'] else 'follow'
    emergency=gap<max(3.,speed*.35) or (speed>lead+1 and gap/max(.1,speed-lead)<1.2)
    return float(target),reason,bool(emergency)


def trajectory_from_route(route,speed,target,emergency=False):
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1))]
    start=int(np.argmin(np.abs(route[:,0])));arc=arc-arc[start]
    position=0.;velocity=speed;out=[]
    for step in range(50):
        a=float(np.clip((target-velocity)/.8,-(7. if emergency else 4.),1.8))
        new=max(0.,velocity+.1*a);position+=.05*(velocity+new);velocity=new
        out.append([np.interp(position,arc,route[:,0]),np.interp(position,arc,route[:,1]),
                    np.interp(position,arc,np.unwrap(route[:,3])),velocity,(step+1)*.1])
    return np.array(out)


def collision_risk(trajectory,tracks,horizon=5.):
    """Conservative swept samples with observed-box size and short-time CV bands."""
    for tr in tracks:
        # Do not pretend uncertain velocity is measured.
        vx=tr['vx'] if tr['hits']>=3 else 0.;vy=np.clip(tr['vy'],-2.,2.) if tr['hits']>=3 else 0.
        for q in trajectory:
            t=q[4]
            if t>horizon:break
            if 'predicted_xy' in tr:
                predicted=tr['predicted_xy'][min(len(tr['predicted_xy'])-1,max(0,round(t/.1)-1))]
                dx=predicted[0]-q[0];dy=predicted[1]-q[1]
            else:dx=tr['x']+vx*t-q[0];dy=tr['y']+vy*t-q[1]
            co,si=math.cos(q[2]),math.sin(q[2]);long=dx*co+dy*si;lat=-dx*si+dy*co
            pad_x=.8+.25*t+tr['age_s']*5.;pad_y=.25+.10*t+tr['age_s']
            if abs(long)<2.5+tr['half_length']+pad_x and abs(lat)<1.+tr['half_width']+pad_y:
                return {'track_id':tr['id'],'time_s':float(t)}
    return None


class HighwayPlanner:
    def __init__(self,cruise_mps=33.333,home_lane=-2):
        self.cruise=cruise_mps;self.home_lane=home_lane;self.target_lane=home_lane
        self.maneuver=None;self.last_change=-100.;self.slow_since=None;self.overtake_id=None
        self.events=[];self.started=None

    def plan(self,occ,tracks,state,lanes,lane_permissions,now):
        if self.started is None:self.started=now
        speed=state['speed'];current=state['lane'];station=state['s'];events=[]
        infos={l:lane_traffic(tracks,r,speed) for l,r in lanes.items()}
        views={l:visibility(occ,r) for l,r in lanes.items()}
        # Road-conditioned motion extrapolation; no true future actor states.
        for tr in tracks:
            nearest=None
            for route in lanes.values():
                _,lat,idx=project(np.array([[tr['x'],tr['y']]]),route)
                if nearest is None or abs(lat[0])<abs(nearest[1]):nearest=(route,float(lat[0]),int(idx[0]))
            route,lat,idx=nearest;heading=route[idx,3]
            long_v=max(0.,tr['vx']*math.cos(heading)+tr['vy']*math.sin(heading)) if tr['hits']>=3 else 0.
            side_v=np.clip(-tr['vx']*math.sin(heading)+tr['vy']*math.cos(heading),-1.5,1.5) if tr['hits']>=4 else 0.
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1))]
            predictions=[]
            for t in np.arange(.1,5.01,.1):
                station_t=arc[idx]+long_v*t;angle=np.interp(station_t,arc,np.unwrap(route[:,3]))
                yoffset=lat+side_v*min(t,2.)
                predictions.append([np.interp(station_t,arc,route[:,0])-math.sin(angle)*yoffset,
                                    np.interp(station_t,arc,route[:,1])+math.cos(angle)*yoffset])
            tr['predicted_xy']=predictions
        if self.target_lane not in lanes:self.target_lane=current
        if self.maneuver:
            move=self.maneuver;progress=np.clip((station-move['start_s'])/move['length'],0,1)
            if progress>.98 and current==move['to'] and abs(state['lateral_error'])<.3:
                event={'type':'lane_change_completed','from':move['from'],'to':move['to'],'kind':move['kind'],'time':now}
                events.append(event);self.maneuver=None;self.last_change=now
            elif move['kind']!='abort' and progress<.35 and not permitted_gap(infos[move['to']],speed):
                # Smoothly return only if source lane is also clear.
                if permitted_gap(infos[move['from']],speed):
                    self.maneuver={'from':move['to'],'to':move['from'],'start_s':station,
                                   'length':max(35.,speed*4.),'kind':'abort',
                                   'start_y':-float(lanes[move['to']][np.argmin(np.abs(lanes[move['to']][:,0])),1])}
                    self.target_lane=move['from'];events.append({'type':'abort_started','time':now})
        if not self.maneuver:
            self.target_lane=current
            info=infos[current];limited=info['front_gap']<max(80.,speed*3.) and info['front_speed']<self.cruise-2.
            if limited:self.slow_since=now if self.slow_since is None else self.slow_since
            else:self.slow_since=None
            choices=[]
            if limited and self.slow_since is not None and now-self.slow_since>.8 and speed>4 and now-self.last_change>5:
                left=current+1
                if left in lanes and lane_permissions.get((current,left),False):choices.append((left,'overtake'))
            if current!=self.home_lane and self.overtake_id is not None and now-self.last_change>3:
                passed=next((t for t in tracks if t['id']==self.overtake_id and t['x']<-max(18.,speed)),None)
                next_lane=current-1 if current>self.home_lane else current+1
                if passed is not None and next_lane in lanes and lane_permissions.get((current,next_lane),False):choices.append((next_lane,'return'))
            for destination,kind in choices:
                view=views[destination]
                rear_needed=max(35.,speed*1.5)
                if not permitted_gap(infos[destination],speed) or view['rear']<rear_needed or view['front']<max(30.,speed*1.5):continue
                length=max(35.,speed*5.)
                route=self.blend(lanes[current],lanes[destination],station,station,length,0.)
                target,_,_=speed_limit(infos[destination],speed,self.cruise,view['front'])
                proposal=trajectory_from_route(route,speed,min(target,max(speed,infos[current]['front_speed'])))
                if collision_risk(proposal,tracks):continue
                self.maneuver={'from':current,'to':destination,'start_s':station,'length':length,'kind':kind,'start_y':0.}
                self.target_lane=destination
                if kind=='overtake':self.overtake_id=info['lead_id']
                events.append({'type':'lane_change_started','from':current,'to':destination,'kind':kind,'time':now})
                break
        reference=lanes[self.target_lane]
        if self.maneuver:
            move=self.maneuver
            reference=self.blend(lanes[move['from']],lanes[move['to']],station,move['start_s'],move['length'],move.get('start_y',0.))
        info=infos[self.target_lane];target,reason,emergency=speed_limit(info,speed,self.cruise,views[self.target_lane]['front'])
        if self.maneuver:
            # During early transition retain longitudinal constraint of source lane.
            move=self.maneuver;u=(station-move['start_s'])/move['length']
            if u<.55:
                source_speed,_,source_emergency=speed_limit(infos[move['from']],speed,self.cruise,views[move['from']]['front'])
                target=min(target,source_speed);emergency|=source_emergency
            reason=move['kind']
        # Static occupied voxels are checked along the actual chosen corridor.
        static=(~np.isin(occ.tags,(*GROUND_TAGS,*VEHICLE_TAGS)))&(occ.xyz[:,2]>.3)&(occ.xyz[:,2]<2.4)
        points=occ.xyz[static]
        if len(points):
            along,lat,indices=project(points,reference)
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(reference[:,:2],axis=0),axis=1))]
            along-=arc[np.argmin(np.abs(reference[:,0]))]
            near=(along>3)&(np.abs(lat)<1.5)
            if near.any():
                gap=float(along[near].min()-3.)
                cap=max(0.,-3+math.sqrt(9+12*max(0,gap-4)))
                if cap<target:target=cap;reason='static_obstacle'
                emergency|=gap<max(3.,speed*.5)
        if now-occ.timestamp>.25:target=0.;emergency=True;reason='sensor_stale'
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(reference[:,:2],axis=0),axis=1))]
        curvature=np.abs(np.gradient(np.unwrap(reference[:,3]))/np.maximum(.5,np.gradient(arc)))
        ahead=(reference[:,0]>0)&(reference[:,0]<max(50.,speed*3.))
        if ahead.any():
            curve_cap=math.sqrt(3./max(1e-5,float(curvature[ahead].max())))
            if target>curve_cap:target=curve_cap;reason='curvature_limit'
        if emergency:
            target=0.
            if reason!='sensor_stale':reason='emergency_stop'
        traj=trajectory_from_route(reference,speed,target,emergency)
        risk=collision_risk(traj,tracks,horizon=3.)
        if risk:
            target=0.;reason='predicted_collision_brake';emergency=risk['time_s']<1.5
            traj=trajectory_from_route(reference,speed,0.,emergency)
        self.events.extend(events)
        def finite_info(d):return {k:(None if isinstance(v,float) and not math.isfinite(v) else v) for k,v in d.items()}
        return {'target_speed':float(target),'reason':reason,'emergency':bool(emergency),'target_lane':self.target_lane,
                'trajectory':traj.tolist(),'reference':reference,'events':events,'maneuver':self.maneuver,
                'lane_traffic':{str(k):finite_info(v) for k,v in infos.items()},
                'visibility':{str(k):{'front':v['front'],'rear':v['rear']} for k,v in views.items()},
                'collision_risk':risk,'sensor_frame':occ.frame}

    @staticmethod
    def blend(source,destination,station,start,length,start_y):
        # Equal stations in source/destination arrays. Quintic lateral shift.
        count=min(len(source),len(destination));a=source[:count];b=destination[:count]
        offsets=a[:,4] if a.shape[1]>4 else a[:,0]
        u=np.clip((station+offsets-start)/length,0,1)
        f=10*u**3-15*u**4+6*u**5
        out=a.copy();out[:,:3]=a[:,:3]+(b[:,:3]-a[:,:3])*f[:,None]
        if start_y:out[:,1]+=start_y*(1-f)
        delta=np.gradient(out[:,:2],axis=0);out[:,3]=np.arctan2(delta[:,1],delta[:,0])
        return out


class HighwayController:
    def __init__(self,wheelbase,rear_x,max_angle):
        self.wheelbase=wheelbase;self.rear_x=rear_x;self.max_angle=max_angle
        self.integral=0.;self.reference=0.;self.last_steer=0.

    def update(self,speed,plan,dt=.05,steering_scale=1.):
        desired=plan['target_speed']
        if plan['emergency']:self.reference=0.
        else:self.reference+=float(np.clip(desired-self.reference,-4.*dt,1.8*dt))
        error=self.reference-speed
        self.integral=float(np.clip(self.integral+error*dt,-3,4))
        effort=.12+.25*error+.055*self.integral
        throttle=float(np.clip(effort,0,.85));brake=float(np.clip(-effort,0,.7))
        if desired<.1 and speed<.3:throttle=0.;brake=.5;self.integral=0.
        if plan['emergency']:throttle=0.;brake=.9;self.integral=0.
        reference=plan['reference'];look=max(6.,min(34.,speed*.8+5.))
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(reference[:,:2],axis=0),axis=1))]
        arc-=arc[np.argmin(np.abs(reference[:,0]))]
        x=np.interp(look,arc,reference[:,0])-self.rear_x;y=np.interp(look,arc,reference[:,1])
        steer=-math.atan2(2*self.wheelbase*y,max(1.,x*x+y*y))/self.max_angle/max(.1,steering_scale)
        steer=float(np.clip(steer,self.last_steer-.5*dt,self.last_steer+.5*dt));steer=float(np.clip(steer,-.5,.5));self.last_steer=steer
        return {'throttle':throttle,'brake':brake,'steer':steer,'speed_reference':self.reference}
