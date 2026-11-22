"""Road-level navigation sketched on the independent nadir survey.
Coordinates are map-wide, rounded to five metres; there are no lane IDs or
precise lane-center samples in these routes. Directions are mission inputs.
"""
import json

import carla
import numpy as np
from .geometry import local

TOWN10 = {
    'straight': [(45,20),(0,20),(-45,20),(-80,20),(-105,20)],
    'curve': [(40,20),(40,45),(40,70),(70,70),(105,70),(105,95),(100,115),(90,130)],
    'fork': [(-45,0),(-45,-30),(-45,-65),(-10,-65),(40,-65)],
    'pole': [(15,20),(-15,20),(-45,20),(-45,55),(-45,95)],
}
# Eastern perimeter beyond y=105 m is reserved for future held-out route trials.
def sd_route(name):
    xy=np.array(TOWN10[name],dtype=float)
    return np.column_stack((xy,np.zeros(len(xy))))

def sd_routes(map_name,survey=None):
    if survey is None:
        if map_name!='Town10HD_Opt':
            raise ValueError(f'{map_name} requires its own independent SD route survey')
        return {name:sd_route(name) for name in TOWN10}
    data=json.loads(survey.read_text(encoding='utf-8'))
    if data['map']!=map_name:
        raise ValueError(f'SD survey for {data["map"]!r} cannot be used on {map_name!r}')
    routes={}
    for name,coordinates in data['routes'].items():
        xy=np.asarray(coordinates,dtype=float)
        if (xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<2 or not np.isfinite(xy).all()
                or np.any(np.linalg.norm(np.diff(xy,axis=0),axis=1)<1e-6)):
            raise ValueError(f'Invalid world XY polyline for {name!r} in {survey}')
        routes[name]=np.column_stack((xy,np.zeros(len(xy))))
    return routes

def gps_drift(phase,t_seconds):
    return 2.5*np.sin(phase+.12*t_seconds)


def waypoint_forward(waypoint):
    v=waypoint.transform.get_forward_vector()
    return np.array([v.x,v.y,v.z])


def directed_step(before,after):
    return after.lane_type == carla.LaneType.Driving and float(waypoint_forward(before)@waypoint_forward(after))>.5


class NoDirectedRoute(ValueError):
    pass


class DirectedRoute:
    """Certify the simulator route, then expose only three coarse navigation anchors."""
    def __init__(self, waypoints):
        if len(waypoints)<2 or waypoints[0].lane_type != carla.LaneType.Driving:
            raise NoDirectedRoute('A connected driving route is required')
        if not all(directed_step(a,b) for a,b in zip(waypoints,waypoints[1:])):
            raise NoDirectedRoute('Selected route reverses direction or exits driving lanes')
        self.xyz = np.array([[w.transform.location.x,w.transform.location.y,w.transform.location.z] for w in waypoints])
        self.forward = np.stack([waypoint_forward(w) for w in waypoints])
        self.margin = np.array([max(0.,.5*w.lane_width-.30) for w in waypoints])
        self.delta = np.diff(self.xyz[:,:2],axis=0)
        self.length = np.linalg.norm(self.delta,axis=1)
        if np.any(self.length<1e-4):
            raise NoDirectedRoute('Duplicated directed route vertices')
        if np.any((self.delta*self.forward[:-1,:2]).sum(1)/self.length <= .5):
            raise NoDirectedRoute('Directed route vertices do not progress forward')
        self.arc = np.r_[0.,self.length.cumsum()]

    def match(self,pose,previous=None):
        eligible = self.forward[:-1]@pose[:3,0] > .5
        if previous is not None:
            eligible &= (self.arc[:-1]<=previous+85.)&(self.arc[1:]>=previous-15.)
        ix = np.flatnonzero(eligible)
        if not len(ix):
            raise NoDirectedRoute('No local driving segment satisfies forward dot > 0.5')
        offset = pose[:2,3]-self.xyz[ix,:2]
        fraction = np.clip((offset*self.delta[ix]).sum(1)/self.length[ix]**2,0,1)
        lateral = np.linalg.norm(offset-fraction[:,None]*self.delta[ix],axis=1)
        # Prefer the same deck at stacked interchanges.
        height = self.xyz[ix,2]+fraction*(self.xyz[ix+1,2]-self.xyz[ix,2])-pose[2,3]
        nearest = int(np.argmin(lateral**2+height**2))
        j = ix[nearest]
        return float(self.arc[j]+fraction[nearest]*self.length[j]),float(np.hypot(lateral[nearest],height[nearest]))

    def positions(self,station,offsets,drift):
        target = np.minimum(station+np.asarray(offsets),self.arc[-1])
        point = np.column_stack([np.interp(target,self.arc,c) for c in self.xyz.T])
        axis = np.column_stack([np.interp(target,self.arc,c) for c in self.forward[:,:2].T])
        normal = axis[:,::-1]*[1,-1]/np.linalg.norm(axis,axis=1,keepdims=True)
        limit = self.drift_limit(station)
        point[:,:2] += normal*np.clip(drift,-limit,limit)
        return point

    def drift_limit(self,station):
        first=max(0,self.arc.searchsorted(station)-1)
        last=min(len(self.arc),self.arc.searchsorted(station+60)+1)
        return float(self.margin[first:last].min())

    def anchors(self,station,pose,drift,require_full=False):
        if require_full and station+60>self.arc[-1]:
            raise NoDirectedRoute('Directed route ends before the 60 m anchor')
        return local(self.positions(station,(20,40,60),drift),pose)[:,:2].astype(np.float32)

    def preview(self,station,pose,drift):
        return local(self.positions(station,np.arange(0.,61.,2.),drift),pose)[:,:2].astype(np.float32)

def navigation(coarse,pose,drift):
    a,b=coarse[:-1,:2],coarse[1:,:2]
    delta=b-a; length=np.linalg.norm(delta,axis=1)
    frac=np.clip(((pose[:2,3]-a)*delta).sum(1)/np.maximum(length**2,1e-6),0,1)
    i=np.argmin(np.linalg.norm(a+frac[:,None]*delta-pose[:2,3],axis=1))
    station=np.r_[0,length.cumsum()]
    target=station[i]+frac[i]*length[i]+np.array([20,40,60])
    if target[-1]>station[-1]:raise ValueError('SD route ends before last anchor')
    points=np.stack([np.interp(target,station,c) for c in coarse.T],axis=1)
    tangent=delta[np.searchsorted(station[1:],target)]
    # CARLA y points right. Drift along local road normals varies slowly in time.
    normal=np.stack((tangent[:,1],-tangent[:,0]),1)/np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-6)
    points[:,:2]+=normal*drift
    return local(points,pose)[:,:2].astype(np.float32)
