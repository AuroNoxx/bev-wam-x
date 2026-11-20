"""Road-level navigation sketched on the independent nadir survey.
Coordinates are map-wide, rounded to five metres; there are no lane IDs or
precise lane-center samples in these routes. Directions are mission inputs.
"""
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

def gps_drift(phase,t_seconds):
    return 2.5*np.sin(phase+.12*t_seconds)

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
