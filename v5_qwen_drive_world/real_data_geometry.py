"""Physical labels from actual simulator poses and actor boxes, in current FLU."""
import numpy as np


def convex_equations(points):
    """Convex hull of eight box vertices; no raster or image encoder is involved."""
    points=sorted(map(tuple,points))
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    halves=[]
    for order in (points,points[::-1]):
        half=[]
        for p in order:
            while len(half)>=2 and cross(half[-2],half[-1],p)<=0:half.pop()
            half.append(p)
        halves+=half[:-1]
    hull=np.asarray(halves)
    if len(hull)<3:raise ValueError('Degenerate source actor footprint')
    edge=np.roll(hull,-1,axis=0)-hull;normal=np.c_[edge[:,1],-edge[:,0]]
    return np.c_[normal,-(normal*hull).sum(1)]

X=-16+(np.arange(128)+.5);Y=-19.2+(np.arange(128)+.5)*.3
XX,YY=np.meshgrid(X,Y)


def ego(row):return next(b for b in row['bounding_boxes'] if b['class']=='ego_vehicle')


def world_pose(row):return np.linalg.inv(np.asarray(ego(row)['world2ego'],np.float64))


def relative_poses(rows,reference):
    inv=np.asarray(ego(reference)['world2ego'],np.float64)
    output=[]
    for row in rows:
        local=inv@world_pose(row)
        output.append([local[0,3],-local[1,3],np.arctan2(-local[1,0],local[0,0])])
    return np.asarray(output,np.float32)


def occupancy(row,reference,radius=90.):
    inv=np.asarray(ego(reference)['world2ego'],np.float64)
    centre=relative_poses([row],reference)[0,:2]
    support=((XX-centre[0])**2+(YY-centre[1])**2 <= radius**2)
    output=np.zeros((128,128),np.uint8)
    for b in row['bounding_boxes']:
        if b['class'] not in ('vehicle','walker','pedestrian'):continue
        vertices=np.asarray(b['world_cord'],np.float64)
        local=(inv@np.c_[vertices,np.ones(len(vertices))].T).T[:,:2];local[:,1]*=-1
        points=np.unique(local,axis=0)
        if len(points)<3:continue
        equations=convex_equations(points)
        lo=points.min(0);hi=points.max(0)
        js=np.flatnonzero((X>=lo[0])&(X<=hi[0]));ii=np.flatnonzero((Y>=lo[1])&(Y<=hi[1]))
        if not len(js) or not len(ii):continue
        gx,gy=np.meshgrid(X[js],Y[ii]);xy=np.stack((gx,gy),-1)
        inside=(xy@equations[:,:2].T+equations[:,2]<=1e-6).all(-1)
        output[np.ix_(ii,js)] |= inside.astype(np.uint8)
    return output,support


def labels(rows,index,source):
    now=rows[index];future=rows[index+1:index+51]
    poses=relative_poses([now]+future,now)
    speeds=np.array([r['speed'] for r in [now]+future],np.float64)
    ax=np.diff(speeds)/.1
    yaw=np.unwrap(poses[:,2].astype(np.float64));rate=np.diff(yaw)/.1
    ay=rate*np.maximum(speeds[:-1]**2,25)/np.maximum(speeds[:-1],.1)
    ay=np.where(speeds[:-1]<.1,0,ay)
    raw=np.stack((ax,ay),-1).astype(np.float32)
    bounded=np.clip(raw,[-8,-3.5],[4,3.5]).astype(np.float32)
    radius=90 if source=='bench2drive' else 190
    current,current_support=occupancy(now,now,radius)
    occ=[];support=[];ego_future=[]
    for k in (10,20,30):
        image,mask=occupancy(rows[index+k],now,radius);occ.append(image);support.append(mask)
        ego_future.append(relative_poses([rows[index+k]],now)[0])
    return {'trajectory':poses[1:,:2],'controls':bounded,'raw_controls':raw,'v0':np.float32(speeds[0]),
            'occupancy':np.asarray(occ),'support':np.asarray(support),'current_occupancy':current,
            'current_support':current_support,'future_ego_poses':np.asarray(ego_future),
            'clipped_control_fraction':np.float32(np.mean(np.abs(raw-bounded)>1e-5))}
