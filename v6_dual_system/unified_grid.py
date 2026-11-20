"""Metric rasters for the unified driver. Columns are FLU x and rows are FLU y."""
import numpy as np

from .sensors import CALIBRATION, ego_points

HORIZONS = (.5, 1., 2., 3., 4.)
X = np.arange(128, dtype=np.float32) - 15.5
Y = (np.arange(128, dtype=np.float32) - 63.5)*.4


def camera_grid():
    _, position, settings = CALIBRATION["rgb"]
    width, height = (int(settings[name]) for name in ("image_size_x","image_size_y"))
    focal = width / (2*np.tan(np.deg2rad(float(settings["fov"]))/2))
    x,y = np.meshgrid(X,Y)
    depth = x-position[0]
    u = width/2 - focal*(y+position[1])/np.maximum(depth,.01)
    v = height/2 + focal*position[2]/np.maximum(depth,.01)
    valid = (depth>0)&(x>=0)&(x<=45)&(u>=0)&(u<=width-1)&(v>=0)&(v<=height-1)
    uv = np.stack((2*(u+.5)/width-1,2*(v+.5)/height-1),axis=-1)
    return np.where(valid[...,None],uv,2.).astype(np.float32),valid


def ground_plane(lidar):
    """Fit z=ax+by+c to current road returns only, in ego FLU metres."""
    row,col=np.nonzero((lidar[1]>0)&(lidar[0]>0))
    x,y=X[col],Y[row]
    z=lidar[0,row,col].astype(np.float64)*4./255.-.5
    valid=(x>=3)&(x<=48)&(abs(y)<=7)&(abs(z)<.45)
    a=np.column_stack((x[valid],y[valid],np.ones(valid.sum())))
    h=z[valid]
    if len(h)<18 or np.ptp(a[:,0])<12:
        return np.zeros(3,np.float32)
    weights=np.ones(len(h))
    for _ in range(4):
        fitted=np.linalg.lstsq(a*weights[:,None],h*weights,rcond=None)[0]
        residual=abs(a@fitted-h)
        weights=np.minimum(1.,.035/np.maximum(residual,1e-5))
    return fitted.astype(np.float32)


def corrected_camera_grid(plane):
    x,y=np.meshgrid(X,Y)
    f=320/np.tan(np.deg2rad(40.))
    depth=x-1.3
    z=plane[0]*x+plane[1]*y+plane[2]
    u=320.-f*y/np.maximum(depth,.01)
    v=180.+f*(2.-z)/np.maximum(depth,.01)
    valid=(depth>0)&(x>=0)&(x<=45)&(u>=0)&(u<=639)&(v>=0)&(v<=359)
    uv=np.stack((2*(u+.5)/640.-1,2*(v+.5)/360.-1),axis=-1)
    return np.where(valid[...,None],uv,2.).astype(np.float32),valid


def lidar_raster(scan):
    points=ego_points(scan)
    col=np.floor(points[:,0]+16).astype(np.int32)
    row=np.floor((-points[:,1]+25.6)/.4).astype(np.int32)
    inside=(col>=0)&(col<128)&(row>=0)&(row<128)
    cells=row[inside]*128+col[inside]
    count=np.bincount(cells,minlength=128**2)
    height=np.zeros(128**2,np.float32)
    np.maximum.at(height,cells,np.clip((points[inside,2]+.5)/4,0,1))
    intensity=np.bincount(cells,weights=scan[inside,3],minlength=128**2)/np.maximum(1,count)
    return np.rint(255*np.stack((height,np.minimum(1,np.log1p(count)/5),
                                 np.clip(intensity,0,1)))).astype(np.uint8).reshape(3,128,128)


def corridor(polyline,shift=0.):
    """Gaussian distance to the road-normal shifted route, in metres."""
    pts=np.asarray(polyline,dtype=np.float32)
    pts=pts[np.r_[True,np.linalg.norm(np.diff(pts,axis=0),axis=1)>.001]]
    if len(pts)<2:raise ValueError("Route requires two distinct points")
    tangent=np.gradient(pts,axis=0)
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-6)
    pts=pts+shift*np.column_stack((-tangent[:,1],tangent[:,0]))
    # Outside 6 sigma the quantized corridor is zero; compute exact distances locally.
    d2=np.full((128,128),np.inf,np.float32)
    for a,b in zip(pts,pts[1:]):
        low=np.minimum(a,b)-7.2
        high=np.maximum(a,b)+7.2
        x0,x1=np.searchsorted(X,[low[0],high[0]])
        y0,y1=np.searchsorted(Y,[low[1],high[1]])
        if x0>=x1 or y0>=y1:continue
        dx=X[None,x0:x1]-a[0]
        dy=Y[y0:y1,None]-a[1]
        sx,sy=b-a
        t=np.clip((dx*sx+dy*sy)/max(sx*sx+sy*sy,1e-6),0,1)
        area=d2[y0:y1,x0:x1]
        np.minimum(area,(dx-t*sx)**2+(dy-t*sy)**2,out=area)
    return np.rint(255*np.exp(-.5*d2/1.2**2)).astype(np.uint8)


def occupancy(boxes,pose):
    import cv2
    out=np.zeros((128,128),np.uint8)
    inverse=np.linalg.inv(pose)
    for box in boxes:
        local=box@inverse[:3,:3].T+inverse[:3,3]
        if local[:,2].min()>3 or local[:,2].max() < -1:continue
        uv=np.column_stack((local[:,0]+15.5,-local[:,1]/.4+63.5))
        hull=cv2.convexHull(np.rint(uv*256).astype(np.int32))
        cv2.fillConvexPoly(out,hull,1,shift=8)
    return out
