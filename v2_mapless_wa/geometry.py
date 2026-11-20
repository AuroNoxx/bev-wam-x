"""CARLA world coordinates are left-handed. Stored local coordinates are FLU.
Raster row 0 is farthest forward and column 0 farthest left.
"""
import cv2
import numpy as np

W,H=384,224
CAMS=((1.6,0,1.5,0,-10,100),(1.1,-.9,1.4,-45,-22,110),(1.1,.9,1.4,45,-22,110))
NAMES=('front','front_left','front_right')
Z=(-.2,0.,.6,1.4,2.4)
DX,DY=.75,.30
X,Y=np.meshgrid(80-(np.arange(128)+.5)*DX,19.2-(np.arange(128)+.5)*DY,indexing='ij')

def local(world,pose):
    p=(world-pose[:3,3])@pose[:3,:3]
    return p*np.array([1.,-1.,1.])

def world_at(local_xyz,pose):
    return (local_xyz*np.array([1.,-1.,1.]))@pose[:3,:3].T+pose[:3,3]

def pixels(xy):
    return np.stack(((19.2-xy[...,1])/DY-.5,(80-xy[...,0])/DX-.5),axis=-1)

def raster(scan):
    xy=scan[:,:2]
    pix=np.floor(pixels(xy)+.5).astype(int)
    take=((pix>=0)&(pix<128)).all(1)
    idx=pix[take,1]*128+pix[take,0]
    z=scan[take,2]; power=scan[take,3]
    n=np.bincount(idx,minlength=16384)
    lo=np.full(16384,np.inf); hi=-lo.copy()
    np.minimum.at(lo,idx,z); np.maximum.at(hi,idx,z)
    lo[n==0]=hi[n==0]=0
    counts=lambda mask:np.bincount(idx[mask],minlength=16384)/(DX*DY)
    return np.stack((lo,hi,counts((z>=.2)&(z<1.2)),counts((z>=1.2)&(z<=2.5)),
                     np.bincount(idx,power,minlength=16384)/np.maximum(n,1),np.log1p(n))).reshape(6,128,128).astype(np.float16)

def projection_tables(extrinsics,heights=Z):
    tables=[];valid=[]
    for height in heights:
        points=np.stack((X,-Y,np.broadcast_to(height,X.shape)),axis=-1)
        coords=[];masks=[]
        for cam,pose in zip(CAMS,extrinsics):
            p=(points-pose[:3,3])@pose[:3,:3]
            depth=np.maximum(p[...,0],1e-6)
            focal=W/(2*np.tan(np.deg2rad(cam[-1]/2)))
            uv=np.stack((W/2+focal*p[...,1]/depth,H/2-focal*p[...,2]/depth),axis=-1)
            mask=(p[...,0]>0)&(uv[...,0]>=0)&(uv[...,0]<W-1)&(uv[...,1]>=0)&(uv[...,1]<H-1)
            coords.append(np.where(mask[...,None],uv,-1).astype(np.float32));masks.append(mask)
        tables.append(coords);valid.append(masks)
    return np.asarray(tables),np.asarray(valid)

def road_surface(scan):
    """Fit road altitude in ego FLU. Give each occupied road patch one vote."""
    x,y,z=scan[:,:3].T
    use=(x>=1.8)&(x<30)&(abs(y)<3.4)&(abs(z)<.55)
    pts=scan[use,:3]
    cells=np.floor(pts[:,:2]/[2.,1.]).astype(int)
    _,groups=np.unique(cells,axis=0,return_inverse=True)
    patches=[]
    for group in range(groups.max()+1 if groups.size else 0):
        patch=pts[groups==group]
        if len(patch)>=4:
            patches.append([*np.median(patch[:,:2],axis=0),np.quantile(patch[:,2],.25)])
    if len(patches)<9:raise ValueError('Insufficient same-frame road support')
    pts=np.asarray(patches)
    design=np.column_stack((pts[:,:2],np.ones(len(pts))))
    keep=np.ones(len(pts),bool)
    for _ in range(4):
        abc,_,rank,_=np.linalg.lstsq(design[keep],pts[keep,2],rcond=None)
        if rank!=3:raise ValueError('Road plane is underconstrained')
        residual=pts[:,2]-design@abc
        mad=np.median(abs(residual[keep]-np.median(residual[keep])))
        keep=abs(residual)<=max(.035,3*1.4826*mad)
        if keep.sum()<6:raise ValueError('Insufficient inlier patches')
    return abc[0]*X+abc[1]*Y+abc[2],abc,float(np.median(abs(residual[keep])))

def sample_arc(points,distances):
    d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1))]
    keep=np.r_[True,np.diff(d)>1e-5]
    if d[-1]<max(distances):raise ValueError('incomplete future arc')
    return np.column_stack([np.interp(distances,d[keep],c[keep]) for c in points.T])

def polygon(mask,xy,value=1):
    uv=np.rint(pixels(xy)*16).astype(np.int32)
    cv2.fillPoly(mask,[uv],value,shift=4)
