"""Offline teacher targets. This module is not part of the actor input path."""
import argparse,json,time
from pathlib import Path
import carla,cv2,numpy as np
from .geometry import local,pixels,polygon,sample_arc
from .scenarios import xyz,same_direction
from .navigation import navigation,sd_route,gps_drift

def right(w):
    r=w.transform.get_right_vector()
    return np.array([r.x,r.y,r.z])

def ends(w):
    radius=right(w)*w.lane_width/2
    return xyz(w)-radius,xyz(w)+radius

def marked(mark):
    return str(mark.type).lower() not in ('none','other') and mark.width>0

def projection(pts3,pose):
    return local(pts3,pose)[...,:2]

class Surfaces:
    def __init__(self,m):
        quads=[]; borders=[]; curb=[]
        for w in m.generate_waypoints(1.):
            if w.lane_type!=carla.LaneType.Driving:continue
            for n in w.next(1.02):
                a,b=ends(w); c,d=ends(n)
                if np.linalg.norm(a-c)>4 or np.linalg.norm(b-d)>4:continue
                quads.append([a,b,d,c])
                borders.append([marked(w.left_lane_marking),marked(w.right_lane_marking)])
                adj=[w.get_left_lane(),w.get_right_lane()]
                curb.append([p is not None and p.lane_type==carla.LaneType.Sidewalk for p in adj])
        self.quads=np.asarray(quads);self.centers=self.quads.mean(1)
        self.borders=np.asarray(borders);self.curb=np.asarray(curb)

    def draw(self,pose):
        mask=np.zeros((4,128,128),np.uint8)
        centers=local(self.centers,pose)
        near=(centers[:,0]>-25)&(centers[:,0]<88)&(abs(centers[:,1])<27)&(abs(centers[:,2])<3)
        pts=local(self.quads[near],pose)
        uv=np.rint(pixels(pts[...,:2])*16).astype(np.int32)
        for q,mark,curb in zip(uv,self.borders[near],self.curb[near]):
            cv2.fillConvexPoly(mask[0],q,1,shift=4)
            for side,(i,j) in enumerate(((0,3),(1,2))):
                if mark[side]:cv2.line(mask[1],q[i],q[j],1,1,shift=4)
                if curb[side]:cv2.line(mask[2],q[i],q[j],1,1,shift=4)
        return mask

def map_vectors(m,pose,driven,static,mask):
    w=m.get_waypoint(carla.Location(*pose[:3,3]),lane_type=carla.LaneType.Driving)
    ref=[w]
    for i in range(1,64):
        opts=ref[-1].next(1.)
        if not opts:break
        # At junctions use the branch the demonstrator actually traversed.
        # Only the teacher sees this trace. Examine the outgoing road several metres
        # beyond the common branch point, where the candidates can be distinguished.
        def error(p):
            return min(np.linalg.norm(driven[:,:2]-xyz(q)[:2],axis=1).min()
                       for q in (p.next(8.) or [p]))
        ref.append(opts[0] if len(opts)==1 else min(opts,key=error))
    # Account for the exact source station of the nearest road waypoint.
    centers=np.array([xyz(w) for w in ref])
    station=np.r_[0,np.linalg.norm(np.diff(centers[:,:2],axis=0),axis=1).cumsum()]
    targets=np.linspace(0,60,20)
    valid=np.zeros((7,20),bool);lines=np.zeros((7,20,2),np.float32)
    for a,b in zip(ref,ref[1:]):
        al,ar=ends(a);bl,br=ends(b)
        polygon(mask[3],projection(np.array([al,ar,br,bl]),pose))
    # Also rasterize the current lane behind ego and between 60 m and grid end.
    for forward,endpoint,steps in ((False,ref[0],22),(True,ref[-1],26)):
        for _ in range(steps):
            opts=endpoint.next(1.) if forward else endpoint.previous(1.)
            if not opts or len(opts)>1:break
            n=opts[0];a,b=ends(endpoint);c,d=ends(n)
            polygon(mask[3],projection(np.array([a,b,d,c]),pose));endpoint=n
    for j,s in enumerate(targets):
        if s>station[-1]:continue
        k=min(np.searchsorted(station,s,side='right')-1,len(ref)-2)
        w,n=ref[k:k+2]; f=(s-station[k])/max(1e-6,station[k+1]-station[k])
        for idx,get in ((0,lambda p:xyz(p)),(1,lambda p:ends(p)[0]),(2,lambda p:ends(p)[1]),
                        (5,lambda p:xyz(q) if (q:=same_direction(p,'left')) else None),
                        (6,lambda p:xyz(q) if (q:=same_direction(p,'right')) else None)):
            a=get(w);b=get(n)
            if a is None or b is None:continue
            if idx in (1,2) and not marked(w.left_lane_marking if idx==1 else w.right_lane_marking):continue
            lines[idx,j]=projection((1-f)*a+f*b,pose);valid[idx,j]=True
        center=lines[0,j];r=projection(xyz(w)+right(w),pose)-projection(xyz(w),pose)
        tangent=np.array([-r[1],r[0]])
        for idx,sgn in ((3,-1),(4,1)):
            delta=static-center; side=delta@r*sgn
            take=(abs(delta@tangent)<1.6)&(side>w.lane_width*.35)&(side<18)
            # The nearest surveyed hard surface in this cross section.
            if take.any():
                points=static[take];lines[idx,j]=points[side[take].argmin()];valid[idx,j]=True
    return lines,valid

def save_episode(folder,m,surf,tags):
    meta=json.loads((folder/'episode.json').read_text())
    coarse=sd_route(meta['scenario'])
    with np.load(folder/'teacher.npz') as f:data={k:f[k] for k in f}
    ids=data['fid'];poses=data['pose'];boxes=data['boxes']; assert np.all(np.diff(ids)==1)
    sidx=np.repeat(data['sem_fid'],data['sem_count'])
    # Disjoint survey for static boundaries: no current pilot-frame lidar returns in labels.
    use=(sidx>max(meta['frames']))&np.isin(data['sem_tag'],[tags[k] for k in ('Poles','Walls','Fences','GuardRail','TrafficSigns')])
    world=data['sem_xyz'][use]
    for fid in meta['frames']:
        fd=folder/f'{fid:07d}';i=int(fid-ids[0]);pose=poses[i]
        path=projection(sample_arc(poses[i:,:3,3],np.arange(1,41)*1.5),pose).astype(np.float32)
        occ=np.zeros((5,128,128),np.uint8)
        for grid,k in zip(occ,(10,20,40,60,80)):
            for corners in boxes[i+k]:
                pts=local(corners,pose)
                if abs(pts[:,2].mean())>3:continue
                polygon(grid,cv2.convexHull(pts[:,:2].astype(np.float32)).reshape(-1,2))
        mask=surf.draw(pose)
        points=local(world,pose)
        keep=(points[:,2]>.15)&(points[:,2]<2.5)
        uv=np.rint(pixels(points[keep,:2])).astype(np.int32)
        uv=uv[((uv>=0)&(uv<128)).all(1)]
        mask[2,uv[:,1],uv[:,0]]=1
        rows,cols=np.nonzero(mask[2])
        static=np.column_stack((80-(rows+.5)*.75,19.2-(cols+.5)*.3))
        lines,valid=map_vectors(m,pose,poses[i:,:3,3],static,mask)
        drift=gps_drift(meta['drift_phase'],.05*i)
        nav=navigation(coarse,pose,drift)
        np.savez_compressed(fd/'input.npz',sd_nav=nav,cmd_one_hot=np.array([1,0,0],np.uint8))
        np.savez_compressed(fd/'labels.npz',map_lines=lines,exists=valid.any(1),point_valid=valid,
                            masks=mask,path_gt=path,p_occ=occ)
        yaw=np.rad2deg(np.arctan2(pose[1,0],pose[0,0]))
        np.savez_compressed(fd/'index.npz',frame=np.int64(fid),pose_xyzyaw=np.r_[pose[:3,3],yaw],
                            episode_tick=i,world_from_ego=pose,sd_drift_m=drift)
    print(folder.name,'labels',len(meta['frames']),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);out=p.parse_args().out
    start=time.perf_counter();m=carla.Map('pilot',(out/'map.xodr').read_text(encoding='utf8')); surf=Surfaces(m)
    tags=json.loads((out/'sim.json').read_text())['tags']
    for ep in sorted(out.glob('*/episode.json')):save_episode(ep.parent,m,surf,tags)
    (out/'offline-seconds.txt').write_text(str(time.perf_counter()-start))

if __name__=='__main__':main()
