"""Offline road survey and scenario selection. Only coarse navigation is exposed
to the actor. Full waypoints and static geometry stay on the teacher side.
"""
import carla
import numpy as np

def xyz(w):
    a=w.transform.location
    return np.array([a.x,a.y,a.z])

def same_direction(w,side):
    if w is None:return None
    n=w.get_left_lane() if side=='left' else w.get_right_lane()
    if n is None or n.lane_type!=carla.LaneType.Driving or w.lane_id*n.lane_id<=0:return None
    return n

def routes(road_map,poles):
    choices=[]
    for i,s in enumerate(road_map.get_spawn_points()):
        start=road_map.get_waypoint(s.location)
        for branch in (0,-1):
            path=[start]; forks=[]
            for step in range(90):
                opts=sorted(path[-1].next(2),key=lambda w:(w.road_id,w.lane_id,w.s))
                if not opts:break
                if len(opts)>1:forks.append(step*2)
                path.append(opts[branch])
            if len(path)<82:continue
            pts=np.stack([xyz(w) for w in path]);hd=np.unwrap(np.arctan2(*np.diff(pts[:,:2],axis=0).T[::-1]))
            turn=np.max(abs(hd[1:23]-hd[:22]))/2
            pole=np.linalg.norm(pts[3:14,None,:2]-poles[None,:,:2],axis=2).min() if len(poles) else np.inf
            choices.append((i,start,path,turn,pole,min(forks,default=1e4),branch))
    out={};used=[]
    scores={'straight':lambda c:c[3]+.01*(c[5]<45),'curve':lambda c:-c[3],
            'fork':lambda c:abs(c[5]-16),'pole':lambda c:abs(c[4]-3)}
    for name,score in scores.items():
        pool=[c for c in choices if all(np.linalg.norm(xyz(c[1])[:2]-p)>22 for p in used)]
        sel=min(pool,key=score); used.append(xyz(sel[1])[:2]);out[name]=sel
    return out
