"""Choose a connected expert route for an already frozen, independently surveyed SD mission."""
import numpy as np

from .navigation import directed_step, waypoint_forward
from .scenarios import xyz


def survey_match(mission,xy,expected):
    lo=max(0,mission.station.searchsorted(expected-25)-1)
    hi=min(len(mission.delta),mission.station.searchsorted(expected+25)+1)
    d=mission.delta[lo:hi];length=mission.length[lo:hi]
    off=xy-mission.xy[lo:hi]
    t=np.clip((off*d).sum(1)/length**2,0,1)
    stations=mission.station[lo:hi]+t*length
    dist=np.linalg.norm(off-t[:,None]*d,axis=1)
    j=(dist+.03*abs(stations-expected)).argmin()
    return stations[j],dist[j],d[j]/length[j]


def route(road,mission,bounds,sampled):
    from .bulk import allowed
    distance=min(mission.station[-1]-72,mission.speed/3.6*107+95)
    candidates=[]
    for reverse in (True,False):
        s0=distance if reverse else 0.
        center=np.array([np.interp(s0,mission.station,c) for c in mission.xy.T])
        starts=[]
        for wp in sampled:
            loc=xyz(wp)[:2]
            if np.linalg.norm(loc-center)>30:continue
            station,error,heading=survey_match(mission,loc,s0)
            if waypoint_forward(wp)[:2]@heading<.65 or not allowed(loc,mission.split,bounds):continue
            starts.append((error+.2*abs(station-s0),station,wp))
        beam=[(v,s,[w],0.,0.) for v,s,w in sorted(starts,key=lambda v:v[0])[:38]]
        for _ in range(int(distance/4)+85):
            unique={}
            for cost,station,path,peak,accumulated in beam:
                current=path[-1]
                if (reverse and station<5) or (not reverse and station>=distance):
                    result=path[::-1] if reverse else path
                    roads={w.road_id for w in result}
                    if mission.required_roads.issubset(roads):
                        candidates.append((.45*peak+accumulated/len(path),result))
                    continue
                for wp in (current.previous(4) if reverse else current.next(4)):
                    if not (directed_step(wp,current) if reverse else directed_step(current,wp)):continue
                    xy=xyz(wp)[:2]
                    if not allowed(xy,mission.split,bounds):continue
                    expect=station+(-4 if reverse else 4)
                    st,err,heading=survey_match(mission,xy,expect)
                    align=waypoint_forward(wp)[:2]@heading
                    if align<.38 or err>18 or (st-station)*(-1 if reverse else 1)<-1:continue
                    score=err+2*(1-align)+.32*abs(st-expect)
                    score_sum=accumulated+score
                    rank=score_sum/len(path)+.45*max(peak,err)+.01*abs(st-expect)
                    key=(wp.road_id,wp.lane_id,round(wp.s/4))
                    if key not in unique or rank<unique[key][0]:
                        unique[key]=(rank,st,path+[wp],max(peak,err),score_sum)
            beam=sorted(unique.values(),key=lambda v:v[0])[:30]
            if not beam:break
    if not candidates:
        raise ValueError(f'No connected expert route for independent survey {mission.name}')
    return min(candidates,key=lambda v:v[0])[1]
