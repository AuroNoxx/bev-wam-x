"""Measure progress across the lane for confirmed same-road, stable crossings."""
from itertools import groupby

import carla
import numpy as np
from .scenarios import same_direction,xyz


def certify(rows,poses,road):
    counters={'left':0,'right':0}
    for side,seq in groupby(enumerate(rows),key=lambda x:x[1]['lane_change']):
        entries=list(seq)
        if not side:continue
        lanes=[r['lane_id'] for _,r in entries]
        transitions=np.flatnonzero(np.diff(lanes))
        if len(transitions)!=1 or len(set(r['road_id'] for _,r in entries))!=1:
            for _,r in entries:r['lane_change']=None
            continue
        source=lanes[0];target=lanes[-1]
        event=entries[transitions[0]+1][1]['frame']
        for i,r in entries:
            p=poses[i,:3,3]
            w=road.get_waypoint(carla.Location(*p))
            if w.lane_id==source:a,b=w,same_direction(w,side)
            elif w.lane_id==target:a,b=same_direction(w,'right' if side=='left' else 'left'),w
            else:a=b=None
            if a is None or b is None or a.is_junction or b.is_junction or a.lane_id!=source or b.lane_id!=target:
                r['lane_change']=None
                continue
            start=xyz(a)[:2];span=xyz(b)[:2]-start
            fraction=float((p[:2]-start)@span/max(1e-6,span@span))
            if .05<=fraction<=.95:
                r['lane_change_event']=event
                r['lane_change_fraction']=round(fraction,4)
                counters[side]+=1
            else:r['lane_change']=None
    return counters
