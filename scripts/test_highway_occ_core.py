"""Meaningful negative controls for highway planning and observation isolation."""
import ast
import json
from pathlib import Path
import numpy as np
from highway_occ_core import HighwayOcc,SHAPE,TrafficTracker,HighwayPlanner,permitted_gap,collision_risk,make_occ,ORIGIN,RES


def main():
    clear=np.ones(tuple(SHAPE),bool)
    obs=HighwayOcc(np.empty((0,3)),np.empty(0,int),clear,1,0.,1000)
    x=np.arange(-100,201,dtype=float)
    lanes={l:np.c_[x,np.full(len(x),(l+3)*3.5),np.zeros((len(x),2)),x] for l in [-4,-3,-2]}
    permissions={(a,b):True for a in lanes for b in lanes if abs(a-b)==1}
    ego={'speed':20.,'lane':-3,'s':0.,'lateral_error':0.}
    def vehicle(i,x,y,v):return {'id':i,'x':x,'y':y,'vx':v,'vy':0.,'hits':12,'age_s':0.,'observed':True,'voxel_count':10,'half_length':2.4,'half_width':.95}
    p=HighwayPlanner(30.,-3)
    a=p.plan(obs,[vehicle(1,50,0,10)],ego,lanes,permissions,0.)
    obs.timestamp=1.;b=p.plan(obs,[vehicle(1,50,0,10)],ego,lanes,permissions,1.)
    assert b['maneuver'] and b['target_lane']==-2,b
    json.dumps({k:v for k,v in b.items() if k!='reference'},allow_nan=False)
    p=HighwayPlanner(30.,-3);tracks=[vehicle(1,50,0,10),vehicle(2,-20,3.5,35)]
    obs.timestamp=0.;p.plan(obs,tracks,ego,lanes,permissions,0.)
    obs.timestamp=1.;blocked=p.plan(obs,tracks,ego,lanes,permissions,1.)
    assert blocked['maneuver'] is None,'Must reject fast rear vehicle'
    p=HighwayPlanner(30.,-3);obs.timestamp=0.;p.plan(obs,[vehicle(1,50,0,10)],ego,lanes,{},0.)
    obs.timestamp=1.;assert p.plan(obs,[vehicle(1,50,0,10)],ego,lanes,{},1.)['maneuver'] is None
    unknown=HighwayOcc(obs.xyz,obs.tags,np.zeros(tuple(SHAPE),bool),1,0.,0)
    assert HighwayPlanner(30.).plan(unknown,[],ego,lanes,permissions,0.)['target_speed']==0
    assert HighwayPlanner(30.).plan(obs,[],ego,lanes,permissions,2.)['reason']=='sensor_stale'
    tracker=TrafficTracker();points=np.array([[20+dx,y,1.] for dx in [0,.5,1] for y in [-.5,0,.5]])
    for i in range(10):
        # Absolute object speed36m/s, ego speed30m/s; >5m displacement per update.
        q=points.copy();q[:,0]+=i*1.2
        tracks=tracker.update(HighwayOcc(q,np.full(len(q),14),clear,i,i*.2,9),np.array([i*6.,0.]),0.)
    assert len(tracks)==1 and tracks[0]['hits']==10 and abs(tracks[0]['vx']-36)<1,tracks
    carved=make_occ(np.array([[180.,0.,1.]]),np.array([14]),np.array([0.,0.,1.]),1,0.)
    def cell(x):return tuple(np.floor((np.array([x,0,1])-ORIGIN)/RES).astype(int))
    assert carved.cleared[cell(150)] and not carved.cleared[cell(181)]
    source=Path(__file__).with_name('highway_occ_core.py').read_text()
    tree=ast.parse(source)
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and 'carla' in ast.get_source_segment(source,n) for n in ast.walk(tree))
    print(json.dumps({'pass':True,'overtake_when_clear':True,'fast_rear_blocks':True,'marking_permission_required':True,
                      'unknown_and_stale_stop':True,'high_speed_tracking_mps':tracks[0]['vx'],'long_range_ray_no_behind_hit':True,'no_simulator_import':True}))


if __name__=='__main__':main()
