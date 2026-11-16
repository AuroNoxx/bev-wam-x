"""Contract tests: ray occlusion, ego-motion compensation and control signs."""
import json
import numpy as np
from occ_classical_core import Occupancy, voxelize, OccTracker, LanePlanner, Controller, SHAPE, ORIGIN, RES


def main():
    hit=np.array([[12.,0.,1.]])
    occ=voxelize(hit,np.array([14]),np.array([0.,0.,1.]),1,0.)
    def cell(p):return tuple(np.floor((np.array(p)-ORIGIN)/RES).astype(int))
    assert occ.cleared[cell([6.,0.,1.])]
    assert not occ.cleared[cell([13.,0.,1.])]
    assert not occ.cleared[cell([12.,0.,1.])]
    route=np.column_stack([np.arange(66),np.zeros((66,3))]).astype(float)
    empty=Occupancy(np.empty((0,3)),np.empty(0,dtype=int),np.ones(tuple(SHAPE),bool),1,0.,1000)
    planner=LanePlanner(8.33)
    clear=planner.plan(empty,[],{'speed':5.},route,0.)
    assert abs(clear['target_speed']-8.33)<1e-5
    blocked=Occupancy(np.array([[10.,0.,1.]]),np.array([3]),empty.cleared,1,0.,1000)
    result=planner.plan(blocked,[],{'speed':5.},route,0.)
    assert result['target_speed']==0 and result['gap_m']<8
    unknown=Occupancy(empty.xyz,empty.tags,np.zeros(tuple(SHAPE),bool),1,0.,0)
    assert planner.plan(unknown,[],{'speed':5.},route,0.)['target_speed']==0
    assert planner.plan(empty,[],{'speed':5.},route,1.)['emergency']
    points=np.array([[x,y,1.] for x in [10,10.5,11] for y in [-.5,0,.5]])
    tracker=OccTracker()
    for i in range(12):
        q=points.copy();q[:,0]-=i*.5
        tracked=tracker.update(Occupancy(q,np.full(len(q),14),empty.cleared,i,i*.2,100),np.array([i*.5,0]),0.)
    assert len(tracked)==1 and np.linalg.norm(tracked[0]['velocity'])<1e-5
    tracker=OccTracker()
    for i in range(30):
        q=points.copy();q[:,0]+=i*.6
        tracked=tracker.update(Occupancy(q,np.full(len(q),14),empty.cleared,i,i*.2,100),np.zeros(2),0.)
    assert abs(tracked[0]['velocity'][0]-3)<.3
    left=route.copy();left[:,1]=1.
    c=Controller(2.8,-1.4,.7);u=c.update({'speed':5.},left,clear)
    assert u['steer']<0
    stopped=c.update({'speed':5.},route,{**clear,'emergency':True,'target_speed':0})
    assert stopped['throttle']==0 and stopped['brake']>0
    print(json.dumps({'pass':True,'ray_before_hit_only':True,'unknown_stops':True,'static_obstacle_stops':True,
                      'stale_sensor_stops':True,'ego_motion_compensated':True,'tracked_velocity_mps':tracked[0]['velocity'][0],
                      'FLU_to_CARLA_steer_sign':True}))


if __name__=='__main__':main()
