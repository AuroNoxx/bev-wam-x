"""Geometry/causality/control regression tests; no training or CARLA required."""
import numpy as np
from native_v2_carla_contract import (native_points_grid,align_history_grid,history_payload,
                                     trajectory_to_world,TrajectoryController,state20,vehicle_geometry,merge_semantic_scans)


def main():
    eye=np.eye(4)
    grid=native_points_grid(np.array([[4.2,-2.2,.4],[4.2,2.2,.4]]),np.array([14,12]))
    assert grid[13,69,74]==4  # positive left FLU, larger storage Y
    assert grid[13,58,74]==5
    assert np.array_equal(align_history_grid(grid,eye,eye),grid)
    moved=eye.copy();moved[0,3]=.4
    scan=np.array([[1.,2.,3.,.5,0.,0.]],np.float32);scan[:,4]=np.array([271],np.uint32).view(np.float32);scan[:,5]=np.array([14],np.uint32).view(np.float32)
    merged=merge_semantic_scans(scan,scan,eye,moved)
    np.testing.assert_allclose(merged[0,:3],[.6,2.,3.],atol=1e-6)
    assert merged[:,4].view(np.uint32).tolist()==[271,271] and merged[:,5].view(np.uint32).tolist()==[14,14]
    aligned=align_history_grid(grid,eye,moved)
    assert aligned[13,69,73]==4 and (aligned[:,:,-1]==255).all()
    # Cross-check directly against the authoritative training alignment, including yaw.
    from b2d_native_occ import align_native
    yaw=.37;pose=np.array([[np.cos(yaw),-np.sin(yaw),0,.8],
                           [np.sin(yaw),np.cos(yaw),0,-.4],[0,0,1,0],[0,0,0,1]])
    raw=grid[:,::-1,:]
    expected,_=align_native(raw,eye,pose)
    assert np.array_equal(align_history_grid(grid,eye,pose),expected)
    local_wheels=np.array([[1.5,-.8,.3],[1.5,.8,.3],[-1.36,-.8,.3],[-1.36,.8,.3]])
    for transform in [eye,pose]:
        world_wheels=(np.column_stack([local_wheels,np.ones(4)])@transform.T)[:,:3]*100
        wb,geometry=vehicle_geometry(world_wheels,[70,70,0,0],transform)
        assert abs(wb-2.86)<1e-8
    frames=[{'timestamp':i*.1,'frame':i,'pose':eye,'grid':grid} for i in range(7)]
    history,meta=history_payload(frames)
    assert meta['frames']==[0,2,4,6] and history.shape==(4,20,128,128)
    bad=[dict(x) for x in frames];bad[4]['timestamp']+=.1
    try:history_payload(bad)
    except ValueError:pass
    else:raise AssertionError('Bad cadence accepted')
    state=state20(6,[2,3,9.81],.4,eye,[5,-2,0],[30,3,0],[1,4])
    np.testing.assert_allclose(state[:8],[.2,.2,-.3,-.4,.1,.04,.6,-.06],atol=1e-7)
    assert np.array_equal(state[8:14],[1,0,0,0,0,0])
    assert np.array_equal(state[14:],[0,0,0,1,0,0])
    xy=np.stack([np.arange(1,26)*.8,np.arange(1,26)*.05],-1)
    world=trajectory_to_world(xy,eye);assert world[0,1]<0
    c=TrajectoryController(2.8,.6,[(0,1),(100,.5)])
    ctl,detail=c.control(world,np.tile([4,0],(25,1)),eye,0,0)
    assert ctl['steer']<0 and ctl['throttle']>0 and ctl['brake']==0
    # High jerk and lateral displacement are executed; no veto or brake imposed.
    velocity=np.tile([4.,0.],(25,1));velocity[1::2,1]=100
    ctl,_=c.control(world,velocity,eye,0,0)
    assert ctl['throttle']>0 and ctl['brake']==0
    ctl,detail=c.control(world,np.tile([.01,0],(25,1)),eye,0,0)
    assert ctl['throttle']>0 and ctl['brake']==0  # no artificial low-speed veto
    ctl,_=c.control(world,np.tile([-2.,0],(25,1)),eye,2,0)
    assert ctl['brake']>0  # model-signed speed error, explicitly not takeover
    print('PASS native geometry, training alignment parity, history cadence, state, steering, no-quality-veto controller')


if __name__=='__main__':main()
