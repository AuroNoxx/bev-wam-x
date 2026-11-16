"""Checks real units, the initial-state boundary and differentiable constraints."""
import json
import torch
from stage1_objective import kinematic_residual,trajectory_metrics


def main():
    state=torch.zeros(2,20);state[:,0]=10/30
    t=torch.arange(1,26)*.2
    path=torch.zeros(2,25,5);path[...,0]=10*t/50;path[...,3]=1;path[...,4]=10/30
    assert kinematic_residual(path,state).abs().max()<1e-4
    accelerating=path.clone();speed=10+2*t
    accelerating[...,0]=(10*t+t*t)/50;accelerating[...,4]=speed/30
    assert kinematic_residual(accelerating,state).abs().max()<1e-4
    incorrect=path.clone();incorrect[...,4]=5/30
    assert kinematic_residual(incorrect,state).norm(dim=-1).mean()>4.8
    differentiable=incorrect.clone().requires_grad_();kinematic_residual(differentiable,state).square().mean().backward()
    assert torch.isfinite(differentiable.grad).all()
    assert differentiable.grad[...,:2].abs().sum()>0 and differentiable.grad[...,4].abs().sum()>0
    batch={'trajectory':path,'state':state};pred=path[:,None].repeat(1,8,1,1)
    assert trajectory_metrics(pred,batch)['meanADE8_m'].max()==0
    shifted=pred.clone();shifted[...,0]+=1/50
    assert torch.allclose(trajectory_metrics(shifted,batch)['meanADE8_m'],torch.ones(2),atol=1e-5)
    shifted=pred.clone();shifted[...,1]+=1/10
    assert torch.allclose(trajectory_metrics(shifted,batch)['meanADE8_m'],torch.ones(2),atol=1e-5)
    print(json.dumps({'pass':True,'straight_and_accelerating_consistent':True,'speed_mismatch_detected':True,'finite_position_speed_gradients':True,'equal_meter_XY_metric':True}))


if __name__=='__main__':main()
