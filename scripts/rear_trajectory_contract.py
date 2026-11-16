"""Isolated differentiable kinematic prototype. NOT wired to training or controller."""
from dataclasses import dataclass
import torch
from torch import nn
import numpy as np
from audit_signed_curvature_expressivity import basis,matrices,D
@dataclass(frozen=True)
class Geometry:
 wheelbase_m:float=2.874995708465576
 rear_distance_m:float=1.4375089406967163
 episode_certified:bool=False
class RearTrajectoryDecoder(nn.Module):
 def __init__(self,geometry=Geometry()):
  super().__init__();self.geometry=geometry
  self.register_buffer('velocity_basis',torch.tensor(matrices(6,D)[0],dtype=torch.float64))
  self.register_buffer('curvature_basis',torch.tensor(basis(D),dtype=torch.float64))
 def forward(self,current_signed_speed,coefficients):
  # Only causal current speed and generated controls. No future/teacher data accepted.
  assert coefficients.shape[-2:]==(6,2)
  v=current_signed_speed[:,None]+torch.einsum('tk,bk->bt',self.velocity_basis.to(coefficients),coefficients[:,:,0]);k=torch.einsum('tk,bk->bt',self.curvature_basis.to(coefficients),coefficients[:,:,1]);omega=v*k
  zero=torch.zeros_like(v[:,:1]);yaw=torch.cat([zero,torch.cumsum((omega[:,1:]+omega[:,:-1])*.005,1)],1)
  direction=torch.stack([yaw.cos(),yaw.sin()],-1);vv=v[...,None]*direction
  rear=torch.cat([torch.zeros_like(vv[:,:1]),torch.cumsum((vv[:,1:]+vv[:,:-1])*.005,1)],1)
  ref=direction.clone();ref[:,:,0]-=1;actor=rear+self.geometry.rear_distance_m*ref
  return {'signed_speed':v[:,20::20],'body_yaw':yaw[:,20::20],'actor_xy':actor[:,20::20],'yaw_rate':omega[:,20::20],'curvature':k[:,20::20],'initial_signed_speed':current_signed_speed}

def supervision(pred,label):
 # label masks are fixed measured validity, never functions of predictions.
 # Dimensions are explicit SI; scales are provisional diagnostics, not tuned weights.
 valid=label['valid'].bool();yaw_valid=valid & label['yaw_observable'].bool()
 terms={'position_m2':(pred['actor_xy']-label['actor_xy']).square().sum(-1),'signed_speed_m2ps2':(pred['signed_speed']-label['signed_speed']).square(),'body_yaw_rad2':torch.atan2(torch.sin(pred['body_yaw']-label['body_yaw']),torch.cos(pred['body_yaw']-label['body_yaw'])).square()}
 out={}
 for name,err in terms.items():
  mask=yaw_valid if name=='body_yaw_rad2' else valid;out[name]=(torch.where(mask,err,torch.zeros_like(err)).sum()/mask.sum().clamp_min(1),int(mask.sum()))
 # Interval yaw rate uses t0 yaw=0. Sensor validity must include t0.
 p_yaw=torch.cat([torch.zeros_like(pred['body_yaw'][:,:1]),pred['body_yaw']],1);py=(p_yaw[:,1:]-p_yaw[:,:-1])/.2
 mask=label['curvature_identifiable'].bool() & label['interval_valid'].bool()
 err=(py-label['interval_yaw_rate']).square();out['interval_yaw_rate_rad2ps2']=(torch.where(mask,err,torch.zeros_like(err)).sum()/mask.sum().clamp_min(1),int(mask.sum()))
 # No curvature coefficient loss: underdetermined nullspace is not ground truth.
 return out
