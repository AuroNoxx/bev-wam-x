"""Experimental train-calibrated XY residual representation; never production default."""
import torch
from masked_history_joint import MaskedHistoryJoint
from train_jepa_nojepa_paired import NoJEPA
class ResidualMixin:
 def set_scales(self,scales):
  z=torch.tensor(scales,dtype=torch.float32)
  assert z.shape==(25,2) and torch.isfinite(z).all() and (z>0).all()
  self.register_buffer('residual_xy_scale',z,persistent=False)
 def prior(self,b):
  t=torch.arange(1,26,device=b['state'].device,dtype=torch.float32)*.2
  v=b['state'][:,0].float()*30
  return torch.stack([v[:,None]*t[None],torch.zeros_like(v[:,None]*t[None])],-1)
 def to_flow_space(self,b,target):
  q=target.float().clone();q[...,:2]=(target[...,:2]*target.new_tensor([50.,10.])-self.prior(b))/self.residual_xy_scale;return q
 def from_flow_space(self,b,q):
  x=q.float().clone();x[...,:2]=(q[...,:2]*self.residual_xy_scale+self.prior(b))/q.new_tensor([50.,10.]);return x
 @torch.no_grad()
 def sample(self,batch,k=8,steps=16):
  context=self.context(batch).repeat_interleave(k,0);x=torch.randn(len(context),25,5,device=context.device)
  for i in range(steps):x=x+self.velocity(context,x,torch.full((len(x),1),i/steps,device=x.device))/steps
  b={'state':batch['state'].repeat_interleave(k,0)};x=self.from_flow_space(b,x);x[...,2:4]=torch.nn.functional.normalize(x[...,2:4],dim=-1,eps=1e-6);x[...,4]=x[...,4].clamp_min(0)
  return x.reshape(len(batch['state']),k,25,5)
class ResidualJEPA(ResidualMixin,MaskedHistoryJoint):pass
class ResidualNoJEPA(ResidualMixin,NoJEPA):pass
