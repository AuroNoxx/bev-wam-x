"""Isolated prototypes: causal vector-velocity action and label-only future-positive support."""
import torch
from torch import nn
from torch.nn import functional as F
from experimental_residual_representation import ResidualJEPA

def future_positive_targets(b,horizon=2):
 future=b['future'][:,:horizon].clone();future=torch.where((future>0)&(future<8),future,255)
 counts=torch.cat([F.adaptive_avg_pool2d((future==c).float().mean(2).flatten(0,1)[:,None],(8,20)) for c in range(1,8)],1).flatten(2).transpose(1,2).reshape(len(future),horizon,160,7);valid=counts.sum(-1)>0;fractions=counts/counts.sum(-1,keepdim=True).clamp_min(1e-8)
 return future,valid,fractions
class PrototypeJoint(ResidualJEPA):
 def __init__(self,width=128,layers=2,heads=4,action='xy5',world='current_support'):
  super().__init__(width,layers,heads);assert action in ['xy5','velocity2'];assert world in ['current_support','future_positive'];self.action_kind=action;self.world_kind=world
  if action=='velocity2':self.trajectory=nn.Linear(2,width);self.output=nn.Linear(width,2)
 def target_data(self,b):return future_positive_targets(b,self.horizon) if self.world_kind=='future_positive' else super().target_data(b)
 def initial_velocity(self,b):return torch.stack([b['state'][:,0].float()*30,torch.zeros_like(b['state'][:,0])],-1)[:,None]
 def to_flow_space(self,b,p):
  if self.action_kind=='xy5':return super().to_flow_space(b,p)
  velocity=p[...,4:5]*30*p[...,[3,2]];return (velocity-self.initial_velocity(b))/10
 def from_flow_space(self,b,q):
  if self.action_kind=='xy5':return super().from_flow_space(b,q)
  v=self.initial_velocity(b)+q.float()*10;prev=torch.cat([self.initial_velocity(b),v[:,:-1]],1);xy=((prev+v)*.1).cumsum(1);speed=v.norm(dim=-1,keepdim=True);direction=v/speed.clamp_min(1e-8);fallback=torch.zeros_like(direction);fallback[...,0]=1;direction=torch.where(speed>1e-8,direction,fallback);return torch.cat([xy/q.new_tensor([50,10]),direction[...,[1,0]],speed/30],-1)
 @torch.no_grad()
 def sample(self,b,k=8,steps=16):
  context=self.context(b).repeat_interleave(k,0);raw=torch.randn(len(context),25,5,device=context.device);x=raw[...,:2] if self.action_kind=='velocity2' else raw
  for i in range(steps):x=x+self.velocity(context,x,torch.full((len(x),1),i/steps,device=x.device))/steps
  p=self.from_flow_space({'state':b['state'].repeat_interleave(k,0)},x);p[...,2:4]=F.normalize(p[...,2:4],dim=-1,eps=1e-6);p[...,4]=p[...,4].clamp_min(0);return p.reshape(len(b['state']),k,25,5)

def prototype_loss(m,b,cfg):
 from stage1_objective import kinematic_residual
 p=b['trajectory'].float();target=m.to_flow_space(b,p);raw=torch.randn(len(target),25,5,device=target.device);noise=raw[...,:2] if m.action_kind=='velocity2' else raw;t=torch.rand(len(target),1,device=target.device);xt=(1-t[:,:,None])*noise+t[:,:,None]*target;pred=m(b,xt,t).float();fm=F.mse_loss(pred,target-noise);endpoint=m.from_flow_space(b,xt+(1-t[:,:,None])*pred);gate=(t[:,0]>=cfg['auxiliary_min_t']).float()/(1-cfg['auxiliary_min_t'])
 def gated(x):return (x.flatten(1).mean(1)*gate).mean()
 err=(endpoint[...,:2]-p[...,:2])*p.new_tensor([50,10]);position=gated(F.smooth_l1_loss(err/cfg['position_scale_m'],torch.zeros_like(err),reduction='none'));res=kinematic_residual(endpoint,b['state'])/cfg['velocity_scale_mps'];kin=gated(F.smooth_l1_loss(res,torch.zeros_like(res),reduction='none'));heading=gated((endpoint[...,2:4].square().sum(-1)-1).square());negative=gated(F.relu(-endpoint[...,4]*30/cfg['velocity_scale_mps']).square());terms={'fm':fm,'position':position,'kinematics':kin,'unit_heading':heading,'negative_speed':negative};return sum(cfg['weights'][k]*v for k,v in terms.items()),terms
