"""Differentiable pure-noise actor rollout regularizer; labels are never read."""
import torch
from torch.nn import functional as F
from stage1_objective import kinematic_residual
ACTOR_KEYS=('history','state','history_valid','history_age_seconds','geometry_valid')
def rollout(m,b,steps=4,k=1):
 actor={key:b[key] for key in ACTOR_KEYS};context=m.context(actor).repeat_interleave(k,0);x=torch.randn(len(context),25,5,device=context.device)
 for i in range(steps):
  t=torch.full((len(x),1),i/steps,device=x.device);x=x+m.velocity(context,x,t)/steps
 x=x.float();heading=F.normalize(x[...,2:4],dim=-1,eps=1e-6);speed=x[...,4:5].clamp_min(0)
 return torch.cat([x[...,:2],heading,speed],-1)
def dynamics_loss(p,state):
 kin=kinematic_residual(p,state);v=p[...,4]*30;v0=state[:,0:1]*30;acc=torch.diff(torch.cat([v0,v],1),dim=1)/.2;jerk=torch.diff(acc,dim=1)/.2
 terms={'rollout_kin':F.smooth_l1_loss(kin/2,torch.zeros_like(kin)),'rollout_acc':F.smooth_l1_loss(F.relu(acc.abs()-4)/4,torch.zeros_like(acc)),'rollout_jerk':F.smooth_l1_loss(F.relu(jerk.abs()-10)/10,torch.zeros_like(jerk))}
 return sum(terms.values()),terms
