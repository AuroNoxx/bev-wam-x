from direct_state_velocity import DirectStateVelocity
import torch
from torch import nn
class FrozenLateralResidual(DirectStateVelocity):
 def __init__(self,rms):
  super().__init__();self.requires_grad_(False);self.lateral_residual=nn.Linear(4,25,bias=False);nn.init.zeros_(self.lateral_residual.weight);self.interaction_rms=float(rms)
 def update_teacher(self,*args,**kwargs):pass
 def lateral_features(self,s):
  x=self.direct_features(s);return torch.cat([x[:,:3],x[:,3:4]/self.interaction_rms],1)
 def forward(self,b,xt,t):
  q=super().forward(b,xt,t);y=self.lateral_residual(self.lateral_features(b['state']));return q+torch.stack([torch.zeros_like(y),y],-1)
