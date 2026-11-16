from deterministic_velocity_model import DeterministicVelocityFlow
import torch
from torch import nn
class DirectStateVelocity(DeterministicVelocityFlow):
 def __init__(self):
  super().__init__();self.direct_state=nn.Linear(4,50,bias=False);nn.init.zeros_(self.direct_state.weight)
 def direct_features(self,s):
  v=s[:,0]*3;nx=s[:,4]*5;ny=s[:,5]*50;return torch.stack([v,nx,ny,v*ny],-1)
 def forward(self,b,xt,t):return super().forward(b,xt,t)+self.direct_state(self.direct_features(b['state'])).reshape(-1,25,2)
 @torch.no_grad()
 def sample(self,b,k=4,steps=16):
  q=self.forward(b,torch.zeros(len(b['state']),25,2,device=b['state'].device),torch.zeros(len(b['state']),1,device=b['state'].device));return self.from_flow_space(b,q)[:,None].expand(-1,k,-1,-1).clone()
