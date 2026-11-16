from train_mixed_regression_flow512 import CompareModel
import torch
class ScaledVelocityFlow(CompareModel):
 def __init__(self,lateral_scale_mps=.5):super().__init__('fm');self.lateral_scale_mps=float(lateral_scale_mps)
 def to_flow_space(self,b,p):
  v=p[...,4:5]*30*p[...,[3,2]];return (v-self.initial_velocity(b))/p.new_tensor([10,self.lateral_scale_mps])
 def from_flow_space(self,b,q):
  v=self.initial_velocity(b)+q.float()*q.new_tensor([10,self.lateral_scale_mps]);prev=torch.cat([self.initial_velocity(b),v[:,:-1]],1);xy=((prev+v)*.1).cumsum(1);speed=v.norm(dim=-1,keepdim=True);direction=v/speed.clamp_min(1e-8);fallback=torch.zeros_like(direction);fallback[...,0]=1;direction=torch.where(speed>1e-8,direction,fallback);return torch.cat([xy/q.new_tensor([50,10]),direction[...,[1,0]],speed/30],-1)
