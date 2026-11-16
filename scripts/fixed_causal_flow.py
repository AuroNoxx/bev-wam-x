"""Isolated fixed-time-causal variant. Not default, no label-length input."""
import torch
from navigation_scaled_flow import NavigationScaledFlow
class FixedCausalFlow(NavigationScaledFlow):
 def velocity(self,context,xt,t):
  q=self.trajectory(xt)+self.position+self.time(t)[:,None]
  fixed=torch.triu(torch.ones(xt.shape[1],xt.shape[1],device=xt.device,dtype=torch.bool),1)
  return self.output(self.transformer(q,context,tgt_mask=fixed))
