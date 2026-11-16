"""Observed front motion as additive action-memory conditioning; no teacher inputs."""
import torch
from torch import nn
from navigation_scaled_flow import NavigationScaledFlow
class ObservedMotionFlow(NavigationScaledFlow):
 def __init__(self):
  super().__init__(50);self.front_condition=nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,128));nn.init.zeros_(self.front_condition[-1].weight);nn.init.zeros_(self.front_condition[-1].bias)
 def context(self,b):
  memory=super().context(b);return memory+self.front_condition(b['front_observation'].float()).to(memory.dtype)[:,None,:]
