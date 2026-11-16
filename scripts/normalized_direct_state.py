from direct_state_velocity import DirectStateVelocity
import torch
class NormalizedDirectState(DirectStateVelocity):
 def __init__(self,interaction_rms):super().__init__();self.interaction_rms=float(interaction_rms)
 def direct_features(self,s):
  x=super().direct_features(s);return torch.cat([x[:,:3],x[:,3:4]/self.interaction_rms],-1)
