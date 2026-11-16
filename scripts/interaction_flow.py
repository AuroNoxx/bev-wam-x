from navigation_scaled_flow import NavigationScaledFlow
class InteractionFlow(NavigationScaledFlow):
 def __init__(self,interaction=False):super().__init__(50);self.interaction=bool(interaction)
 def encoded_state(self,state):
  x=super().encoded_state(state)
  if self.interaction:x[...,19]=x[...,0]*3*x[...,5]
  return x
