from scaled_velocity_flow import ScaledVelocityFlow
class NavigationScaledFlow(ScaledVelocityFlow):
 def __init__(self,nav_y_gain=1.):super().__init__(.5);self.nav_y_gain=float(nav_y_gain)
 def encoded_state(self,state):
  x=state.clone();x[...,5]=x[...,5]*self.nav_y_gain;return x
 def forecast_batch(self,b):return super().forecast_batch({**b,'state':self.encoded_state(b['state'])})
