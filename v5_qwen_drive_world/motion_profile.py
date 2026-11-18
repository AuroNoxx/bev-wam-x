"""Continuous velocity-profile readout of the sampled longitudinal Flow controls."""
import torch
from torch import nn
from torch.nn import functional as F
from .geometry_repair import GeometryRepair
from .kinematics import integrate


def profile_features(features):
    bev=features['bev'].float()
    state=features['state'].float()
    # Use the ego-aligned change of native camera features close to the current lane.
    # Keep the spatial distribution and the signed channel changes at the lead vehicle.
    near=bev[:,96:,12:20,9:25]
    magnitude=near.square().mean(1,keepdim=True).sqrt()
    motion=F.adaptive_avg_pool2d(magnitude,(4,8)).flatten(1)
    signed=near[:,:,2:6,:6].mean((2,3))
    # Keep stopped and creeping observations distinct while retaining high-speed inputs.
    units=state.new_tensor([120,120,8,8,80,80,20,20,20,20,20,20])
    physical=torch.asinh(state*units)
    raw=features['ctrl'].float()
    v0=features['v0'].flatten().float()
    base_v=integrate(raw,v0)[:,4::5,3]
    controls=raw.reshape(-1,10,5,2).mean(2).flatten(1)/4
    return torch.cat([motion,signed,physical,controls,torch.asinh(base_v-v0[:,None])],1),base_v


class MotionProfile(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(122,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.1),
                               nn.Linear(64,40),nn.SiLU(),nn.Linear(40,30))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self,observation,base_v,raw,v0):
        amp,confidence,prior=self.mlp(observation).float().chunk(3,1)
        with torch.autocast(device_type=observation.device.type,enabled=False):
            v0=v0.flatten().float()
            activity=1.5*torch.asinh(v0[:,None])-1+confidence
            wanted=F.softplus(v0[:,None]+prior.sigmoid()*(base_v-v0[:,None])+amp,
                              beta=1.5)*activity.sigmoid()
            # Reachable half-second knots imply a bounded acceleration sequence.
            # The same longitudinal readout and integrator apply at every speed.
            previous=v0
            acceleration=[]
            for target in wanted.unbind(1):
                change=(target-previous).clamp(-4,2)
                acceleration.append(change*2)
                previous=previous+change
            ax=torch.stack(acceleration,1).repeat_interleave(5,dim=1)
            ax=F.avg_pool1d(F.pad(ax[:,None],(1,1),mode='replicate'),3,stride=1)[:,0]
            control=torch.stack([ax,raw[...,1].float()],dim=-1)
            return control,integrate(control,v0),activity


class ProfileActor(GeometryRepair):
    def __init__(self,world):
        super().__init__(world)
        self.profile=MotionProfile()

    def forward(self,features,candidate=None,frozen_logits=None):
        observation,base_v=profile_features(features)
        ctrl,traj,_=self.profile(observation,base_v,features['ctrl'],features['v0'])
        encoded=self.appearance(features['bev'].float())
        xy=traj[...,:2] if candidate is None else candidate
        return ctrl,traj,self.occupancy(encoded,features['latent'],xy,features['v0'],frozen_logits)
