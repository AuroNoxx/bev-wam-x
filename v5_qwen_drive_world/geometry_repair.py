"""Trainable geometry-guided readouts on the frozen step-750 actor and World."""
import torch
from torch import nn
from torch.nn import functional as F
from .geometry_repair_features import CFG, actor_features
from .kinematics import integrate


class GeometryRepair(nn.Module):
    def __init__(self,world):
        super().__init__()
        self.world=world.eval().requires_grad_(False)
        channels=CFG['projection_channels']*3
        self.appearance=nn.Sequential(nn.Conv2d(channels,32,1),nn.GroupNorm(4,32),nn.SiLU(),
                                      nn.Conv2d(32,32,3,padding=1),nn.GroupNorm(4,32),nn.SiLU())
        self.controls=nn.Sequential(nn.Linear(32*12+12+10,96),nn.LayerNorm(96),nn.SiLU(),nn.Linear(96,12))
        self.map_logits=nn.Sequential(nn.Conv2d(32+5,32,3,padding=1),nn.GroupNorm(4,32),nn.SiLU(),
                                      nn.Conv2d(32,16,3,padding=2,dilation=2),nn.SiLU(),nn.Conv2d(16,1,1))
        nn.init.zeros_(self.controls[-1].weight)
        nn.init.zeros_(self.controls[-1].bias)
        nn.init.zeros_(self.map_logits[-1].weight)
        nn.init.zeros_(self.map_logits[-1].bias)
        y,x=torch.meshgrid(-19.2+(torch.arange(32)+.5)*1.2,-16+(torch.arange(64)+.5)*2,indexing='ij')
        self.register_buffer('x',x);self.register_buffer('y',y)

    def plan(self,encoded,state,controls,v0):
        # Pool the projected front corridor in spatial order, preserving local changes over time.
        front=F.adaptive_avg_pool2d(encoded[:,:,10:22,8:40],(3,4)).flatten(1)
        control_context=controls.reshape(-1,5,10,2).mean(2).flatten(1)/4
        knots=self.controls(torch.cat([front,state,control_context],1)).float()
        with torch.autocast(device_type=state.device.type,enabled=False):
            bias,gain=knots.chunk(2,dim=1)
            bias=F.interpolate(bias[:,None],size=50,mode='linear',align_corners=True)[:,0]
            gain=F.interpolate(gain[:,None],size=50,mode='linear',align_corners=True)[:,0]
            longitudinal=(controls[...,0].float()*(1+gain.tanh())+4*bias.tanh()).clamp(-8,4)
            corrected=torch.stack([longitudinal,controls[...,1].float()],dim=-1)
            return corrected,integrate(corrected,v0.flatten())

    def frozen_world_logits(self,latent,xy,v0):
        with torch.no_grad():
            current=self.world.decoder(latent).float()
            # Each horizon receives only its candidate prefix, padded by its last pose.
            prefixes=[torch.cat([xy[:,:end],xy[:,end-1:end].expand(-1,50-end,-1)],dim=1) for end in (10,20,30)]
            horizon=latent.new_tensor([1.,2.,3.])[None,:,None].expand(len(xy),-1,-1)/3
            action=torch.cat([torch.stack(prefixes,1).flatten(2)/160,
                              v0.reshape(-1,1,1).expand(-1,3,-1)/35,horizon],2)
            shift,scale=self.world.action_condition(action.flatten(0,1)).chunk(2,-1)
            base=latent[:,None].expand(-1,3,-1,-1,-1).flatten(0,1)
            future=base+self.world.delta(base*(1+scale[:,:,None,None])+shift[:,:,None,None])
            return torch.cat([current,self.world.decoder(future).float().reshape(-1,3,128,128)],1)

    def occupancy(self,encoded,latent,xy,v0,original=None):
        if original is None:
            original=self.frozen_world_logits(latent,xy,v0)
        horizons=xy.new_tensor([0.,1.,2.,3.])
        poses=torch.cat([torch.zeros_like(xy[:,:1]),xy[:,[9,19,29]]],1)
        b=xy.shape[0]
        x=self.x.expand(b,4,-1,-1)
        y=self.y.expand_as(x)
        offsets=torch.stack([x/112,y/19.2,(x-poses[:,:,0,None,None])/60,
                             (y-poses[:,:,1,None,None])/12,horizons[None,:,None,None].expand_as(x)/3],2)
        context=encoded[:,None].expand(-1,4,-1,-1,-1)
        logits=self.map_logits(torch.cat([context,offsets],2).flatten(0,1))
        return original+F.interpolate(logits.float(),size=(128,128),mode='bilinear',align_corners=False).reshape_as(original)

    def forward(self,features,candidate=None,frozen_logits=None):
        encoded=self.appearance(features['bev'].float())
        ctrl,traj=self.plan(encoded,features['state'],features['ctrl'],features['v0'])
        logits=self.occupancy(encoded,features['latent'],traj[...,:2] if candidate is None else candidate,
                              features['v0'],frozen_logits)
        return ctrl,traj,logits

    @torch.no_grad()
    def predict_actor(self,base,record,seed):
        """Raw actor entry point. The proposed trajectory conditions the future occupancy."""
        from .runtime import math_attention
        with math_attention(),torch.autocast('cuda',dtype=torch.bfloat16):
            features={k:v[None].cuda() for k,v in actor_features(base,record,seed).items()}
            return self(features)


class WorldRepairActor(GeometryRepair):
    """Pair the corrected occupancy map with the source checkpoint's exact Flow sample."""
    def plan(self,encoded,state,controls,v0):
        return controls,integrate(controls,v0.flatten())
