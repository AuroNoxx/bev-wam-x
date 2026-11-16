"""Experimental velocity-vector FM; no production/baseline changes."""
import torch
from torch.nn import functional as F
from masked_history_joint import MaskedHistoryJoint

class VelocityIntegratedJoint(MaskedHistoryJoint):
    def to_flow_space(self,b,trajectory):
        direction=F.normalize(trajectory[...,[3,2]],dim=-1,eps=1e-6)
        return direction*trajectory[...,4:5] # normalized by30m/s
    def from_flow_space(self,b,v):
        v=v.float();speed=v.norm(dim=-1,keepdim=True)
        initial=torch.stack([b['state'][:,0].float(),torch.zeros_like(b['state'][:,0])],-1)[:,None]
        previous=torch.cat([initial,v[:,:-1]],1)
        xy=((previous+v)*.5*30*.2).cumsum(1)/v.new_tensor([50.,10.])
        heading=F.normalize(v,dim=-1,eps=1e-6)
        # At exact rest choose forward heading, avoiding arbitrary unit vector.
        heading=torch.where((speed>1e-6).expand_as(heading),heading,heading.new_tensor([1.,0.]))
        return torch.cat([xy,heading[...,[1,0]],speed],-1)
    def velocity(self,context,xt,t):
        return super().velocity(context,F.pad(xt,(0,3)),t)[...,:2]
    @torch.no_grad()
    def sample(self,batch,k=4,steps=16):
        context=self.context(batch).repeat_interleave(k,0)
        x=torch.randn(len(context),25,2,device=context.device)
        for i in range(steps):
            t=torch.full((len(x),1),i/steps,device=x.device)
            x=x+self.velocity(context,x,t)/steps
        out=self.from_flow_space({'state':batch['state'].repeat_interleave(k,0)},x)
        return out.reshape(len(batch['state']),k,25,5)
