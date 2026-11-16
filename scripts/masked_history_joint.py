"""Isolated research actor: explicit causal validity/time; no baseline edits."""
import torch
from torch import nn
from torch.nn import functional as F
from probe_joint_occ_capacity import CapacityModel

class MaskedHistoryJoint(CapacityModel):
    def __init__(self,width=128,layers=2,heads=4):
        super().__init__(width,layers,heads)
        self.history_time=nn.Sequential(nn.Linear(2,width),nn.SiLU(),nn.Linear(width,width))

    def forecast(self,history,state):
        raise ValueError("Use forecast_batch with explicit validity, geometry and age")

    def canonical_history(self,b):
        h=b['history']; valid=b['history_valid'].bool()
        if not valid[:,-1].all():raise ValueError('Current frame must be valid')
        geometry=b['geometry_valid'].bool()
        return torch.where(geometry & valid[:,:,None,None,None] & (h>0) & (h<8),h,255)

    def forecast_batch(self,b):
        history=self.canonical_history(b); valid=b['history_valid'].bool(); age=b['history_age_seconds'].float()
        if not torch.isfinite(age).all() or (age<0).any() or (age[:,1:]>age[:,:-1]).any():raise ValueError('Invalid causal age order')
        frames=self.encoder(history); bs,t,n,d=frames.shape; h=frames.new_zeros(bs,n,d)
        last_age=age[:,0]; seen=torch.zeros(bs,device=age.device,dtype=torch.bool)
        for i in range(t):
            delta=torch.where(seen,last_age-age[:,i],torch.zeros_like(last_age)).clamp_min(0)
            timing=self.history_time(torch.stack([age[:,i]/5.,delta/5.],-1).to(frames.dtype))[:,None]
            x=frames[:,i]+timing
            reset,update=self.temporal_gate(torch.cat([x,h],-1)).sigmoid().chunk(2,-1)
            candidate=self.temporal_candidate(torch.cat([x,reset*h],-1)).tanh()
            proposal=(1-update)*h+update*candidate
            h=torch.where(valid[:,i,None,None],proposal,h)
            last_age=torch.where(valid[:,i],age[:,i],last_age);seen=seen|valid[:,i]
        h=h+self.spatial+self.state(b['state'])[:,None]
        return h,self.predictor(h[:,None]+self.future_time)

    def context(self,b):
        h,pred=self.forecast_batch(b)
        return torch.cat([h,pred.flatten(1,2)],1)

    def target_data(self,b):
        # Canonical current geometry is label-side support; never future -> actor.
        return super().target_data({**b,'history':self.canonical_history(b)})

    def auxiliary_loss(self,batch):
        _,pred=self.forecast_batch(batch)
        future,valid,fractions=self.target_data(batch)
        with torch.no_grad():target=self.teacher(future).float()
        # EMA alone is not a non-collapse proof; same-location cross-example
        # variance regularizes online and predicted features, avoiding spatial
        # positional embeddings falsely satisfying the variance term.
        online=self.encoder(future).float()
        mask=valid.float();den=mask.sum().clamp_min(1)
        latent=((pred.float()-target).square().mean(-1)*mask).sum()/den
        variance=0.
        for z in [online,pred.float()]:
            w=mask[...,None];n=w.sum(0);mean=(z*w).sum(0)/n.clamp_min(1)
            var=((z-mean).square()*w).sum(0)/(n-1).clamp_min(1)
            eligible=(n>1).expand_as(var)
            variance=variance+(F.relu(.25-(var+1e-4).sqrt())*eligible).sum()/eligible.sum().clamp_min(1)
        logits=self.occ_head(pred).float()
        semantic=(-(fractions*F.log_softmax(logits,-1)).sum(-1)*mask).sum()/den
        return latent,variance,semantic
