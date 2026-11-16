"""OCC-only trajectory flow and trajectory-conditioned occupancy forecaster."""
import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self,c):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(c,c,3,padding=1,bias=False),nn.GroupNorm(8,c),nn.SiLU(),nn.Conv2d(c,c,3,padding=1,bias=False),nn.GroupNorm(8,c))
    def forward(self,x):return F.silu(x+self.net(x))

class OccEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(9,2);self.channels_last=False
        self.net=nn.Sequential(nn.Conv2d(162,48,5,2,2),nn.GroupNorm(8,48),nn.SiLU(),Residual(48),
                               nn.Conv2d(48,96,3,2,1),nn.GroupNorm(8,96),nn.SiLU(),Residual(96),
                               nn.Conv2d(96,192,3,1,1),nn.GroupNorm(8,192),nn.SiLU(),Residual(192))
        y,x=torch.meshgrid(torch.linspace(-1,1,64),torch.linspace(-1,1,160),indexing='ij')
        self.register_buffer('coordinates',torch.stack([x,y])[None])
    def forward(self,history):
        labels=torch.where(history==255,8,history).long()
        x=self.embedding(labels).permute(0,1,2,5,3,4).flatten(1,3)
        x=torch.cat([x,self.coordinates.expand(len(x),-1,-1,-1)],1)
        if self.channels_last:x=x.contiguous(memory_format=torch.channels_last)
        return self.net(x)

class FlowPlanner(nn.Module):
    def __init__(self,parameterization='absolute'):
        super().__init__();self.encoder=OccEncoder();self.state=nn.Linear(20,192)
        if parameterization not in ('absolute','cv_residual'):raise ValueError(parameterization)
        self.parameterization=parameterization
        self.time=nn.Sequential(nn.Linear(1,192),nn.SiLU(),nn.Linear(192,192))
        self.trajectory=nn.Linear(5,192);self.position=nn.Parameter(torch.randn(1,25,192)*.02)
        layer=nn.TransformerDecoderLayer(192,6,768,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.transformer=nn.TransformerDecoder(layer,4);self.output=nn.Linear(192,5)
    def context(self,batch):
        # Retain spatial tokens so a left obstacle differs from a right obstacle.
        return F.adaptive_avg_pool2d(self.encoder(batch['history']),(8,20)).flatten(2).transpose(1,2)+self.state(batch['state'])[:,None]
    def velocity(self,context,xt,t):
        return self.output(self.transformer(self.trajectory(xt)+self.position+self.time(t)[:,None],context))
    def forward(self,batch,xt,t):return self.velocity(self.context(batch),xt,t)
    def reference(self,batch,like):
        # Only currently observed speed; no labels or future information.
        base=torch.zeros_like(like)
        speed=batch['state'][:,0].to(like)
        times=torch.arange(1,26,device=like.device,dtype=like.dtype)*.2
        base[...,0]=speed[:,None]*30*times[None]/50
        base[...,3]=1
        base[...,4]=speed[:,None]
        return base
    def to_flow_space(self,batch,trajectory):
        if self.parameterization=='absolute':return trajectory
        return trajectory-self.reference(batch,trajectory)
    def from_flow_space(self,batch,trajectory):
        if self.parameterization=='absolute':return trajectory
        return trajectory+self.reference(batch,trajectory)
    @torch.no_grad()
    def sample(self,batch,k=8,steps=16):
        b=len(batch['state']);context=self.context(batch).repeat_interleave(k,0)
        x=torch.randn(b*k,25,5,device=context.device)
        for i in range(steps):
            t=torch.full((len(x),1),i/steps,device=x.device)
            x=x+self.velocity(context,x,t)/steps
        if self.parameterization!='absolute':
            x=self.from_flow_space({'state':batch['state'].repeat_interleave(k,0)},x)
        angles=F.normalize(x[...,2:4].float(),dim=-1,eps=1e-6)
        x=x.float();x[...,2:4]=angles;x[...,4]=x[...,4].clamp_min(0)
        return x.reshape(b,k,25,5)

class TrajectoryWorld(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=OccEncoder()
        self.condition=nn.Sequential(nn.Linear(25*5+20,384),nn.SiLU(),nn.Linear(384,192))
        self.time=nn.Parameter(torch.randn(5,192)*.02)
        self.decoder=nn.Sequential(Residual(192),nn.ConvTranspose2d(192,96,4,2,1),nn.GroupNorm(8,96),nn.SiLU(),Residual(96),
                                   nn.ConvTranspose2d(96,48,4,2,1),nn.GroupNorm(8,48),nn.SiLU(),Residual(48),nn.Conv2d(48,20*8,1))
    def forward(self,batch):
        latent=self.encoder(batch['history'])
        c=self.condition(torch.cat([batch['trajectory'].flatten(1),batch['state']],1))
        x=latent[:,None]+c[:,None,:,None,None]+self.time[None,:,:,None,None]
        return self.decoder(x.flatten(0,1)).reshape(len(latent),5,20,8,64,160)

def loss(mode,model,batch):
    if mode=='flow':
        target=batch['trajectory'];noise=torch.randn_like(target);t=torch.rand(len(target),1,device=target.device)
        return F.mse_loss(model(batch,(1-t[:,:,None])*noise+t[:,:,None]*target,t).float(),target-noise)
    logits=model(batch).flatten(0,2).float();target=batch['future'].flatten(0,2).long()
    weights=logits.new_tensor([.2,.3,.7,1.,4.,6.,1.,2.])
    return F.cross_entropy(logits,target,weight=weights,ignore_index=255)

MODELS={'flow':FlowPlanner,'world':TrajectoryWorld}
