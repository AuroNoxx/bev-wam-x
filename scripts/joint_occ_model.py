"""Small OCC-only research model. No connection to private XPeng code.

Actor APIs receive only history/state. EMA targets and future visibility live
exclusively in auxiliary_loss. Trajectory output keeps the original v3 units.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class FrameEncoder(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.embedding=nn.Embedding(9,2)
        self.conv=nn.Sequential(nn.Conv2d(40,width,3,2,1),nn.GroupNorm(8,width),nn.SiLU(),
                                nn.Conv2d(width,width,3,2,1),nn.GroupNorm(8,width),nn.SiLU())
        self.norm=nn.LayerNorm(width)

    def forward(self, occ):
        # B,T,Z,Y,X -> B,T,160,D; retain all 20 height layers as channels.
        b,t=occ.shape[:2]
        x=self.embedding(torch.where(occ==255,8,occ).long())
        x=x.permute(0,1,2,5,3,4).reshape(b*t,40,64,160)
        x=F.avg_pool2d(x,2)  # 1.6 m cells before convolution; not new observations.
        x=self.conv(x).flatten(2).transpose(1,2)
        return self.norm(x).reshape(b,t,160,-1)


class JointOccFlow(nn.Module):
    def __init__(self, variant='auxiliary', width=64, horizon=2):
        super().__init__()
        assert variant in ['no_occ','temporal_flow','auxiliary','fusion']
        self.variant=variant;self.width=width;self.horizon=horizon
        self.encoder=FrameEncoder(width)
        # Explicit gated recurrence: fused MIOpen GRU failed on this Windows
        # runtime. Linear/pointwise ops keep the same causal dependency contract.
        self.temporal_gate=nn.Linear(width*2,width*2)
        self.temporal_candidate=nn.Linear(width*2,width)
        self.state=nn.Sequential(nn.Linear(20,width),nn.SiLU(),nn.Linear(width,width))
        self.spatial=nn.Parameter(torch.randn(1,160,width)*.02)
        self.future_time=nn.Parameter(torch.randn(1,horizon,1,width)*.02)
        self.predictor=nn.Sequential(nn.Linear(width,width*2),nn.SiLU(),nn.Linear(width*2,width),nn.LayerNorm(width))
        self.occ_head=nn.Linear(width,7)  # conditional occupied semantic fractions; NOT free-space detector.
        self.teacher=copy.deepcopy(self.encoder).requires_grad_(False)
        self.time=nn.Sequential(nn.Linear(1,width),nn.SiLU(),nn.Linear(width,width))
        self.trajectory=nn.Linear(5,width);self.position=nn.Parameter(torch.randn(1,25,width)*.02)
        layer=nn.TransformerDecoderLayer(width,4,width*4,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.transformer=nn.TransformerDecoder(layer,2);self.output=nn.Linear(width,5)

    def history_features(self, history):
        frames=self.encoder(history);b,t,n,d=frames.shape
        h=frames.new_zeros(b,n,d);outputs=[]
        for i in range(t):
            x=frames[:,i];reset,update=self.temporal_gate(torch.cat([x,h],-1)).sigmoid().chunk(2,-1)
            candidate=self.temporal_candidate(torch.cat([x,reset*h],-1)).tanh()
            h=(1-update)*h+update*candidate;outputs.append(h)
        return torch.stack(outputs,1)

    def forecast(self, history, state):
        h=self.history_features(history)[:,-1]+self.spatial+self.state(state)[:,None]
        return h,self.predictor(h[:,None]+self.future_time)

    def context(self, batch):
        if self.variant=='no_occ':return self.state(batch['state'])[:,None]+self.spatial
        h,pred=self.forecast(batch['history'],batch['state'])
        return torch.cat([h,pred.flatten(1,2)],1) if self.variant=='fusion' else h

    def velocity(self, context, xt, t):
        q=self.trajectory(xt)+self.position+self.time(t)[:,None]
        return self.output(self.transformer(q,context))

    def forward(self,batch,xt,t):
        return self.velocity(self.context(batch),xt,t)

    @torch.no_grad()
    def sample(self,batch,k=8,steps=16):
        context=self.context(batch).repeat_interleave(k,0)
        x=torch.randn(len(context),25,5,device=context.device)
        for i in range(steps):
            t=torch.full((len(x),1),i/steps,device=x.device)
            x=x+self.velocity(context,x,t)/steps
        x=x.float();x[...,2:4]=F.normalize(x[...,2:4],dim=-1,eps=1e-6);x[...,4]=x[...,4].clamp_min(0)
        return x.reshape(len(batch['state']),k,25,5)

    @torch.no_grad()
    def update_teacher(self, momentum=.99):
        for teacher,online in zip(self.teacher.parameters(),self.encoder.parameters()):
            teacher.lerp_(online,1-momentum)

    def target_data(self,batch):
        future=batch['future'][:,:self.horizon].clone()
        # Current-source support is inference-available. No future mask is sent
        # to actor. Ignore future outside current occupied observation support;
        # source0 and 255 are NOT supervised as free or occupied negatives.
        current=batch['history'][:,-1:]
        support=(current!=255)&(current!=0)
        future=torch.where(support,future,255)
        known=(future>0)&(future<8)
        counts=[]
        for c in range(1,8):
            counts.append(F.adaptive_avg_pool2d((future==c).float().mean(2).flatten(0,1)[:,None],(8,20)))
        counts=torch.cat(counts,1).flatten(2).transpose(1,2).reshape(len(future),self.horizon,160,7)
        valid=counts.sum(-1)>0
        fractions=counts/counts.sum(-1,keepdim=True).clamp_min(1e-8)
        return future,valid,fractions

    def auxiliary_loss(self,batch):
        _,pred=self.forecast(batch['history'],batch['state'])
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


@torch.no_grad()
def representation_report(model,batch):
    _,pred=model.forecast(batch['history'],batch['state'])
    future,valid,fractions=model.target_data(batch);target=model.teacher(future).float()
    result={}
    for name,z in [('predicted',pred.float()),('target',target)]:
        # Across examples at fixed spatial/time location; constants and pure
        # position/time patterns cannot create this variance.
        std=z.std(0,unbiased=False).mean()
        pooled=z.mean((1,2));centered=pooled-pooled.mean(0)
        s=torch.linalg.svdvals(centered);p=s.square()/s.square().sum().clamp_min(1e-12)
        result[name+'_cross_example_std']=float(std)
        result[name+'_effective_rank']=float(torch.exp(-(p*p.clamp_min(1e-12).log()).sum())) if s.square().sum()>1e-12 else 0.
    den=valid.sum().clamp_min(1)
    result['latent_mse']=float(((pred.float()-target).square().mean(-1)*valid).sum()/den)
    result['shuffled_target_mse']=float(((pred.float()-target.roll(1,0)).square().mean(-1)*valid).sum()/den)
    persistence=model.teacher(batch['history'][:,-1:]).float().expand_as(target)
    result['persistence_latent_mse']=float(((persistence-target).square().mean(-1)*valid).sum()/den)
    mask_only=torch.where((batch['history']>0)&(batch['history']<8),1,batch['history'])
    _,mp=model.forecast(mask_only,batch['state'])
    result['mask_only_predicted_latent_change']=float((mp.float()-pred.float()).square().mean().sqrt())
    prob=model.occ_head(pred).float().softmax(-1)
    result['occupied_semantic_cross_entropy']=float((-(fractions*prob.clamp_min(1e-8).log()).sum(-1)*valid).sum()/den)
    result['vehicle_fraction_mae_on_occupied_support']=float(((prob[...,3]-fractions[...,3]).abs()*valid).sum()/den)
    result['supervised_token_fraction']=float(valid.float().mean())
    result['noncollapse_probe_pass']=result['predicted_cross_example_std']>.01 and result['target_cross_example_std']>.01 and result['predicted_effective_rank']>1.2 and result['target_effective_rank']>1.2
    return result
