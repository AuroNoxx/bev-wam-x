"""Isolated native OCC v2. Deterministic zero-source residual-world flow.
No future OCC decoder. Input is 3D OCC represented by height-channel Conv2d.
Frozen first 12 Qwen text blocks encode history once; no LoRA or text generation.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from occ_qwen_native_flow import NativeFrameEncoder
from occ_qwen_joint_flow import load_text_backbone,trajectory_to_velocity,velocity_to_trajectory


def load_v2_backbone(path, layers=12, dtype=torch.bfloat16):
    backbone=load_text_backbone(path,dtype=dtype)
    if not 1<=layers<=len(backbone.layers):raise ValueError('invalid Qwen layer count')
    backbone.layers=nn.ModuleList(list(backbone.layers[:layers]))
    backbone.config.num_hidden_layers=layers
    return backbone.requires_grad_(False).eval()


class NativeHistoryTokens(nn.Module):
    def __init__(self,width=64):
        super().__init__();self.frame=NativeFrameEncoder(width);self.tokens=64
        self.time=nn.Parameter(torch.randn(1,4,1,width)*.02)
        self.position=nn.Parameter(torch.randn(1,1,64,width)*.02)
    def frames(self,occ):
        z=self.frame(occ);b,t,n,d=z.shape
        return F.adaptive_avg_pool2d(z.reshape(b*t,32,32,d).permute(0,3,1,2),(8,8)).flatten(2).transpose(1,2).reshape(b,t,64,d)
    def forward(self,history):
        if history.shape[1:]!=(4,20,128,128):raise ValueError('history must be native 4x20x128x128')
        return (self.frames(history)+self.time+self.position).flatten(1,2)


class AdaLNZeroBlock(nn.Module):
    def __init__(self,d,heads):
        super().__init__()
        self.norms=nn.ModuleList([nn.LayerNorm(d,elementwise_affine=False) for _ in range(3)])
        self.self_attn=nn.MultiheadAttention(d,heads,batch_first=True,dropout=0)
        self.cross_attn=nn.MultiheadAttention(d,heads,batch_first=True,dropout=0)
        self.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
        self.modulation=nn.Sequential(nn.SiLU(),nn.Linear(d,9*d))
        nn.init.zeros_(self.modulation[-1].weight);nn.init.zeros_(self.modulation[-1].bias)
    def forward(self,x,scene,condition):
        m=self.modulation(condition).chunk(9,-1)
        q=self.norms[0](x)*(1+m[1][:,None])+m[0][:,None]
        x=x+m[2][:,None]*self.self_attn(q,q,q,need_weights=False)[0]
        q=self.norms[1](x)*(1+m[4][:,None])+m[3][:,None]
        x=x+m[5][:,None]*self.cross_attn(q,scene,scene,need_weights=False)[0]
        q=self.norms[2](x)*(1+m[7][:,None])+m[6][:,None]
        return x+m[8][:,None]*self.mlp(q)


class NativeOccQwenV2(nn.Module):
    def __init__(self,backbone,width=64,decoder_width=256,decoder_layers=3,heads=8,
                 world_weight=.1,accel_weight=.001,jerk_weight=.001,variance_weight=.01,
                 accel_scale=3.,jerk_scale=10.):
        super().__init__();self.backbone=backbone.requires_grad_(False).eval()
        self.width=width;self.scene_tokens=64;self.future_steps=2
        self.encoder=NativeHistoryTokens(width);self.teacher=copy.deepcopy(self.encoder).requires_grad_(False).eval()
        self.scene_in=nn.Linear(width,backbone.config.hidden_size)
        self.scene_out=nn.Linear(backbone.config.hidden_size,decoder_width)
        self.scene_norm=nn.LayerNorm(decoder_width)
        self.action_in=nn.Linear(2,decoder_width);self.world_in=nn.Linear(width,decoder_width)
        self.action_position=nn.Parameter(torch.randn(1,25,decoder_width)*.02)
        self.world_position=nn.Parameter(torch.randn(1,128,decoder_width)*.02)
        self.time_in=nn.Sequential(nn.Linear(1,decoder_width),nn.SiLU(),nn.Linear(decoder_width,decoder_width))
        self.state_in=nn.Sequential(nn.Linear(20,decoder_width),nn.SiLU(),nn.Linear(decoder_width,decoder_width))
        self.action_blocks=nn.ModuleList([AdaLNZeroBlock(decoder_width,heads) for _ in range(decoder_layers)])
        self.world_blocks=nn.ModuleList([nn.TransformerDecoderLayer(decoder_width,heads,4*decoder_width,dropout=0,batch_first=True,norm_first=True) for _ in range(decoder_layers)])
        self.world_action_attention=nn.MultiheadAttention(decoder_width,heads,batch_first=True,dropout=0)
        self.world_action_gate=nn.Parameter(torch.zeros(()))
        self.action_norm=nn.LayerNorm(decoder_width);self.world_norm=nn.LayerNorm(decoder_width)
        self.action_out=nn.Linear(decoder_width,2);self.world_out=nn.Linear(decoder_width,width)
        # Zero WORLD velocity + zero source => persistence at initialization for all Euler step counts.
        nn.init.zeros_(self.world_out.weight);nn.init.zeros_(self.world_out.bias)
        self.loss_weights={'world':world_weight,'accel':accel_weight,'jerk':jerk_weight,'variance':variance_weight}
        self.accel_scale=accel_scale;self.jerk_scale=jerk_scale
    def train(self,mode=True):
        super().train(mode);self.backbone.eval();self.teacher.eval();return self
    def encode_history(self,history):
        z=self.encoder(history);x=self.scene_in(z);dtype=next(self.backbone.parameters()).dtype
        mask=torch.zeros(len(x),1,256,256,device=x.device,dtype=dtype)
        h=self.backbone(inputs_embeds=x.to(dtype),attention_mask=mask,use_cache=False).last_hidden_state.float()
        return self.scene_norm(self.scene_out(h)),z
    def velocity(self,scene,state,action,t,world):
        b=len(state)
        if state.shape!=(b,20) or action.shape!=(b,25,2) or world.shape!=(b,128,self.width):raise ValueError('v2 flow shape contract')
        time=self.time_in(t.reshape(b,1));condition=time+self.state_in(state)
        w=self.world_in(world)+self.world_position+time[:,None]
        for block in self.world_blocks:w=block(w,scene)
        world_velocity=self.world_out(self.world_norm(w))
        a=self.action_in(action)+self.action_position+time[:,None]
        a=a+self.world_action_gate.tanh()*self.world_action_attention(a,w,w,need_weights=False)[0]
        for block in self.action_blocks:a=block(a,scene,condition)
        return self.action_out(self.action_norm(a)),world_velocity
    def forward(self,history,state,action,t,world):
        scene,_=self.encode_history(history);return self.velocity(scene,state,action,t,world)
    @torch.no_grad()
    def world_target(self,future):return self.teacher.frames(future[:,:2]).flatten(1,2).float()
    @torch.no_grad()
    def persistence(self,history):return self.teacher.frames(history[:,-1:]).float().expand(-1,2,-1,-1).flatten(1,2)
    @torch.no_grad()
    def update_teacher(self,momentum=.99):
        for target,source in zip(self.teacher.parameters(),self.encoder.parameters()):target.lerp_(source,1-momentum)
    def loss(self,history,state,trajectory,future):
        action_target=trajectory_to_velocity(trajectory,state)
        delta=self.world_target(future)-self.persistence(history) # both EMA, detached
        t=torch.rand(len(state),1,device=state.device);mix=t[:,:,None];noise=torch.randn_like(action_target)
        at=(1-mix)*noise+mix*action_target;wt=mix*delta # deterministic Dirac-base world flow
        scene,z=self.encode_history(history);va,vw=self.velocity(scene,state,at,t,wt)
        afm=F.mse_loss(va,action_target-noise)
        known=((future[:,:2]>0)&(future[:,:2]<8)).float().mean(2)
        support=F.adaptive_avg_pool2d(known.flatten(0,1)[:,None],(8,8)).reshape(len(state),128)>0
        wfm=((vw-delta).square().mean(-1)*support).sum()/support.sum().clamp_min(1)
        # Endpoint estimate from sampled interpolation time; training regularizer, not a hard constraint.
        residual_endpoint=at+(1-mix)*va
        velocity=residual_endpoint*10+torch.stack([state[:,0]*30,torch.zeros_like(state[:,0])],-1)[:,None]
        acceleration=torch.diff(velocity,dim=1)/.2;jerk=torch.diff(velocity,n=2,dim=1)/(.2**2)
        accel_penalty=F.huber_loss(acceleration/self.accel_scale,torch.zeros_like(acceleration),delta=1.)
        jerk_penalty=F.huber_loss(jerk/self.jerk_scale,torch.zeros_like(jerk),delta=1.)
        variance=F.relu(.25-(z.var(0,unbiased=False)+1e-4).sqrt()).mean() if len(state)>1 else z.sum()*0
        loss=afm+self.loss_weights['world']*wfm+self.loss_weights['accel']*accel_penalty+self.loss_weights['jerk']*jerk_penalty+self.loss_weights['variance']*variance
        return loss,{'action_fm':afm,'world_fm':wfm,'accel_huber':accel_penalty,'jerk_huber':jerk_penalty,'variance':variance,'world_action_gate':self.world_action_gate.tanh().detach()}
    @torch.no_grad()
    def sample(self,history,state,steps=8,generator=None):
        if steps<1:raise ValueError('positive Euler steps required')
        scene,_=self.encode_history(history);b=len(state)
        a=torch.randn(b,25,2,device=state.device,generator=generator);w=torch.zeros(b,128,self.width,device=state.device)
        for i in range(steps):
            t=torch.full((b,1),i/steps,device=state.device);va,vw=self.velocity(scene,state,a,t,w)
            a=a+va/steps;w=w+vw/steps
        return {'trajectory_xy':velocity_to_trajectory(a,state),'world_latent':self.persistence(history)+w,'world_delta':w,'residual_velocity':a}
