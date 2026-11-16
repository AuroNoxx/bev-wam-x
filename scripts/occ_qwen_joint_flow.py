"""OCC-Qwen joint flow research prototype. Not a trained driving policy.

No RGB tower, text generation, future observations in actor inputs, or OCC decoder.
Shared frozen Qwen blocks jointly update scene/action/world tokens using a full
bidirectional mask. Freezing parameters does NOT disable input gradients.
"""
import copy
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from joint_occ_model import FrameEncoder


class OccTokens(nn.Module):
    def __init__(self, width=64, tokens=16):
        super().__init__()
        self.frame = FrameEncoder(width)
        self.time = nn.Parameter(torch.randn(1,4,1,width)*.02)
        self.fuse = nn.Linear(4*width,width)
        self.tokens = tokens

    def frames(self, occ):
        z = self.frame(occ)
        b,t,n,d = z.shape
        return F.adaptive_avg_pool1d(z.reshape(b*t,n,d).transpose(1,2),self.tokens).transpose(1,2).reshape(b,t,self.tokens,d)

    def forward(self, history):
        if history.shape[1:] != (4,20,64,160):
            raise ValueError('history must be B,4,20,64,160 in current ego FLU')
        z = self.frames(history) + self.time
        return self.fuse(z.permute(0,2,1,3).flatten(2))


def load_text_backbone(path, dtype=torch.bfloat16):
    """Strict local loading of text blocks only; omit unused embedding/vision tower."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    from safetensors import safe_open
    path=Path(path)
    cfg=Qwen3VLTextConfig(**json.loads((path/'config.json').read_text())['text_config'])
    cfg._attn_implementation='sdpa'
    with torch.device('meta'):
        model=Qwen3VLTextModel(cfg)
    # Input is continuous embeddings, not text token IDs. Saving 311M unused params.
    model.embed_tokens=nn.Identity()
    weights={}
    prefix='model.language_model.'
    for shard in sorted(path.glob('*.safetensors')):
        with safe_open(shard,framework='pt',device='cpu') as f:
            for k in f.keys():
                if k.startswith(prefix) and not k.startswith(prefix+'embed_tokens.'):
                    weights[k[len(prefix):]]=f.get_tensor(k).to(dtype)
    model.load_state_dict(weights,strict=True,assign=True)
    # Rotary frequency buffers are nonpersistent and must not remain on meta.
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
    model.rotary_emb=Qwen3VLTextRotaryEmbedding(cfg,device='cpu')
    return model.requires_grad_(False).eval()


class OccQwenJointFlow(nn.Module):
    def __init__(self, backbone, width=64, scene_tokens=16, future_steps=2):
        super().__init__()
        self.backbone=backbone.requires_grad_(False).eval()
        d=backbone.config.hidden_size
        self.encoder=OccTokens(width,scene_tokens)
        self.teacher=copy.deepcopy(self.encoder).requires_grad_(False).eval()
        self.scene_in=nn.Linear(width,d)
        self.state_in=nn.Linear(20,d)
        self.action_in=nn.Linear(2,d)
        self.world_in=nn.Linear(width,d)
        self.time_in=nn.Sequential(nn.Linear(1,d),nn.SiLU(),nn.Linear(d,d))
        self.position=nn.Parameter(torch.randn(1,scene_tokens+1+25+future_steps*scene_tokens,d)*.01)
        self.action_out=nn.Linear(d,2)
        self.world_out=nn.Linear(d,width)
        self.width=width; self.scene_tokens=scene_tokens; self.future_steps=future_steps

    def train(self,mode=True):
        super().train(mode)
        self.teacher.eval(); self.backbone.eval()
        return self

    def encode_history(self,history,state):
        return torch.cat([self.scene_in(self.encoder(history)),self.state_in(state)[:,None]],1)

    def velocity(self,scene,action,t,world,return_scene=False):
        b=scene.shape[0]; n=self.scene_tokens+1
        if action.shape!=(b,25,2) or world.shape!=(b,self.future_steps*self.scene_tokens,self.width):
            raise ValueError('invalid flow tensor shapes')
        z=torch.cat([scene,self.action_in(action),self.world_in(world)],1)
        z=z+self.position+self.time_in(t.reshape(b,1))[:,None]
        dtype=next(self.backbone.parameters()).dtype
        z=z.to(dtype)
        # Explicit additive 4D mask bypasses HF causal-mask construction.
        # All modalities at this flow time can attend to all other modalities.
        mask=torch.zeros(b,1,z.shape[1],z.shape[1],device=z.device,dtype=dtype)
        h=self.backbone(inputs_embeds=z,attention_mask=mask,use_cache=False).last_hidden_state.float()
        a=self.action_out(h[:,n:n+25]); w=self.world_out(h[:,n+25:])
        return (a,w,h[:,:n]) if return_scene else (a,w)

    def forward(self,history,state,action,t,world):
        return self.velocity(self.encode_history(history,state),action,t,world)

    @torch.no_grad()
    def world_target(self,future):
        if future.shape[1]<self.future_steps: raise ValueError('missing future labels')
        return self.teacher.frames(future[:,:self.future_steps]).flatten(1,2).float()

    @torch.no_grad()
    def update_teacher(self,momentum=.99):
        for target,source in zip(self.teacher.parameters(),self.encoder.parameters()):
            target.lerp_(source,1-momentum)

    def loss(self,history,state,trajectory,future):
        # Only this training API accepts future truth. Flow corruption is the
        # standard training target path; sample() has no future-label argument.
        target_action=trajectory_to_velocity(trajectory,state)
        target_world=self.world_target(future)
        t=torch.rand(len(state),1,device=state.device)
        an=torch.randn_like(target_action); wn=torch.randn_like(target_world)
        mix=t[:,:,None]
        a,w=self(history,state,(1-mix)*an+mix*target_action,t,(1-mix)*wn+mix*target_world)
        action_loss=F.mse_loss(a,target_action-an)
        # Unknown/unfilled never means verified-free. Supervise occupied support
        # only; no claims that these latent targets are calibrated occupancy.
        known=((future[:,:self.future_steps]>0)&(future[:,:self.future_steps]<8)).float().mean(2)
        support=F.adaptive_avg_pool2d(known.flatten(0,1)[:,None],(8,20)).flatten(2)
        support=F.adaptive_avg_pool1d(support,self.scene_tokens).reshape(len(state),-1)>0
        world_loss=((w-(target_world-wn)).square().mean(-1)*support).sum()/support.sum().clamp_min(1)
        # Online history variance is only a collapse guard, not a proof of useful
        # semantics. Target variance/rank still require held-out evaluation.
        z=self.encoder(history)
        variance=F.relu(.25-(z.var(0,unbiased=False)+1e-4).sqrt()).mean() if len(state)>1 else z.sum()*0
        total=action_loss+.1*world_loss+.01*variance
        return total,{'action_fm':action_loss,'world_fm':world_loss,'variance':variance}

    @torch.no_grad()
    def sample(self,history,state,steps=8,generator=None):
        if steps<1: raise ValueError('steps must be positive')
        scene=self.encode_history(history,state); b=len(state)
        a=torch.randn(b,25,2,device=state.device,generator=generator)
        w=torch.randn(b,self.future_steps*self.scene_tokens,self.width,device=state.device,generator=generator)
        for i in range(steps):
            t=torch.full((b,1),i/steps,device=state.device)
            va,vw=self.velocity(scene,a,t,w)
            a=a+va/steps; w=w+vw/steps
        return {'trajectory_xy':velocity_to_trajectory(a,state),'world_latent':w,'residual_velocity':a}


def trajectory_to_velocity(trajectory,state,dt=.2):
    # Existing OCC cache labels: x/50, y/10, sin(yaw), cos(yaw), speed/30.
    xy=trajectory[...,:2]*trajectory.new_tensor([50.,10.])
    velocity=torch.diff(torch.cat([torch.zeros_like(xy[:,:1]),xy],1),dim=1)/dt
    baseline=torch.stack([state[:,0]*30,torch.zeros_like(state[:,0])],-1)
    return (velocity-baseline[:,None])/10


def velocity_to_trajectory(residual,state,dt=.2):
    baseline=torch.stack([state[:,0]*30,torch.zeros_like(state[:,0])],-1)
    return ((residual*10+baseline[:,None])*dt).cumsum(1)
