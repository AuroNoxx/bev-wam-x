"""Native-grid variant; preserves old resampled model as historical baseline."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from occ_qwen_joint_flow import OccQwenJointFlow,OccTokens,load_text_backbone,trajectory_to_velocity,velocity_to_trajectory

class NativeFrameEncoder(nn.Module):
    def __init__(self,width=64):
        super().__init__();self.embedding=nn.Embedding(9,2)
        self.conv=nn.Sequential(nn.Conv2d(40,width,3,2,1),nn.GroupNorm(8,width),nn.SiLU(),nn.Conv2d(width,width,3,2,1),nn.GroupNorm(8,width),nn.SiLU())
        self.norm=nn.LayerNorm(width)
    def forward(self,occ):
        if occ.shape[2:]!=(20,128,128): raise ValueError('native grid must be 20x128x128')
        b,t=occ.shape[:2]
        x=self.embedding(torch.where(occ==255,8,occ).long())
        x=x.permute(0,1,2,5,3,4).reshape(b*t,40,128,128)
        # Learned encoder feature compression, NOT preprocessing resize of labels.
        x=self.conv(x).flatten(2).transpose(1,2)
        return self.norm(x).reshape(b,t,1024,-1)

class NativeTokens(OccTokens):
    def __init__(self,width=64,tokens=16):
        super().__init__(width,tokens);self.frame=NativeFrameEncoder(width)
    def frames(self,occ):
        z=self.frame(occ);b,t,n,d=z.shape
        side=int(self.tokens**.5)
        if side*side!=self.tokens:raise ValueError('native tokens require a square spatial lattice')
        # Preserve both X and Y: flatten-then-1D pooling would merge whole rows.
        z=z.reshape(b*t,32,32,d).permute(0,3,1,2)
        return F.adaptive_avg_pool2d(z,(side,side)).flatten(2).transpose(1,2).reshape(b,t,self.tokens,d)
    def forward(self,history):
        if history.shape[1:]!=(4,20,128,128):raise ValueError('native historical OCC shape mismatch')
        z=self.frames(history)+self.time
        return self.fuse(z.permute(0,2,1,3).flatten(2))

class NativeOccQwenJointFlow(OccQwenJointFlow):
    def __init__(self,backbone,width=64,scene_tokens=16,future_steps=2):
        super().__init__(backbone,width,scene_tokens,future_steps)
        self.encoder=NativeTokens(width,scene_tokens)
        self.teacher=copy.deepcopy(self.encoder).requires_grad_(False).eval()
    def loss(self,history,state,trajectory,future):
        target_action=trajectory_to_velocity(trajectory,state);target_world=self.world_target(future)
        t=torch.rand(len(state),1,device=state.device);an=torch.randn_like(target_action);wn=torch.randn_like(target_world);mix=t[:,:,None]
        a,w=self(history,state,(1-mix)*an+mix*target_action,t,(1-mix)*wn+mix*target_world)
        action_loss=F.mse_loss(a,target_action-an)
        known=((future[:,:self.future_steps]>0)&(future[:,:self.future_steps]<8)).float().mean(2)
        side=int(self.scene_tokens**.5)
        support=F.adaptive_avg_pool2d(known.flatten(0,1)[:,None],(side,side)).reshape(len(state),-1)>0
        world_loss=((w-(target_world-wn)).square().mean(-1)*support).sum()/support.sum().clamp_min(1)
        z=self.encoder(history);variance=F.relu(.25-(z.var(0,unbiased=False)+1e-4).sqrt()).mean() if len(state)>1 else z.sum()*0
        return action_loss+.1*world_loss+.01*variance,{'action_fm':action_loss,'world_fm':world_loss,'variance':variance}

