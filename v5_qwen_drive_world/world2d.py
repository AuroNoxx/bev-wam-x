"""Authorized 2D World extension; new decoder is NOT a pretrained OCC head."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .runtime import math_attention


class ResidualUpsample(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.skip=nn.Conv2d(cin,cout,1)
        self.body=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1),nn.GroupNorm(8,cout),nn.SiLU(),
                                nn.Conv2d(cout,cout,3,padding=1),nn.GroupNorm(8,cout))
    def forward(self,x):
        x=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
        return F.silu(self.skip(x)+self.body(x))


class ActionConditionedWorld2D(nn.Module):
    def __init__(self,perception):
        super().__init__()
        native=perception.bev_modeling
        stage=native.adaptor.stages[2]
        # Copy the original 1x1 language projection and channel normalization.
        # Drop the depth/volume network, 3D decoder, detection and map branches.
        self.scene_projection=nn.Sequential(copy.deepcopy(stage[0]),copy.deepcopy(stage[1])).float()
        head=native.head
        old=head.bev_embedding.weight.detach().float().reshape(head.bev_h,head.bev_w,256).permute(2,0,1)[None]
        y=-19.2+(torch.arange(8)+.5)*(38.4/8)
        x=-16+(torch.arange(8)+.5)*(128/8)
        yy,xx=torch.meshgrid(y,x,indexing='ij')
        xy=torch.stack((xx,yy),-1)
        pc=head.pc_range
        grid=torch.stack((2*(xx-pc[0])/(pc[3]-pc[0])-1,2*(yy-pc[1])/(pc[4]-pc[1])-1),-1)
        query=F.grid_sample(old,grid[None],align_corners=False,padding_mode='zeros')
        self.query=nn.Parameter(query[0].flatten(1).T.contiguous())
        self.register_buffer('query_xy',xy.reshape(64,2)/torch.tensor([128.,38.4]))
        self.query_position=nn.Linear(2,256)
        self.cross_attention=nn.MultiheadAttention(256,8,batch_first=True,dropout=0.)
        self.norm=nn.LayerNorm(256)
        self.action_condition=nn.Sequential(nn.Linear(102,256),nn.SiLU(),nn.Linear(256,512))
        self.delta=nn.Sequential(nn.Conv2d(256,256,3,padding=1),nn.SiLU(),nn.Conv2d(256,256,1))
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
        self.decoder=nn.Sequential(ResidualUpsample(256,128),ResidualUpsample(128,96),
                                   ResidualUpsample(96,64),ResidualUpsample(64,32),nn.Conv2d(32,1,1))
        self.migration={'reused':'official scale1 language 1x1 projection+LayerNorm2d; physically resampled BEV query weights',
                        'new':'BEV cross attention, physical query position, action residual, four2D upsample blocks',
                        'discarded':'UVTR/depth/Conv3d/3D occupancy/detection/map branches',
                        'latent_shape':[256,8,8],'occupancy_shape':[3,128,128],
                        'pretrained_2d_occupancy_decoder':False,'far_range_capability_validated':False}
        assert not any(isinstance(m,nn.Conv3d) for m in self.modules())

    def encode_current(self,hidden,attention_mask=None):
        memory=self.scene_projection(hidden.transpose(1,2).unsqueeze(2)).squeeze(2).transpose(1,2)
        query=(self.query+self.query_position(self.query_xy))[None].expand(len(hidden),-1,-1).to(memory.dtype)
        with math_attention():
            value=self.cross_attention(query,memory,memory,need_weights=False,
                    key_padding_mask=None if attention_mask is None else ~attention_mask.bool())[0]
        return self.norm(query+value).transpose(1,2).reshape(len(hidden),256,8,8)

    def forward(self,current,trajectory_xy,v0):
        if trajectory_xy.shape[1:]!=(50,2):raise ValueError('Expected physical five-second XY trajectory')
        batch=len(current)
        horizon=current.new_tensor([1.,2.,3.])[None,:,None].expand(batch,-1,-1)/3
        action=torch.cat((trajectory_xy.flatten(1)[:,None].expand(-1,3,-1)/160,
                          v0[:,None,None].expand(-1,3,-1)/35,horizon),-1).reshape(batch*3,102)
        shift,scale=self.action_condition(action).chunk(2,-1)
        base=current[:,None].expand(-1,3,-1,-1,-1).reshape(batch*3,256,8,8)
        delta=self.delta(base*(1+scale[:,:,None,None])+shift[:,:,None,None])
        future=base.detach()+delta
        logits=self.decoder(future).reshape(batch,3,128,128).float()
        return {'occupancy':logits.sigmoid(),'logits':logits,
                'delta_latent':delta.reshape(batch,3,256,8,8)}
