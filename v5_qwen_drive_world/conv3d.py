"""Workspace-bounded native Conv3d calls, preserving complete depth halos."""
import torch
from torch import nn
from torch.nn import functional as F


class DepthTiledConv3d(nn.Conv3d):
    def forward(self, x):
        # Native NCDHW path: NDHWC backward selection stalled on the full encoder.
        x=x.contiguous()
        if (self.stride[0]!=1 or self.dilation[0]!=1 or self.padding_mode!='zeros'
                or self.padding[0]!=(self.kernel_size[0]-1)//2):
            raise ValueError('Unsupported depth tiling geometry')
        estimated_workspace=x.numel()*self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2]*2
        if x.shape[2]<=4 or estimated_workspace<=512*2**20:
            return super().forward(x)
        pad=self.padding[0]
        padded=F.pad(x,(0,0,0,0,pad,pad)) if pad else x
        outputs=[]
        for start in range(0,x.shape[2],4):
            end=min(start+4,x.shape[2])
            tile=padded[:,:,start:end+2*pad].contiguous()
            outputs.append(F.conv3d(tile,self.weight,self.bias,self.stride,
                                    (0,self.padding[1],self.padding[2]),self.dilation,self.groups))
        return torch.cat(outputs,dim=2)


def use_depth_tiled_conv3d(module):
    count=0
    for layer in module.modules():
        if type(layer) is nn.Conv3d:
            layer.__class__=DepthTiledConv3d
            count+=1
    return count
