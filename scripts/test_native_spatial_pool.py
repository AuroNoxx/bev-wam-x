"""Regression: feature compression must not erase the longitudinal token axis."""
import torch
from torch import nn
from occ_qwen_native_flow import NativeTokens
class Probe(nn.Module):
    def forward(self,occ):
        z=torch.zeros(1,1,1024,64)
        z[:,:,0,:]=1
        z[:,:,31,:]=2
        return z
m=NativeTokens();m.frame=Probe()
z=m.frames(torch.empty(1,1,20,128,128))
assert z.shape==(1,1,16,64)
assert torch.allclose(z[0,0,0],torch.full((64,),1/64))
assert torch.allclose(z[0,0,3],torch.full((64,),2/64))
assert (z[0,0].abs().sum(-1)>0).sum()==2
print('PASS: left/right grid landmarks occupy distinct XY tokens')
