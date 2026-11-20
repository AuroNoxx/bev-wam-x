"""Single-frame RGB IPM and seven-channel BEV planner."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from .unified_grid import X,Y
import numpy as np


class NativeBatchNorm(nn.BatchNorm2d):
    """Use PyTorch's native 2D BatchNorm on this ROCm build, with ResNet's fixed momentum."""
    def forward(self,x):
        if self.training:
            self.num_batches_tracked.add_(1)
        y,_,_=torch.native_batch_norm(x,self.weight,self.bias,self.running_mean,
                                     self.running_var,self.training,self.momentum,self.eps)
        return y


def backbone(model):
    return nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,
                         model.layer1,model.layer2,model.layer3,model.layer4)


class GroundProjection(nn.Module):
    def __init__(self):
        super().__init__()
        x,y=np.meshgrid(X,Y)
        focal=320./np.tan(np.deg2rad(40.))
        depth=np.maximum(x-1.3,.01)
        u=2*(320.-focal*y/depth+.5)/640.-1.
        v=2*(180.+2*focal/depth+.5)/360.-1.
        valid=(x>1.3)&(x<=45)&(np.abs(u)<=1.-1./640)
        coeff=-2*focal/(360.*depth)
        basis=np.stack((coeff*x,coeff*y,coeff)).reshape(3,-1)
        self.register_buffer("u",torch.from_numpy(u.astype(np.float32))[None])
        self.register_buffer("v",torch.from_numpy(v.astype(np.float32))[None])
        self.register_buffer("basis",torch.from_numpy(basis.astype(np.float32)))
        self.register_buffer("footprint",torch.from_numpy(valid)[None])

    def sample(self,rgb,plane):
        with torch.autocast(device_type=rgb.device.type,enabled=False):
            vertical=self.v+(plane.float()@self.basis).view(-1,128,128)
            valid=self.footprint & (vertical.abs()<=1.-1./360)
            grid=torch.stack((self.u.expand_as(vertical),vertical),dim=-1)
            projected=F.grid_sample(rgb.float()/255.,grid,align_corners=False,padding_mode="zeros")
            return projected*valid[:,None],valid[:,None]

    def forward(self,rgb,plane):
        return self.sample(rgb,plane)[0]


class UnifiedDriver(nn.Module):
    def __init__(self):
        super().__init__()
        self.ipm=GroundProjection()
        encoder=resnet18(weights=None,norm_layer=NativeBatchNorm)
        encoder.conv1=nn.Conv2d(7,64,7,stride=2,padding=3,bias=False)
        nn.init.kaiming_normal_(encoder.conv1.weight,mode="fan_out",nonlinearity="relu")
        self.bev=backbone(encoder)
        self.condition=nn.Sequential(nn.Linear(4,64),nn.SiLU())
        self.trajectory=nn.Sequential(nn.Flatten(),nn.Linear(512*4*4+64,256),nn.SiLU(),nn.Linear(256,80))
        self.coarse=nn.Sequential(nn.Conv2d(512,64,3,padding=1),nn.GroupNorm(8,64),nn.SiLU())
        self.occupancy=nn.Sequential(nn.Conv2d(128,64,3,padding=1),nn.GroupNorm(8,64),
                                     nn.SiLU(),nn.Conv2d(64,5,1))

    def encode_input(self,rgb,lidar,nav,plane):
        return torch.cat((self.ipm(rgb,plane),lidar.float()/255.,nav.float()/255.),dim=1)

    def plan(self,bev,v0,intent):
        near=self.bev[:5](bev)
        h=self.bev[5:](near)
        cond=self.condition(torch.cat((v0.reshape(-1,1)/35.,intent.float()),dim=1))
        params=self.trajectory(torch.cat((h.flatten(1),cond),dim=1)).float().reshape(-1,40,2)
        # Integrate bounded acceleration and yaw rate at 10 Hz in FP32.
        speed=(v0.reshape(-1,1)+.1*(3.5*params[:,:,0].tanh()).cumsum(1)).clamp_min(0.)
        theta=(.1*params[:,:,1].tanh()).cumsum(1)
        step=torch.stack((theta.cos(),theta.sin()),dim=-1)*speed[:,:,None]*.1
        low=F.interpolate(self.coarse(h),size=near.shape[-2:],mode="bilinear",align_corners=False)
        logits=self.occupancy(torch.cat((near,low),1))
        return step.cumsum(1),F.interpolate(logits,size=(128,128),mode="bilinear",align_corners=False)

    def forward(self,rgb,lidar,nav,v0,intent,plane):
        return self.plan(self.encode_input(rgb,lidar,nav,plane),v0,intent)
