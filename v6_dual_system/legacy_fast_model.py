"""Two-dimensional fast planner, trained and deployed at native sensor resolution."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, resnet34

from v5_qwen_drive_world.runtime import math_attention


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


class FastDriver(nn.Module):
    def __init__(self,corrective=False,lane_supervision=False,road_roi=False,navigation=False,spatial=False,film=False):
        super().__init__()
        self.layout=torch.channels_last
        self.register_buffer("state_scale",torch.tensor([33.33,5.,1.]),persistent=False)
        image=resnet34(weights=None,norm_layer=NativeBatchNorm)
        # Both native frames enter a single learned temporal stem.
        image.conv1=nn.Conv2d(6,64,7,stride=4,padding=3,bias=False)
        self.rgb=backbone(image)
        self.bev=backbone(resnet18(weights=None,norm_layer=NativeBatchNorm))
        self.condition=nn.Sequential(nn.Linear(7,256),nn.GELU(),nn.Linear(256,512))
        self.position=nn.Parameter(torch.empty(1,1+5*9+4*4,512))
        nn.init.normal_(self.position,std=.02)
        self.fusion=nn.ModuleList(nn.TransformerEncoderLayer(
            512,8,2048,dropout=0,activation="gelu",batch_first=True,norm_first=True) for _ in range(3))
        self.norm=nn.LayerNorm(512)
        self.head=nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Linear(512,20))
        # Ten local displacement corrections interpolate to 40 future increments.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.skip=nn.Conv2d(64,32,1)
        self.decode=nn.Sequential(nn.Conv2d(512,128,3,padding=1),nn.GroupNorm(8,128),nn.SiLU(),
                                  nn.Conv2d(128,64,3,padding=1),nn.GroupNorm(8,64),nn.SiLU())
        self.occupancy=nn.Sequential(nn.Conv2d(96,48,3,padding=1),nn.GroupNorm(8,48),
                                     nn.SiLU(),nn.Conv2d(48,4,1))
        self.corrective=False
        self.road_roi=False
        self.navigation=False
        self.spatial=False
        if corrective:self.enable_metric_fusion()
        if lane_supervision:self.enable_lane_supervision()
        if road_roi:self.enable_road_roi()
        if navigation:self.enable_navigation()
        if spatial:self.enable_spatial()
        self.film=False
        if film:self.enable_film()

    def enable_navigation(self):
        if not self.road_roi:
            raise ValueError("Navigation requires the calibrated road ROI and lane head")
        # Preserve speed and command weights. Acceleration/yaw-rate columns disappear.
        for network in (self.condition,self.head):
            old=network[0]
            prefix=old.in_features-7
            new=nn.Linear(old.in_features+2,old.out_features)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:,:prefix+1].copy_(old.weight[:,:prefix+1])
                new.weight[:,prefix+5:].copy_(old.weight[:,prefix+3:])
                new.bias.copy_(old.bias)
            network[0]=new
        self.state_scale=torch.tensor([33.33,1.,1.,1.,1.])
        self.register_buffer("times",torch.arange(1,41,dtype=torch.float32)[None]*.1,persistent=False)
        self.navigation=True

    def enable_spatial(self):
        if not self.navigation:raise ValueError("Spatial navigation requires the navigation checkpoint")
        old=self.bev[0]
        conv=nn.Conv2d(4,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[:,:3].copy_(old.weight)
        self.bev[0]=conv
        for network in (self.condition,self.head):
            old=network[0]
            prefix=old.in_features-9
            new=nn.Linear(old.in_features-4,old.out_features)
            with torch.no_grad():
                new.weight.copy_(torch.cat((old.weight[:,:prefix+1],old.weight[:,prefix+5:]),1))
                new.bias.copy_(old.bias)
            network[0]=new
        self.register_buffer("grid_x",(torch.arange(128,dtype=torch.float32)-15.5)[None,None,:],persistent=False)
        self.register_buffer("grid_y",((torch.arange(128,dtype=torch.float32)-63.5)*.3)[None,:,None],persistent=False)
        self.spatial=True

    def enable_film(self):
        if not self.spatial:
            raise ValueError("Dense navigation requires the spatial parent")
        self.nav_encoder=nn.Sequential(nn.Linear(8,96),nn.GELU())
        self.lane_film=nn.Linear(96,2*96)
        self.plan_film=nn.Linear(96,2*512)
        for projection in (self.nav_encoder[0],self.lane_film,self.plan_film):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.state_scale=torch.tensor([33.33]+[1.]*12)
        self.film=True

    def route_channel(self, normalized):
        """Distance to two finite FLU segments; Gaussian full width at half maximum 1.5 m."""
        x1,y1,x2,y2=(v[:,None,None]*60. for v in normalized.unbind(1))
        def distance(x,y,dx,dy):
            dot=x*dx+y*dy
            t=(dot/(dx.square()+dy.square()).clamp_min(1e-6)).clamp(0.,1.)
            return (x-t*dx).square()+(y-t*dy).square()
        first=distance(self.grid_x,self.grid_y,x1,y1)
        second=distance(self.grid_x-x1,self.grid_y-y1,x2-x1,y2-y1)
        return torch.exp((-4.*0.6931471805599453/1.5**2)*torch.minimum(first,second))[:,None]

    def enable_road_roi(self):
        if not hasattr(self,"lane_head"):raise ValueError("Road ROI requires lane supervision")
        self.mid_projection=nn.Conv2d(128,64,kernel_size=1,bias=False)
        self.register_buffer("mirror_command_order",torch.tensor([0,2,1,3]),persistent=False)
        nn.init.zeros_(self.mid_projection.weight)
        # align_corners=False: transform native pixel centres through crop and resize.
        v=8*((self.ground_grid[...,1]+1)*45/2-.5)
        v=(v-140+.5)*(360/220)-.5
        self.ground_grid[...,1]=2*(v/8+.5)/45-1
        self.road_roi=True

    def enable_lane_supervision(self):
        if not self.corrective:
            raise ValueError("Lane supervision requires the calibrated RGB feature grid")
        self.lane_head=nn.Linear(96,2)

    def enable_metric_fusion(self):
        """Read the existing RGB ResNet features at calibrated road-plane locations."""
        if self.corrective:return
        x=torch.tensor([6.,7.5,9.,12.,16.,22.,30.,42.])
        y=torch.linspace(-6.,6.,24)
        x,y=torch.meshgrid(x,y,indexing="ij")
        import math
        focal=320/math.tan(math.radians(40))
        u=320-focal*y/(x-1.3)
        v=180+focal*2./(x-1.3)
        # ResNet stem has stride eight here, with its first receptive field centred at pixel zero.
        self.register_buffer("ground_grid",torch.stack((2*(u/8+.5)/80-1,2*(v/8+.5)/45-1),-1)[None])
        self.ground_readout=nn.Sequential(nn.Conv2d(64,8,1),nn.GELU(),nn.Flatten(),
                                          nn.Linear(8*8*24,96),nn.LayerNorm(96))
        old=self.head[0]
        expanded=nn.Linear(old.in_features+96+7,old.out_features)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:,:512].copy_(old.weight)
            expanded.bias.copy_(old.bias)
        self.head[0]=expanded
        self.corrective=True

    def forward(self,rgb,bev,state,command,return_lane=False,return_residual=False):
        if self.road_roi:
            right=command[:,2].bool()
            flip=right[:,None,None,None]
            rgb=torch.where(flip,rgb[:,:,140:,:].flip((-1,)),rgb[:,:,140:,:])
            bev=torch.where(flip,bev.flip((-2,)),bev)
            sign=(1-2*right.float())[:,None]
            if self.navigation:
                nav=torch.stack((state[:,1::2],state[:,2::2]*sign),dim=-1).flatten(1)
                state=torch.cat((state[:,:1],nav),1)
            else:
                state=torch.cat((state[:,:2],state[:,2:]*sign),1)
            command=torch.where(right[:,None],command.index_select(1,self.mirror_command_order),command)
        rgb=(rgb.float()*(2./255)-1).contiguous(memory_format=self.layout)
        bev=(bev.float()*(2./255)-1).contiguous(memory_format=self.layout)
        if self.spatial:
            bev=torch.cat((bev,self.route_channel(state[:,1:5])),1).contiguous(memory_format=self.layout)
        if self.road_roi:
            rgb=F.interpolate(rgb,size=(360,640),mode="bilinear",align_corners=False)
        if self.corrective:
            shallow=self.rgb[:5](rgb)
            if self.road_roi:
                middle=self.rgb[5](shallow)
                encoded=self.rgb[6:](middle)
                shallow=shallow+F.interpolate(self.mid_projection(middle),size=shallow.shape[-2:],
                                               mode="bilinear",align_corners=False)
            else:
                encoded=self.rgb[5:](shallow)
            metric=F.grid_sample(shallow,self.ground_grid.expand(len(rgb),-1,-1,-1),
                                 align_corners=False)
            metric=self.ground_readout(metric)
        else:
            encoded=self.rgb(rgb)
        image=F.adaptive_avg_pool2d(encoded,(5,9))
        x=bev
        for layer in self.bev[:5]:
            x=layer(x)
        skip=self.skip(x)
        for layer in self.bev[5:]:
            x=layer(x)
        scaled=state/self.state_scale
        cond=torch.cat([scaled[:,:1] if self.spatial else scaled,command[:,:3],command[:,3:4]/120],1)
        tokens=torch.cat((self.condition(cond)[:,None],image.flatten(2).transpose(1,2),
                          x.flatten(2).transpose(1,2)),1)+self.position
        with math_attention():
            for layer in self.fusion:
                tokens=layer(tokens)
        tokens=self.norm(tokens)
        plan_features=tokens[:,0]
        if self.corrective:
            plan_features=torch.cat((plan_features,metric,cond*4),1)
        if self.film:
            embedding=self.nav_encoder(state[:,5:])
            lane_scale,lane_shift=self.lane_film(embedding).chunk(2,1)
            metric=metric*(1.+lane_scale)+lane_shift
            plan_scale,plan_shift=self.plan_film(embedding).chunk(2,1)
            hidden=self.head[1](self.head[0](plan_features))
            projection=self.head[2]
            longitudinal=F.linear(hidden,projection.weight[:10],projection.bias[:10])
            lateral=F.linear(hidden*(1.+plan_scale)+plan_shift,projection.weight[10:],projection.bias[10:])
            knots=torch.stack((longitudinal,lateral),1).float()
        else:
            knots=self.head(plan_features).reshape(-1,2,10).float()
        delta=F.interpolate(knots,size=40,mode="linear",align_corners=True).transpose(1,2)
        base=torch.stack((state[:,0]*.1,torch.zeros_like(state[:,0])),-1)
        xy=torch.cumsum(base[:,None]+delta,dim=1)
        if self.navigation:
            lane=self.lane_head(metric).float()
            yaw=(lane[:,1:2]*.1).clamp(-1.,1.)
            u=(self.times/1.5).clamp_max(1.)
            recovery=u.square()*(3.-2.*u)
            baseline=-xy[:,:,0]*yaw.tan()-lane[:,:1]*recovery/yaw.cos()
            # Quadratic onset leaves the analytic initial tangent intact.
            residual=xy[:,:,1]*(self.times/.5).clamp_max(1.).square()
            keep=command[:,:1]
            lateral=keep*(baseline+residual)+(1.-keep)*xy[:,:,1]
            xy=torch.stack((xy[:,:,0],lateral),-1)
        context=tokens[:,-16:].transpose(1,2).reshape(-1,512,4,4)
        coarse=F.interpolate(self.decode(context),size=skip.shape[-2:],mode="bilinear",align_corners=False)
        occ=self.occupancy(torch.cat((coarse,skip),1))
        occ=F.interpolate(occ,size=(128,128),mode="bilinear",align_corners=False)
        if self.road_roi:
            xy=torch.stack((xy[:,:,0],xy[:,:,1]*sign),dim=-1)
            occ=torch.where(flip,occ.flip((-2,)),occ)
        if return_lane:
            if not self.navigation:lane=self.lane_head(metric).float()
            outputs=(xy,occ,lane*sign if self.road_roi else lane)
            return (*outputs,residual) if return_residual else outputs
        return xy,occ


def losses(pred,logits,xy,truth,emphasis=None,near_lateral=False,navigation=False):
    p=pred.float()
    if emphasis is None:
        error=F.smooth_l1_loss(p,xy.float(),reduction="none").mean((0,1))
    else:
        speed,command,class_weights=emphasis
        # Resolve centimetre-scale errors near Pure Pursuit's actual lookahead.
        times=torch.arange(1,41,device=p.device)*.1
        lookahead=(3.5+(24.-3.5)*speed/33.333).clamp(3.5,24.)
        tau=(lookahead/speed.clamp_min(2.)).clamp(.4,2.)
        w=1+6*torch.exp(-((times[None]-tau[:,None])/.4).square())
        w=w*class_weights[command[:,:3].argmax(1),None]
        point_error=F.smooth_l1_loss(p,xy.float(),beta=.2,reduction="none")
        error=(point_error*w[:,:,None]).sum((0,1))/w.sum()
        if near_lateral:
            wy=w*(torch.where(times<=2.,1.,.1) if navigation else torch.exp(-torch.relu(times-1.5)/.65))
            error=torch.stack((error[0],(point_error[:,:,1]*wy).sum()/wy.sum()))
    # dt is constant. Penalize the third finite difference in metres.
    jerk=p.diff(n=3,dim=1).square().mean()
    logits=logits.float()
    truth=truth.float()
    bce=F.binary_cross_entropy_with_logits(logits,truth,pos_weight=logits.new_tensor(16.))
    prob=logits.sigmoid()
    dice=1-((2*(prob*truth).sum((-2,-1))+1)/(prob.sum((-2,-1))+truth.sum((-2,-1))+1)).mean()
    return error[0]+4*error[1]+2*jerk+bce+dice
