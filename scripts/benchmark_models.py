"""Small OCC planning models for a measured training-cost pilot, not a driving claim."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights


class Residual(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(c,c,3,padding=1,bias=False),nn.GroupNorm(8,c),nn.SiLU(),
                               nn.Conv2d(c,c,3,padding=1,bias=False),nn.GroupNorm(8,c))
    def forward(self,x): return F.silu(x+self.net(x))


class OccEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(7,4)
        self.net=nn.Sequential(nn.Conv2d(64,64,5,stride=2,padding=2),nn.GroupNorm(8,64),nn.SiLU(),Residual(64),
                               nn.Conv2d(64,128,3,stride=2,padding=1),nn.GroupNorm(8,128),nn.SiLU(),Residual(128),
                               nn.Conv2d(128,256,3,padding=1),nn.GroupNorm(8,256),nn.SiLU(),
                               Residual(256),Residual(256),Residual(256))
    def forward(self,x):
        x=torch.where(x==255,6,x).long()
        x=self.embedding(x).permute(0,1,2,5,3,4).flatten(1,3)
        return self.net(x)


class FlowPlanner(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=OccEncoder()
        self.state=nn.Linear(6,256)
        self.time=nn.Sequential(nn.Linear(1,256),nn.SiLU(),nn.Linear(256,256))
        self.trajectory=nn.Linear(5,256)
        self.position=nn.Parameter(torch.randn(1,25,256)*.02)
        layer=nn.TransformerEncoderLayer(256,8,1024,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,4,enable_nested_tensor=False)
        self.output=nn.Linear(256,5)
    def forward(self,batch,xt,t):
        context=self.encoder(batch['history']).mean((-1,-2))+self.state(batch['state'])+self.time(t)
        return self.output(self.transformer(self.trajectory(xt)+self.position+context[:,None]))


class ActionWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=OccEncoder()
        self.condition=nn.Sequential(nn.Linear(25*5+25*3+6,512),nn.SiLU(),nn.Linear(512,256))
        self.future_embedding=nn.Parameter(torch.randn(5,256)*.02)
        self.decoder=nn.Sequential(Residual(256),nn.ConvTranspose2d(256,128,4,2,1),nn.GroupNorm(8,128),nn.SiLU(),
                                   Residual(128),nn.ConvTranspose2d(128,64,4,2,1),nn.GroupNorm(8,64),nn.SiLU(),
                                   Residual(64),nn.Conv2d(64,4*6,1))
    def forward(self,batch):
        latent=self.encoder(batch['history'])
        condition=self.condition(torch.cat([batch['trajectory'].flatten(1),batch['controls'].flatten(1),batch['state']],1))
        x=latent[:,None]+condition[:,None,:,None,None]+self.future_embedding[None,:,:,None,None]
        x=self.decoder(x.flatten(0,1))
        return x.reshape(len(latent),5,4,6,128,128)


class VisualOcc(nn.Module):
    def __init__(self):
        super().__init__()
        pretrained=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Frozen eval BatchNorm is exactly foldable into its preceding conv.
        # Avoid the Windows ROCm BF16 MIOpen BatchNorm inference failure.
        pretrained.eval()
        fuse=torch.nn.utils.fuse_conv_bn_eval
        pretrained.conv1=fuse(pretrained.conv1,pretrained.bn1)
        pretrained.bn1=nn.Identity()
        for stage in [pretrained.layer1,pretrained.layer2,pretrained.layer3,pretrained.layer4]:
            for block in stage:
                block.conv1=fuse(block.conv1,block.bn1); block.bn1=nn.Identity()
                block.conv2=fuse(block.conv2,block.bn2); block.bn2=nn.Identity()
                if block.downsample is not None:
                    block.downsample=nn.Sequential(fuse(block.downsample[0],block.downsample[1]))
        self.backbone=nn.Sequential(*list(pretrained.children())[:7])
        self.backbone.requires_grad_(False)
        self.adapter=nn.Conv2d(256,32,1)
        self.decoder=nn.Sequential(nn.Conv2d(128,128,3,padding=1),nn.GroupNorm(8,128),nn.SiLU(),
                                   Residual(128),Residual(128),nn.Conv2d(128,24,1))
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
    def forward(self,batch):
        images=batch['rgb'].permute(0,1,4,2,3).flatten(0,1).float()/255
        self.backbone.eval()
        with torch.no_grad(): features=self.backbone((images-self.mean)/self.std)
        features=self.adapter(features)
        b,cameras=batch['rgb'].shape[:2]
        grid=batch['projection'].flatten(0,1).unsqueeze(2)
        samples=F.grid_sample(features,grid.to(features.dtype),align_corners=True,padding_mode='zeros').squeeze(-1)
        samples=samples.reshape(b,cameras,32,4*128*128)
        mask=batch['visible'][:,:,None].to(samples.dtype)
        merged=(samples*mask).sum(1)/mask.sum(1).clamp_min(1)
        merged=merged.reshape(b,32,4,128,128).flatten(1,2)
        return self.decoder(merged).reshape(b,4,6,128,128)


def compute_loss(mode,model,batch):
    if mode=='flow':
        target=batch['trajectory']
        noise=torch.randn_like(target)
        t=torch.rand(len(target),1,device=target.device)
        xt=(1-t[:,:,None])*noise+t[:,:,None]*target
        predicted=model(batch,xt,t)
        return F.mse_loss(predicted.float(),target-noise)
    logits=model(batch)
    if mode=='world':
        target=batch['future'].long()
        # Equal treatment of frames; preserve invalid labels where supplied.
        return F.cross_entropy(logits.flatten(0,2).float(),target.flatten(0,2),ignore_index=255)
    target=batch['current'].long().clone()
    visible=batch['visible'].any(1).reshape_as(target)
    target[~visible]=255
    return F.cross_entropy(logits.flatten(0,1).float(),target.flatten(0,1),ignore_index=255)
