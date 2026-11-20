"""PyTorch deployment optimization for the two frozen ResNet branches."""
import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchvision.models.resnet import BasicBlock

from .legacy_fast_model import FastDriver
from .sensors import CAL_ID


def load_driver(path):
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    if checkpoint["task_state"]["calibration"]!=CAL_ID:raise ValueError("sensor calibration mismatch")
    driver=FastDriver(corrective=checkpoint["task_state"].get("corrective",False),
                      lane_supervision=checkpoint["task_state"].get("lane_supervision",False),
                      film=checkpoint["task_state"].get("film",False),
                      spatial=checkpoint["task_state"].get("spatial",False),
                      navigation=checkpoint["task_state"].get("navigation",False),
                      road_roi=checkpoint["task_state"].get("road_roi",False))
    driver.load_state_dict({k.removeprefix("driver."):v for k,v in checkpoint["state_dict"].items()})
    return driver.eval(),int(checkpoint["global_step"]),checkpoint["task_state"]["identity"]


def fold_batch_norm(driver):
    if driver.training:raise ValueError("BatchNorm folding requires evaluation statistics")
    pairs=0
    for encoder in (driver.rgb,driver.bev):
        encoder[0]=fuse_conv_bn_eval(encoder[0],encoder[1])
        encoder[1]=nn.Identity()
        pairs+=1
        for block in list(encoder.modules()):
            if not isinstance(block,BasicBlock):continue
            block.conv1=fuse_conv_bn_eval(block.conv1,block.bn1)
            block.conv2=fuse_conv_bn_eval(block.conv2,block.bn2)
            block.bn1=nn.Identity()
            block.bn2=nn.Identity()
            pairs+=2
            if block.downsample is not None:
                block.downsample[0]=fuse_conv_bn_eval(block.downsample[0],block.downsample[1])
                block.downsample[1]=nn.Identity()
                pairs+=1
    return pairs


class GraphDriver:
    """Replay identical PyTorch operators through a single fixed-shape ROCm graph."""
    def __init__(self,model,state_width=3):
        self.inputs=(torch.zeros((1,6,360,640),dtype=torch.uint8,device="cuda"),
                     torch.zeros((1,3,128,128),dtype=torch.uint8,device="cuda"),
                     torch.zeros((1,state_width),device="cuda"),torch.zeros((1,4),device="cuda"))
        current=torch.cuda.current_stream()
        warm=torch.cuda.Stream()
        warm.wait_stream(current)
        with torch.cuda.stream(warm):
            for _ in range(3):model(*self.inputs)
        current.wait_stream(warm)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outputs=model(*self.inputs)

    def __call__(self,*obs):
        for static,value in zip(self.inputs,obs):static.copy_(value)
        self.graph.replay()
        return self.outputs
