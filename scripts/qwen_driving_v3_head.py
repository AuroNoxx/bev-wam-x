"""Small cached-Qwen trajectory/world heads, Lightning-owned training, no deployment."""
from pathlib import Path
import json,time,random,hashlib
import numpy as np
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel,SDPBackend
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchdata.stateful_dataloader import StatefulDataLoader
from lightning_task import VerifiedCheckpointIO,random_state,restore_random
R=Path(__file__).resolve().parents[1];O=R/'artifacts/qwen-driving-v3';O.mkdir(exist_ok=True)
class Head(pl.LightningModule):
 def __init__(self,method):
  super().__init__();self.method=method;self.proj=nn.Linear(2048,256);self.state=nn.Linear(20,256);self.action=nn.Linear(2,256);self.world=nn.Linear(32,256);self.time=nn.Linear(1,256);self.pos=nn.Parameter(torch.randn(1,58,256)*.01);self.tr=nn.TransformerEncoder(nn.TransformerEncoderLayer(256,8,1024,dropout=0,batch_first=True,norm_first=True),4,enable_nested_tensor=False);self.ao=nn.Linear(256,2);self.wo=nn.Linear(256,32)
 def forward(self,f,state,x,t,w):
  a=self.action(x);tokens=[self.proj(f),self.state(state).unsqueeze(1),a]
  if self.method=='joint_fm':tokens.append(self.world(w))
  z=torch.cat(tokens,1);z=self.tr(z+self.pos[:,:z.shape[1]]+self.time(t).unsqueeze(1));return self.ao(z[:,17:42]),self.wo(z[:,42:]) if self.method=='joint_fm' else w*0
 def training_step(self,b,idx):
  f,st,y,wy=b; t=torch.rand(len(y),1,device=y.device);noise=torch.randn_like(y);wn=torch.randn_like(wy)
  if self.method=='regression':x=torch.zeros_like(y);t=t*0;target=y
  else:x=(1-t[:,:,None])*noise+t[:,:,None]*y;target=y-noise
  w=(1-t[:,:,None])*wn+t[:,:,None]*wy;a,v=self(f,st,x,t,w);loss=(a-target).square().mean()
  if self.method=='joint_fm':loss=loss+.1*(v-(wy-wn)).square().mean()
  self.log('train_loss',loss,on_step=True);return loss
 def configure_optimizers(self):return torch.optim.AdamW(self.parameters(),lr=1e-4,weight_decay=.01)
