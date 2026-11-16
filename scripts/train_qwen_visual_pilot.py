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
R=Path(__file__).resolve().parents[1];O=R/'artifacts/qwen-visual-pilot-v1'
class Head(pl.LightningModule):
 def __init__(self,method):
  super().__init__();self.method=method;self.proj=nn.Linear(2048,128);self.state=nn.Linear(20,128);self.action=nn.Linear(2,128);self.world=nn.Linear(32,128);self.time=nn.Linear(1,128);self.pos=nn.Parameter(torch.randn(1,58,128)*.01);self.tr=nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,512,dropout=0,batch_first=True,norm_first=True),2,enable_nested_tensor=False);self.ao=nn.Linear(128,2);self.wo=nn.Linear(128,32)
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
 def on_save_checkpoint(self,c):c['task_state']=random_state();c['contract']={'method':self.method,'cache_sha':hashlib.sha256((O/'visual_features.pt').read_bytes()).hexdigest(),'resume':'StatefulDataLoader state saved; full resumed trajectory not yet tested'}
 def on_load_checkpoint(self,c):restore_random(c['task_state'])
class Data(pl.LightningDataModule):
 def __init__(self,ds):super().__init__();self.loader=StatefulDataLoader(ds,batch_size=8,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(9101))
 def train_dataloader(self):return self.loader
 def state_dict(self):return {'loader':self.loader.state_dict()}
 def load_state_dict(self,s):self.loader.load_state_dict(s['loader'])
def main():
 torch.set_num_threads(6);rows=torch.load(O/'visual_features.pt',weights_only=False);g=torch.Generator().manual_seed(9101);projection=torch.randn(2048,32,generator=g)/2048**.5
 train=[r for r in rows if r['meta']['split']=='train'];val=[r for r in rows if r['meta']['split']=='validation'];assert not set(x['meta']['route'] for x in train)&set(x['meta']['route'] for x in val)
 def pack(rr):
  f=torch.stack([x['features'] for x in rr]);state=torch.stack([x['state'] for x in rr]);pos=torch.stack([x['trajectory'][:,:2]*torch.tensor([50.,10.]) for x in rr]);velocity=torch.diff(torch.cat([torch.zeros(len(rr),1,2),pos],1),dim=1)/.2; residual=(velocity-torch.stack([state[:,0]*30,torch.zeros(len(rr))],-1)[:,None,:])/10
  world=torch.stack([x['future_features'] for x in rr])@projection;return f,state,residual,world,pos
 tf,ts,ty,tw,tp=pack(train);vf,vs,vy,vw,vp=pack(val);mu=tw.mean((0,1));scale=tw.std((0,1)).clamp_min(.1);tw=(tw-mu)/scale;vw=(vw-mu)/scale;persist=(vf@projection-mu)/scale
 results={'n_train':len(train),'n_val':len(val),'steps':256,'batch':8,'seed':9101,'front_only':True,'history_frames':4,'world_target':'future2s frozen visual feature 16x32 random projection; no future OCC','caution':'tiny chain probe not full train or driving safety; one seed; future labels do not enter actor','world_persistence_mse':float((persist-vw).square().mean()),'arms':{}}
 cv=vs[:,0,None,None]*30*torch.arange(1,26)[None,:,None]*.2*torch.tensor([1.,0.]);results['CV_ADE_m']=float(torch.linalg.vector_norm(cv-vp,dim=-1).mean())
 initial=None
 for method in ['regression','fm','joint_fm']:
  pl.seed_everything(9101);task=Head(method)
  if initial is None:initial={k:v.clone() for k,v in task.state_dict().items()}
  else:task.load_state_dict(initial)
  # Causal API: explicit past features/state/noisy states only, no future targets accepted.
  task.eval();f0=vf[:2];s0=vs[:2];x0=torch.zeros(2,25,2);t0=torch.zeros(2,1);w0=torch.zeros(2,16,32)
  with torch.no_grad():
   before=task(f0,s0,x0,t0,w0)[0];poison=vw[:2]*float('nan');after=task(f0,s0,x0,t0,w0)[0];assert torch.equal(before,after) and torch.isfinite(after).all()
  folder=O/method;folder.mkdir(exist_ok=True);(folder/'preflight.json').write_text(json.dumps({'future_label_poison_output_unchanged':True,'future_target_not_in_actor_signature':True,'note':'single-forward API invariant, not source RGB timestamp proof'}));cb=ModelCheckpoint(dirpath=folder,filename='latest',enable_version_counter=False,save_top_k=1,every_n_train_steps=64,save_on_train_epoch_end=False)
  tr=pl.Trainer(accelerator='gpu',devices=1,precision='32-true',max_steps=256,max_epochs=-1,callbacks=[cb],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(R/'logs/tensorboard'),name='qwen-visual-pilot',version=method),limit_val_batches=0,num_sanity_val_steps=0,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=16)
  torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  with sdpa_kernel(SDPBackend.MATH):tr.fit(task,datamodule=Data(torch.utils.data.TensorDataset(tf,ts,ty,tw)))
  elapsed=time.perf_counter()-start;task=task.cuda().eval();f=vf.cuda();st=vs.cuda();gen=torch.Generator(device='cuda').manual_seed(9188);x=torch.randn(vy.shape,generator=gen,device='cuda');w=torch.randn(vw.shape,generator=gen,device='cuda')
  with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
   if method=='regression':x,_=task(f,st,torch.zeros_like(x),torch.zeros(len(x),1,device='cuda'),torch.zeros_like(w))
   else:
    for j in range(16):a,b=task(f,st,x,torch.full((len(x),1),j/16,device='cuda'),w);x=x+a/16;w=w+b/16
  velocity=x.cpu()*10+torch.stack([vs[:,0]*30,torch.zeros(len(vs))],-1)[:,None,:];pos=velocity.cumsum(1)*.2;err=torch.linalg.vector_norm(pos-vp,dim=-1);acc=torch.diff(velocity,dim=1)/.2;jerk=torch.diff(acc,dim=1)/.2
  results['arms'][method]={'parameters':sum(p.numel() for p in task.parameters()),'steps':tr.global_step,'future_label_poison_passed':True,'train_seconds':elapsed,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'ADE_m':float(err.mean()),'endpoint_1_2_3_5s_m':err[:,[4,9,14,24]].mean(0).tolist(),'speed_MAE_mps':float(abs(torch.linalg.vector_norm(velocity,dim=-1)-torch.linalg.vector_norm(vy*10+torch.stack([vs[:,0]*30,torch.zeros(len(vs))],-1)[:,None,:],dim=-1)).mean()),'jerk_mean_mps3':float(torch.linalg.vector_norm(jerk,dim=-1).mean()),'world_MSE':float((w.cpu()-vw).square().mean()) if method=='joint_fm' else None}
  (O/'results.json').write_text(json.dumps(results,indent=2));print(method,results['arms'][method],flush=True);del task,tr;torch.cuda.empty_cache()
if __name__=='__main__':main()
