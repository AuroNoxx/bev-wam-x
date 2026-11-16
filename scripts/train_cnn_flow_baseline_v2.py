"""Non-LLM CNN+Flow baseline, same selected past RGB/state windows as Qwen v2."""
from pathlib import Path
import json,time,io,tarfile,hashlib,sys
import numpy as np
from PIL import Image
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.attention import sdpa_kernel,SDPBackend
from train_qwen_visual_pilot_v2 import Head,Data
from lightning_task import VerifiedCheckpointIO,random_state,restore_random
R=Path(__file__).resolve().parents[1];O=R/'artifacts/cnn-flow-pilot-v2';O.mkdir(exist_ok=True)
SELECTION=R/'artifacts/qwen-visual-pilot-v2/selection.json'
def cache():
 plan=json.loads(SELECTION.read_text());rows=[]
 for route in dict.fromkeys(x['route'] for x in plan):
  selected=[x for x in plan if x['route']==route];needed={k for x in selected for k in x['history']};images={}
  with tarfile.open(R/'data/public/bench2drive/archives'/(route+'.tar.gz'),'r|gz') as tar:
   for z in tar:
    if '/camera/rgb_front/' in z.name and int(Path(z.name).stem) in needed:
     im=Image.open(io.BytesIO(tar.extractfile(z).read())).convert('RGB').resize((448,252)).resize((112,64));images[int(Path(z.name).stem)]=torch.from_numpy(np.array(im).copy()).permute(2,0,1)
  assert needed<=images.keys()
  for x in selected:
   assert max(x['history'])==x['frame']
   with np.load(R/'data/occ_v3'/x['cache_source']) as z:state=torch.tensor(z['state'].copy()).float();trajectory=torch.tensor(z['trajectory'].copy()).float()
   rows.append({'meta':x,'images':torch.stack([images[k] for k in x['history']]),'state':state,'trajectory':trajectory})
  print('CACHED',route,len(selected),flush=True)
 torch.save(rows,O/'rgb_windows.pt');(O/'selection.json').write_text(json.dumps(plan,indent=2))
def add_world_labels():
 rows=torch.load(O/'rgb_windows.pt',weights_only=False)
 if 'world_label' in rows[0]:return
 projection=torch.randn(12,32,generator=torch.Generator().manual_seed(9101))/12**.5
 for route in dict.fromkeys(x['meta']['route'] for x in rows):
  rr=[x for x in rows if x['meta']['route']==route];needed={k for x in rr for k in x['meta']['future_label_frames']};images={}
  with tarfile.open(R/'data/public/bench2drive/archives'/(route+'.tar.gz'),'r|gz') as tar:
   for z in tar:
    if '/camera/rgb_front/' in z.name and int(Path(z.name).stem) in needed:
     im=Image.open(io.BytesIO(tar.extractfile(z).read())).convert('RGB').resize((4,4));images[int(Path(z.name).stem)]=torch.tensor(np.array(im).copy()).float().reshape(4,12)/255
  for x in rr:x['world_label']=torch.cat([images[k] for k in x['meta']['future_label_frames']],0)@projection
  print('WORLD_LABELS',route,flush=True)
 torch.save(rows,O/'rgb_windows.pt')
class CNNFlow(pl.LightningModule):
 def __init__(self):
  super().__init__();self.encoder=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU(),nn.Conv2d(64,128,3,2,1),nn.GELU(),nn.AdaptiveAvgPool2d((2,2)));self.visual=nn.Linear(128,2048);self.head=Head('joint_fm')
 def features(self,images):
  b=len(images);z=self.encoder(images.float().reshape(-1,3,64,112)/255);z=z.flatten(2).transpose(1,2).reshape(b,16,128);return self.visual(z)
 def training_step(self,b,idx):
  images,state,y,wy=b;f=self.features(images);t=torch.rand(len(y),1,device=y.device);noise=torch.randn_like(y);x=(1-t[:,:,None])*noise+t[:,:,None]*y;wn=torch.randn_like(wy);w=(1-t[:,:,None])*wn+t[:,:,None]*wy;out,pw=self.head(f,state,x,t,w);loss=(out-(y-noise)).square().mean()+.1*(pw-(wy-wn)).square().mean();self.log('train_loss',loss,on_step=True);self.last_loss=float(loss.detach().cpu())
  if (self.global_step+1)%128==0:print('PROGRESS',self.global_step+1,float(loss.detach()),flush=True)
  return loss
 def configure_optimizers(self):return torch.optim.AdamW((p for p in self.parameters() if p.requires_grad),lr=1e-4,weight_decay=.01)
 def on_save_checkpoint(self,c):c['task_state']=random_state();c['selection_sha']=hashlib.sha256(SELECTION.read_bytes()).hexdigest()
 def on_load_checkpoint(self,c):restore_random(c['task_state'])
def main():
 torch.set_num_threads(6)
 if not (O/'rgb_windows.pt').exists():cache()
 add_world_labels()
 if '--cache-only' in sys.argv:return
 rows=torch.load(O/'rgb_windows.pt',weights_only=False);train=[x for x in rows if x['meta']['split']=='train'];val=[x for x in rows if x['meta']['split']=='validation'];assert not {x['meta']['route'] for x in train}&{x['meta']['route'] for x in val}
 def pack(rr):
  im=torch.stack([x['images'] for x in rr]);st=torch.stack([x['state'] for x in rr]);pos=torch.stack([x['trajectory'][:,:2]*torch.tensor([50.,10.]) for x in rr]);v=torch.diff(torch.cat([torch.zeros(len(rr),1,2),pos],1),dim=1)/.2;y=(v-torch.stack([st[:,0]*30,torch.zeros(len(rr))],-1)[:,None,:])/10;wy=torch.stack([x['world_label'] for x in rr]);return im,st,y,pos,wy
 ti,ts,ty,tp,tw=pack(train);vi,vs,vy,vp,vw=pack(val);mu=tw.mean((0,1));scale=tw.std((0,1)).clamp_min(.1);tw=(tw-mu)/scale;vw=(vw-mu)/scale;torch.save({'mu':mu,'scale':scale},O/'world_normalization.pt');pl.seed_everything(9101);task=CNNFlow();cb=ModelCheckpoint(dirpath=O,filename='latest',enable_version_counter=False,save_top_k=1,every_n_train_steps=512,save_on_train_epoch_end=False)
 tr=pl.Trainer(accelerator='gpu',devices=1,precision='32-true',max_steps=2048,max_epochs=-1,callbacks=[cb],plugins=[VerifiedCheckpointIO()],logger=TensorBoardLogger(str(R/'logs/tensorboard'),name='cnn-flow-pilot-v2'),limit_val_batches=0,num_sanity_val_steps=0,gradient_clip_val=1,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=16)
 torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 with sdpa_kernel(SDPBackend.MATH):tr.fit(task,datamodule=Data(torch.utils.data.TensorDataset(ti,ts,ty,tw)))
 elapsed=time.perf_counter()-start;task.cuda().eval();gen=torch.Generator(device='cuda').manual_seed(9188);x=torch.randn(vy.shape,generator=gen,device='cuda');w=torch.randn(len(val),16,32,generator=gen,device='cuda')
 with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
  f=task.features(vi.cuda());st=vs.cuda()
  for j in range(16):a,b=task.head(f,st,x,torch.full((len(val),1),j/16,device='cuda'),w);x=x+a/16;w=w+b/16
 velocity=x.cpu()*10+torch.stack([vs[:,0]*30,torch.zeros(len(vs))],-1)[:,None,:];pos=velocity.cumsum(1)*.2;err=torch.linalg.vector_norm(pos-vp,dim=-1);jerk=torch.diff(velocity,n=2,dim=1)/.04;truthv=vy*10+torch.stack([vs[:,0]*30,torch.zeros(len(vs))],-1)[:,None,:];cv=vs[:,0,None,None]*30*torch.arange(1,26)[None,:,None]*.2*torch.tensor([1.,0.])
 result={'n_train':len(train),'n_val':len(val),'steps':tr.global_step,'batch':32,'seed':9101,'lr':1e-4,'parameters_total':sum(p.numel() for p in task.parameters()),'parameters_trainable':sum(p.numel() for p in task.parameters() if p.requires_grad),'head':'same128width2layers4heads512FFN jointFM; world loss weight0.1','visual':'random-init jointly trained3conv CNN,4past RGB448x252 resized112x64;16tokens projected2048','world_MSE':float((w.cpu()-vw).square().mean()),'world_target':'future4frames each4x4 RGB,16x12 tokens fixed12to32 projection,train-only normalization; differs from Qwen future latent target','LLM':False,'future_inputs':False,'selection_sha':hashlib.sha256(SELECTION.read_bytes()).hexdigest(),'last_train_loss':task.last_loss,'train_seconds':elapsed,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'ADE_m':float(err.mean()),'endpoint_1_2_3_5s_m':err[:,[4,9,14,24]].mean(0).tolist(),'speed_MAE_mps':float(abs(torch.linalg.vector_norm(velocity,dim=-1)-torch.linalg.vector_norm(truthv,dim=-1)).mean()),'jerk_mean_mps3':float(torch.linalg.vector_norm(jerk,dim=-1).mean()),'CV_ADE_m':float(torch.linalg.vector_norm(cv-vp,dim=-1).mean()),'caution':'not a controlled LLM-only ablation: CNN resolution/learning/world targets differ; no closed-loop safety inference'}
 (O/'results.json').write_text(json.dumps(result,indent=2));torch.save({'predicted_xy':pos,'truth_xy':vp,'meta':[x['meta'] for x in val]},O/'validation_predictions.pt');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
