"""CNN jointFlow worker; no LLM or future input."""
import sys,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from train_cnn_flow_baseline_v2 import CNNFlow
R=Path(__file__).resolve().parents[1];P=R/'artifacts/cnn-flow-pilot-v2';torch.set_num_threads(6);h=CNNFlow();h.load_state_dict(torch.load(P/'latest.ckpt',map_location='cpu',weights_only=False)['state_dict']);h.cuda().eval();print(json.dumps({'ready':True}),flush=True)
for line in sys.stdin:
 try:
  a=json.loads(line);assert set(a)=={'images','state'} and len(a['images'])==4 and len(a['state'])==20;images=torch.stack([torch.tensor(np.array(Image.open(p).convert('RGB').resize((448,252)).resize((112,64))),dtype=torch.uint8).permute(2,0,1) for p in a['images']])[None].cuda();st=torch.tensor([a['state']],device='cuda');start=time.perf_counter()
  with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
   f=h.features(images);gen=torch.Generator(device='cuda').manual_seed(9188);x=torch.randn(1,25,2,generator=gen,device='cuda');w=torch.randn(1,16,32,generator=gen,device='cuda')
   for j in range(16):v,vw=h.head(f,st,x,torch.full((1,1),j/16,device='cuda'),w);x=x+v/16;w=w+vw/16
   vel=x[0]*10+torch.tensor([a['state'][0]*30,0],device='cuda');pos=vel.cumsum(0)*.2;torch.cuda.synchronize()
  print(json.dumps({'xy':pos.cpu().tolist(),'velocity':vel.cpu().tolist(),'model_seconds':time.perf_counter()-start}),flush=True)
 except Exception as e:print(json.dumps({'error':repr(e)}),flush=True)
