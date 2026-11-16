import sys,json,time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from torch.nn.attention import sdpa_kernel,SDPBackend
from qwen_driving_v3_head import Head
R=Path(__file__).resolve().parents[1];M=R/'models/Qwen3-VL-2B-Instruct';P=R/'artifacts/qwen-driving-v4'
torch.set_num_threads(6);pr=AutoProcessor.from_pretrained(M,local_files_only=True);q=Qwen3VLForConditionalGeneration.from_pretrained(M,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();q.requires_grad_(False);h=Head('joint_fm');h.load_state_dict(torch.load(P/'weights/best.ckpt',map_location='cpu',weights_only=False)['state_dict']);h.cuda().eval();print(json.dumps({'ready':True}),flush=True)
for line in sys.stdin:
 try:
  a=json.loads(line);assert set(a)=={'images','state'} and len(a['images'])==4 and len(a['state'])==20;ims=[Image.open(p).convert('RGB').resize((448,252)) for p in a['images']];b=pr(text=['<|vision_start|><|image_pad|><|vision_end|>'*4],images=ims,return_tensors='pt').to('cuda');st=torch.tensor([a['state']],device='cuda');start=time.perf_counter()
  with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
   z=q.model(**b,use_cache=False).last_hidden_state.float();f=torch.nn.functional.adaptive_avg_pool1d(z.transpose(1,2),16).transpose(1,2);gen=torch.Generator(device='cuda').manual_seed(9188);x=torch.randn(1,25,2,generator=gen,device='cuda');w=torch.randn(1,16,32,generator=gen,device='cuda')
   for j in range(16):v,vw=h(f,st,x,torch.full((1,1),j/16,device='cuda'),w);x=x+v/16;w=w+vw/16
   vel=x[0]*10+torch.tensor([a['state'][0]*30,0],device='cuda');pos=vel.cumsum(0)*.2;torch.cuda.synchronize()
  print(json.dumps({'xy':pos.cpu().tolist(),'velocity':vel.cpu().tolist(),'model_seconds':time.perf_counter()-start}),flush=True)
 except Exception as e:print(json.dumps({'error':repr(e)}),flush=True)
