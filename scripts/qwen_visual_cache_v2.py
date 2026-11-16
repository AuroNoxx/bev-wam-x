"""Bounded visual pilot cache. Future image features are label-only."""
from pathlib import Path
import os,json,io,tarfile,time,gzip,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];os.environ['HF_HOME']=str(R/'.cache/huggingface')
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from torch.nn.attention import sdpa_kernel,SDPBackend
O=R/'artifacts/qwen-visual-pilot-v2';O.mkdir(exist_ok=True)
m=json.loads((R/'data/occ_v3/manifest.json').read_text());chosen=[]
chosen=[r for r in m['routes'] if r['split'] in ['train','validation']]
plan=[]
for r in chosen:
 names=r['samples'];sel=sorted(set(np.linspace(0,len(names)-1,min(8,len(names)),dtype=int).tolist()))
 for j in sel:
  i=int(Path(names[j]).stem);plan.append({'route':r['route'],'split':r['split'],'family':r['family'],'frame':i,'history':[i-6,i-4,i-2,i],'future_label_frames':[i+14,i+16,i+18,i+20],'cache_source':names[j]})
(O/'selection.json').write_text(json.dumps(plan,indent=2))
path=R/'models/Qwen3-VL-2B-Instruct';pr=AutoProcessor.from_pretrained(path,local_files_only=True);model=Qwen3VLForConditionalGeneration.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();model.requires_grad_(False)
rows=[];start=time.perf_counter()
@torch.inference_mode()
def features(ims):
 b=pr(text=[''.join('<|vision_start|><|image_pad|><|vision_end|>' for _ in ims)],images=ims,return_tensors='pt').to('cuda')
 with sdpa_kernel(SDPBackend.MATH):h=model.model(**b,use_cache=False).last_hidden_state.float()
 # preserve ordered 16 coarse token regions; deterministic pooling not a trained adapter
 return torch.nn.functional.adaptive_avg_pool1d(h.transpose(1,2),16).transpose(1,2)[0].cpu()
for route in [r['route'] for r in chosen]:
 pp=[x for x in plan if x['route']==route];needed=set(k for x in pp for k in x['history']+x['future_label_frames']);images={}
 with tarfile.open(R/'data/public/bench2drive/archives'/(route+'.tar.gz'),'r|gz') as tar:
  for z in tar:
   if '/camera/rgb_front/' in z.name and int(Path(z.name).stem) in needed:
    images[int(Path(z.name).stem)]=Image.open(io.BytesIO(tar.extractfile(z).read())).convert('RGB').resize((448,252))
 assert needed<=images.keys()
 for x in pp:
  f=features([images[k] for k in x['history']]);future=features([images[k] for k in x['future_label_frames']]);assert torch.isfinite(f).all() and torch.isfinite(future).all()
  with np.load(R/'data/occ_v3'/x['cache_source']) as z:
   state=torch.tensor(z['state'].copy()).float();trajectory=torch.tensor(z['trajectory'].copy()).float()
  # Existing state[:8]: ego kinematics and near/far navigation; never teacher policy fields.
  rows.append({'meta':x,'features':f,'future_features':future,'state':state,'trajectory':trajectory})
 print('CACHED',route,len(pp),'elapsed',time.perf_counter()-start,flush=True)
torch.save(rows,O/'visual_features.pt');print('CACHE_DONE',len(rows),time.perf_counter()-start,flush=True)
