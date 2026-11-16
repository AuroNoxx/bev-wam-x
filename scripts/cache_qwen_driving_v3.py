"""All available valid windows, frozen Qwen, resumable per-route feature cache."""
from pathlib import Path
import json,time,tarfile,io,os
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from torch.nn.attention import sdpa_kernel,SDPBackend
R=Path(__file__).resolve().parents[1];O=R/'artifacts/qwen-driving-v3';O.mkdir(exist_ok=True);C=O/'cache';C.mkdir(exist_ok=True)
m=json.loads((R/'data/occ_v3/manifest.json').read_text());routes=[r for r in m['routes'] if r['split'] in ['train','validation']];plan=[]
for r in routes:
 for n in r['samples']:
  i=int(Path(n).stem);plan.append(dict(route=r['route'],split=r['split'],family=r['family'],frame=i,history=[i-6,i-4,i-2,i],future_label_frames=[i+14,i+16,i+18,i+20],cache_source=n))
(O/'selection.json').write_text(json.dumps(plan,indent=2));reuse={}
for x in torch.load(R/'artifacts/qwen-visual-pilot-v2/visual_features.pt',weights_only=False):reuse[(x['meta']['route'],x['meta']['frame'])]=x
p=R/'models/Qwen3-VL-2B-Instruct';pr=AutoProcessor.from_pretrained(p,local_files_only=True);model=Qwen3VLForConditionalGeneration.from_pretrained(p,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.requires_grad_(False);torch.set_num_threads(6);start=time.perf_counter();done=0
@torch.inference_mode()
def features(groups):
 b=pr(text=['<|vision_start|><|image_pad|><|vision_end|>'*4]*len(groups),images=[im for g in groups for im in g],return_tensors='pt',padding=True).to('cuda')
 with sdpa_kernel(SDPBackend.MATH):h=model.model(**b,use_cache=False).last_hidden_state.float()
 return torch.nn.functional.adaptive_avg_pool1d(h.transpose(1,2),16).transpose(1,2).cpu()
for route in [r['route'] for r in routes]:
 target=C/(route+'.pt');pp=[x for x in plan if x['route']==route]
 if target.exists():done+=len(pp);continue
 needed={k for x in pp if (route,x['frame']) not in reuse for k in x['history']+x['future_label_frames']};images={}
 with tarfile.open(R/'data/public/bench2drive/archives'/(route+'.tar.gz'),'r|gz') as tar:
  for z in tar:
   if '/camera/rgb_front/' in z.name and int(Path(z.name).stem) in needed:images[int(Path(z.name).stem)]=Image.open(io.BytesIO(tar.extractfile(z).read())).convert('RGB').resize((448,252))
 assert needed<=images.keys();rows=[reuse[(route,x['frame'])] for x in pp if (route,x['frame']) in reuse];fresh=[x for x in pp if (route,x['frame']) not in reuse]
 for offset in range(0,len(fresh),4):
  xx=fresh[offset:offset+4];groups=[[images[k] for k in x['history']] for x in xx]+[[images[k] for k in x['future_label_frames']] for x in xx];f=features(groups)
  assert torch.isfinite(f).all()
  for j,x in enumerate(xx):
   with np.load(R/'data/occ_v3'/x['cache_source']) as z:st=torch.tensor(z['state'].copy()).float();tr=torch.tensor(z['trajectory'].copy()).float()
   rows.append(dict(meta=x,features=f[j],future_features=f[j+len(xx)],state=st,trajectory=tr))
 rows.sort(key=lambda x:x['meta']['frame']);torch.save(rows,target);done+=len(rows);info=dict(phase='caching',completed=done,total=len(plan),elapsed_seconds=time.perf_counter()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,route=route);(O/'live_progress.json').write_text(json.dumps(info,indent=2));print('CACHE',info,flush=True)
print('CACHE_COMPLETE',done,flush=True)
