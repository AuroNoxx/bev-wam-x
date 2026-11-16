"""Frozen backbone resource probe; no text generation, no training, train route only."""
from pathlib import Path
import os,json,io,tarfile,time,hashlib
R=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(R/'.cache/huggingface')
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from torch.nn.attention import sdpa_kernel,SDPBackend
O=R/'artifacts/qwen-visual-pilot-v1';O.mkdir(exist_ok=True)
modelpath=R/'models/Qwen3-VL-2B-Instruct'
route='HardBreakRoute_Town12_Route22647_Weather20_02-09-07-52-05'
images={}
with tarfile.open(R/'data/public/bench2drive/archives'/(route+'.tar.gz'),'r|gz') as t:
 for m in t:
  if '/camera/rgb_front/' in m.name and int(Path(m.name).stem) in [144,146,148,150]:
   im=Image.open(io.BytesIO(t.extractfile(m).read())).convert('RGB');images[int(Path(m.name).stem)]=im.resize((448,252))
assert len(images)==4
processor=AutoProcessor.from_pretrained(modelpath,local_files_only=True)
text=''.join('<|vision_start|><|image_pad|><|vision_end|>' for _ in images)
batch=processor(text=[text],images=[images[i] for i in sorted(images)],return_tensors='pt').to('cuda')
start=time.perf_counter();model=Qwen3VLForConditionalGeneration.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();model.requires_grad_(False)
load_s=time.perf_counter()-start
measure=[];torch.cuda.reset_peak_memory_stats()
with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
 for rep in range(3):
  torch.cuda.synchronize();st=time.perf_counter();out=model.model(**batch,use_cache=False);torch.cuda.synchronize();measure.append(time.perf_counter()-st)
  hidden=out.last_hidden_state;assert torch.isfinite(hidden).all()
result={'source':'ModelScope Qwen/Qwen3-VL-2B-Instruct','parameters':sum(p.numel() for p in model.parameters()),'dtype':'bfloat16','attention':'SDPA MATH portability baseline','image_size_before_processor':[448,252],'history_frames':4,'input_tokens':batch['input_ids'].shape[-1],'image_grid_thw':batch['image_grid_thw'].tolist(),'hidden_shape':list(hidden.shape),'load_seconds':load_s,'forward_seconds':measure,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'text_generation':False,'trainable_backbone_parameters':0,'limitations':'front-only; no driving quality claim; filename pairing not timestamp certificate'}
(O/'resource_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
