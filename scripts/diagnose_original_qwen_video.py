import json,time,hashlib
from pathlib import Path
import torch
import numpy as np
from transformers.video_utils import VideoMetadata
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from torch.nn.attention import sdpa_kernel,SDPBackend
R=Path(__file__).resolve().parents[1]; O=R/'artifacts/qwen-original-video-diagnostic';O.mkdir(exist_ok=True)
cases=[x for x in json.loads((R/'artifacts/qwen-original-history-diagnostic/inputs.json').read_text()) if x['id'].startswith(('follow','brake'))]
(O/'inputs.json').write_text(json.dumps(cases,indent=2))
torch.set_num_threads(6); M=R/'models/Qwen3-VL-2B-Instruct'; p=AutoProcessor.from_pretrained(M,local_files_only=True);q=Qwen3VLForConditionalGeneration.from_pretrained(M,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();q.requires_grad_(False)
results=[]
for c in cases:
 ims=[Image.open(x).convert('RGB').resize((448,252)) for x in c['images']]
 msgs=[{'role':'user','content':[{'type':'video'}]+[{'type':'text','text':c['prompt']}]}]
 text=p.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True);indices=[f-c['frame_ids'][0] for f in c['frame_ids']]; metadata=VideoMetadata(total_num_frames=indices[-1]+1,fps=20.0,frames_indices=indices,width=448,height=252); b=p(text=[text],videos=[np.stack([np.array(im) for im in ims])],video_metadata=[metadata],do_sample_frames=False,return_tensors='pt').to('cuda');start=time.perf_counter()
 with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
  y=q.generate(**b,max_new_tokens=180,do_sample=False,repetition_penalty=1.1,use_cache=True);torch.cuda.synchronize()
 out=p.batch_decode(y[:,b['input_ids'].shape[1]:],skip_special_tokens=True)[0]
 v={'id':c['id'],'output':out,'seconds':time.perf_counter()-start,'input_tokens':int(b['input_ids'].shape[1]),'output_tokens':int(y.shape[1]-b['input_ids'].shape[1]),'rendered_prompt':text,'processed_text':p.tokenizer.decode(b['input_ids'][0]),'video_metadata':dict(metadata)};results.append(v);(O/'outputs.json').write_text(json.dumps(results,indent=2));print(json.dumps(v),flush=True)
(O/'completed.json').write_text(json.dumps({'cases':len(results),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'training':False,'weights_modified':False,'note':'Original pretrained text generation; added text interface differs from frozen hidden-feature driving head.'},indent=2))
