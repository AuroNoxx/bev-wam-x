import json,time,hashlib
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from torch.nn.attention import sdpa_kernel,SDPBackend
R=Path(__file__).resolve().parents[1]; O=R/'artifacts/qwen-original-history-diagnostic';O.mkdir(exist_ok=True)
cases=[]
for scene in ['follow','brake','lanechange']:
 for repeat in [0,1]:
  d=R/f'artifacts/qwen-driving-v4/scenarios/{scene}-{repeat}-model'; rows=json.loads((d/'steps.json').read_text());row=rows[30];st=row['state20']
  decoded={'ego_speed_m_s':st[0]*30,'ego_accel_forward_m_s2':st[1]*10,'ego_accel_left_m_s2':st[2]*10,'ego_yaw_rate_left_rad_s':st[3],'nav_near_forward_left_m':[st[4]*50,st[5]*50],'nav_far_forward_left_m':[st[6]*50,st[7]*50],'nav_command':'change lane left' if st[12] else 'follow lane'}
  for span,offsets in [('0.6s',[12,8,4,0]),('3s',[60,40,20,0])]:
   ids=[row['frame']-x for x in offsets];paths=[d/f'rgb-{i}.png' for i in ids];assert all(p.exists() for p in paths)
   times=[-x*.05 for x in offsets]
   prompt='Four chronological front-camera images at relative seconds '+str(times)+'. Coordinates x forward,y left. The following measurements are ONLY the EGO car, never the vehicle seen ahead: '+json.dumps(decoded)+'. In four short lines state: visible vehicles; road direction; LEAD vehicle moving/stopped/unknown and visual evidence; recommended ego steering and speed change. Do not invent distances or speed. Say unknown if motion cannot be distinguished from ego motion.'
   cases.append({'id':f'{scene}-repeat{repeat}-{span}','images':[str(p) for p in paths],'frame_ids':ids,'sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'state20':st,'prompt':prompt,'truth_not_in_prompt':{'lead_gap_m':row['lead_gap_m'],'TTC_s':row['TTC_s'],'scene':scene}})
(O/'inputs.json').write_text(json.dumps(cases,indent=2))
torch.set_num_threads(6); M=R/'models/Qwen3-VL-2B-Instruct'; p=AutoProcessor.from_pretrained(M,local_files_only=True);q=Qwen3VLForConditionalGeneration.from_pretrained(M,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();q.requires_grad_(False)
results=[]
for c in cases:
 ims=[Image.open(x).convert('RGB').resize((448,252)) for x in c['images']]
 msgs=[{'role':'user','content':[{'type':'image'} for _ in ims]+[{'type':'text','text':c['prompt']}]}]
 text=p.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True);b=p(text=[text],images=ims,return_tensors='pt').to('cuda');start=time.perf_counter()
 with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
  y=q.generate(**b,max_new_tokens=180,do_sample=False,repetition_penalty=1.1,use_cache=True);torch.cuda.synchronize()
 out=p.batch_decode(y[:,b['input_ids'].shape[1]:],skip_special_tokens=True)[0]
 v={'id':c['id'],'output':out,'seconds':time.perf_counter()-start,'input_tokens':int(b['input_ids'].shape[1]),'output_tokens':int(y.shape[1]-b['input_ids'].shape[1]),'rendered_prompt':text};results.append(v);(O/'outputs.json').write_text(json.dumps(results,indent=2));print(json.dumps(v),flush=True)
(O/'completed.json').write_text(json.dumps({'cases':len(results),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'training':False,'weights_modified':False,'note':'Original pretrained text generation; added text interface differs from frozen hidden-feature driving head.'},indent=2))
