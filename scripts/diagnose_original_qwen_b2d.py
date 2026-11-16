import json,time,hashlib
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from torch.nn.attention import sdpa_kernel,SDPBackend
R=Path(__file__).resolve().parents[1]; O=R/'artifacts/qwen-original-b2d-diagnostic';O.mkdir(exist_ok=True)
cases=[]
for m in json.loads((R/'artifacts/qwen-original-b2d-inputs/manifest.json').read_text()):
 rows=torch.load(R/'artifacts/qwen-driving-v3/cache'/(m['route']+'.pt'),weights_only=False);st=next(x['state'].tolist() for x in rows if x['meta']['frame']==m['anchor'])
 decoded={'ego_speed_m_s':st[0]*30,'ego_accel_forward_m_s2':st[1]*10,'ego_accel_left_m_s2':st[2]*10,'ego_yaw_rate_left_rad_s':st[3],'nav_near_forward_left_m':[st[4]*50,st[5]*50],'nav_far_forward_left_m':[st[6]*50,st[7]*50],'nav_command_codes':[st[8:14].index(max(st[8:14]))+1,st[14:20].index(max(st[14:20]))+1]}
 front=sorted([x for x in m['files'] if x['camera']=='rgb_front'],key=lambda x:x['frame']);cams=['rgb_front','rgb_front_left','rgb_front_right','rgb_back','rgb_back_left','rgb_back_right'];multi=[next(x for x in m['files'] if x['camera']==c and x['frame']==m['anchor']) for c in cams]
 for arm,files,size,context in [('history448',front,[448,252],'4 chronological FRONT frames at -0.6,-0.4,-0.2,0 seconds'),('history896',front,[896,504],'4 chronological FRONT frames at -0.6,-0.4,-0.2,0 seconds'),('single448',[front[-1]],[448,252],'one current FRONT frame'),('surround448',multi,[448,252],'6 current views ordered FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT; these are views at the same time, not a temporal sequence')]:
  prompt='Images: '+context+'. Coordinates x forward,y left. ONLY ego telemetry: '+json.dumps(decoded)+'. Nav codes1left,2right,3straight,4lane-follow,5change-left,6change-right. In four short lines: visible vehicles and relative direction; road/lane geometry; lead motion moving/stopped/unknown with evidence; recommended ego steering and speed change. Do not treat ego telemetry as lead speed. Do not invent distances; a single time cannot prove motion.'
  cases.append({'id':m['family']+'-'+arm,'images':[x['extracted_path'] for x in files],'resize':size,'prompt':prompt,'route':m['route'],'anchor':m['anchor'],'state20':st})
(O/'inputs.json').write_text(json.dumps(cases,indent=2))
torch.set_num_threads(6); M=R/'models/Qwen3-VL-2B-Instruct'; p=AutoProcessor.from_pretrained(M,local_files_only=True);q=Qwen3VLForConditionalGeneration.from_pretrained(M,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();q.requires_grad_(False)
results=[]
for c in cases:
 ims=[Image.open(x).convert('RGB').resize(tuple(c['resize'])) for x in c['images']]
 msgs=[{'role':'user','content':[{'type':'image'} for _ in ims]+[{'type':'text','text':c['prompt']}]}]
 text=p.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True);b=p(text=[text],images=ims,return_tensors='pt').to('cuda');start=time.perf_counter()
 with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
  y=q.generate(**b,max_new_tokens=180,do_sample=False,repetition_penalty=1.1,use_cache=True);torch.cuda.synchronize()
 out=p.batch_decode(y[:,b['input_ids'].shape[1]:],skip_special_tokens=True)[0]
 v={'id':c['id'],'output':out,'seconds':time.perf_counter()-start,'input_tokens':int(b['input_ids'].shape[1]),'output_tokens':int(y.shape[1]-b['input_ids'].shape[1]),'rendered_prompt':text};results.append(v);(O/'outputs.json').write_text(json.dumps(results,indent=2));print(json.dumps(v),flush=True)
(O/'completed.json').write_text(json.dumps({'cases':len(results),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'training':False,'weights_modified':False,'note':'Original pretrained text generation; added text interface differs from frozen hidden-feature driving head.'},indent=2))
