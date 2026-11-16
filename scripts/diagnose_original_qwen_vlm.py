import json,time,hashlib
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from torch.nn.attention import sdpa_kernel,SDPBackend
R=Path(__file__).resolve().parents[1]; O=R/'artifacts/qwen-original-vlm-diagnostic';O.mkdir(exist_ok=True)
cases=[]
for scene in ['follow','brake','lanechange']:
 d=R/f'artifacts/qwen-driving-v4/scenarios/{scene}-0-model'; rows=json.loads((d/'steps.json').read_text())
 for index in [0,30]:
  row=rows[index]; paths=[d/f'rgb-{f}.png' for f in row['history_frame_ids']]; assert all(p.exists() for p in paths)
  st=row['state20']; decoded={'speed_m_s':st[0]*30,'acceleration_forward_m_s2':st[1]*10,'acceleration_left_m_s2':st[2]*10,'yaw_rate_left_rad_s':st[3],'navigation_near_forward_left_m':[st[4]*50,st[5]*50],'navigation_far_forward_left_m':[st[6]*50,st[7]*50],'navigation_command':'change lane left' if st[12] else 'follow lane'}
  prompt='These four front-camera frames are chronological at t=-0.6,-0.4,-0.2,0 seconds. Describe only visible evidence; distinguish uncertainty from fact. Do not infer exact distances or other vehicle speeds from appearance alone. Ego-relative coordinates: x forward, y left. Current measured state/navigation: '+json.dumps(decoded)+'. In at most 180 words answer: 1) visible vehicle(s) and relative direction, road/lane geometry; 2) can lead vehicle motion or stopping be determined? 3) immediate driving recommendation (steering direction and maintain/reduce/increase speed), with evidence and uncertainty. Do not claim safe execution or output throttle/brake numbers.'
  cases.append({'id':f'{scene}-step{index}','scene':scene,'index':index,'frame':row['frame'],'images':[str(p) for p in paths],'sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],'state20':st,'decoded_state':decoded,'prompt':prompt,'recorded_truth_not_given_to_model':{'lead_gap_m':row['lead_gap_m'],'scene':scene,'elapsed_since_first_control_s':row['timestamp']-rows[0]['timestamp']},'original_head_output':row['output']})
(O/'inputs.json').write_text(json.dumps(cases,indent=2))
torch.set_num_threads(6); M=R/'models/Qwen3-VL-2B-Instruct'; p=AutoProcessor.from_pretrained(M,local_files_only=True);q=Qwen3VLForConditionalGeneration.from_pretrained(M,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();q.requires_grad_(False)
results=[]
for c in cases:
 ims=[Image.open(x).convert('RGB').resize((448,252)) for x in c['images']]
 msgs=[{'role':'user','content':[{'type':'image'} for _ in ims]+[{'type':'text','text':c['prompt']}]}]
 text=p.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True);b=p(text=[text],images=ims,return_tensors='pt').to('cuda');start=time.perf_counter()
 with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
  y=q.generate(**b,max_new_tokens=300,do_sample=False,use_cache=True);torch.cuda.synchronize()
 out=p.batch_decode(y[:,b['input_ids'].shape[1]:],skip_special_tokens=True)[0]
 v={'id':c['id'],'output':out,'seconds':time.perf_counter()-start,'input_tokens':int(b['input_ids'].shape[1]),'output_tokens':int(y.shape[1]-b['input_ids'].shape[1]),'rendered_prompt':text};results.append(v);(O/'outputs.json').write_text(json.dumps(results,indent=2));print(json.dumps(v),flush=True)
(O/'completed.json').write_text(json.dumps({'cases':len(results),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'training':False,'weights_modified':False,'note':'Original pretrained text generation; added text interface differs from frozen hidden-feature driving head.'},indent=2))
