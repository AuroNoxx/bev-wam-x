import os,sys,time,json,hashlib,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'runtime/qwen-drive-deps'),str(R/'runtime/qwen-drive-official/src')]
O=R/'artifacts/qwen-drive-official-diagnostic';O.mkdir(exist_ok=True)
M=R/'models/Qwen-Drive-1.0-4B'; target=M/'model.safetensors';expected=9078630512

def save(n,d):
 (O/n).write_text(json.dumps(d,indent=2),encoding='utf8')
try:
 deadline=time.time()+7200
 while not target.exists() or target.stat().st_size!=expected:
  save('progress.json',{'phase':'waiting_for_download','bytes':target.stat().st_size if target.exists() else 0,'total':expected,'training':False})
  if time.time()>deadline:raise TimeoutError('weight download not complete after two hours')
  time.sleep(15)
 h=hashlib.sha256()
 with target.open('rb') as f:
  while b:=f.read(16*1024*1024):h.update(b)
 assert h.hexdigest()=='b9de4bf448f57485fdaa45c60b1eea8e41a4b6ae82ec0cee8855a1e0301caccc',h.hexdigest()
 import torch,transformers
 from PIL import Image
 from qwen_drive import QwenDriveForPlanning
 from qwen_drive.modeling_qwen_drive import VQA_DECODE_DEFAULTS
 from torch.nn.attention import sdpa_kernel,SDPBackend
 torch.set_num_threads(6)
 save('progress.json',{'phase':'loading','training':False,'weight_sha256':h.hexdigest()})
 start=time.perf_counter()
 q=QwenDriveForPlanning.from_pretrained(M,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval();q.requires_grad_(False)
 load_seconds=time.perf_counter()-start
 cases=json.loads((R/'artifacts/qwen-original-history-diagnostic/inputs.json').read_text());save('inputs.json',cases)
 results=[]
 for c in cases:
  ims=[Image.open(x).convert('RGB').resize((448,252)) for x in c['images']]
  assert [hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in c['images']]==c['sha256']
  inp=q.processor.encode_vqa(ims,c['prompt'],device=q.device)
  kw=dict(VQA_DECODE_DEFAULTS);seed=kw.pop('seed');kw.pop('presence_penalty',None);kw['max_new_tokens']=384;torch.manual_seed(seed)
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
  with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
   out=q.vlm.generate(input_ids=inp['input_ids'],pixel_values=inp['pixel_values'],image_grid_thw=inp['image_grid_thw'],mm_token_type_ids=q._modality_ids(inp['input_ids']),**kw)
  torch.cuda.synchronize();elapsed=time.perf_counter()-t
  new=out[0,inp['input_ids'].shape[1]:];raw=q.processor.tokenizer.decode(new,skip_special_tokens=True)
  row={'id':c['id'],'raw_output':raw,'answer':raw.split('</think>')[-1].strip(),'seconds':elapsed,'input_tokens':int(inp['input_ids'].shape[1]),'output_tokens':len(new),'tokens_per_s':len(new)/elapsed,'hit_cap':len(new)>=384,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'image_grid_thw':inp['image_grid_thw'].tolist()};results.append(row);save('outputs.json',results);save('progress.json',{'phase':'inference','completed':len(results),'total':len(cases),'training':False});print(json.dumps(row),flush=True)
 save('completed.json',{'cases':len(results),'training':False,'weights_modified':False,'model':'Qwen/Qwen-Drive-1.0-4B','weight_sha256':h.hexdigest(),'precision':'BF16 official, not quantized','torch':torch.__version__,'hip':torch.version.hip,'transformers':transformers.__version__,'device':torch.cuda.get_device_name(0),'load_seconds':load_seconds,'official_commit':'28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0','limitations':['VQA only; no planner loaded','same source images and prompt as earlier diagnostic, different model processor','max_new_tokens limited to384; marked capped outputs','SDPA MATH rather than official CUDA flash attention']})
except Exception as e:
 save('failed.json',{'error':repr(e),'traceback':traceback.format_exc(),'training':False});traceback.print_exc();raise
