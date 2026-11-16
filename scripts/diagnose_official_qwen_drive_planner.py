import sys,time,json,hashlib,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R/'runtime/qwen-drive-official'
sys.path[:0]=[str(R/'runtime/qwen-drive-deps'),str(S/'src')]
M=R/'models/Qwen-Drive-1.0-4B';O=R/'artifacts/qwen-drive-official-planner';O.mkdir(exist_ok=True)
def save(n,d): (O/n).write_text(json.dumps(d,indent=2),encoding='utf8')
try:
 files=[(M/'model.safetensors',9078630512,'b9de4bf448f57485fdaa45c60b1eea8e41a4b6ae82ec0cee8855a1e0301caccc'),(M/'planner-sft/model.safetensors',2079739550,'dcf5989ed292799e77f539b21e5d8b701566676c1c5d13038135e97f96e2d7b4')]
 deadline=time.time()+7200
 while any(not p.exists() or p.stat().st_size!=size for p,size,h in files):
  save('progress.json',{'phase':'waiting_for_download','files':[{'file':str(p),'bytes':p.stat().st_size if p.exists() else 0,'total':s} for p,s,h in files],'training':False})
  if time.time()>deadline: raise TimeoutError('download deadline')
  time.sleep(15)
 for p,size,expected in files:
  h=hashlib.sha256()
  with p.open('rb') as f:
   while b:=f.read(16*1024*1024):h.update(b)
  assert h.hexdigest()==expected,(str(p),h.hexdigest())
 import numpy as np,torch,transformers
 from qwen_drive import QwenDriveForPlanning,InferenceMode
 from qwen_drive.benchmarks import read_scene_file
 from qwen_drive.images import ImageArchive
 from torch.nn.attention import sdpa_kernel,SDPBackend
 torch.set_num_threads(6);save('progress.json',{'phase':'loading_official_sft_planner','training':False});start=time.perf_counter()
 q=QwenDriveForPlanning.from_pretrained(M,planner=M/'planner-sft',dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval();q.requires_grad_(False);load_s=time.perf_counter()-start
 archive=ImageArchive.open(S/'data/demo/frames.parquet');samples=list(read_scene_file(S/'data/demo/planning_scenes.jsonl',image_archive=archive,num_history_points=q.config.num_history_points));results=[]
 for i,sample in enumerate(samples):
  scene=sample.scene;inputs=q.processor(scene,with_reasoning=False,device='cpu');save('progress.json',{'phase':'direct_planning','scene':i,'token':sample.token,'input_tokens':int(inputs['input_ids'].shape[1]),'training':False})
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):out=q.plan_from_inputs(inputs,mode=InferenceMode.DIRECT_PLANNING,num_samples=1)
  torch.cuda.synchronize();elapsed=time.perf_counter()-start;traj=out.trajectory;gt=sample.future_trajectory
  assert np.isfinite(traj).all()
  row={'scene':i,'token':sample.token,'shape':list(traj.shape),'trajectory':traj.tolist(),'ground_truth':None if gt is None else gt.tolist(),'seconds':elapsed,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'input_tokens':int(inputs['input_ids'].shape[1]),'image_grid_thw':inputs['image_grid_thw'].tolist(),'views':list(scene.views),'frames_per_view':scene.num_camera_frames,'nav_command':int(scene.nav_command),'mode':'DIRECT_PLANNING','num_samples':1}
  if gt is not None:
   err=np.linalg.norm(traj[:,:2]-gt[:len(traj),:2],axis=-1);row.update(ADE_m=float(err.mean()),FDE_m=float(err[-1]))
  results.append(row);save('outputs.json',results);print(json.dumps(row),flush=True)
 save('completed.json',{'training':False,'weights_modified':False,'cases':len(results),'model':'Qwen/Qwen-Drive-1.0-4B','planner':'official planner-sft','precision':'BF16','load_seconds':load_s,'torch':torch.__version__,'hip':torch.version.hip,'transformers':transformers.__version__,'device':torch.cuda.get_device_name(0),'official_commit':'28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0','limitations':['official demo open-loop only, not CARLA closed-loop','SDPA MATH on AMD rather than reference CUDA flash attention','one candidate per scene, no Judge/world model/custom driving head']})
except Exception as e:
 save('failed.json',{'error':repr(e),'traceback':traceback.format_exc(),'training':False});traceback.print_exc();raise
