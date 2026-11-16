import sys,json,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'runtime/qwen-drive-deps'),str(R/'runtime/qwen-drive-official/src')]
import numpy as np,torch
from qwen_drive import QwenDriveForPlanning,InferenceMode,CameraFrame,DrivingScene
from torch.nn.attention import sdpa_kernel,SDPBackend
torch.set_num_threads(6)
M=R/'models/Qwen-Drive-1.0-4B'
q=QwenDriveForPlanning.from_pretrained(M,planner=M/'planner-sft',dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval();q.requires_grad_(False)
print(json.dumps({'ready':True,'model':'Qwen-Drive-1.0-4B/planner-sft','mode':'direct','seed':42}),flush=True)
for line in sys.stdin:
 try:
  a=json.loads(line);a['views']={k:[CameraFrame(p) for p in v] for k,v in a['views'].items()}
  scene=DrivingScene(**a);inputs=q.processor(scene,with_reasoning=False,device='cpu');torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
  with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH): out=q.plan_from_inputs(inputs,mode=InferenceMode.DIRECT_PLANNING,num_samples=1,seed=42)
  torch.cuda.synchronize();tr=np.asarray(out.trajectory)
  assert tr.shape==(50,3) and np.isfinite(tr).all()
  print(json.dumps({'trajectory':tr.tolist(),'model_seconds':time.perf_counter()-t,'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'tokens':int(inputs['input_ids'].shape[1])}),flush=True)
 except Exception as e:traceback.print_exc(file=sys.stderr);print(json.dumps({'error':repr(e)}),flush=True)
