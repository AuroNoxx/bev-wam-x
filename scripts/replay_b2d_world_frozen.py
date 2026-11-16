"""Frozen joint-FM world diagnostic; cross-run differences are NOT causal version effects."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from torch.nn.attention import sdpa_kernel,SDPBackend
from qwen_driving_v3_head import Head
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',default=['artifacts/qwen-driving-v4/scenarios/follow-0-model','artifacts/b2d-0915/smoke-scenarios/follow-0-model']);p.add_argument('--count',type=int,default=10);p.add_argument('--out',default='artifacts/b2d-0915/world-replay');a=p.parse_args();out=R/a.out;out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(6);modelpath=R/'models/Qwen3-VL-2B-Instruct';pr=AutoProcessor.from_pretrained(modelpath,local_files_only=True);q=Qwen3VLForConditionalGeneration.from_pretrained(modelpath,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();q.requires_grad_(False)
 h=Head('joint_fm');weight=R/'artifacts/qwen-driving-v4/weights/best.ckpt';h.load_state_dict(torch.load(weight,map_location='cpu',weights_only=False)['state_dict']);h.cuda().eval();h.requires_grad_(False)
 norm=torch.load(R/'artifacts/qwen-driving-v4/world_normalization.pt',map_location='cuda',weights_only=False)
 @torch.inference_mode()
 def features(paths):
  ims=[Image.open(x).convert('RGB').resize((448,252)) for x in paths];b=pr(text=['<|vision_start|><|image_pad|><|vision_end|>'*4],images=ims,return_tensors='pt').to('cuda')
  with sdpa_kernel(SDPBackend.MATH):z=q.model(**b,use_cache=False).last_hidden_state.float()
  return torch.nn.functional.adaptive_avg_pool1d(z.transpose(1,2),16).transpose(1,2)
 results=[]
 with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
  for runid,runstr in enumerate(a.runs):
   run=R/runstr;steps=json.loads((run/'steps.json').read_text());valid=[]
   # Live simulation logged every .05s: dataset offsets +14/+16/+18/+20 at .1s => +28/+32/+36/+40 frames.
   for row in steps:
    future=[row['frame']+x for x in (28,32,36,40)]
    ids=row['history_frame_ids']+future
    if all((run/f'rgb-{i}.png').exists() for i in ids):valid.append((row,future))
   selected=sorted(set(np.linspace(0,len(valid)-1,min(a.count,len(valid)),dtype=int).tolist())) if valid else []
   report=json.loads((run/'report.json').read_text()) if (run/'report.json').exists() else {}
   for idx in selected:
    row,future=valid[idx];start=time.perf_counter();f=features([run/f'rgb-{i}.png' for i in row['history_frame_ids']]);ff=features([run/f'rgb-{i}.png' for i in future]);target=(ff@norm['projection']-norm['mu'])/norm['scale'];persistence=(f@norm['projection']-norm['mu'])/norm['scale'];st=torch.tensor([row['state20']],device='cuda');g=torch.Generator(device='cuda').manual_seed(9188);x=torch.randn(1,25,2,generator=g,device='cuda');w=torch.randn(1,16,32,generator=g,device='cuda')
    for j in range(16):v,vw=h(f,st,x,torch.full((1,1),j/16,device='cuda'),w);x=x+v/16;w=w+vw/16
    vel=x[0]*10+torch.tensor([row['state20'][0]*30,0],device='cuda');xy=vel.cumsum(0)*.2;original=np.asarray(row['output']['xy']);xy_np=xy.cpu().numpy();record=dict(run=runstr,simulator=report.get('simulator'),frame=row['frame'],timestamp=row['timestamp'],history_frames=row['history_frame_ids'],future_frames=future,world_mse=float((w-target).square().mean()),persistence_mse=float((persistence-target).square().mean()),xy_replay_max_abs_diff_m=float(np.abs(xy_np-original).max()),elapsed_s=time.perf_counter()-start)
    name=f'run{runid}-frame{row["frame"]}.npz';np.savez_compressed(out/name,world=w.cpu().numpy(),target=target.cpu().numpy(),persistence=persistence.cpu().numpy(),xy=xy_np,original_xy=original,state20=np.asarray(row['state20']),feature=f.cpu().numpy());record['arrays']=name;results.append(record);(out/'samples.json').write_text(json.dumps(results,indent=2));print(json.dumps(record),flush=True)
 summary={'training':False,'weights_modified':False,'weight_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'world_shape':[1,16,32],'future_seconds':[1.4,1.6,1.8,2.0],'normalization':'saved v4 train normalization, unchanged','caution':'Different scenes/closed-loop paths: not causal simulator-version comparison. Future labels follow executed controls, not predicted trajectory counterfactual.','runs':[]}
 for run in a.runs:
  rr=[r for r in results if r['run']==run];summary['runs'].append(dict(run=run,n=len(rr),world_mse=float(np.mean([r['world_mse'] for r in rr])) if rr else None,persistence_mse=float(np.mean([r['persistence_mse'] for r in rr])) if rr else None,xy_max_diff=max([r['xy_replay_max_abs_diff_m'] for r in rr],default=None)))
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
