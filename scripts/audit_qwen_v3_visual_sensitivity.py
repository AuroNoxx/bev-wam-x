"""Read-only frozen v3 visual intervention audit on eight held-out windows."""
from pathlib import Path
import json,tarfile,io,time,sys,hashlib
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from torch.nn.attention import sdpa_kernel,SDPBackend
from qwen_driving_v3_head import Head
R=Path(__file__).resolve().parents[1];P=R/'artifacts/qwen-driving-v3';O=R/'artifacts/qwen-driving-v3-visual-audit';O.mkdir(exist_ok=True);torch.set_num_threads(6)
plan=json.loads((P/'selection.json').read_text());routes=list(dict.fromkeys(x['route'] for x in plan if x['split']=='validation'));chosen=[]
# One midpoint from each of eight route-disjoint held-out routes; no result-based cherry picking.
for route in routes[:8]:
 rr=[x for x in plan if x['route']==route];chosen.append(rr[len(rr)//2])
(O/'selection.json').write_text(json.dumps(chosen,indent=2));items=[]
for x in chosen:
 images={};need=set(x['history'])
 with tarfile.open(R/'data/public/bench2drive/archives'/(x['route']+'.tar.gz'),'r|gz') as tar:
  for z in tar:
   if '/camera/rgb_front/' in z.name and int(Path(z.name).stem) in need:images[int(Path(z.name).stem)]=Image.open(io.BytesIO(tar.extractfile(z).read())).convert('RGB').resize((448,252))
 cached=next(z for z in torch.load(P/'cache'/(x['route']+'.pt'),weights_only=False) if z['meta']['frame']==x['frame']);items.append((x,[images[k] for k in x['history']],cached));print('PREPARED',x['route'],x['frame'],flush=True)
if '--prepare-only' in sys.argv:sys.exit()
pr=AutoProcessor.from_pretrained(R/'models/Qwen3-VL-2B-Instruct',local_files_only=True);q=Qwen3VLForConditionalGeneration.from_pretrained(R/'models/Qwen3-VL-2B-Instruct',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();q.requires_grad_(False);h=Head('joint_fm');h.load_state_dict(torch.load(P/'weights/best.ckpt',map_location='cpu',weights_only=False)['state_dict']);h.cuda().eval();norm=torch.load(P/'world_normalization.pt',weights_only=False);results=[]
@torch.inference_mode()
def encode(ims):
 b=pr(text=['<|vision_start|><|image_pad|><|vision_end|>'*4],images=ims,return_tensors='pt').to('cuda')
 with sdpa_kernel(SDPBackend.MATH):z=q.model(**b,use_cache=False).last_hidden_state.float()
 return torch.nn.functional.adaptive_avg_pool1d(z.transpose(1,2),16).transpose(1,2)
@torch.inference_mode()
def rollout(f,st):
 gen=torch.Generator(device='cuda').manual_seed(9188);x=torch.randn(1,25,2,generator=gen,device='cuda');w=torch.randn(1,16,32,generator=gen,device='cuda')
 with sdpa_kernel(SDPBackend.MATH):
  for j in range(16):v,vw=h(f,st,x,torch.full((1,1),j/16,device='cuda'),w);x=x+v/16;w=w+vw/16
 vel=x[0]*10+torch.tensor([float(st[0,0])*30,0],device='cuda');return (vel.cumsum(0)*.2).cpu(),vel.cpu(),w[0].cpu()
for meta,ims,cached in items:
 st=cached['state'][None].cuda();truth=cached['trajectory'][:,:2]*torch.tensor([50.,10.]);outputs={};original_f=None
 for label,pics in [('original',ims),('all_black',[Image.new('RGB',(448,252),(0,0,0)) for _ in range(4)]),('history_shuffle',[ims[k] for k in [3,1,2,0]])]:
  f=encode(pics);xy,vel,w=rollout(f,st)
  if label=='original':original_f=f;original_xy=xy
  outputs[label]={'xy':xy.tolist(),'velocity':vel.tolist(),'trajectory_change_mean_m':float((xy-original_xy).norm(dim=-1).mean()),'trajectory_change_endpoint5_m':float((xy[-1]-original_xy[-1]).norm()),'GT_ADE_m':float((xy-truth).norm(dim=-1).mean()),'predicted_first_vx_mps':float(vel[0,0]),'predicted_mean_speed_mps':float(vel.norm(dim=-1).mean()),'feature_relative_L2_change':float((f-original_f).norm()/original_f.norm())}
  if label=='original':predworld=w
 target=(cached['future_features']@norm['projection']-norm['mu'])/norm['scale'];persist=(cached['features']@norm['projection']-norm['mu'])/norm['scale']
 with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
  wx=torch.zeros(1,16,32,device='cuda');a=torch.zeros(1,25,2,device='cuda');t=torch.full((1,1),.5,device='cuda');_,w0=h(original_f,st,a,t,wx);_,w1=h(original_f,st,a+.5,t,wx);coupling=float((w1-w0).square().mean())
 row={'meta':meta,'conditions':outputs,'world_MSE':float((predworld-target).square().mean()),'world_persistence_MSE':float((persist-target).square().mean()),'world_velocity_response_to_action_token_shift_MSE':coupling,'new_vs_cached_feature_relative_L2':float((original_f.cpu()[0]-cached['features']).norm()/cached['features'].norm())};results.append(row);(O/'results.json').write_text(json.dumps(results,indent=2));print('AUDITED',meta['route'],flush=True)
summary={'windows':len(results),'checkpoint_sha256':hashlib.sha256((P/'weights/best.ckpt').read_bytes()).hexdigest(),'conditions':{},'region_occlusion':'not performed: no independently verified front-vehicle pixel bbox; do not substitute guessed box','state_navigation_seed_fixed':True,'seed':9188,'world_MSE_mean':sum(x['world_MSE'] for x in results)/len(results),'world_persistence_MSE_mean':sum(x['world_persistence_MSE'] for x in results)/len(results),'full_validation_world_persistence_MSE':json.loads((P/'config.json').read_text())['world_persistence_mse'],'full_validation_final_world_MSE':json.loads((P/'epoch_results.jsonl').read_text().splitlines()[-1])['world_MSE'],'caution':'interventions are out-of-distribution; sensitivity is not causal competence or safety; joint token attention is not counterfactual world-model validation'}
for name in ['original','all_black','history_shuffle']:
 summary['conditions'][name]={k:sum(x['conditions'][name][k] for x in results)/len(results) for k in ['trajectory_change_mean_m','trajectory_change_endpoint5_m','GT_ADE_m','predicted_first_vx_mps','predicted_mean_speed_mps','feature_relative_L2_change']}
(O/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
