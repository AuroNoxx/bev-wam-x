import sys,json,time,gc
from pathlib import Path
import torch
sys.path.insert(0,'scripts')
from lightning_task import ROOT,configure_runtime
from probe_teacher_history_matrix import idle
from train_masked_teacher_bounded import Task,DiskSamples
from train_joint_occ import batch_to
from stage1_objective import consistent_flow_loss
from rollout_dynamics_regularizer import rollout,dynamics_loss
idle();configure_runtime();src=ROOT/'artifacts/masked-teacher-155m-continue2048';out=ROOT/'artifacts/rollout-dynamics-preflight';out.mkdir(exist_ok=False);ck=torch.load(src/'best.ckpt',map_location='cpu',weights_only=False);cfg=json.loads((src/'config.json').read_text());task=Task(cfg,ck['task_state']['manifest_sha'],ck['task_state']['settings']['selection_sha']);task.load_state_dict(ck['state_dict']);m=task.model.cuda().train();del ck
sel=json.loads((ROOT/'artifacts/teacher-boundary-short-training/selection.json').read_text());ds=DiskSamples(sel['train']);b=batch_to([ds[i] for i in range(2)],'cuda');params=[p for p in m.parameters() if p.requires_grad];torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
with torch.autocast('cuda',dtype=torch.bfloat16):
 torch.manual_seed(1810);fm,_=consistent_flow_loss(m,b,cfg['objective']);aux=m.auxiliary_loss(b);base=fm+.1*sum(aux)
g1=torch.autograd.grad(base,params,allow_unused=True);n1=sum(float(g.float().square().sum()) for g in g1 if g is not None)**.5;del g1,base,fm,aux
with torch.autocast('cuda',dtype=torch.bfloat16):
 torch.manual_seed(991810);p=rollout(m,b,4);loss,terms=dynamics_loss(p,b['state'])
g2=torch.autograd.grad(loss,params,allow_unused=True);n2=sum(float(g.float().square().sum()) for g in g2 if g is not None)**.5;finite=all(torch.isfinite(g).all().item() for g in g2 if g is not None);del g2
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
 torch.manual_seed(991810);p4=rollout(m,b,4);torch.manual_seed(991810);p16=rollout(m,b,16);torch.manual_seed(991810);poison=rollout(m,{**b,'future':torch.full_like(b['future'],255),'trajectory':torch.randn_like(b['trajectory'])},4)
 # Reset after constructing poison noise, so random perturbation construction cannot change rollout seed.
 torch.manual_seed(991810);poison=rollout(m,{**b,'future':torch.full_like(b['future'],255),'trajectory':torch.zeros_like(b['trajectory'])},4)
 diff=((p4[...,:2]-p16[...,:2])*p4.new_tensor([50.,10.])).norm(dim=-1).mean()
torch.cuda.synchronize();r={'train_windows':sel['train'][:2],'gradient_finite':finite,'baseline_grad_norm':n1,'rollout_grad_norm':n2,'weighted_rollout_grad_ratio_at_point01':.01*n2/n1,'rollout_loss':float(loss),'terms':{k:float(v) for k,v in terms.items()},'future_label_isolation_exact':torch.equal(p4,poison),'same_noise_4vs16_mean_position_difference_m':float(diff),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'wall_s':time.perf_counter()-start,'optimizer_updates':0,'limits':'2train windows only; no initial-acceleration jerk boundary,scalar t0 velocity assumption; diagnostic not safety'};(out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
