"""Read-only checkpoint/validation evidence; no optimization or test access."""
import json,torch,time
from pathlib import Path
from torchdata.stateful_dataloader import StatefulDataLoader
from train_masked_teacher_bounded import Task,DiskSamples
from train_joint_occ import batch_to
from lightning_task import ROOT,configure_runtime
from stage1_objective import kinematic_residual
from probe_teacher_history_matrix import idle
idle();configure_runtime();folder=ROOT/'artifacts/masked-teacher-155m-first256';out=folder/'postfit-evidence';out.mkdir(exist_ok=False)
ck=torch.load(folder/'best.ckpt',map_location='cpu',weights_only=False);cfg=json.loads((folder/'config.json').read_text());sel=json.loads((folder/'selection.json').read_text());task=Task(cfg,ck['task_state']['manifest_sha'],ck['task_state']['settings']['selection_sha']);task.load_state_dict(ck['state_dict'],strict=True);task.on_load_checkpoint(ck)
state=ck['loops']['fit_loop']['state_dict']['combined_loader'][0]
def loader():return StatefulDataLoader(list(range(len(sel['train']))),batch_size=8,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(1701))
a=loader();it=iter(a)
for _ in range(256):next(it)
expected=next(it).tolist();b=loader();b.load_state_dict(state);actual=next(iter(b)).tolist();assert expected==actual
report={'checkpoint':'best.ckpt','global_step':ck['global_step'],'state_dict_strict_load':True,'config_rng_hook_load':True,'loader_next_indices_equal':expected==actual,'next_indices':actual,'optimizer_step_executed':False,'full_interrupted_gradient_equivalence_tested':False,'latest_loader_reset_warning':True,'metrics':[]}
m=task.model.cuda().eval();ds=DiskSamples(sel['validation'][:16]);samples=[];started=time.perf_counter()
for start in range(0,16,4):
 b=batch_to([ds[i] for i in range(start,start+4)],'cuda')
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  torch.manual_seed(8181+start);p=m.sample(b,k=4,steps=16)
  xt=torch.randn(4,25,5,device='cuda');tt=torch.full((4,1),.3,device='cuda');isolated=torch.equal(m(b,xt,tt),m({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.randn_like(b['trajectory'])},xt,tt));assert isolated
  _,z=m.forecast_batch(b);future,valid,frac=m.target_data(b);target=m.teacher(future).float();current=m.teacher(m.canonical_history(b)[:,-1:]).float().expand_as(target);mask=valid.float();den=mask.sum().clamp_min(1)
  latent=((z.float()-target).square().mean(-1)*mask).sum()/den;persistence=((current-target).square().mean(-1)*mask).sum()/den
  # Same observed-current support as JEPA target; persistence fractions from repeated current occupancy.
  _,_,persistfrac=m.target_data({**b,'future':b['history'][:,-1:].expand(-1,2,-1,-1,-1)})
  prob=m.occ_head(z).float().softmax(-1);sem=((prob-frac).square().mean(-1)*mask).sum()/den;sem_p=((persistfrac-frac).square().mean(-1)*mask).sum()/den
  kin=kinematic_residual(p.flatten(0,1),b['state'].repeat_interleave(4,0)).norm(dim=-1).mean();speed=p[...,4]*30;acc=(speed[...,1:]-speed[...,:-1])/.2;jerk=(acc[...,1:]-acc[...,:-1])/.2
  report['metrics'].append({'batch_start':start,'future_label_isolation':isolated,'latent_mse':float(latent),'persistent_current_latent_mse':float(persistence),'semantic_brier':float(sem),'persistent_semantic_brier':float(sem_p),'kinematic_mps':float(kin),'mean_abs_scalar_accel_mps2':float(acc.abs().mean()),'max_abs_scalar_accel_mps2':float(acc.abs().max()),'mean_abs_scalar_jerk_mps3':float(jerk.abs().mean()),'candidate_finite':bool(torch.isfinite(p).all()),'collision':None,'offroad':None,'takeover':None})
  if start==0:
   samples=[{'validation_file':sel['validation'][i],'ego_state':b['state'][i].float().cpu().tolist(),'candidate_xy_m_sin_cos_speed_mps':(p[i]*p.new_tensor([50,10,1,1,30])).cpu().tolist()} for i in range(2)]
report.update(wall_s=time.perf_counter()-started,scope='16 previously used validation windows, open-loop predicted candidates not executed; current occupied support only, future-unobserved cells ignored; latent/semantic comparisons not full world dynamics or collision tests',samples=samples)
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))
