"""Frozen inference convergence and weighted gradient decomposition; zero updates."""
import json,time,gc,unittest
import numpy as np,torch
from torch.nn import functional as F
from lightning_task import ROOT,configure_runtime
from probe_teacher_history_matrix import idle
from train_masked_teacher_bounded import Task,DiskSamples
from train_joint_occ import batch_to
from train_rollout_dynamics_paired import SeedData
from stage1_objective import consistent_flow_loss
from rollout_dynamics_regularizer import rollout,dynamics_loss,ACTOR_KEYS
from audit_candidate_boundary import audit,decode,Tests

def main():
 idle();configure_runtime();out=ROOT/'artifacts/frozen-rollout-convergence-gradient';out.mkdir(exist_ok=False);started=time.perf_counter()
 test=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));assert test.wasSuccessful()
 src=ROOT/'artifacts/masked-teacher-155m-continue2048';paired=ROOT/'artifacts/teacher-rollout-dynamics-paired';sel=json.loads((paired/'selection.json').read_text());cfg=json.loads((paired/'config.json').read_text());ck=torch.load(src/'best.ckpt',map_location='cpu',weights_only=False);task=Task(cfg,ck['task_state']['manifest_sha'],ck['task_state']['settings']['selection_sha']);task.load_state_dict(ck['state_dict']);del ck;m=task.model.cuda().eval()
 paths=[('original2048',src/'best.ckpt')]+[(f'{s}-{a}',paired/f'{s}-{a}/latest.ckpt') for s in [1810,1811] for a in ['baseline','rollout']];rows=[]
 for label,path in paths:
  ck=torch.load(path,map_location='cpu',weights_only=False);task.load_state_dict(ck['state_dict']);del ck;ds=DiskSamples(sel['validation']);records=[]
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   for wi in range(16):
    b=batch_to([ds[wi]],'cuda');v0=float(b['state'][0,0])*30;context=m.context({k:b[k] for k in ACTOR_KEYS}).repeat_interleave(4,0);torch.manual_seed(8181+wi);noise=torch.randn(4,25,5,device='cuda');pred={}
    for steps in [4,16,32]:
     x=noise.clone()
     for i in range(steps):x=x+m.velocity(context,x,torch.full((4,1),i/steps,device='cuda'))/steps
     raw=x.float();p=torch.cat([raw[...,:2],F.normalize(raw[...,2:4],dim=-1,eps=1e-6),raw[...,4:5].clamp_min(0)],-1);pred[steps]=p.cpu().numpy()
     for k in range(4):
      q=decode(pred[steps][k]);a=audit(q,v0);derived=np.linalg.norm(np.diff(np.vstack(([0.,0.],q[:,:2])),axis=0),axis=1)/.2
      records.append({'window':wi,'candidate':k,'steps':steps,'v0_mps':v0,'speed_bin':'0-2' if v0<2 else '2-5' if v0<5 else '5+','first_residual':a['first_segment']['kinematic_residual_norm_mps'],'full_residual':a['full']['mean_kinematic_residual_mps'],'first_accel':a['first_segment']['signed_speed_accel_mps2'],'peak_accel':a['full']['max_abs_scalar_accel_mps2'],'peak_jerk':a['full']['max_abs_scalar_jerk_mps3'],'forward_m':float(q[-1,0]),'speed_disagreement':float(abs(derived-q[:,4]).mean()),'nearzero':float((q[:,4]<.5).mean()),'raw_negative_fraction':float((raw[k,:,4]<0).float().mean()),'first_raw_speed_mps':float(raw[k,0,4])*30})
    d416=np.linalg.norm((pred[4][...,:2]-pred[16][...,:2])*[50,10],axis=-1).mean();d1632=np.linalg.norm((pred[16][...,:2]-pred[32][...,:2])*[50,10],axis=-1).mean()
    for r in records[-12:]:r.update(delta4to16_m=float(d416),delta16to32_m=float(d1632))
  groups=[]
  for sb in ['all','0-2','2-5','5+']:
   for steps in [4,16,32]:
    rr=[r for r in records if r['steps']==steps and (sb=='all' or r['speed_bin']==sb)]
    if rr:groups.append({'speed_bin':sb,'steps':steps,'n_candidates':len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in ['first_residual','full_residual','peak_accel','peak_jerk','forward_m','speed_disagreement','nearzero','raw_negative_fraction','delta4to16_m','delta16to32_m']},'global_first_abs_accel':max(abs(r['first_accel']) for r in rr)})
  rows.append({'model':label,'groups':groups,'records':records});(out/'convergence.json').write_text(json.dumps(rows,indent=2));print('INFERENCE '+label+' '+json.dumps([g for g in groups if g['speed_bin']=='all']),flush=True)
 # Replay source initialization and first paired batches, not post-update weights.
 ck=torch.load(src/'best.ckpt',map_location='cpu',weights_only=False);task.load_state_dict(ck['state_dict']);del ck;m.train();params=[p for p in m.parameters() if p.requires_grad];gradrows=[]
 for seed in [1810,1811]:
  b={k:v.cuda() for k,v in next(iter(SeedData(sel,seed).loader('train'))).items()};small={k:v[:2] for k,v in b.items()};buffers={k:v.clone() for k,v in m.named_buffers()}
  with torch.autocast('cuda',dtype=torch.bfloat16):
   torch.manual_seed(seed*100000);base,terms=consistent_flow_loss(m,b,cfg['objective']);aux=m.auxiliary_loss(b);base=base+sum(cfg['aux_weights'][n]*v for n,v in zip(['latent','variance','semantic'],aux));fm=terms['fm'];other=base-fm
   torch.manual_seed(seed*100000+50000);p=rollout(m,small,4);_,rt=dynamics_loss(p,small['state']);losses={'fm':fm,'base_other':other,**{k:.01*v for k,v in rt.items()}}
  grads={}
  for name,value in losses.items():grads[name]=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
  names=list(grads);dots={}
  for i,a in enumerate(names):
   for bb in names[i:]:dots[a+'|'+bb]=sum(float((x.float()*y.float()).sum()) for x,y in zip(grads[a],grads[bb]) if x is not None and y is not None)
  def dot(a,b):return dots[a+'|'+b] if a+'|'+b in dots else dots[b+'|'+a]
  norm={a:dot(a,a)**.5 for a in names};total2=sum(dot(a,b) for a in names for b in names);base2=sum(dot(a,b) for a in names[:2] for b in names[:2]);total=total2**.5;cos={a+'|'+b:dot(a,b)/max(norm[a]*norm[b],1e-20) for i,a in enumerate(names) for b in names[i+1:]}
  row={'seed':seed,'weighted_loss_values':{k:float(v.detach()) for k,v in losses.items()},'weighted_gradient_norm':norm,'pairwise_cosine':cos,'dot_products':dots,'base_gradient_norm':base2**.5,'combined_gradient_norm':total,'clip_coefficient':min(1,1/(total+1e-6)),'component_projection_onto_total':{a:sum(dot(a,b) for b in names)/total for a in names},'buffer_unchanged':all(torch.equal(v,buffers[k]) for k,v in m.named_buffers()),'updates':0};gradrows.append(row);print('GRADIENT '+json.dumps(row),flush=True);del grads,losses,base,fm,other,terms,aux,p,rt;gc.collect();torch.cuda.empty_cache()
 (out/'gradient.json').write_text(json.dumps(gradrows,indent=2));(out/'completion.json').write_text(json.dumps({'updates':0,'synthetic_tests_passed':test.testsRun,'wall_s':time.perf_counter()-started,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'test_access':False,'limits':'16 repeatedly-used development windows; source OCC causality unknown; t0 forward scalar assumption; no prehistory jerk; prediction only not safety'},indent=2));print('DONE',flush=True)
if __name__=='__main__':main()
