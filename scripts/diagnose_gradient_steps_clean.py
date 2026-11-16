"""Same-noise 4/16-step weighted gradient comparison. No updates or test data."""
from diagnose_frozen_rollout import *
import hashlib

def dot(a,b):return sum(float((x.float()*y.float()).sum()) for x,y in zip(a,b) if x is not None and y is not None)
def add(a,b):return tuple(y if x is None else x if y is None else x+y for x,y in zip(a,b))
def norm(a):return dot(a,a)**.5
def cosine(a,b):return dot(a,b)/max(norm(a)*norm(b),1e-20)
def sample(m,b,noise,steps):
 context=m.context({k:b[k] for k in ACTOR_KEYS});x=noise.clone()
 for i in range(steps):x=x+m.velocity(context,x,torch.full((len(x),1),i/steps,device=x.device))/steps
 raw=x.float();return torch.cat([raw[...,:2],F.normalize(raw[...,2:4],dim=-1,eps=1e-6),raw[...,4:5].clamp_min(0)],-1),context,raw

def main():
 idle();configure_runtime();src=ROOT/'artifacts/masked-teacher-155m-continue2048';out=ROOT/'artifacts/frozen-gradient-4vs16-clean';out.mkdir(exist_ok=False);cfg=json.loads((ROOT/'artifacts/teacher-rollout-dynamics-paired/config.json').read_text());sel=json.loads((ROOT/'artifacts/teacher-rollout-dynamics-paired/selection.json').read_text());ck=torch.load(src/'best.ckpt',map_location='cpu',weights_only=False);task=Task(cfg,ck['task_state']['manifest_sha'],ck['task_state']['settings']['selection_sha']);task.load_state_dict(ck['state_dict']);del ck;m=task.model.cuda().train();params=[p for p in m.parameters() if p.requires_grad];versions=[p._version for p in params];buffers={k:v.clone() for k,v in m.named_buffers()};rows=[];started=time.perf_counter()
 for seed in [1810,1811]:
  b={k:v.cuda() for k,v in next(iter(SeedData(sel,seed).loader('train'))).items()};small={k:v[:2] for k,v in b.items()};torch.manual_seed(seed*100000+50000);noise=torch.randn(2,25,5,device='cuda');fingerprints=[hashlib.sha256(b['history'][i].cpu().numpy().tobytes()+b['state'][i].cpu().numpy().tobytes()).hexdigest() for i in range(8)]
  with torch.autocast('cuda',dtype=torch.bfloat16):
   torch.manual_seed(seed*100000);base,terms=consistent_flow_loss(m,b,cfg['objective']);aux=m.auxiliary_loss(b);base=base+sum(cfg['aux_weights'][n]*v for n,v in zip(['latent','variance','semantic'],aux))
  gb=torch.autograd.grad(base,params,allow_unused=True);del base,terms,aux
  bystep={};stored={};context_reference=None
  # Warm up each execution shape, timing subsequently includes forward+one combined backward only.
  for steps in [4,16]:
   with torch.autocast('cuda',dtype=torch.bfloat16):p,context,raw=sample(m,small,noise,steps);reg,_=dynamics_loss(p,small['state'])
   warm=torch.autograd.grad(.01*reg,params,allow_unused=True);del warm,p,context,raw,reg;gc.collect();torch.cuda.synchronize()
  for steps in ([4,16] if seed==1810 else [16,4]):
   torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
   with torch.autocast('cuda',dtype=torch.bfloat16):p,context,raw=sample(m,small,noise,steps);reg,terms=dynamics_loss(p,small['state']);weighted={k:.01*v for k,v in terms.items()}
   gr=torch.autograd.grad(.01*reg,params,retain_graph=True,allow_unused=True);torch.cuda.synchronize();elapsed=time.perf_counter()-start;combined=add(gb,gr);nc=norm(combined);peak=torch.cuda.max_memory_allocated()/2**30
   if context_reference is None:context_reference=context.detach().clone()
   else:assert torch.equal(context_reference,context.detach()),'context mismatch'
   components={};stored[steps]={}
   for name,value in weighted.items():
    g=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True);components[name]={'weighted_loss':float(value.detach()),'norm':norm(g),'cosine_to_base':cosine(g,gb),'projection_onto_combined':dot(g,combined)/nc};stored[steps][name]=tuple(None if x is None else x.detach().cpu() for x in g);del g
   q=p.detach().cpu().numpy();records=[]
   for i in range(2):
    ph=decode(q[i]);v0=float(small['state'][i,0])*30;a=audit(ph,v0);derived=np.linalg.norm(np.diff(np.vstack(([0.,0.],ph[:,:2])),axis=0),axis=1)/.2;records.append({'fingerprint':fingerprints[i],'v0_mps':v0,'first_accel_mps2':a['first_segment']['signed_speed_accel_mps2'],'first_residual_mps':a['first_segment']['kinematic_residual_norm_mps'],'full_residual_mps':a['full']['mean_kinematic_residual_mps'],'forward_m':float(ph[-1,0]),'nearzero_fraction':float((ph[:,4]<.5).mean()),'negative_raw_speed_fraction':float((raw[i,:,4]<0).float().mean()),'speed_disagreement_mps':float(abs(derived-ph[:,4]).mean())})
   bystep[steps]={'combined_regularizer_norm':norm(gr),'base_norm':norm(gb),'combined_norm':nc,'clip_coefficient':min(1,1/(nc+1e-6)),'regularizer_cosine_to_base':cosine(gr,gb),'forward_backward_seconds':elapsed,'peak_allocated_gib_with_base_gradient':peak,'components':components,'samples':records};stored[steps]['total']=tuple(None if x is None else x.detach().cpu() for x in gr);print('STEP '+str(seed)+' '+str(steps)+' '+str(elapsed),flush=True);del gr,combined,p,context,raw,reg,terms,weighted,value;gc.collect();torch.cuda.empty_cache()
  comparisons={name:{'cosine_4_vs_16':cosine(stored[4][name],stored[16][name]),'norm_ratio_16_over_4':norm(stored[16][name])/max(norm(stored[4][name]),1e-20)} for name in stored[4]}
  row={'seed':seed,'batch_fingerprints':fingerprints,'effective_rollout_samples':2,'base_samples':8,'bystep':bystep,'direction_comparison':comparisons,'same_context_exact':True,'same_explicit_noise':True};rows.append(row);(out/'report.json').write_text(json.dumps(rows,indent=2));print('RESULT '+json.dumps(comparisons),flush=True);del stored,gb,b,small,context_reference;gc.collect();torch.cuda.empty_cache()
 assert versions==[p._version for p in params];assert all(torch.equal(v,buffers[k]) for k,v in m.named_buffers());(out/'completion.json').write_text(json.dumps({'optimizer_updates':0,'parameters_version_unchanged':True,'buffers_unchanged':True,'test_access':False,'wall_seconds':time.perf_counter()-started,'limits':'Two initial training batches, four rollout examples, one noise each; not full-data attribution or driving safety. Timing warmed one observation each and includes retained graph, no optimizer/data loader. CPU stored gradients excluded from GPU peak, arm order counterbalanced; memory baseline includes retained base gradients.'},indent=2));print('DONE',flush=True)
if __name__=='__main__':main()

