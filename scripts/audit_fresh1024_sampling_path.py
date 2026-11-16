"""Frozen endpoint/rollout diagnostic on reused dev only; no updates."""
from train_fresh_jepa_first1024_v2 import *
from stage1_objective import kinematic_residual
@torch.no_grad()
def metrics(x,b,target):
 r=kinematic_residual(x,b['state']);xy=x[...,:2]*x.new_tensor([50,10]);prev=torch.cat([torch.zeros_like(xy[:,:1]),xy[:,:-1]],1);derived=(xy-prev).norm(dim=-1)/.2;v=x[...,4]*30;heading=x[...,2:4].norm(dim=-1)
 return {'first_residual_mps':float(r[:,0].norm(dim=-1).mean()),'full_residual_mps':float(r.norm(dim=-1).mean()),'position_label_error_diagnostic_m':float(((x[...,:2]-target[...,:2])*x.new_tensor([50,10])).norm(dim=-1).mean()),'speed_label_mae_mps':float(((x[...,4]-target[...,4])*30).abs().mean()),'first_xy_norm_m':float(xy[:,0].norm(dim=-1).mean()),'first_speed_mps':float(v[:,0].mean()),'derived_mean_speed_mps':float(derived.mean()),'predicted_mean_speed_mps':float(v.mean()),'negative_speed_fraction':float((v<0).float().mean()),'heading_norm_mean':float(heading.mean()),'heading_norm_min':float(heading.min()),'backward_segments_mean':float(((xy-prev)[...,0]<0).float().sum(-1).mean()),'forward_proxy_m':float(xy[:,-1,0].mean())}
def main():
 idle();configure_runtime();src=ROOT/'artifacts/fresh-jepa-paired-first1024';out=ROOT/'artifacts/fresh1024-frozen-path-audit';out.mkdir(exist_ok=False);cfg=json.loads((src/'config.json').read_text());sel=json.loads((src/'selection.json').read_text());names=[sel['validation'][i] for i in np.linspace(0,26,8,dtype=int)];b=batch_to([DiskSamples(names)[i] for i in range(8)],'cuda');target=b['trajectory'].float();results=[];start=time.perf_counter()
 for seed,mode in [(1960,'jepa'),(1960,'no_jepa'),(1961,'jepa'),(1961,'no_jepa')]:
  task=FreshTask(cfg,'fresh-no-checkpoint',hashlib.sha256(json.dumps(sel,sort_keys=True).encode()).hexdigest(),seed,mode);ck=torch.load(src/f'{seed}-{mode}/step1024.ckpt',map_location='cpu',weights_only=False);task.load_state_dict(ck['state_dict'],strict=True);del ck;m=task.model.cuda().eval();m.requires_grad_(False);rows=[]
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   context=m.context(b);poison={**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(target,999)};assert torch.equal(context,m.context(poison));torch.manual_seed(1970);noise=torch.randn_like(target)
   for tval in [.25,.5,.75,.95]:
    t=torch.full((len(target),1),tval,device='cuda');xt=(1-tval)*noise+tval*target;pred=m.velocity(context,xt,t).float();endpoint=xt+(1-tval)*pred;rows.append({'path':'target_informed_bridge','t':tval,'fm_mse':float((pred-(target-noise)).square().mean()),**metrics(endpoint,b,target)})
   for steps in [16,32]:
    x=noise.clone()
    for i in range(steps):x=x+m.velocity(context,x,torch.full((len(x),1),i/steps,device='cuda')).float()/steps
    raw=x.clone();post=x.clone();post[...,2:4]=torch.nn.functional.normalize(post[...,2:4],dim=-1,eps=1e-6);post[...,4]=post[...,4].clamp_min(0)
    rows.extend([{'path':'pure_noise_raw','steps':steps,**metrics(raw,b,target)},{'path':'pure_noise_postprocessed','steps':steps,**metrics(post,b,target)}])
  result={'seed':seed,'mode':mode,'rows':rows,'actor_poison_exact':True};results.append(result);print('RESULT '+json.dumps(result),flush=True);del task,m,context,poison;gc.collect();torch.cuda.empty_cache()
 report={'windows':names,'t0_scalar_speed_mps':(b['state'][:,0]*30).tolist(),'target_metrics':metrics(target,b,target),'results':results,'wall_s':time.perf_counter()-start,'limits':'8 reused development windows,one noise; target-informed bridge is diagnostic only, never online input; all frozen; no test; t0 lateral velocity assumed0; sourceOCCcausality unverified; not driving acceptance'};(out/'report.json').write_text(json.dumps(report,indent=2));print('DONE',flush=True)
if __name__=='__main__':main()
