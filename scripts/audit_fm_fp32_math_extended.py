"""Extended train-only frozen backend validation and cost; no optimizer."""
from train_carla_reactive_history256 import *
from fixed_causal_flow import FixedCausalFlow
from contextlib import nullcontext
from torch.nn.attention import sdpa_kernel,SDPBackend
O=ROOT/'artifacts/fm-fp32-math-extended-audit-v1'
def backend(name):return sdpa_kernel(SDPBackend.MATH) if name=='math' else nullcontext()
def run():
 safe_idle();configure_runtime();O.mkdir(exist_ok=False);start=time.perf_counter();d=json.loads((ROOT/'data/carla_reactive_hold_cache_v1/manifest.json').read_text());ReactiveData.acceleration=True;ds=ReactiveData(d['train']);indices=list(range(0,491,70));design={'train_window_indices':indices,'raw_anchor_indices':[i+36 for i in indices],'independence':'nonoverlapping time windows, SAME one train episode/road, not independent scenarios','seeds':[2070,2071],'suffix_normal_std':[.1,1,3,10],'threshold':'max(1e-6,max_parameter_gradient*1e-5), unchanged','updates':0,'precision':'FP32','paired_repeats':3};(O/'design.json').write_text(json.dumps(design,indent=2));results=[];samples=[];perf={}
 for seed in [2070,2071]:
  source=ROOT/f'artifacts/carla-reactive-history-paired256/{seed}-history/latest.ckpt';sha=hashlib.sha256(source.read_bytes()).hexdigest();ck=torch.load(source,map_location='cpu',weights_only=False);m=FixedCausalFlow(50).cuda();m.load_state_dict({k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')});snapshot={k:v.clone() for k,v in m.state_dict().items()};pars=[p for p in m.parameters() if p.requires_grad];names=[n for n,p in m.named_parameters() if p.requires_grad]
  for pair in range(4):
   b=batch_to([ds[indices[2*pair]],ds[indices[2*pair+1]]],'cuda');mask=torch.arange(25,device='cuda')[None]<torch.tensor([5+pair,18-pair],device='cuda')[:,None];torch.manual_seed(2900+pair);noise=torch.randn(2,25,2,device='cuda');suffix=torch.randn_like(noise);t=torch.tensor([[.15+.2*pair],[.85-.15*pair]],device='cuda');safe=torch.where(mask[:,:,None],m.to_flow_space(b,b['trajectory']),noise);xt=(1-t[:,:,None])*noise+t[:,:,None]*safe;target=safe-noise
   for kind in ['math']:
    m.train()
    def gradients(x):
     x=x.detach().clone().requires_grad_()
     with backend(kind):
      with torch.autocast('cuda',enabled=False):
       p=m(b,x,t);ep=m.from_flow_space(b,x+(1-t[:,:,None])*p);err=torch.where(mask[:,:,None],p.float()-target,torch.zeros_like(p.float()));physical=torch.where(mask[:,:,None],ep[:,:,:2],torch.zeros_like(ep[:,:,:2]));loss=err.square().sum()/p.numel()+physical.square().sum()/physical.numel()
      g=torch.autograd.grad(loss,pars+[x],allow_unused=True)
     return p.detach(),g
    for repeat in range(3):
     p,g=gradients(xt);refmax=max(float(v.abs().max()) for v in g[:-1] if v is not None);tol=max(1e-6,refmax*1e-5)
     rp,rg=gradients(xt);repeatmax=max(float((a-z).abs().max()) for a,z in zip(g[:-1],rg[:-1]) if a is not None)
     for amp in [.1,1,3,10]:
      xp=xt.clone();xp[~mask]=suffix[~mask]*amp;q,h=gradients(xp);deltas=[(name,float((a-z).abs().max()),float((a-z).norm()/a.norm().clamp_min(1e-20))) for name,a,z in zip(names,g[:-1],h[:-1]) if a is not None];top=max(deltas,key=lambda x:x[1]);rec={'seed':seed,'pair':pair,'repeat':repeat,'same_input_repeat_max':repeatmax,'backend':kind,'amplitude':amp,'prefix_maxdiff':float((p[mask]-q[mask]).abs().max()),'gradient_maxdiff':top[1],'top_relative_l2':top[2],'top_parameter':top[0],'tolerance':tol,'suffix_input_gradient_max':float(h[-1][~mask].abs().max()),'finite':all(torch.isfinite(a).all().item() for a in h if a is not None)};rec['pass']=rec['prefix_maxdiff']==0 and rec['gradient_maxdiff']<=tol and rec['suffix_input_gradient_max']==0 and rec['finite'];results.append(rec)
    del g,h,p,q
    m.eval()
    with torch.no_grad(),backend(kind),torch.autocast('cuda',enabled=False):
     context=m.context(b)
     for steps in [4,16]:
      a=noise.clone();z=noise.clone();z[~mask]=suffix[~mask]
      for j in range(steps):
       tt=torch.full((2,1),j/steps,device='cuda');a=a+m.velocity(context,a,tt)/steps;z=z+m.velocity(context,z,tt)/steps
      pa=m.from_flow_space(b,a);pz=m.from_flow_space(b,z);samples.append({'seed':seed,'pair':pair,'backend':kind,'steps':steps,'latent_maxdiff':float((a[mask]-z[mask]).abs().max()),'physical_maxdiff':float((pa[mask]-pz[mask]).abs().max()),'finite':bool(torch.isfinite(a).all() and torch.isfinite(z).all())})
   (O/'partial.json').write_text(json.dumps({'gradients':results,'sampling':samples},indent=2));print('PAIR_DONE',seed,pair,flush=True)
  # Controlled wallclock/peak runs at two batch sizes, same examples and fullprefix loss.
  if seed==2070:
   for bs in [2,12]:
    b=batch_to([ds[indices[j%8]] for j in range(bs)],'cuda');x=torch.randn(bs,25,2,device='cuda');tt=torch.full((bs,1),.5,device='cuda')
    for kind in ['math']:
     times=[];peaks=[];m.train()
     for rep in range(12):
      torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();st=time.perf_counter()
      with backend(kind):
       with torch.autocast('cuda',enabled=False):p=m(b,x,tt);loss=p.float().square().mean()
       grads=torch.autograd.grad(loss,pars,allow_unused=True)
      torch.cuda.synchronize();elapsed=time.perf_counter()-st;peak=torch.cuda.max_memory_allocated()/2**30;del grads,p,loss
      if rep>=2:times.append(elapsed);peaks.append(peak)
     perf[f'batch{bs}_{kind}_forward_backward']={'p50_s':float(np.median(times)),'p95_s':float(np.quantile(times,.95)),'max_allocated_gib':max(peaks),'updates':0,'repeats':10,'includes_context':True}
     m.eval();times=[];peaks=[]
     for rep in range(6):
      torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();st=time.perf_counter()
      with torch.no_grad(),backend(kind),torch.autocast('cuda',enabled=False):
       ctx=m.context(b);xx=x.clone()
       for j in range(16):xx=xx+m.velocity(ctx,xx,torch.full((bs,1),j/16,device='cuda'))/16
       physical=m.from_flow_space(b,xx)
      torch.cuda.synchronize();elapsed=time.perf_counter()-st;peak=torch.cuda.max_memory_allocated()/2**30;del ctx,xx,physical
      if rep>=2:times.append(elapsed);peaks.append(peak)
     perf[f'batch{bs}_{kind}_sample16_K1']={'p50_s':float(np.median(times)),'p95_s':float(np.quantile(times,.95)),'max_allocated_gib':max(peaks),'repeats':4,'includes_context_decode':True}
  assert all(torch.equal(snapshot[k],v) for k,v in m.state_dict().items());assert hashlib.sha256(source.read_bytes()).hexdigest()==sha;del m,ck,snapshot;gc.collect();torch.cuda.empty_cache()
 report={'design':design,'gradients':results,'sampling':samples,'performance':perf,'state_unchanged':True,'updates':0,'wall_s':time.perf_counter()-start};(O/'report.json').write_text(json.dumps(report,indent=2));print('COMPLETE',report['wall_s'],json.dumps(perf),flush=True)
if __name__=='__main__':run()
