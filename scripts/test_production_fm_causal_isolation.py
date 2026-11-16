"""Production-code frozen causal-isolation tests. No optimizer or training."""
from train_carla_reactive_history256 import *
from fixed_causal_flow import FixedCausalFlow
O=ROOT/'artifacts/production-fm-causal-isolation-v3'
def run():
 safe_idle();configure_runtime();O.mkdir(exist_ok=False);start=time.perf_counter();source=ROOT/'artifacts/carla-reactive-history-paired256/2070-history/latest.ckpt';sha=hashlib.sha256(source.read_bytes()).hexdigest();ck=torch.load(source,map_location='cpu',weights_only=False);w={k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')};m=FixedCausalFlow(50).cuda();m.load_state_dict(w);old=NavigationScaledFlow(50).cuda();old.load_state_dict(w);before={k:v.clone() for k,v in m.state_dict().items()};d=json.loads((ROOT/'data/carla_reactive_hold_cache_v1/manifest.json').read_text());ReactiveData.acceleration=True;ds=ReactiveData(d['train']);b=batch_to([ds[130],ds[340]],'cuda');mask=torch.arange(25,device='cuda')[None]<torch.tensor([7,16],device='cuda')[:,None];torch.manual_seed(2802);noise=torch.randn(2,25,2,device='cuda');t=torch.tensor([[.35],[.8]],device='cuda');clean=m.to_flow_space(b,b['trajectory']);safe=torch.where(mask[:,:,None],clean,noise);xt=(1-t[:,:,None])*noise+t[:,:,None]*safe;target=safe-noise;xp=xt.clone();xp[~mask]=torch.randn_like(xp[~mask])*1000;result={}
 for precision in ['fp32','bf16']:
  enabled=precision=='bf16';entry={};m.train()
  def grads(x):
   x=x.detach().clone().requires_grad_()
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled):
    p=m(b,x,t);endpoint=m.from_flow_space(b,x+(1-t[:,:,None])*p);err=torch.where(mask[:,:,None],p.float()-target,torch.zeros_like(p.float()));loss=err.square().sum()/p.numel();physical=torch.where(mask[:,:,None],endpoint[:,:,:2],torch.zeros_like(endpoint[:,:,:2]));loss=loss+physical.square().sum()/physical.numel()
   pars=[p for p in m.parameters() if p.requires_grad];g=torch.autograd.grad(loss,pars+[x],allow_unused=True);return p.detach(),endpoint.detach(),g
  p,ep,g=grads(xt);repeat_p,repeat_ep,repeat_g=grads(xt);entry["same_input_repeat_output_maxdiff"]=float((p-repeat_p).abs().max());entry["same_input_repeat_gradient_maxdiff"]=max(float((a-b).abs().max()) for a,b in zip(g[:-1],repeat_g[:-1]) if a is not None);q,eq,h=grads(xp);outdiff=float((p[mask]-q[mask]).abs().max());physicaldiff=float((ep[mask]-eq[mask]).abs().max());gd=max(float((a-b).abs().max()) for a,b in zip(g[:-1],h[:-1]) if a is not None);gn=max(float(a.abs().max()) for a in g[:-1] if a is not None);suffixgrad=float(g[-1][~mask].abs().max());assert outdiff==0 and physicaldiff==0 and suffixgrad==0;entry["parameter_gradient_original_tolerance_pass"]=gd<=max(1e-6,gn*1e-5);entry["parameter_gradient_original_tolerance"]=max(1e-6,gn*1e-5)
  names=[name for name,p in m.named_parameters() if p.requires_grad];entry["gradient_diff_top5"]=sorted([{ "name":name,"maxdiff":float((a-b).abs().max())} for name,a,b in zip(names,g[:-1],h[:-1]) if a is not None],key=lambda x:x["maxdiff"],reverse=True)[:5]
  entry.update(prefix_output_maxdiff=outdiff,decoded_prefix_maxdiff=physicaldiff,parameter_gradient_maxdiff=gd,parameter_gradient_maxabs=gn,suffix_input_gradient_max=suffixgrad,finite_gradients=all(torch.isfinite(a).all().item() for a in g if a is not None));m.eval()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled):
   ctx=m.context(b);other=m.context({**b,'future':torch.full_like(b['future'],255),'trajectory':torch.full_like(b['trajectory'],999)});assert torch.equal(ctx,other);entry['context_future_poison_exact']=True;oldp=old(b,xt,t);oldq=old(b,xp,t);entry['original_bidirectional_prefix_change']=float((oldp[mask]-oldq[mask]).abs().max());assert entry['original_bidirectional_prefix_change']>1e-5
   for steps in [4,16]:
    a=noise.clone();z=noise.clone();z[~mask]=torch.randn_like(z[~mask])*1000
    for i in range(steps):
     tt=torch.full((2,1),i/steps,device='cuda');a=a+m.velocity(ctx,a,tt)/steps;z=z+m.velocity(ctx,z,tt)/steps
    aa=m.from_flow_space(b,a);zz=m.from_flow_space(b,z);diff=float((a[mask]-z[mask]).abs().max());phys=float((aa[mask]-zz[mask]).abs().max());assert diff==0 and phys==0;entry[f'euler{steps}']={'latent_prefix_maxdiff':diff,'physical_prefix_maxdiff':phys,'finite_all_tokens':bool(torch.isfinite(a).all() and torch.isfinite(z).all())}
  result[precision]=entry;print(precision,json.dumps(entry),flush=True)
 assert all(torch.equal(before[k],v) for k,v in m.state_dict().items());assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
 report={'results':result,'source_checkpoint':str(source),'source_sha256':sha,'state_and_buffers_unchanged':True,'optimizer_steps':0,'fixed_same25token_topology':True,'label_mask_not_forward_argument':True,'decoder':'existing2Dvelocity cumulative trapezoid; not12globalcurve','limits':['one2070checkpoint,twoactualtraincontexts','architectureonlytest,notmodelquality','FP32/BF16 nativePyTorchdecoder onAMD; no separatecompile/backend sweep','oldscalarinitialvelocity/unknownOCC/physicslimitationsremain'],'wall_s':time.perf_counter()-start,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30};(O/'report.json').write_text(json.dumps(report,indent=2));print('COMPLETE',report['wall_s'])
if __name__=='__main__':run()
