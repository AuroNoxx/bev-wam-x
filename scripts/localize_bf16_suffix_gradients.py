"""Frozen BF16 backward localization; no optimizer."""
from train_carla_reactive_history256 import *
from fixed_causal_flow import FixedCausalFlow
from contextlib import nullcontext
from torch.nn.attention import sdpa_kernel,SDPBackend
O=ROOT/'artifacts/bf16-suffix-gradient-localization-v1'
def diff(a,b):
 if a is None:return None
 x=a.float();y=b.float();return {'max_abs':float((x-y).abs().max()),'relative_l2':float((x-y).norm()/x.norm().clamp_min(1e-20)),'reference_max':float(x.abs().max())}
def run():
 safe_idle();configure_runtime();O.mkdir(exist_ok=False);start=time.perf_counter();source=ROOT/'artifacts/carla-reactive-history-paired256/2070-history/latest.ckpt';sha=hashlib.sha256(source.read_bytes()).hexdigest();ck=torch.load(source,map_location='cpu',weights_only=False);m=FixedCausalFlow(50).cuda().train();m.load_state_dict({k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')});snapshot={k:v.clone() for k,v in m.state_dict().items()};d=json.loads((ROOT/'data/carla_reactive_hold_cache_v1/manifest.json').read_text());ReactiveData.acceleration=True;ds=ReactiveData(d['train']);b=batch_to([ds[130],ds[340]],'cuda');mask=torch.arange(25,device='cuda')[None]<torch.tensor([7,16],device='cuda')[:,None];torch.manual_seed(2802);noise=torch.randn(2,25,2,device='cuda');t=torch.tensor([[.35],[.8]],device='cuda');clean=m.to_flow_space(b,b['trajectory']);safe=torch.where(mask[:,:,None],clean,noise);xt=(1-t[:,:,None])*noise+t[:,:,None]*safe;target=safe-noise;xp=xt.clone();xp[~mask]=torch.randn_like(xp[~mask])*1000
 configs=[('fp32',False,False,'combined','default'),('bf16',True,True,'combined','default'),('context_bf16_decoder_fp32',True,False,'combined','default'),('context_fp32_decoder_bf16',False,True,'combined','default'),('bf16_FM_only',True,True,'fm','default'),('bf16_position_only',True,True,'position','default'),('bf16_math_attention',True,True,'combined','math')];results={};names=[name for name,p in m.named_parameters() if p.requires_grad];pars=[p for p in m.parameters() if p.requires_grad]
 for name,ctxbf,velbf,losskind,backend in configs:
  def calc(x):
   x=x.detach().clone().requires_grad_()
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=ctxbf):ctx=m.context(b)
   with (sdpa_kernel(SDPBackend.MATH) if backend=='math' else nullcontext()):
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=velbf):p=m.velocity(ctx.float() if not velbf else ctx,x,t)
    # Always explicitly FP32 loss/reconstruction, preserving original objective.
    with torch.autocast('cuda',enabled=False):
     p32=p.float();ep=m.from_flow_space(b,x+(1-t[:,:,None])*p32);fm=torch.where(mask[:,:,None],p32-target,torch.zeros_like(p32)).square().sum()/p.numel();position=torch.where(mask[:,:,None],ep[:,:,:2],torch.zeros_like(ep[:,:,:2])).square().sum()/(2*25*2);loss=fm+position if losskind=='combined' else fm if losskind=='fm' else position
    grads=torch.autograd.grad(loss,pars+[ctx,x,p],allow_unused=True)
   return p.detach(),grads
  p,a=calc(xt);pr,ar=calc(xt);q,z=calc(xp);pd=[(n,diff(g,h)) for n,g,h in zip(names,a[:-3],z[:-3]) if g is not None];pd.sort(key=lambda x:x[1]['max_abs'],reverse=True);repeat=max(diff(g,h)['max_abs'] for g,h in zip(a[:-3],ar[:-3]) if g is not None)
  r={'prefix_output_maxdiff':float((p[mask]-q[mask]).abs().max()),'parameter_top8':[{'name':n,**v} for n,v in pd[:8]],'same_input_repeat_max_parameter_diff':repeat,'context_gradient':diff(a[-3],z[-3]),'flow_output_gradient':diff(a[-1],z[-1]),'suffix_input_gradient_max':float(a[-2][~mask].abs().max()),'all_gradients_finite':all(torch.isfinite(g).all().item() for g in a if g is not None),'original_threshold_pass':pd[0][1]['max_abs']<=max(1e-6,max(float(g.abs().max()) for g in a[:-3] if g is not None)*1e-5)};results[name]=r;(O/'partial.json').write_text(json.dumps(results,indent=2));print(name,json.dumps(r),flush=True)
 assert all(torch.equal(snapshot[k],v) for k,v in m.state_dict().items());assert hashlib.sha256(source.read_bytes()).hexdigest()==sha;(O/'report.json').write_text(json.dumps({'results':results,'optimizer_steps':0,'state_unchanged':True,'source_sha':sha,'wall_s':time.perf_counter()-start,'mask_same_batch_fixed':True},indent=2));print('COMPLETE',time.perf_counter()-start)
if __name__=='__main__':run()
