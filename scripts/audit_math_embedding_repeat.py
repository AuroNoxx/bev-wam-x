"""Exact original graph/reduction/backend comparison. No updates."""
from train_carla_reactive_history256 import *
from fixed_causal_flow import FixedCausalFlow
from contextlib import nullcontext
from torch.nn.attention import sdpa_kernel,SDPBackend
O=ROOT/'artifacts/fm-math-embedding-repeat-v1'
def run():
 safe_idle();configure_runtime();O.mkdir(exist_ok=False);ck=torch.load(ROOT/'artifacts/carla-reactive-history-paired256/2071-history/latest.ckpt',map_location='cpu',weights_only=False);m=FixedCausalFlow(50).cuda().train();m.load_state_dict({k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')});d=json.loads((ROOT/'data/carla_reactive_hold_cache_v1/manifest.json').read_text());ReactiveData.acceleration=True;ds=ReactiveData(d['train']);b=batch_to([ds[0],ds[70]],'cuda');mask=torch.arange(25,device='cuda')[None]<torch.tensor([5,18],device='cuda')[:,None];torch.manual_seed(2900);noise=torch.randn(2,25,2,device='cuda');t=torch.tensor([[.15],[.85]],device='cuda');safe=torch.where(mask[:,:,None],m.to_flow_space(b,b['trajectory']),noise);xt=(1-t[:,:,None])*noise+t[:,:,None]*safe;target=safe-noise;suffix=torch.randn_like(noise);xp=xt.clone();xp[~mask]=suffix[~mask];pars=[p for p in m.parameters() if p.requires_grad];names=[n for n,p in m.named_parameters() if p.requires_grad];result={}
 for mode,lossfp,backend in [('bf16_math',False,'math'),('fp32_math',False,'math')]:
  def calc(x):
   x=x.detach().clone().requires_grad_()
   with (sdpa_kernel(SDPBackend.MATH) if backend=='math' else nullcontext()):
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mode!='fp32_math'):
     p=m(b,x,t)
     with (torch.autocast('cuda',enabled=False) if lossfp else nullcontext()):
      ep=m.from_flow_space(b,x+(1-t[:,:,None])*p);err=torch.where(mask[:,:,None],p.float()-target,torch.zeros_like(p.float()));physical=torch.where(mask[:,:,None],ep[:,:,:2],torch.zeros_like(ep[:,:,:2]));loss=err.square().sum()/p.numel()+physical.square().sum()/physical.numel()
    g=torch.autograd.grad(loss,pars+[x],allow_unused=True,retain_graph=True)
   return p.detach(),g
  runs=[]
  for repeat in range(3):
   p,a=calc(xt);r,ar=calc(xt);q,z=calc(xp);values=[(n,float((g-h).abs().max()),float((g-h).norm()/g.norm().clamp_min(1e-20))) for n,g,h in zip(names,a[:-1],z[:-1]) if g is not None];values.sort(key=lambda v:v[1],reverse=True);repeatmax=max(float((g-h).abs().max()) for g,h in zip(a[:-1],ar[:-1]) if g is not None);maxref=max(float(g.abs().max()) for g in a[:-1] if g is not None);runs.append({'max_abs':values[0][1],'top_parameter':values[0][0],'top_relative_l2':values[0][2],'repeat_input_max':repeatmax,'prefix_output_max':float((p[mask]-q[mask]).abs().max()),'suffix_input_gradient_max':float(a[-1][~mask].abs().max()),'original_tolerance':max(1e-6,maxref*1e-5),'passes_original_tolerance':values[0][1]<=max(1e-6,maxref*1e-5)})
  result[mode]=runs;(O/'report.json').write_text(json.dumps({'results':result,'optimizer_steps':0},indent=2));print(mode,json.dumps(runs),flush=True)
if __name__=='__main__':run()
