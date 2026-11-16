"""Isolated structural FM missing-suffix tests. No optimizer/model checkpoint or training."""
import torch,json
from torch import nn
from pathlib import Path
O=Path('artifacts/fm-prefix-causality-tests-v1');O.mkdir(exist_ok=False);torch.manual_seed(2801)
class Toy(nn.Module):
 def __init__(self,causal=True,global_mix=False):
  super().__init__();self.inp=nn.Linear(2,16);self.qkv=nn.Linear(16,48);self.norm=nn.LayerNorm(16);self.out=nn.Linear(16,2);self.causal=causal;self.global_mix=global_mix
 def forward(self,x,t,context):
  # Full fixed length for every example. No validity/cutoff argument.
  h=self.inp(x)+context[:,None,:]+t[:,None,None];q,k,v=self.qkv(h).chunk(3,-1);logits=q@k.transpose(-1,-2)/4
  if self.causal:logits=logits.masked_fill(torch.triu(torch.ones(x.shape[1],x.shape[1],dtype=torch.bool),1),-torch.inf)
  h=self.norm(h+logits.softmax(-1)@v)
  if self.global_mix:h=h+h.mean(1,keepdim=True)
  return self.out(h)
def bridge(clean,noise,t,label_valid):
 # Missing labels removed BEFORE arithmetic. Invalid bridge is source noise for all flow-times.
 safe=torch.where(label_valid[...,None],clean,noise)
 return (1-t[:,None,None])*noise+t[:,None,None]*safe,safe-noise

def prefix_loss(pred,target,label_valid):
 safe_target=torch.where(label_valid[...,None],target,pred.detach())
 # Original denominator, no mask/valid-length input to network.
 return (pred-safe_target).square().sum()/pred.numel()

m=Toy();clean=torch.randn(2,25,2);noise=torch.randn_like(clean);t=torch.tensor([.35,.8]);context=torch.randn(2,16);cutoff=[7,16];mask=torch.arange(25)[None,:]<torch.tensor(cutoff)[:,None];xt,target=bridge(clean,noise,t,mask);poison=xt.clone();poison[~mask]=torch.randn_like(poison[~mask])*1000;xt.requires_grad_();p=m(xt,t,context);q=m(poison,t,context);assert torch.equal(p[mask],q[mask]);l=prefix_loss(p,target,mask);gp=torch.autograd.grad(l,tuple(m.parameters())+(xt,),retain_graph=True);gq=torch.autograd.grad(prefix_loss(q,target,mask),tuple(m.parameters()));assert all(torch.equal(a,b) for a,b in zip(gp[:-1],gq));assert torch.count_nonzero(gp[-1][~mask])==0
nan_clean=clean.clone();nan_clean[~mask]=float('nan');xn,tn=bridge(nan_clean,noise,t,mask);assert torch.equal(xn,xt.detach()) and torch.isfinite(tn).all()
# Same forward independent of evaluation mask choice; lengths never reach it.
assert torch.equal(m(xt,t,context),m(xt,t,context))
negative={}
for name,a,b in [('bidirectional',False,False),('causal_then_global_pool',True,True)]:
 bad=Toy(a,b);bad.load_state_dict(m.state_dict());delta=float((bad(xt,t,context)[mask]-bad(poison,t,context)[mask]).abs().max().detach());assert delta>1e-4;negative[name]=delta
# Prefix decoder cumulative integration does not read future token controls.
v=torch.randn(2,25,2);vbad=v.clone();vbad[~mask]=999;xy=v.cumsum(1)*.2;xybad=vbad.cumsum(1)*.2;assert torch.equal(xy[mask],xybad[mask])
r={'fixed_causal_prefix_output_poison_exact':True,'parameter_gradient_poison_exact':True,'gradient_to_suffix_zero':True,'missing_nan_sanitized_before_bridge':True,'decoder_prefix_integration_exact':True,'negative_controls_max_prefix_change':negative,'optimizer_steps':0,'production_model_tested':False,'scope':'structural toy proof only, not current bidirectional FM implementation or driving guarantee','known_initialization_seed':2801};(O/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
