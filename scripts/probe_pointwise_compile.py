"""Bounded pointwise compile feasibility; not a full-model benchmark."""
import json,time,torch
from lightning_task import ROOT,configure_runtime
from train_joint_occ import resource_guard

def gate(x,y):return torch.sigmoid(x)*torch.tanh(y)+(1-torch.sigmoid(x))*y

def main():
    resource_guard();configure_runtime();torch.manual_seed(10);a=torch.randn(8,160,768,device='cuda',dtype=torch.bfloat16);b=torch.randn_like(a);result={'scope':'pointwise gate only, not full model/attention compilation','torch':torch.__version__}
    refs=None
    for name,fn in [('eager',gate),('compiled',torch.compile(gate,fullgraph=True))]:
        x=a.detach().clone().requires_grad_();y=b.detach().clone().requires_grad_();times=[];first=None
        for i in range(23):
            x.grad=None;y.grad=None;torch.cuda.synchronize();t=time.perf_counter();z=fn(x,y);z.float().square().mean().backward();torch.cuda.synchronize();elapsed=time.perf_counter()-t
            if i==0:first=elapsed
            if i>=3:times.append(elapsed*1000)
        row={'first_forward_backward_seconds':first,'hot_forward_backward_ms':sum(times)/len(times),'finite_gradients':bool(torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all())}
        if refs is None:refs=[z.detach().clone(),x.grad.clone(),y.grad.clone()]
        else:row['max_abs_differences_output_dx_dy']=[float((v.float()-r.float()).abs().max()) for v,r in zip([z,x.grad,y.grad],refs)]
        result[name]=row;print(json.dumps({name:row}),flush=True)
    p=ROOT/'artifacts/pointwise-compile-probe-20261115.json';assert not p.exists();p.write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
