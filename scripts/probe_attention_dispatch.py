"""Isolated SDPA dispatch comparison, not whole-model speed claim."""
import json,time,torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from contextlib import nullcontext
from lightning_task import ROOT,configure_runtime
from train_joint_occ import resource_guard

def main():
    resource_guard();configure_runtime();torch.manual_seed(12);q=torch.randn(8,12,25,64,device='cuda');k=torch.randn(8,12,480,64,device='cuda');v=torch.randn_like(k)
    result={'torch':torch.__version__,'hip':torch.version.hip,'shapes':[list(x.shape) for x in [q,k,v]],'scope':'isolated attention forward/backward, no model training','runs':[]}
    with sdpa_kernel(SDPBackend.MATH):ref=torch.nn.functional.scaled_dot_product_attention(q,k,v)
    gradient_reference=None
    for dtype in [torch.float32,torch.bfloat16]:
        for name in ['math','default']:
            x=[z.to(dtype).detach().requires_grad_() for z in [q,k,v]];times=[];ys=None
            try:
                for i in range(23):
                    for z in x:z.grad=None
                    torch.cuda.synchronize();t=time.perf_counter()
                    with (sdpa_kernel(SDPBackend.MATH) if name=='math' else nullcontext()):y=torch.nn.functional.scaled_dot_product_attention(*x);y.float().square().mean().backward()
                    torch.cuda.synchronize()
                    if i>=3:times.append((time.perf_counter()-t)*1000)
                r={'dtype':str(dtype),'backend_request':name,'mean_forward_backward_ms':sum(times)/len(times),'output_max_abs_diff_vs_fp32_math':float((y.float()-ref).abs().max()),'output_rms_diff_vs_fp32_math':float((y.float()-ref).square().mean().sqrt()),'finite_gradients':all(bool(torch.isfinite(z.grad).all()) for z in x)}
            except Exception as e:r={'dtype':str(dtype),'backend_request':name,'error':str(e)}
            if 'error' not in r:
                if gradient_reference is None:gradient_reference=[z.grad.detach().float().clone() for z in x]
                r['gradient_max_abs_diff_vs_fp32_math']=[float((z.grad.float()-g).abs().max()) for z,g in zip(x,gradient_reference)]
            result['runs'].append(r);print(json.dumps(r),flush=True)
    p=ROOT/'artifacts/attention-dispatch-probe-20261115-v2.json';assert not p.exists();p.write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()

