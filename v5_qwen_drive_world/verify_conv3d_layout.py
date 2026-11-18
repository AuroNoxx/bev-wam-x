"""Native MIOpen layout verification at the actual failing World conv shape."""
from .runtime import configure,ROOT
configure()
import os,json,time,gc
import torch
from torch.nn import functional as F


def main():
    torch.manual_seed(71)
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(.4)
    os.environ['PYTORCH_MIOPEN_SUGGEST_NHWC']='1'
    x=torch.randn(1,192,16,128,128)*.1
    w=torch.randn(64,192,3,3,3)*.01
    rows=[]; results=[]
    for name,fmt in [('contiguous',torch.contiguous_format),('channels_last_3d',torch.channels_last_3d)]:
        a=x.cuda().contiguous(memory_format=fmt).requires_grad_()
        b=w.cuda().contiguous(memory_format=fmt).requires_grad_()
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            y=F.conv3d(a,b,padding=1)
            grad=torch.autograd.grad(y.float().square().mean(),(a,b))
        torch.cuda.synchronize()
        rows.append(dict(layout=name,seconds=time.perf_counter()-started,
                         peak_GiB=torch.cuda.max_memory_allocated()/2**30))
        results.append([t.detach().cpu().float() for t in (y,*grad)])
        del a,b,y,grad
        gc.collect();torch.cuda.empty_cache()
    errors=[]
    for a,b in zip(*results):
        assert torch.isfinite(b).all()
        rel=float((a-b).norm()/a.norm().clamp_min(1e-12));assert rel<.03,rel
        errors.append(rel)
    result=dict(phase='passed',shape=list(x.shape),kernel=list(w.shape),
                measurements=rows,output_and_gradient_relative_l2=errors)
    (ROOT/'artifacts/v5-qwen-drive-world/conv3d-layout.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
