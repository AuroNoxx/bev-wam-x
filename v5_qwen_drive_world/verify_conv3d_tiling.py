"""Actual World encoder dimensions: native full vs depth-tiled Conv3d gradients."""
from .runtime import configure,ROOT
configure()
import os,json,time,gc
import torch
from .conv3d import DepthTiledConv3d


def main():
    os.environ['PYTORCH_MIOPEN_SUGGEST_NHWC']='0'
    torch.manual_seed(71);torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(.4)
    x=torch.randn(1,256,16,128,128)*.1
    model=torch.nn.Conv3d(256,256,3,padding=1)
    state=model.state_dict()
    results=[];rows=[]
    for cls in (torch.nn.Conv3d,DepthTiledConv3d):
        layer=cls(256,256,3,padding=1).cuda();layer.load_state_dict(state)
        a=x.cuda().requires_grad_()
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            y=layer(a)
            gradients=torch.autograd.grad(y.float().square().mean(),(a,layer.weight,layer.bias))
        torch.cuda.synchronize()
        rows.append(dict(operator=cls.__name__,seconds=time.perf_counter()-started,
                         peak_GiB=torch.cuda.max_memory_allocated()/2**30))
        results.append([t.detach().cpu().float() for t in (y,*gradients)])
        del layer,a,y,gradients
        gc.collect();torch.cuda.empty_cache()
    errors=[]
    for a,b in zip(*results):
        assert torch.isfinite(b).all()
        rel=float((a-b).norm()/a.norm().clamp_min(1e-12));assert rel<.03,rel
        errors.append(rel)
    result=dict(phase='passed',shape=list(x.shape),measurements=rows,
                output_and_gradients_relative_l2=errors)
    (ROOT/'artifacts/v5-qwen-drive-world/conv3d-depth-tiling.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
