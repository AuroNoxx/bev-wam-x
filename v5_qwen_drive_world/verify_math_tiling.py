"""Verify native SDPA query tiling, masks, BF16 gradients and causality."""
from .runtime import configure,ROOT,math_attention
configure()
import json
from types import SimpleNamespace
import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from .accelerators import tiled_native_sdpa


def main():
    torch.manual_seed(71)
    rows=[]
    for kind in ('causal','noncausal','boolean','additive'):
        module=SimpleNamespace(num_key_value_groups=4,is_causal=kind=='causal')
        tensors=[torch.randn(1,h,777,64,device='cuda',dtype=torch.bfloat16)*.1 for h in (8,2,2)]
        mask=None
        if kind in ('boolean','additive'):
            mask=torch.ones(1,1,777,777,device='cuda',dtype=torch.bool).tril()
            if kind=='additive':
                mask=torch.zeros_like(mask,dtype=torch.float32).masked_fill(~mask,float('-inf'))
        results=[]
        for fn in (sdpa_attention_forward,tiled_native_sdpa):
            inputs=[x.clone().requires_grad_() for x in tensors]
            with math_attention():
                y=fn(module,*inputs,mask)[0]
                grad=torch.autograd.grad(y[:,:300].float().square().sum(),inputs)
            results.append((y.detach(),grad))
        diffs=[]
        for a,b in zip((results[0][0],*results[0][1]),(results[1][0],*results[1][1])):
            assert torch.isfinite(b).all()
            error=float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-12))
            assert error<.02,(kind,error)
            diffs.append(error)
        suffix=[float(g[:,:,300:].abs().max()) for g in results[1][1]]
        if kind!='noncausal': assert max(suffix)==0,suffix
        rows.append(dict(mask=kind,relative_l2_output_and_gradients=diffs,suffix_gradient_max=suffix))
    result=dict(phase='passed',query_chunk=256,tests=rows)
    (ROOT/'artifacts/v5-qwen-drive-world/math-query-tiling.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
