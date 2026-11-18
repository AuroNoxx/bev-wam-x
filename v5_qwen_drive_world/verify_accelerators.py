"""ROCm operator forward/backward, causal suffix, and optimizer checks."""
from .runtime import ROOT, configure, math_attention
configure()
import json
import time
import traceback
import torch
from torch.nn import functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules import FusedRMSNormGated
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule, Qwen3_5RMSNormGated
from .accelerators import fla_causal_conv1d


def difference(actual, reference):
    actual, reference = actual.detach().float(), reference.detach().float()
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    delta=(actual-reference)
    relative=float(delta.norm()/reference.norm().clamp_min(1e-12))
    maximum=float(delta.abs().max())
    assert relative < .03 and maximum < .03*max(1.,float(reference.abs().max())), (relative,maximum)
    return dict(relative_l2=relative,max_abs=maximum)


def compare(function,reference,values,prefix_axis=1):
    inputs=[value.detach().clone().requires_grad_() for value in values]
    truth=[value.detach().clone().requires_grad_() for value in values]
    actual=function(*inputs)
    expected=reference(*truth)
    output=difference(actual,expected)
    probe=torch.randn_like(actual)
    grads=torch.autograd.grad((actual.float()*probe.float()).sum(),inputs,retain_graph=True)
    refgrads=torch.autograd.grad((expected.float()*probe.float()).sum(),truth)
    metrics=[difference(a,b) for a,b in zip(grads,refgrads)]
    stop=actual.shape[prefix_axis]//2
    selection=[slice(None)]*actual.ndim;selection[prefix_axis]=slice(0,stop)
    causal=torch.autograd.grad(actual[tuple(selection)].float().square().sum(),inputs)
    # Only tensors carrying the sequence dimension have a suffix.
    suffix=[]
    for gradient in causal:
        if gradient.ndim > prefix_axis and gradient.shape[prefix_axis] == actual.shape[prefix_axis]:
            selection=[slice(None)]*gradient.ndim;selection[prefix_axis]=slice(stop,None)
            value=float(gradient[tuple(selection)].abs().max())
            assert value == 0, value
            suffix.append(value)
    return dict(output=output,gradients=metrics,suffix_gradient_max=suffix)


def main():
    torch.set_num_threads(6);torch.manual_seed(71)
    torch.cuda.set_per_process_memory_fraction(.4,0)
    report=dict(phase='starting',formal_training=False)
    path=ROOT/'artifacts/v5-qwen-drive-world/fused-operator-verification.json'
    def save():path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save(); started=time.perf_counter()
    try:
        report['phase']='gated_delta_rule';save()
        shape=(1,256,32,128)
        values=[torch.randn(shape,device='cuda',dtype=torch.bfloat16)*.5 for _ in range(3)]
        values += [-F.softplus(torch.randn(1,256,32,device='cuda')),torch.rand(1,256,32,device='cuda',dtype=torch.bfloat16)]
        def call(fn,*xs):return fn(*xs,use_qk_l2norm_in_kernel=True)[0]
        report['gated_delta_rule']=compare(lambda *xs:call(chunk_gated_delta_rule,*xs),lambda *xs:call(torch_chunk_gated_delta_rule,*xs),values)
        report['phase']='causal_conv1d';save()
        x=torch.randn(1,8192,256,device='cuda',dtype=torch.bfloat16)*.2
        weight=torch.randn(8192,4,device='cuda',dtype=torch.bfloat16)*.1
        bias=torch.randn(8192,device='cuda',dtype=torch.bfloat16)*.1
        report['causal_conv1d']=compare(lambda a,w,b:fla_causal_conv1d(a,w,b,activation='silu'),
             lambda a,w,b:F.silu(F.conv1d(a,w[:,None],b,padding=3,groups=a.shape[1])[...,:a.shape[-1]]),[x,weight,bias],prefix_axis=2)
        report['phase']='gated_norm';save()
        norm=FusedRMSNormGated(128,eps=1e-6,activation='silu').cuda().to(torch.bfloat16)
        ref=Qwen3_5RMSNormGated(128,eps=1e-6).cuda().to(torch.bfloat16)
        ref.load_state_dict(norm.state_dict(),strict=True)
        values=[torch.randn(128,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
        report['gated_norm']=compare(norm,ref,values,prefix_axis=0)
        report['phase']='fused_adamw';save()
        a=torch.nn.Parameter(torch.randn(128,128,device='cuda'));b=torch.nn.Parameter(a.detach().clone())
        fused=torch.optim.AdamW([a],lr=1e-4,fused=True)
        reference=torch.optim.AdamW([b],lr=1e-4,foreach=True)
        for _ in range(3):
            grad=torch.randn_like(a);a.grad=grad;b.grad=grad.clone()
            fused.step();reference.step()
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        for field in ('exp_avg','exp_avg_sq'):
            assert fused.state[a][field].dtype == torch.float32
            torch.testing.assert_close(fused.state[a][field],reference.state[b][field],atol=1e-6,rtol=1e-6)
        report['fused_adamw']=dict(passed=True,fp32_states=True)
        torch.cuda.synchronize()
        report['phase']='passed'
    except Exception:
        report.update(failed_at=report['phase'],phase='failed',error=traceback.format_exc())
        raise
    finally:
        report['seconds']=time.perf_counter()-started
        save();print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
