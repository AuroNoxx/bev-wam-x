"""Value/gradient equivalence for query tiling of the official grid sampler."""
from .runtime import configure, ROOT
configure()
import json
import torch
from qwen_drive_perception.layers import multi_scale_deformable_attn_pytorch
from .perception_ops import deformable_attention
from .test_contracts import test_voxel, test_voxel_bf16


def check_deformable(device):
    torch.manual_seed(71)
    shapes = torch.tensor([[4,4],[2,2]],device=device)
    starts = torch.tensor([0,16],device=device)
    value = torch.randn(2,20,4,8,device=device,requires_grad=True)
    points = torch.rand(2,517,4,2,3,2,device=device,requires_grad=True)
    weights = torch.randn(2,517,4,2,3,device=device).flatten(-2).softmax(-1).reshape(2,517,4,2,3).requires_grad_()
    reference = multi_scale_deformable_attn_pytorch(value,shapes,points,weights)
    tiled = deformable_attention(value,shapes,starts,points,weights,64)
    probe = torch.randn_like(reference)
    ref_grads = torch.autograd.grad((reference*probe).sum(),(value,points,weights))
    new_grads = torch.autograd.grad((tiled*probe).sum(),(value,points,weights))
    torch.testing.assert_close(tiled,reference,atol=1e-6,rtol=1e-5)
    errors=[]
    for actual,expected in zip(new_grads,ref_grads):
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=1e-4)
        errors.append(float((actual-expected).abs().max()))
    return dict(queries=517,chunk=256,value_max_abs=float((tiled.detach()-reference.detach()).abs().max()),
                gradient_max_abs=errors)


def main():
    results={}
    for device in ('cpu','cuda'):
        results[device]={'deformable':check_deformable(device),'voxel':test_voxel(device),'voxel_bf16':test_voxel_bf16(device)}
    torch.cuda.synchronize()
    (ROOT/'artifacts/v5-qwen-drive-world/perception-memory-operators.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)


if __name__ == '__main__':
    main()
