"""Necessary operator, sampler-resume and label-boundary checks; no model training."""
from .runtime import ROOT, configure
configure()
import copy
import json
import traceback
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from .data import SpeedBinBalancedSampler, intent_labels
from .perception_ops import voxel_pool_depth, rocm_perception_ops
from .verify_components import check_perception_operator
from .objectives import footprint_collision, relative_advantages


def test_sampler():
    rows = [{"source": source, "speed_kmh": speed, "index": i}
            for source, speeds in [("bench2drive", [0, 25, 45, 65]), ("highway", [75, 95, 115])]
            for i, speed in enumerate(speeds)]
    sampler = SpeedBinBalancedSampler(rows, num_batches=20)
    loader = StatefulDataLoader(list(range(len(rows))), batch_sampler=sampler, num_workers=0)
    iterator = iter(loader)
    first = [next(iterator) for _ in range(3)]
    state = copy.deepcopy(loader.state_dict())
    expected = [next(iterator) for _ in range(5)]
    restored = StatefulDataLoader(list(range(len(rows))), batch_sampler=SpeedBinBalancedSampler(rows, num_batches=20), num_workers=0)
    restored.load_state_dict(state)
    actual = iter(restored)
    for wanted in expected:
        torch.testing.assert_close(next(actual), wanted, rtol=0, atol=0)
    full = list(SpeedBinBalancedSampler(rows, num_batches=20))
    highway = sum(rows[i]["source"] == "highway" for batch in full for i in batch)
    assert highway == 48, highway
    accel = torch.zeros(5, 50, 2)
    accel[:, :, 0] = torch.tensor([-3.01, -3., -.5, .5, .51])[:, None]
    xy = torch.zeros(5, 50, 2)
    xy[:, :, 1] = torch.tensor([0., 1.2, -1.2, 0., 0.])[:, None]
    labels = intent_labels(accel, xy)
    assert labels.tolist() == [0, 4, 8, 6, 9], labels
    return {"stateful_loader_next_five_batches_exact": True, "highway_fraction_20_batches": highway/160,
            "intent_boundaries": labels.tolist(), "coverage": sampler.coverage()}


def test_voxel(device):
    # Two sweeps and two cameras: same voxel sums within cameras, not across them.
    features = torch.tensor([[[[[2.]], [[3.]]], [[[5.]], [[7.]]],
                              [[[11.]], [[13.]]], [[[17.]], [[19.]]]]], device=device, requires_grad=True)
    depth = torch.tensor([.25, .75, .4, .6, .1, .9, .3, .7], device=device).reshape(4, 2, 1, 1).requires_grad_()
    coords = torch.zeros(1, 2, 2, 2, 1, 1, 4, dtype=torch.long, device=device)
    coords[..., 1] = torch.arange(2, device=device).view(1, 1, 1, 2, 1, 1)
    mask = torch.ones(coords.shape[:-1], device=device, dtype=torch.bool)
    output = voxel_pool_depth(features, depth, coords, mask, 1, 2, 1, 1)
    expected = torch.tensor([[[2*.25+11*.1, 3*.25+13*.1], [2*.75+11*.9, 3*.75+13*.9]],
                             [[5*.4+17*.3, 7*.4+19*.3], [5*.6+17*.7, 7*.6+19*.7]]], device=device)
    torch.testing.assert_close(output.reshape(2, 2, 2), expected)
    output.sum().backward()
    torch.testing.assert_close(features.grad, torch.ones_like(features))
    torch.testing.assert_close(depth.grad.flatten(), torch.tensor([5.,5.,12.,12.,24.,24.,36.,36.], device=device))
    empty = voxel_pool_depth(features, depth, coords, torch.zeros_like(mask), 1, 2, 1, 1)
    assert empty.count_nonzero() == 0
    # The only collapsed caller sums cameras immediately in native feat_encoding.
    # Exercise several contribution chunks and nonuniform gradients, including
    # empty support; compare to the explicit per-camera values above.
    f2 = features.detach().clone().requires_grad_()
    d2 = depth.detach().clone().requires_grad_()
    fused = voxel_pool_depth(f2, d2, coords, mask, 1, 2, 1, 1,
                            collapse_camera_sum=True, contribution_chunk=3)
    torch.testing.assert_close(fused.reshape(2,2), expected.sum(0))
    weights = torch.arange(1,5,device=device).reshape_as(fused).float()
    (fused*weights).sum().backward()
    f3 = features.detach().clone().requires_grad_()
    d3 = depth.detach().clone().requires_grad_()
    reference = voxel_pool_depth(f3,d3,coords,mask,1,2,1,1).sum(1,keepdim=True)
    (reference*weights).sum().backward()
    torch.testing.assert_close(f2.grad,f3.grad)
    torch.testing.assert_close(d2.grad,d3.grad)
    return {"sweep_camera_geometry": True, "feature_depth_backward": True, "empty_support": True,
            "fused_camera_sum_values_gradients": True, "chunked_inplace_index_add": True}


def test_voxel_bf16(device):
    generator = torch.Generator().manual_seed(38)
    features = torch.randn(1,6,4,2,2,generator=generator).to(device=device,dtype=torch.bfloat16).requires_grad_()
    depth = torch.rand(6,3,2,2,generator=generator).to(device=device,dtype=torch.bfloat16).requires_grad_()
    coords = torch.zeros(1,2,3,3,2,2,4,device=device,dtype=torch.long)
    coords[...,1] = torch.arange(3,device=device).view(1,1,1,3,1,1)
    mask = torch.ones(coords.shape[:-1],device=device,dtype=torch.bool)
    reference = voxel_pool_depth(features,depth,coords,mask,1,3,1,1).sum(1,keepdim=True)
    weights = torch.arange(1,13,device=device).reshape_as(reference).float()/12
    ref_grads = torch.autograd.grad((reference.float()*weights).sum(),(features,depth))
    fused = voxel_pool_depth(features,depth,coords,mask,1,3,1,1,
                             collapse_camera_sum=True,contribution_chunk=7)
    fused_grads = torch.autograd.grad((fused.float()*weights).sum(),(features,depth))
    torch.testing.assert_close(fused,reference,atol=.04,rtol=.02)
    for actual, expected in zip(fused_grads,ref_grads):
        torch.testing.assert_close(actual,expected,atol=.04,rtol=.02)
    return {"value_max_abs":float((reference.detach().float()-fused.detach().float()).abs().max()),
            "gradient_max_abs":[float((a.float()-b.float()).abs().max()) for a,b in zip(fused_grads,ref_grads)],
            "note":"fused FP32 camera reduction changes BF16 rounding order"}


def main():
    import faulthandler
    faulthandler.dump_traceback_later(30, repeat=True)
    results = {"sampler": test_sampler(), "voxel_cpu": test_voxel("cpu"), "voxel_rocm": test_voxel("cuda"),
               "voxel_fused_bf16_cpu":test_voxel_bf16("cpu"), "voxel_fused_bf16_rocm":test_voxel_bf16("cuda")}
    probability = ((torch.arange(128)+.5)/128)[None,None,None,:].expand(1,3,128,128)
    trajectory = torch.zeros(1,50,7)
    cost, covered = footprint_collision(probability,trajectory,torch.tensor([[4.,2.]]),torch.ones_like(probability))
    torch.testing.assert_close(cost,torch.tensor([3*18/128]))
    assert covered.all()
    trajectory[...,0] = 120
    cost, covered = footprint_collision(probability,trajectory,torch.tensor([[4.,2.]]),torch.ones_like(probability))
    assert not covered.any() and cost.isnan().all()
    assert relative_advantages(torch.ones(2,8)).count_nonzero() == 0
    results["critic_geometry"] = {"FLU_X_Y_pixel_centers":True,"unknown_is_not_free":True,"constant_group_advantage_zero":True}
    try:
        results["original_cuda_extension"] = check_perception_operator("cuda")
    except Exception:
        results["original_cuda_extension"] = {"passed": False, "error": traceback.format_exc()}
    with rocm_perception_ops():
        results["official_pytorch_deformable_attention_rocm"] = check_perception_operator("cuda")
    path = ROOT / "artifacts/v5-qwen-drive-world/contracts.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    torch.cuda.synchronize()
    import gc
    gc.collect()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__": main()
