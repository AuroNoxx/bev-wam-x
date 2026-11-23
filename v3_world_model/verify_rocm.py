"""Compare ROCm BF16 eager and HIP Graph inference under the same fixed input contract."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .geometry import CONFIG_SHA256, H, NAMES, SIZES, W
from .observations import validate_observation
from .perception_model import BEVWAMXPerception, sdpa_context
from .qc_v3 import load_npz, read_image


def observations(frame):
    if frame:
        rgb = tuple(torch.from_numpy(read_image(frame / f"{n}.png")[..., ::-1].copy())
                    .permute(2, 0, 1)[None].cuda().contiguous() for n in NAMES)
        obs = load_npz(frame / "obs.npz")
        validate_observation(obs)
        inputs = {k: torch.from_numpy(v)[None].cuda() for k, v in obs.items()}
    else:
        rgb = tuple(torch.randint(0, 256, (1, 3, h, w), dtype=torch.uint8, device="cuda") for w, h in SIZES)
        inputs = dict(lidar_bev=torch.rand((1, 6, H, W), device="cuda"),
                      ego_delta=torch.tensor([[-1.1, .013, .025]], device="cuda"),
                      cmd=torch.tensor([[0., 1., 0.]], device="cuda"),
                      nav_anchors=torch.tensor([[[12., .2, .1], [23., 2., .3], [34., 4., .6]]], device="cuda"),
                      ego_speed=torch.tensor([[11., .035]], device="cuda"))
    if tuple((int(x.shape[-1]), int(x.shape[-2])) for x in rgb) != SIZES:
        raise ValueError("Wrong input image dimensions")
    return rgb, inputs


def measure(call, n):
    begin = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    wall = []
    for a, b in zip(begin, end, strict=True):
        started = perf_counter()
        a.record()
        call()
        b.record()
        b.synchronize()
        wall.append((perf_counter() - started) * 1e3)
    device = [a.elapsed_time(b) for a, b in zip(begin, end, strict=True)]
    return dict(median=float(np.median(wall)), p95=float(np.quantile(wall, .95)),
                device_median=float(np.median(device)), device_p95=float(np.quantile(device, .95)))


@torch.inference_mode()
def compare(args):
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("Run this verifier on a PyTorch ROCm GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The installed ROCm build does not expose BF16")
    torch.manual_seed(731)
    torch.cuda.reset_peak_memory_stats()
    state = None
    if args.checkpoint:
        from v1_dual_system.checkpoint_io import VerifiedCheckpointIO
        state = VerifiedCheckpointIO().load_checkpoint(args.checkpoint, map_location="cpu")
        if state["task_state"]["config_sha256"] != CONFIG_SHA256:
            raise ValueError("Checkpoint sensor calibration mismatch")
    model = BEVWAMXPerception(lidar_dropout=0., legacy=bool(state and state["task_state"].get("architecture") != "stage31b")).eval().cuda()
    if state:
        model.load_state_dict({k.removeprefix("model."): v for k, v in state["state_dict"].items()})
    rgb, inputs = observations(args.frame)

    def step():
        return model(rgb, prev_latent=previous, **inputs)

    free_before, capacity = torch.cuda.mem_get_info()
    card_used = [capacity - free_before]
    with sdpa_context(args.sdpa), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        previous = model(rgb, **inputs)["z_t"].clone()
        warm_stream = torch.cuda.Stream()
        warm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm_stream):
            for _ in range(5):
                step()
        torch.cuda.current_stream().wait_stream(warm_stream)
        torch.cuda.synchronize()
        expected = {k: v.clone() for k, v in step().items()}
        card_used.append(capacity - torch.cuda.mem_get_info()[0])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = step()
        graph.replay()
        torch.cuda.synchronize()
        diff = {k: float((expected[k].float() - v.float()).abs().amax().item()) for k, v in actual.items()}
        eager_ms, graph_ms = measure(step, args.iterations), measure(graph.replay, args.iterations)
        card_used.append(capacity - torch.cuda.mem_get_info()[0])
    return dict(device=torch.cuda.get_device_name(), rocm=torch.version.hip, sdpa=args.sdpa, precision="BF16",
                batch=1, iterations=args.iterations, input="capture" if args.frame else "synthetic",
                checkpoint=bool(args.checkpoint), eager_ms=eager_ms, hip_graph_ms=graph_ms,
                max_abs_diff=max(diff.values()), per_output_max_abs_diff=diff, control_period_ms=100,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                sampled_card_used_bytes=max(card_used), card_capacity_bytes=capacity,
                graph_p95_budget_ms=args.max_inference_ms, card_budget_gib=args.max_card_gib)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdpa", choices=("aotriton", "math"), default="aotriton")
    parser.add_argument("--frame", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-inference-ms", type=float, default=100.)
    parser.add_argument("--max-card-gib", type=float, default=20.)
    args = parser.parse_args()
    if min(args.iterations, args.max_inference_ms, args.max_card_gib) <= 0:
        parser.error("Positive iteration count and budgets are required")
    report = compare(args)
    report["passed"] = (report["max_abs_diff"] == 0 and report["hip_graph_ms"]["p95"] <= args.max_inference_ms
                        and report["sampled_card_used_bytes"] <= args.max_card_gib * 2**30)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("ROCm inference parity or budget gate failed")


if __name__ == "__main__":
    main()
