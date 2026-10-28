"""GPU forward/backward checks with SDPA MATH and no Triton dependency."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("artifacts/stage1/rocm.json"))
    args = p.parse_args()
    installed = {d.metadata["Name"].lower() for d in metadata.distributions()}
    # The shared environment also serves Unsloth, which requires AMD Triton.
    # Project execution remains eager/SDPA MATH; importing Triton is still rejected below.
    shared_environment = Path(sys.prefix).resolve() == (Path.home()/'.venvs/llm-rocm').resolve()
    optional_accelerators = sorted(n for n in installed if 'triton' in n)
    forbidden = sorted(n for n in installed if not (shared_environment and 'triton' in n) and any(
        s in n for s in ("triton", "flash-attn", "xformers", "nvidia-", "cuda-python", "cupy")
    ))
    assert not forbidden, f"Forbidden packages: {forbidden}"
    assert torch.version.hip and torch.version.cuda is None, "ROCm build required"
    assert torch.cuda.is_available(), "HIP device not available"
    torch.manual_seed(0)
    # PyTorch deliberately uses the cuda device/API spelling for HIP builds.
    device = torch.device("cuda")
    report = dict(torch=torch.__version__, hip=torch.version.hip,
                  cuda_build=torch.version.cuda, available=torch.cuda.is_available(),
                  device=torch.cuda.get_device_name(0),
                  memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                  python=sys.version, seed=0, sdpa_backend="MATH",
                  forbidden_packages=forbidden, shared_environment=shared_environment,
                  installed_optional_accelerators=optional_accelerators,
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    a = torch.randn(512, 512, device=device)
    b = torch.randn_like(a)
    torch.cuda.synchronize()
    start = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    report["matmul_ms_first_call"] = (time.perf_counter() - start) * 1000
    ref = a.cpu().double() @ b.cpu().double()
    report["matmul_relative_l2"] = float((c.cpu().double() - ref).norm() / ref.norm())
    assert report["matmul_relative_l2"] < 1e-4
    q = torch.randn(2, 4, 64, 64, device=device, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(q, k, v)
        expected = ((q.detach().float() @ k.detach().float().transpose(-2, -1)) / 8).softmax(-1) @ v.detach().float()
        report["sdpa_fp32_max_abs"] = float((out.detach() - expected).abs().max())
        assert torch.allclose(out, expected, atol=1e-5, rtol=1e-4)
        out.square().mean().backward()
        assert all(torch.isfinite(x.grad).all() for x in (q, k, v))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mixed = F.scaled_dot_product_attention(q, k, v)
            loss = mixed.float().square().mean()
        loss.backward()
        report["bf16_relative_l2"] = float((mixed.float() - out.detach()).norm() / out.detach().norm())
        report["bf16_finite_forward_backward"] = bool(torch.isfinite(mixed).all() and all(torch.isfinite(x.grad).all() for x in (q, k, v)))
        assert report["bf16_finite_forward_backward"] and report["bf16_relative_l2"] < 0.03
    report["triton_imported"] = any(n == "triton" or n.startswith("triton.") for n in sys.modules)
    assert not report["triton_imported"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
