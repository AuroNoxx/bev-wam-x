"""CLI utility for hardware detection, training parameter calibration, and operator verification.

Usage:
  # Non-intrusive static profile & parameter recommendation:
  python3 scripts/calibrate_hardware.py --output artifacts/hardware_profile.json

  # Verification run (Only after user authorization!):
  python3 scripts/calibrate_hardware.py --verify-attention --trial-run
"""
import argparse
import json
from pathlib import Path
import sys
import time

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from configs.stage4 import Stage4Config
from mini_wa_jepa.calibration import recommend_training_parameters, save_calibration_report
from mini_wa_jepa.hardware import detect_hardware
from mini_wa_jepa.operators import sdpa_context, verify_attention_fidelity


def run_attention_verification(device: torch.device, vendor: str) -> dict:
    """Verify numerical fidelity of candidate attention backends against MATH baseline."""
    print("Testing Attention backend fidelity...", flush=True)
    b, heads, seq, dim = 2, 4, 64, 64
    torch.manual_seed(0)

    results = {}
    for dtype, dtype_name in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
        q = torch.randn(b, heads, seq, dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        atol = 1e-4 if dtype == torch.float32 else 5e-3
        rtol = 1e-3 if dtype == torch.float32 else 1e-2

        results[dtype_name] = {}
        for candidate in ["AUTO", "FLASH", "EFFICIENT"]:
            try:
                res = verify_attention_fidelity(q, k, v, candidate_backend=candidate, vendor=vendor, atol=atol, rtol=rtol)
                results[dtype_name][candidate] = res
                print(f"  [{dtype_name.upper()} | {candidate}] max_abs_err={res['max_abs_error']:.2e}, rel_l2_err={res['relative_l2_error']:.2e}, close={res['is_allclose']}")
            except Exception as e:
                results[dtype_name][candidate] = {"error": str(e), "is_allclose": False}
                print(f"  [{dtype_name.upper()} | {candidate}] Failed: {e}")
    return results


def run_trial_calibration(device: torch.device, candidate_batches=(128, 64, 32)) -> dict:
    """Run forward, backward, and optimizer trial steps on synthetic inputs to measure actual peak VRAM."""
    from mini_wa_jepa.normalization import Normalizer
    from models.joint import JointModel
    from training.joint_loss import joint_loss

    print("Running trial-run calibration across candidate batch sizes...", flush=True)
    config = Stage4Config()
    normalizer = Normalizer()
    model = JointModel(config, normalizer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    results = {}
    for bs in candidate_batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            past = torch.randint(0, 256, (bs, 4, 4, 128, 128), dtype=torch.uint8, device=device)
            future = torch.randint(0, 256, (bs, 4, 4, 128, 128), dtype=torch.uint8, device=device)
            ego = torch.randn(bs, 4, 2, device=device) * 0.05
            trajectory = torch.randn(bs, 10, 4, device=device)
            valid = torch.ones(bs, 10, dtype=torch.bool, device=device)
            action_mask = torch.ones(bs, dtype=torch.bool, device=device)
            world_valid = torch.ones(bs, 4, dtype=torch.bool, device=device)
            batch = dict(bev=past, ego=ego, future=future, trajectory=trajectory, valid=valid, action_mask=action_mask, world_valid=world_valid)

            optimizer.zero_grad()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(past, ego, future)
            loss, _ = joint_loss(out, batch, config)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
            throughput = bs / elapsed
            results[bs] = {
                "peak_allocated_gib": peak_gib,
                "step_latency_seconds": elapsed,
                "throughput_samples_per_sec": throughput,
                "success": True,
            }
            print(f"  Batch {bs:3d}: Peak VRAM = {peak_gib:5.2f} GiB | Latency = {elapsed*1000:6.1f} ms | Throughput = {throughput:5.1f} samples/s")
        except Exception as e:
            results[bs] = {"error": str(e), "success": False}
            print(f"  Batch {bs:3d}: Failed with error: {e}")
            torch.cuda.empty_cache()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=Path("artifacts/hardware_calibration.json"), help="Output JSON path")
    p.add_argument("--verify-attention", action="store_true", help="Run attention fidelity verification")
    p.add_argument("--trial-run", action="store_true", help="Run trial forward/backward calibration on candidates")
    args = p.parse_args()

    # 1. Non-intrusive static hardware detection
    profile = detect_hardware()
    print(profile.summary(), flush=True)

    # 2. Derive recommendations
    recommendation = recommend_training_parameters(profile)
    print(recommendation.summary(), flush=True)

    verification_data = None
    trial_data = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.verify_attention:
        verification_data = run_attention_verification(device, profile.gpu.vendor)

    if args.trial_run:
        trial_data = run_trial_calibration(device)

    # 3. Save report
    save_calibration_report(args.output, recommendation, profile)
    if verification_data or trial_data:
        saved = json.loads(args.output.read_text())
        if verification_data:
            saved["attention_verification"] = verification_data
        if trial_data:
            saved["trial_calibration"] = trial_data
        args.output.write_text(json.dumps(saved, indent=2) + "\n")

    print(f"\nCalibration and profile report saved to: {args.output}")


if __name__ == "__main__":
    main()
