"""Stage 3 estimator diagnostic (read-only; no training, no checkpoint/report changes).

Question: does Stage 3 Gate 1 (E_norm <= 0.85 vs Copy-Current) fail partly because
`evaluate_stage3` scores ONE random flow sample (solve_euler from N(0, I)) with MSE?

For a calibrated conditional sampler y_hat ~ p(y | history):
    E||y_hat - y||^2 = ||mu - y||^2 + E||y_hat - mu||^2       (mu = E[y | history])
                    ~= MSE(conditional mean) + sampling spread
i.e. a single sample pays the conditional variance twice, while Copy-Current pays it once.
At short horizons (copy bias ~ 0) this pushes E_norm toward 2, which matches stage3_001
(+0.5s: 2.12 / 2.26).

For every checkpoint, on the same validation set and teacher, this script reports:
    single : one seeded N(0, I) sample      (reproduces the historical metric)
    zero   : ODE from zero initial noise    (protocol "Candidate 0", deterministic)
    meanK  : average of K seeded samples    (Monte-Carlo conditional-mean estimate)
    spread : unbiased per-sample variance across the K samples, divided by copy MSE
Consistency check: E_single ~= E_meanK + spread * (1 - 1/K).

Usage (from the repo root, ROCm machine):
    .venv/bin/python scripts/diagnose_stage3_estimator.py --max-batches 20 \
        --checkpoints artifacts/bev_wa_alignment/stage3_masked_001/flow_latent_seed_20261022.pt
    .venv/bin/python scripts/diagnose_stage3_estimator.py          # all 8 checkpoints, full val set

Outputs go only to --output-dir (default artifacts/bev_wa_alignment/stage3_estimator_diag_001).
This is a diagnostic. Switching the gate's estimator is a protocol revision that must be
recorded and approved before any result is called a pass; historical numbers stay as they are.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import fields
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.stage3 import Stage3Config
from dataset.bev_wa_dataset import BEWWAJointDataset
from models.bev_wa_latent_flow import BEVWALatentFlowModel

HORIZONS = ["+0.5s", "+1.0s", "+1.5s", "+2.0s"]
VAL_MANIFEST = "artifacts/stage7_integrated/mixed_validation_010/manifest.json"
DEFAULT_RUNS = ("stage3_001", "stage3_residual_001", "stage3_hybrid_001", "stage3_masked_001")
DEFAULT_SEEDS = (20261022, 20261023)
GATE = 0.85


def rebuild_config(saved):
    """Rebuild Stage3Config from a pickled (possibly older) config, keeping known fields only."""
    names = {f.name for f in fields(Stage3Config)}
    src = saved if isinstance(saved, dict) else getattr(saved, "__dict__", {})
    return Stage3Config(**{k: v for k, v in src.items() if k in names})


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = rebuild_config(ckpt["config"])
    model = BEVWALatentFlowModel(cfg)  # weights come from the checkpoint, not encoder_ckpt
    model.online_encoder.load_state_dict(ckpt["online_encoder"], strict=True)
    model.target_encoder.load_state_dict(ckpt["target_encoder"], strict=True)
    missing, unexpected = model.predictor.load_state_dict(ckpt["predictor"], strict=False)
    # Older checkpoints predate the masked-completion branch; it is unused by flow sampling.
    allowed = {"mask_token"} | {k for k in model.predictor.state_dict() if k.startswith("completion_head.")}
    bad = [k for k in missing if k not in allowed] + list(unexpected)
    if bad:
        raise RuntimeError(f"{path}: unexpected predictor key mismatch: {bad}")
    model = model.to(device).eval()
    return model, cfg, ckpt


def add_residual(latents, current, cfg):
    if not getattr(cfg, "residual_target", False):
        return latents
    horizons = tuple(cfg.residual_horizons) if cfg.residual_horizons else (0, 1, 2, 3)
    out = latents.clone()
    for k in horizons:
        out[:, k] = out[:, k] + current
    return out


@torch.no_grad()
def evaluate(model, cfg, loader, device, k_samples, steps, max_batches, noise_seed, use_bf16):
    names = ("single", "zero", "mean", "copy", "spread")
    sums = {n: np.zeros(4) for n in names}
    counts = np.zeros(4)
    gen = torch.Generator(device=device).manual_seed(noise_seed)
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=use_bf16 and device.type == "cuda")
    tokens, dim = model.predictor.token_count, model.predictor.latent_dim

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        h = batch["history_bev"].to(device)
        ego = batch["ego_motion"].to(device)
        f = batch["future_bev"].to(device)
        valid = batch["future_bev_mask"].to(device).bool()
        shape = (h.shape[0], 4, tokens, dim)

        with autocast:
            cond = model.online_encoder(h, ego)
            target = model.target_encoder.encode_future(f).float()
            current = model.target_encoder.encode_current(h[:, -1]).float()
            zero_init = torch.zeros(shape, device=device, dtype=cond.dtype)
            zero = add_residual(model.predictor.solve_euler(cond, steps, initial_noise=zero_init).float(),
                                current, cfg)
            samples = []
            for _ in range(k_samples):
                noise = torch.randn(shape, device=device, generator=gen, dtype=torch.float32).to(cond.dtype)
                y = model.predictor.solve_euler(cond, steps, initial_noise=noise).float()
                samples.append(add_residual(y, current, cfg))

        stack = torch.stack(samples)  # [K, B, 4, 64, 256]
        mean = stack.mean(0)
        spread = (stack - mean).square().mean(dim=(-1, -2)).sum(0) / max(k_samples - 1, 1)  # [B, 4]

        def mse(x):
            return (x - target).square().mean(dim=(-1, -2))  # [B, 4]

        per = {
            "single": mse(stack[0]),
            "zero": mse(zero),
            "mean": mse(mean),
            "copy": (current[:, None] - target).square().mean(dim=(-1, -2)),
            "spread": spread,
        }
        for k in range(4):
            v = valid[:, k]
            counts[k] += float(v.sum().item())
            for n in names:
                sums[n][k] += float(per[n][v, k].sum().item())

    avg = {n: sums[n] / np.maximum(counts, 1) for n in names}
    copy = np.maximum(avg["copy"], 1e-8)
    out = {}
    for k, hz in enumerate(HORIZONS):
        out[hz] = {
            "n": int(counts[k]),
            "copy_mse": float(avg["copy"][k]),
            "E_single": float(avg["single"][k] / copy[k]),
            "E_zero": float(avg["zero"][k] / copy[k]),
            f"E_mean{k_samples}": float(avg["mean"][k] / copy[k]),
            "spread_over_copy": float(avg["spread"][k] / copy[k]),
        }
    return out


def historical_e_norm(ckpt):
    try:
        res = ckpt["final_metrics"]["results_by_horizon"]
        return {h: float(res[h]["normalized_mse"]) for h in HORIZONS}
    except (KeyError, TypeError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", default=None)
    p.add_argument("--samples", type=int, default=8, help="K samples for the Monte-Carlo mean (>=2)")
    p.add_argument("--steps", type=int, default=10, help="Euler steps (historical gate used 10)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=None, help="None = full validation set (gate scope)")
    p.add_argument("--noise-seed", type=int, default=20261024)
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--output-dir", default="artifacts/bev_wa_alignment/stage3_estimator_diag_001")
    args = p.parse_args()
    if args.samples < 2:
        p.error("--samples must be >= 2")

    os.chdir(WORKSPACE)
    ckpts = args.checkpoints or [f"artifacts/bev_wa_alignment/{r}/flow_latent_seed_{s}.pt"
                                 for r in DEFAULT_RUNS for s in DEFAULT_SEEDS]
    ckpts = [c for c in ckpts if Path(c).exists()] or sys.exit("No checkpoints found")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} checkpoints={len(ckpts)} K={args.samples} steps={args.steps} "
          f"max_batches={args.max_batches}", flush=True)

    val_ds = BEWWAJointDataset(WORKSPACE / VAL_MANIFEST, split="validation",
                               eligible_action_only=False, preload_ram=True)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    results = {"generated_at": datetime.now().astimezone().isoformat(), "args": vars(args),
               "gate": GATE, "checkpoints": {}}
    km = f"E_mean{args.samples}"
    for path in ckpts:
        t0 = time.time()
        model, cfg, ckpt = load_model(path, device)
        metrics = evaluate(model, cfg, loader, device, args.samples, args.steps,
                           args.max_batches, args.noise_seed, not args.no_bf16)
        results["checkpoints"][path] = {
            "residual_target": bool(cfg.residual_target),
            "residual_horizons": list(cfg.residual_horizons),
            "historical_E_norm": historical_e_norm(ckpt),
            "metrics": metrics,
            "seconds": round(time.time() - t0, 1),
        }
        row = " | ".join(f"{h} single {m['E_single']:.3f} zero {m['E_zero']:.3f} "
                         f"mean {m[km]:.3f} spread {m['spread_over_copy']:.3f}" for h, m in metrics.items())
        print(f"{path}: {row}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    lines = ["# Stage 3 estimator diagnostic", "",
             f"Generated {results['generated_at']}. K={args.samples}, Euler steps={args.steps}, "
             f"max_batches={args.max_batches} (None = full validation set). Read-only; no gate decision.", "",
             f"| checkpoint | horizon | historical | single | zero | mean{args.samples} | spread/copy |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for path, rec in results["checkpoints"].items():
        hist = rec["historical_E_norm"] or {}
        for h, m in rec["metrics"].items():
            hv = f"{hist[h]:.3f}" if h in hist else "—"
            lines.append(f"| {Path(path).parent.name}/{Path(path).stem[-8:]} | {h} | {hv} | {m['E_single']:.3f} | "
                         f"{m['E_zero']:.3f} | {m[km]:.3f} | {m['spread_over_copy']:.3f} |")
    lines += ["", "## Reading the table", "",
              "- `single` should reproduce `historical` (up to noise seed / batch subset).",
              f"- `single ~= mean{args.samples} + spread*(1-1/K)` confirms the decomposition.",
              f"- If `zero` or `mean{args.samples}` <= {GATE} on every horizon while `single` is not: "
              "the gate failure is largely an estimator artifact. Propose a recorded protocol revision "
              "(score Candidate 0 or the K-mean; keep `single` as a sample-quality metric).",
              f"- If `mean{args.samples}` is still > 1 at +0.5s: the model's conditional mean is worse than "
              "copy-current, so this is a real modelling deficit (see deterministic residual head)."]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_dir / 'results.json'} and {out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
