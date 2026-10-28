"""Evaluate existing remediated A/B/D/D0 weights; never train or instantiate a simulator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from configs.stage3 import Stage3Config
from dataset.bev_wa_dataset import BEWWAJointDataset
from evaluation.bev_wa_metrics import render_stage4_report
from mini_wa_jepa.provenance import provenance
from mini_wa_jepa.operators import sdpa_context, configure_operator_optimizations, verify_attention_fidelity
from models.bev_wa_joint_flow import ArmBActionOnlyFlowModel, ArmDJointFlowModel
from scripts.run_bev_wa_stage4_remediated import DeterministicBaseline, evaluate_open_loop

ARMS = {
    "A": ("arm_a_deterministic_mse", DeterministicBaseline, {}),
    "B": ("arm_b_action-only_flow", ArmBActionOnlyFlowModel, {"use_applied_action": True}),
    "D": ("arm_d_repaired_joint_wa-flow", ArmDJointFlowModel, {"use_applied_action": True}),
    "D0": ("ablation_d-0_repaired_l_scene=0", ArmDJointFlowModel, {"use_applied_action": True, "scene_loss_weight": 0.0}),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not exist; preserves all past artifacts")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "artifacts/bev_wa_alignment/stage4_remediated")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/stage7_integrated/mixed_validation_012/manifest.json")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=["B", "D"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261022, 20261023])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sdpa-backend", choices=("AUTO", "MATH", "FLASH", "EFFICIENT", "CUDNN"), default=Stage3Config().sdpa_backend)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    args = parser.parse_args()
    paths = {(arm, seed): args.checkpoint_dir / f"{ARMS[arm][0]}_seed_{seed}.pt" for arm in args.arms for seed in args.seeds}
    for path in paths.values():
        if not path.is_file():
            parser.error(f"Missing checkpoint (no training fallback): {path}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    ds = BEWWAJointDataset(args.manifest, split="validation", eligible_action_only=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    results = {
        "mode": "existing_weights_open_loop_only", "stage": 4,
        "device": args.device, "torch_version": str(torch.__version__),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "provenance": provenance(), "goal_sha256": sha256(ROOT / "GOAL.md"),
        "protocol_sha256": sha256(ROOT / "docs/BEV_WA_ALIGNED_PROTOCOL.md"),
        "manifest_sha256": sha256(args.manifest), "dataset_stats": ds.stats(),
        "source_sha256": {p: sha256(ROOT / p) for p in (
            "scripts/reevaluate_bev_wa_stage4_open_loop.py", "scripts/run_bev_wa_stage4_remediated.py",
            "evaluation/bev_wa_metrics.py", "dataset/bev_wa_dataset.py", "models/bev_wa_joint_flow.py")},
        "evaluated_checkpoints": {},
        "sdpa_backend": args.sdpa_backend,
        "operator_optimizations": configure_operator_optimizations(vendor="AMD", allow_tf32=args.allow_tf32, cudnn_benchmark=args.cudnn_benchmark, deterministic=True),
        "limitations": ["No training or closed-loop evaluation", "C missing from existing fair retrained checkpoints", "Arm A has no applied-action input; not a matched four-arm comparison", "Noise is restarted per batch as in historical evaluation; batch size affects samples"],
    }
    verify_dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    verify_backend = args.sdpa_backend if args.device == "cuda" else "MATH"
    results["sdpa_verification"] = verify_attention_fidelity(
        torch.randn(16,4,64,64,device=args.device,dtype=verify_dtype),
        torch.randn(16,4,64,64,device=args.device,dtype=verify_dtype),
        torch.randn(16,4,64,64,device=args.device,dtype=verify_dtype),
        candidate_backend=verify_backend, vendor="AMD")
    with sdpa_context(args.sdpa_backend, vendor="AMD"):
        for (arm, seed), path in paths.items():
            torch.manual_seed(seed)
            _, cls, kwargs = ARMS[arm]
            model = cls(cfg=Stage3Config(seed=seed), **kwargs).to(args.device).eval()
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            print(f"Evaluating {arm} seed {seed}, checkpoint epoch {checkpoint.get('epoch')}", flush=True)
            metrics = evaluate_open_loop(model, loader, args.device, noise_seed=seed,
                                         raw_path=args.output_dir / f"{arm}_seed_{seed}_per_anchor.npz")
            metrics = {k: v for k, v in metrics.items() if not k.startswith("sample_")}
            metrics.update(checkpoint=str(path.resolve()), checkpoint_sha256=sha256(path), checkpoint_epoch=checkpoint.get("epoch"))
            results["evaluated_checkpoints"].setdefault(arm, {})[f"seed_{seed}"] = metrics
            (args.output_dir / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
            (args.output_dir / "REPORT.md").write_text(render_stage4_report(results), encoding="utf-8")
            print(json.dumps({k: metrics[k] for k in ("val_ade_2s", "val_fde_2s", "neg_speed_pct", "open_loop_gate")}), flush=True)
            del model, checkpoint


if __name__ == "__main__":
    main()
