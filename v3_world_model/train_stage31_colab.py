"""Train Stage 3.1 using verified, extracted packs and PyTorch Lightning."""
import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2
import pytorch_lightning as pl
import torch
from filelock import FileLock, Timeout
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from v1_dual_system.checkpoint_io import VerifiedCheckpointIO

from .data import Clips
from .geometry import CONFIG_SHA256
from .objectives import stage31_losses
from .pack_colab_v3 import sha256
from .perception_model import BEVWAMXPerception, CLIP_LEN, sdpa_context

LOSS_WEIGHTS = dict(lane=1., vectors=.4, objects=1., occupancy=1., trajectory=.6)


class Stage31(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BEVWAMXPerception(lidar_dropout=config["lidar_dropout"])

    def shared_step(self, batch, stage):
        inputs, targets = batch["inputs"], batch["targets"]
        previous = None
        totals = {}
        with sdpa_context(self.config["sdpa"]):
            for t in range(CLIP_LEN):
                p = self.model(tuple(x[:, t].contiguous() for x in inputs["rgb"]),
                               inputs["lidar_bev"][:, t], previous, inputs["ego_delta"][:, t],
                               inputs["cmd"][:, t])
                previous = p["z_t"]
                losses = stage31_losses(p, {k: v[:, t] for k, v in targets.items()})
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0) + value / CLIP_LEN
        loss = sum(weight * totals[key] for key, weight in LOSS_WEIGHTS.items())
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=inputs["lidar_bev"].shape[0])
        for key, value in totals.items():
            self.log(f"{stage}_{key}", value, on_step=False, on_epoch=True,
                     batch_size=inputs["lidar_bev"].shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config["learning_rate"], weight_decay=.01, foreach=True)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["task_state"] = self.config

    def on_load_checkpoint(self, checkpoint):
        old = checkpoint["task_state"]
        for key in ("config_sha256", "pack_sha256", "learning_rate", "clip_len", "physical_batch",
                    "gradient_accumulation", "lidar_dropout", "seed"):
            if old[key] != self.config[key]:
                raise ValueError(f"Cannot resume with changed {key}")


def atomic_json(path, document):
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


class ValidationLog(pl.Callback):
    def __init__(self, path, cfg):
        self.path = path
        self.cfg = cfg

    def on_validation_end(self, trainer, module):
        if trainer.sanity_checking:
            return
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith("val_")}
        metrics.update(epoch=trainer.current_epoch, actual_updates=trainer.global_step,
                       physical_batch=self.cfg["physical_batch"], gradient_accumulation=self.cfg["gradient_accumulation"],
                       clip_len=CLIP_LEN)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")


def callbacks(out, config):
    best = ModelCheckpoint(dirpath=out, filename="best", monitor="val_loss", mode="min",
                           save_top_k=1, enable_version_counter=False, save_on_train_epoch_end=False)
    latest = ModelCheckpoint(dirpath=out, every_n_train_steps=config["checkpoint_every"],
                             save_top_k=0, save_last=True, enable_version_counter=False)
    # Lightning's canonical last name is set locally per callback.
    latest.CHECKPOINT_NAME_LAST = "latest"
    return ValidationLog(out / "validation.jsonl", config), best, latest


def build_trainer(config, out, **overrides):
    log, best, latest = callbacks(out, config)
    flags = dict(accelerator=config["accelerator"], devices=1, precision=config["precision"],
                 max_steps=config["max_updates"], accumulate_grad_batches=config["gradient_accumulation"],
                 default_root_dir=out, plugins=[VerifiedCheckpointIO()], callbacks=[log, best, latest],
                 num_sanity_val_steps=0, gradient_clip_val=1., logger=False,
                 enable_model_summary=False, log_every_n_steps=1)
    flags.update(overrides)
    return pl.Trainer(**flags), best


def lock():
    return FileLock(Path(tempfile.gettempdir()) / "bev-wam-x-training.lock", timeout=0)


def check_existing_training():
    """Honor the repository rule across stages that have their own lock protocols."""
    import psutil
    modules = ("-m v1_dual_system.", "-m v2_mapless_wa.phase2_train",
               "-m v2_mapless_wa.phase26_train", "-m v3_world_model.train_stage31_colab")
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if proc.info["pid"] == os.getpid() or not (proc.info["name"] or "").lower().startswith("python"):
            continue
        command = " ".join(proc.info["cmdline"] or []).lower()
        if any(x in command for x in modules[1:]) or (modules[0] in command and "train" in command):
            # Windows venv launches a forwarding parent with the same command line.
            if proc.info["pid"] != os.getppid():
                raise RuntimeError(f"Another bev-wam-x training process is active (PID {proc.info['pid']})")


def config_for(args):
    return dict(config_sha256=CONFIG_SHA256, pack_sha256=sha256(args.dataset / "manifest.json"),
                physical_batch=args.batch, gradient_accumulation=args.accumulate, clip_len=CLIP_LEN,
                max_updates=args.max_updates, checkpoint_every=args.checkpoint_every,
                learning_rate=args.lr, lidar_dropout=args.lidar_dropout, seed=args.seed,
                accelerator=args.accelerator, precision=args.precision, sdpa=args.sdpa,
                num_workers=args.workers)


def train(args):
    cfg = config_for(args)
    if args.accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA/ROCm GPU is required")
        if args.precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The GPU PyTorch runtime does not expose BF16")
        torch.cuda.reset_peak_memory_stats()
    pl.seed_everything(args.seed, workers=True)
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    resume = args.out / "latest.ckpt" if args.resume else None
    if args.resume and not resume.is_file():
        raise FileNotFoundError(resume)
    if not args.resume and any((args.out / f).exists() for f in ("latest.ckpt", "best.ckpt", "validation.jsonl")):
        raise ValueError("Use --resume or a fresh run directory")
    module = Stage31(cfg)
    cfg["parameter_count"] = sum(p.numel() for p in module.parameters())
    atomic_json(args.out / "training-config.json", cfg)
    train_data, val_data = (Clips(args.dataset, split) for split in ("train", "val"))
    loaders = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=args.accelerator == "gpu",
                   persistent_workers=args.workers > 0)
    trainer, best = build_trainer(cfg, args.out)
    trainer.fit(module, DataLoader(train_data, shuffle=True, **loaders), DataLoader(val_data, **loaders),
                ckpt_path=str(resume) if resume else None)
    # A run can hit max_steps mid-epoch, before Lightning's first validation/checkpoint trigger.
    if not best.best_model_path or not (args.out / "validation.jsonl").exists():
        metrics = trainer.validate(module, DataLoader(val_data, **loaders), verbose=False)[0]
        if not (args.out / "validation.jsonl").exists():
            with (args.out / "validation.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(metrics, actual_updates=trainer.global_step,
                                            physical_batch=args.batch, gradient_accumulation=args.accumulate,
                                            clip_len=CLIP_LEN), allow_nan=False) + "\n")
        best_path = str(args.out / "best.ckpt")
        if not best.best_model_path or metrics["val_loss"] < float(best.best_model_score):
            best.best_model_path = best.kth_best_model_path = best_path
            best.best_model_score = best.current_score = best.kth_value = torch.tensor(metrics["val_loss"])
            best.best_k_models = {best_path: best.best_model_score}
            trainer.save_checkpoint(best_path)
    trainer.save_checkpoint(args.out / "latest.ckpt")
    atomic_json(args.out / "training-result.json",
                dict(cfg, actual_updates=trainer.global_step,
                     peak_memory_bytes=torch.cuda.max_memory_allocated() if args.accelerator == "gpu" else 0,
                     peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.accelerator == "gpu" else 0,
                     device=torch.cuda.get_device_name(0) if args.accelerator == "gpu" else "CPU",
                     cuda=torch.version.cuda, rocm=torch.version.hip))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--max-updates", type=int, default=10000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lidar-dropout", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=714)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--precision", choices=("32-true", "bf16-mixed"), default="bf16-mixed")
    parser.add_argument("--sdpa", choices=("auto", "math", "aotriton", "cuda_flash"), default="cuda_flash")
    args = parser.parse_args()
    if min(args.batch, args.accumulate, args.max_updates, args.checkpoint_every) < 1 or args.workers < 0:
        parser.error("Batch, accumulation, updates, and save interval must be positive; workers must be nonnegative")
    try:
        with lock():
            check_existing_training()
            train(args)
    except Timeout as exc:
        raise SystemExit("A bev-wam-x training process already holds the host lock") from exc


if __name__ == "__main__":
    main()
