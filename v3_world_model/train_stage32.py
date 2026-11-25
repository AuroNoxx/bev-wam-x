"""Train the Stage 3.2 action-conditioned latent world model using PyTorch Lightning."""
import argparse
import json
import os
from pathlib import Path

import psutil
import pytorch_lightning as pl
import torch
from filelock import Timeout
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from v1_dual_system.checkpoint_io import VerifiedCheckpointIO

from .pack_colab_v3 import sha256
from .perception_model import BEVWAMXPerception, sdpa_context
from .train_stage31_colab import atomic_json, lock, lr_factor
from .world_model import ActionConditionedWorldModel


def check_existing_training():
    """Enforce single-training-job rule across all stages on this host."""
    modules = (
        "-m v1_dual_system.",
        "-m v2_mapless_wa.phase2_train",
        "-m v2_mapless_wa.phase26_train",
        "-m v3_world_model.train_stage31_colab",
        "-m v3_world_model.train_stage32",
    )
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if proc.info["pid"] == os.getpid() or not (proc.info["name"] or "").lower().startswith("python"):
            continue
        cmd = " ".join(proc.info["cmdline"] or []).lower()
        if any(m in cmd for m in modules[1:]) or (modules[0] in cmd and "train" in cmd):
            if proc.info["pid"] != os.getppid():
                raise RuntimeError(f"Another training process is active (PID {proc.info['pid']})")


class LatentRolloutDataset(Dataset):
    """Episode-split fixed-horizon windows of extracted Stage 3.1b latents and vehicle states."""

    def __init__(self, root, split, horizon=10, stride=None, urban_weight=1.0):
        self.root = Path(root)
        self.horizon = int(horizon)
        self.urban_weight = float(urban_weight)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        split_ids = set(self.manifest["splits"][split])
        step = int(stride if stride is not None else (4 if split == "train" else self.horizon))
        self.episodes = {}
        self.windows = []
        self.weights = []
        self._cache = {}
        for entry in self.manifest["episodes"]:
            ep_id = entry["id"]
            if ep_id not in split_ids:
                continue
            n_frames = int(entry["num_frames"])
            if n_frames < self.horizon + 1:
                continue
            self.episodes[ep_id] = entry["file"]
            is_urban_map = entry.get("map") == "Town10HD_Opt"
            ep_data = self._load_episode(ep_id) if (self.urban_weight != 1.0 and not is_urban_map) else None
            for start in range(0, n_frames - self.horizon, step):
                self.windows.append((ep_id, start))
                if self.urban_weight != 1.0:
                    if is_urban_map or bool((ep_data["lane_state"][start + 1:start + self.horizon + 1, 2] > 0).any()):
                        self.weights.append(self.urban_weight)
                    else:
                        self.weights.append(1.0)
                else:
                    self.weights.append(1.0)
        if not self.windows:
            raise ValueError(f"No valid rollout windows of horizon {self.horizon} in split '{split}'")

    def _load_episode(self, ep_id):
        data = self._cache.get(ep_id)
        if data is None:
            data = torch.load(self.root / self.episodes[ep_id], map_location="cpu", weights_only=True)
            self._cache[ep_id] = data
        return data

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ep_id, s = self.windows[idx]
        d = self._load_episode(ep_id)
        e = s + self.horizon + 1
        ego = d["ego_state"][s:e]  # (H+1, 7): v, a_lon, a_lat, yaw_rate, steer, throttle, brake
        kin = torch.stack((ego[:, 0] / 30., ego[:, 3], ego[:, 1] / 10., ego[:, 2] / 10.), dim=-1)
        actions = ego[:-1, 4:7]
        delta = d["ego_delta"][s + 1:e]
        flags = d["flags"][s + 1:e].float()
        thw = d["thw"][s + 1:e]
        inv_thw = torch.where(thw > 0., 1. / (thw.clamp(0.1, 30.) + .5), torch.zeros_like(thw)).unsqueeze(-1)
        nav = d["nav_state"][s + 1:e] / ego.new_tensor([100., 5., 1.])
        return dict(
            z=d["z"][s:e],
            kinematics=kin,
            actions=actions,
            delta=delta,
            flags=flags,
            inv_thw=inv_thw,
            nav=nav,
        )


def _decode_consistency_loss(pred_dec, tgt_dec):
    lane = (
        F.smooth_l1_loss(pred_dec["lane_logits"].float(), tgt_dec["lane_logits"].float()) +
        2.0 * F.mse_loss(pred_dec["lane_logits"].float().sigmoid(), tgt_dec["lane_logits"].float().sigmoid())
    )
    occ = (
        F.smooth_l1_loss(pred_dec["occupancy_logits"].float(), tgt_dec["occupancy_logits"].float()) +
        F.smooth_l1_loss(pred_dec["future_occupancy_logits"].float(), tgt_dec["future_occupancy_logits"].float())
    )
    tgt_prob = tgt_dec["obj_logits"].float().sigmoid()
    box_w = (0.1 + tgt_prob).unsqueeze(-1)
    obj = (
        F.smooth_l1_loss(pred_dec["obj_logits"].float(), tgt_dec["obj_logits"].float()) +
        (F.smooth_l1_loss(pred_dec["boxes"].float(), tgt_dec["boxes"].float(), reduction="none") * box_w).mean() +
        0.5 * F.smooth_l1_loss(pred_dec["cls_logits"].float(), tgt_dec["cls_logits"].float())
    )
    return lane + occ + obj


class Stage32WorldModel(pl.LightningModule):
    """Lightning wrapper for Stage 3.2 action-conditioned dynamics training with frozen decoder consistency."""

    def __init__(self, config, *, perception_state_dict=None):
        super().__init__()
        self.config = config
        self.world_model = ActionConditionedWorldModel(
            layers=config["wm_layers"],
            heads=config["wm_heads"],
        )
        perception_arch = config.get("perception_architecture")
        loaded_sd = perception_state_dict
        if loaded_sd is None and config.get("perception_ckpt"):
            ck = VerifiedCheckpointIO().load_checkpoint(Path(config["perception_ckpt"]), map_location="cpu")
            perception_arch = perception_arch or ck.get("task_state", {}).get("architecture", "stage31b")
            loaded_sd = {k.removeprefix("model."): v for k, v in ck["state_dict"].items()}
        perception_arch = perception_arch or "stage31b"
        if perception_arch not in ("stage31", "stage31b", "stage31c"):
            perception_arch = "stage31"
        self.config["perception_architecture"] = perception_arch
        self.perception = BEVWAMXPerception(lidar_dropout=0.0, arch=perception_arch, pretrained=False)
        if loaded_sd is not None:
            self.perception.load_state_dict(loaded_sd)
        self.perception.eval()
        for param in self.perception.parameters():
            param.requires_grad_(False)
        self.register_buffer("risk_pos_weight", torch.tensor([12.0, 12.0, 12.0]), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.perception.eval()
        return self

    def shared_step(self, batch, stage):
        z_seq = batch["z"]
        kin = batch["kinematics"]
        actions = batch["actions"]
        horizon = actions.shape[1]
        bsz = z_seq.shape[0]
        tf = self.config["teacher_forcing"] if stage == "train" else 0.0

        with sdpa_context(self.config["sdpa"]):
            out = self.world_model.rollout(
                z_seq[:, 0], kin[:, 0, :2], actions, gt_kinematics=kin, teacher_forcing=tf
            )

        z_pred = out["z_next"].float()
        z_tgt = z_seq[:, 1:].float()
        latent_loss = (
            F.smooth_l1_loss(z_pred, z_tgt) +
            (1.0 - F.cosine_similarity(z_pred, z_tgt, dim=-1, eps=1e-4)).mean()
        )
        kin_loss = F.smooth_l1_loss(out["next_kinematics"], kin[:, 1:])
        delta_loss = F.smooth_l1_loss(out["next_delta"], batch["delta"])
        risk_loss = F.binary_cross_entropy_with_logits(
            out["risk_logits"], batch["flags"], pos_weight=self.risk_pos_weight
        )
        thw_loss = F.smooth_l1_loss(out["inv_thw"], batch["inv_thw"])
        nav_loss = F.smooth_l1_loss(out["nav_progress"], batch["nav"])

        step_ids = (0, horizon - 1) if horizon > 1 else (0,)
        z_hat_sub = torch.cat([out["z_next"][:, k] for k in step_ids], dim=0)
        z_ref_sub = torch.cat([z_seq[:, k + 1] for k in step_ids], dim=0)
        with torch.autocast(device_type=z_hat_sub.device.type, enabled=False), sdpa_context("math"):
            pred_dec = self.perception.decode_perception(z_hat_sub.float(), bev=None)
            with torch.no_grad():
                tgt_dec = self.perception.decode_perception(z_ref_sub.float(), bev=None)
            decode_loss = _decode_consistency_loss(pred_dec, tgt_dec)

        loss = (
            1.5 * latent_loss +
            0.35 * decode_loss +
            1.0 * kin_loss +
            1.0 * delta_loss +
            0.25 * risk_loss +
            0.25 * thw_loss +
            0.25 * nav_loss
        )
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=bsz)
        for name, val in (
            ("latent", latent_loss),
            ("decode", decode_loss),
            ("kin", kin_loss),
            ("delta", delta_loss),
            ("risk", risk_loss),
            ("thw", thw_loss),
            ("nav", nav_loss),
        ):
            self.log(f"{stage}_{name}", val, on_step=False, on_epoch=True, batch_size=bsz)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def configure_optimizers(self):
        params = [p for p in self.world_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.config["learning_rate"], weight_decay=0.01, foreach=True)
        warmup, total = self.config["warmup_updates"], self.config["schedule_updates"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total))
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step"))

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        grads = [p.grad for p in self.world_model.parameters() if p.requires_grad and p.grad is not None]
        if any(not torch.isfinite(g).all() for g in grads):
            optimizer.zero_grad(set_to_none=True)
            if self._trainer is not None:
                self.log("train_skipped_nonfinite", 1.0, on_step=False, on_epoch=True,
                         batch_size=self.config["physical_batch"])
            return
        self.clip_gradients(
            optimizer, gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def on_save_checkpoint(self, checkpoint):
        # Drop frozen perception buffers/weights from saved weights only if huge? Keep full VerifiedCheckpointIO safety,
        # OR store lightweight world-model checkpoint via standard Lightning state_dict! Keeping standard state_dict
        # ensures self-contained resume & decode_perception parity (~285 MB per checkpoint).
        checkpoint["task_state"] = self.config

    def on_load_checkpoint(self, checkpoint):
        old = checkpoint["task_state"]
        for key in (
            "architecture", "config_sha256", "pack_sha256", "latents_manifest_sha256",
            "horizon", "wm_layers", "wm_heads", "learning_rate",
            "physical_batch", "gradient_accumulation", "urban_weight", "seed", "warmup_updates", "schedule_updates",
        ):
            if old.get(key) != self.config.get(key):
                raise ValueError(f"Cannot resume with changed {key}: {old.get(key)} vs {self.config.get(key)}")


class Stage32ValidationLog(pl.Callback):
    def __init__(self, path, cfg):
        self.path = path
        self.cfg = cfg

    def on_validation_end(self, trainer, module):
        if trainer.sanity_checking:
            return
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith("val_")}
        metrics.update(
            epoch=trainer.current_epoch,
            actual_updates=trainer.global_step,
            physical_batch=self.cfg["physical_batch"],
            gradient_accumulation=self.cfg["gradient_accumulation"],
            horizon=self.cfg["horizon"],
        )
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")


def _build_trainer(cfg, out, **overrides):
    val_log = Stage32ValidationLog(out / "validation.jsonl", cfg)
    best = ModelCheckpoint(
        dirpath=out, filename="best", monitor="val_loss", mode="min",
        save_top_k=1, enable_version_counter=False, save_on_train_epoch_end=False,
    )
    latest = ModelCheckpoint(
        dirpath=out, every_n_train_steps=cfg["checkpoint_every"],
        save_top_k=0, save_last=True, enable_version_counter=False,
    )
    latest.CHECKPOINT_NAME_LAST = "latest"
    flags = dict(
        accelerator=cfg["accelerator"],
        devices=1,
        precision=cfg["precision"],
        max_steps=cfg["max_steps"],
        accumulate_grad_batches=cfg["gradient_accumulation"],
        val_check_interval=cfg.get("val_check_interval", 1.0),
        default_root_dir=out,
        plugins=[VerifiedCheckpointIO()],
        callbacks=[val_log, best, latest],
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        logger=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )
    flags.update(overrides)
    return pl.Trainer(**flags), best


def _config_for(args):
    latent_manifest_path = args.latents / "manifest.json"
    latent_manifest = json.loads(latent_manifest_path.read_text(encoding="utf-8"))
    saved = json.loads((args.out / "training-config.json").read_text(encoding="utf-8")) if args.resume else {}
    total = args.schedule_updates if args.schedule_updates is not None else saved.get("schedule_updates", args.max_steps)
    warmup = args.warmup_updates if args.warmup_updates is not None else saved.get("warmup_updates", min(500, max(1, total // 20)))
    if not 0 <= warmup < total:
        raise ValueError("warmup_updates must satisfy 0 <= warmup < schedule_updates")
    return dict(
        architecture="stage32-wm-v1",
        perception_architecture=latent_manifest.get("perception_architecture", saved.get("perception_architecture")),
        config_sha256=latent_manifest["config_sha256"],
        pack_sha256=latent_manifest["pack_sha256"],
        latents_manifest_sha256=sha256(latent_manifest_path),
        perception_ckpt=str(args.perception_ckpt) if args.perception_ckpt else None,
        horizon=args.horizon,
        wm_layers=args.layers,
        wm_heads=args.heads,
        physical_batch=args.physical_batch,
        gradient_accumulation=args.grad_accum,
        urban_weight=float(getattr(args, "urban_weight", 1.0)),
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        checkpoint_every=args.checkpoint_every,
        learning_rate=args.lr,
        warmup_updates=warmup,
        schedule_updates=total,
        teacher_forcing=args.teacher_forcing,
        seed=args.seed,
        accelerator=args.accelerator,
        precision=args.precision,
        sdpa=args.sdpa,
        num_workers=args.workers,
    )


def train(args):
    resume = args.out / "latest.ckpt" if args.resume else None
    if args.resume and not resume.is_file():
        raise FileNotFoundError(resume)
    cfg = _config_for(args)
    if args.accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU runtime is required")
        if args.precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not report BF16 support")
        torch.cuda.reset_peak_memory_stats()

    pl.seed_everything(args.seed, workers=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((args.out / f).exists() for f in ("latest.ckpt", "best.ckpt", "validation.jsonl")):
        raise ValueError("Use --resume or a clean output directory")

    module = Stage32WorldModel(cfg)
    cfg["trainable_parameters"] = sum(p.numel() for p in module.world_model.parameters() if p.requires_grad)
    train_data = LatentRolloutDataset(
        args.latents, "train", horizon=args.horizon, stride=args.train_stride, urban_weight=cfg["urban_weight"]
    )
    val_data = LatentRolloutDataset(args.latents, "val", horizon=args.horizon, stride=args.val_stride)

    if not args.resume:
        atomic_json(args.out / "training-config.json", cfg)

    loaders = dict(batch_size=args.physical_batch, num_workers=args.workers,
                   pin_memory=args.accelerator == "gpu", persistent_workers=args.workers > 0)
    use_sampler = cfg["urban_weight"] > 1.0 and len(set(train_data.weights)) > 1
    train_loader = (
        DataLoader(
            train_data,
            sampler=WeightedRandomSampler(train_data.weights, num_samples=len(train_data), replacement=True),
            **loaders,
        )
        if use_sampler else DataLoader(train_data, shuffle=True, **loaders)
    )
    trainer, best = _build_trainer(cfg, args.out)
    trainer.fit(
        module,
        train_loader,
        DataLoader(val_data, **loaders),
        ckpt_path=str(resume) if resume else None,
    )
    if not best.best_model_path or not (args.out / "validation.jsonl").exists():
        metrics = trainer.validate(module, DataLoader(val_data, **loaders), verbose=False)[0]
        if not (args.out / "validation.jsonl").exists():
            with (args.out / "validation.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(
                    metrics, actual_updates=trainer.global_step,
                    physical_batch=args.physical_batch, gradient_accumulation=args.grad_accum,
                    horizon=args.horizon,
                ), allow_nan=False) + "\n")
        best_path = str(args.out / "best.ckpt")
        if not best.best_model_path or metrics["val_loss"] < float(best.best_model_score):
            best.best_model_path = best.kth_best_model_path = best_path
            best.best_model_score = best.current_score = best.kth_value = torch.tensor(metrics["val_loss"])
            best.best_k_models = {best_path: best.best_model_score}
            trainer.save_checkpoint(best_path)

    trainer.save_checkpoint(args.out / "latest.ckpt")
    atomic_json(args.out / "training-config.json", cfg)
    atomic_json(
        args.out / "training-result.json",
        dict(
            cfg,
            actual_updates=trainer.global_step,
            train_windows=len(train_data),
            val_windows=len(val_data),
            learning_rate_after_update=trainer.optimizers[0].param_groups[0]["lr"],
            peak_memory_bytes=torch.cuda.max_memory_allocated() if args.accelerator == "gpu" else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.accelerator == "gpu" else 0,
            device=torch.cuda.get_device_name(0) if args.accelerator == "gpu" else "CPU",
            cuda=torch.version.cuda,
            rocm=torch.version.hip,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--perception-ckpt", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--physical-batch", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--val-stride", type=int, default=10)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--urban-weight", type=float, default=2.5)
    parser.add_argument("--teacher-forcing", type=float, default=0.0)
    parser.add_argument("--warmup-updates", type=int, default=None)
    parser.add_argument("--schedule-updates", type=int, default=None)
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=320)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--precision", choices=("32-true", "bf16-mixed"), default="bf16-mixed")
    parser.add_argument("--sdpa", choices=("auto", "math", "aotriton", "cuda_flash"), default="aotriton")
    args = parser.parse_args()
    if min(args.physical_batch, args.grad_accum, args.max_steps, args.horizon, args.checkpoint_every) < 1 or args.workers < 0 or args.urban_weight <= 0:
        parser.error("Invalid positive numeric argument")
    try:
        with lock():
            check_existing_training()
            train(args)
    except Timeout as exc:
        raise SystemExit("A bev-wam-x training process already holds the host lock") from exc


if __name__ == "__main__":
    main()
