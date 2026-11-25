"""Train Stage 3.1 / Stage 3.1c using verified, extracted packs and PyTorch Lightning."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytorch_lightning as pl
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F
from filelock import FileLock, Timeout
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, WeightedRandomSampler

from v1_dual_system.checkpoint_io import VerifiedCheckpointIO

from .data import Clips
from .geometry import CONFIG_SHA256, grid
from .objectives import stage31_losses, stage31c_losses
from .pack_colab_v3 import sha256
from .perception_model import BEVWAMXPerception, CLIP_LEN, sdpa_context

LOSS_WEIGHTS = dict(lane=1., vectors=.4, objects=2., occupancy=1., trajectory=.6)
STAGE31C_LOSS_WEIGHTS = dict(lane=0.8, maptr=2.0, heatmap=1.5, objects=2.5,
                             vectors=0.3, occupancy=0.8, trajectory=0.6)


def lr_factor(step, warmup, total):
    """Linear ramp followed by a cosine decay to 5% of the base LR."""
    if step < warmup:
        return (step + 1) / warmup
    fraction = min(1., (step - warmup) / (total - warmup))
    return .05 + .95 * (1 + math.cos(math.pi * fraction)) / 2


def _hungarian_tp(pred_xy, gt_xy, radius=3.0):
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return 0
    distances = np.linalg.norm(pred_xy[:, None, :2] - gt_xy[None, :, :2], axis=-1)
    penalty = (min(distances.shape) + 1) * radius
    costs = np.where(distances < radius, distances, penalty)
    rows, cols = linear_sum_assignment(costs)
    return int((distances[rows, cols] < radius).sum())


class Stage31(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.arch = config.get("architecture", "stage31b")
        pretrained = self.arch == "stage31c" and not config.get("no_pretrained", False)
        self.model = BEVWAMXPerception(
            lidar_dropout=config["lidar_dropout"], arch=self.arch, pretrained=pretrained
        )
        g = grid()
        roi = (g[..., 0] >= 0.0) & (g[..., 0] <= 40.0) & (np.abs(g[..., 1]) <= 18.0)
        self.register_buffer("_roi_mask", torch.from_numpy(roi), persistent=False)
        self._val_kpi = {}

    def on_validation_epoch_start(self):
        self._val_kpi = dict(tp=0, pred=0, gt=0, map_l1_sum=0.0, map_cnt=0, dash_inter=0, dash_union=0)
        self._val_z_kpi = dict(tp=0, pred=0, gt=0, driv_inter=0, driv_union=0)

    def _track_z_val(self):
        return bool(self.arch == "stage31c" and (
            float(self.config.get("z_distill_weight", 0.0)) > 0.0
            or bool(self.config.get("z_all_frames", False))
        ))

    def _update_val_z_kpi(self, p_z, tgt_t):
        scores = p_z["obj_logits"].detach().float().sigmoid().cpu().numpy()
        p_xy = p_z["boxes"].detach().float()[..., :2].cpu().numpy()
        gt_b = tgt_t["boxes"].detach().cpu().numpy()
        for b in range(scores.shape[0]):
            sel = p_xy[b, scores[b] > 0.5]
            gt_valid = gt_b[b, gt_b[b, :, -1] > 0.5, :2]
            self._val_z_kpi["pred"] += len(sel)
            self._val_z_kpi["gt"] += len(gt_valid)
            self._val_z_kpi["tp"] += _hungarian_tp(sel, gt_valid, radius=3.0)
        pred_d = p_z["lane_logits"][:, 0].detach().float() > 0.0
        tgt_lanes = tgt_t.get("lanes_cont", tgt_t["lanes"])
        gt_d = tgt_lanes[:, 0].bool()
        roi = self._roi_mask[None]
        self._val_z_kpi["driv_inter"] += int((pred_d & gt_d & roi).sum().item())
        self._val_z_kpi["driv_union"] += int(((pred_d | gt_d) & roi).sum().item())

    def _update_val_kpi(self, p, tgt_t):
        scores = p["obj_logits"].detach().float().sigmoid().cpu().numpy()
        p_xy = p["boxes"].detach().float()[..., :2].cpu().numpy()
        gt_b = tgt_t["boxes"].detach().cpu().numpy()
        for b in range(scores.shape[0]):
            sel = p_xy[b, scores[b] > 0.5]
            gt_valid = gt_b[b, gt_b[b, :, -1] > 0.5, :2]
            self._val_kpi["pred"] += len(sel)
            self._val_kpi["gt"] += len(gt_valid)
            self._val_kpi["tp"] += _hungarian_tp(sel, gt_valid, radius=3.0)

        pred_m = p["lane_logits"][:, 1].detach().float() > 0.0
        tgt_lanes = tgt_t.get("lanes_cont", tgt_t["lanes"])
        gt_m = tgt_lanes[:, 1].bool()
        roi = self._roi_mask[None]
        self._val_kpi["dash_inter"] += int((pred_m & gt_m & roi).sum().item())
        self._val_kpi["dash_union"] += int(((pred_m | gt_m) & roi).sum().item())

        if "map_polylines" in p and "map_valid" in tgt_t:
            mp = p["map_polylines"].detach().float().cpu().numpy()
            ms = p["map_obj_logits"].detach().float().sigmoid().cpu().numpy()
            gp = tgt_t["map_polylines"].detach().float().cpu().numpy()
            gv = tgt_t["map_valid"].detach().cpu().numpy().astype(bool)
            for b in range(mp.shape[0]):
                cols = gv[b].nonzero()[0]
                if not len(cols):
                    continue
                rows = np.flatnonzero(ms[b] > 0.5)
                if not len(rows):
                    rows = np.arange(mp.shape[1])
                df = np.abs(mp[b, rows][:, None] - gp[b, cols][None]).mean(axis=(-2, -1))
                dr = np.abs(mp[b, rows][:, None] - gp[b, cols, ::-1][None]).mean(axis=(-2, -1))
                dmin = np.minimum(df, dr)
                r_i, c_i = linear_sum_assignment(dmin)
                self._val_kpi["map_l1_sum"] += float(dmin[r_i, c_i].sum())
                self._val_kpi["map_cnt"] += len(r_i)

    def shared_step(self, batch, stage):
        inputs, targets = batch["inputs"], batch["targets"]
        previous = None
        totals = {}
        weights = STAGE31C_LOSS_WEIGHTS if self.arch == "stage31c" else LOSS_WEIGHTS
        loss_fn = stage31c_losses if self.arch == "stage31c" else stage31_losses
        z_aux_w = float(self.config.get("z_aux_weight", 0.15))
        z_distill_w = float(self.config.get("z_distill_weight", 0.0))
        z_all_frames = bool(self.config.get("z_all_frames", False))
        need_bev_fine = bool(self.arch == "stage31c" and stage == "train" and z_distill_w > 0.0)
        track_z_val = bool(stage == "val" and self._track_z_val())
        aux_z = 0.0
        z_distill = 0.0
        with sdpa_context(self.config["sdpa"]):
            last_p, last_tgt, last_dn = None, None, None
            for t in range(CLIP_LEN):
                tgt_t = {k: v[:, t] for k, v in targets.items()}
                gt_dn = tgt_t["boxes"] if (self.arch == "stage31c" and stage == "train") else None
                p = self.model(
                    tuple(x[:, t].contiguous() for x in inputs["rgb"]),
                    lidar_bev=inputs["lidar_bev"][:, t], prev_latent=previous,
                    ego_delta=inputs["ego_delta"][:, t], cmd=inputs["cmd"][:, t],
                    nav_anchors=inputs["nav_anchors"][:, t], ego_speed=inputs["ego_speed"][:, t],
                    gt_boxes=gt_dn, return_bev_fine=need_bev_fine,
                )
                previous = p["z_t"]
                losses = loss_fn(p, tgt_t)
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0) + value / CLIP_LEN
                last_p, last_tgt, last_dn = p, tgt_t, gt_dn
                if self.arch == "stage31c" and stage == "train" and z_all_frames:
                    # Supervise z_t latent-only decoding on every clip frame for Stage 3.2 rollout fidelity.
                    p_dec = self.model.decode_perception(
                        p["z_t"], bev=None, gt_boxes=gt_dn, return_bev_fine=need_bev_fine,
                    )
                    p_dec["trajectory"] = p["trajectory"]
                    z_losses = stage31c_losses(p_dec, tgt_t)
                    aux_z = aux_z + (z_aux_w / CLIP_LEN) * sum(weights[k] * z_losses[k] for k in weights if k != "trajectory")
                    if need_bev_fine:
                        z_distill = z_distill + (z_distill_w / CLIP_LEN) * F.smooth_l1_loss(
                            p_dec["bev_fine"].float(), p["bev_fine"].detach().float()
                        )
                if self.arch == "stage31c" and stage == "val":
                    self._update_val_kpi(p, tgt_t)
                    if track_z_val:
                        p_z = self.model.decode_perception(p["z_t"], bev=None)
                        self._update_val_z_kpi(p_z, tgt_t)

            if self.arch == "stage31c" and stage == "train" and not z_all_frames and last_p is not None:
                # Keep z_t latent-only decoding aligned for downstream Stage 3.2 world-model rollout.
                p_dec = self.model.decode_perception(
                    last_p["z_t"], bev=None, gt_boxes=last_dn if z_distill_w > 0.0 else None,
                    return_bev_fine=need_bev_fine,
                )
                p_dec["trajectory"] = last_p["trajectory"]
                z_losses = stage31c_losses(p_dec, last_tgt)
                aux_z = z_aux_w * sum(weights[k] * z_losses[k] for k in weights if k != "trajectory")
                if need_bev_fine:
                    z_distill = z_distill_w * F.smooth_l1_loss(
                        p_dec["bev_fine"].float(), last_p["bev_fine"].detach().float()
                    )

        loss = sum(weight * totals[key] for key, weight in weights.items()) + aux_z + z_distill
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

    def on_validation_epoch_end(self):
        if self.arch != "stage31c" or not self._val_kpi:
            return
        k = self._val_kpi
        p = float(k["tp"] / max(1, k["pred"]))
        r = float(k["tp"] / max(1, k["gt"]))
        f1 = float(2.0 * p * r / max(1e-8, p + r)) if (p + r) > 0 else 0.0
        map_l1 = float(k["map_l1_sum"] / max(1, k["map_cnt"]))
        dash_iou = float(k["dash_inter"] / max(1, k["dash_union"]))
        metrics = dict(val_det_p3m=p, val_det_r3m=r, val_det_f1=f1,
                       val_map_l1_m=map_l1, val_lane_dashed_iou=dash_iou)
        if self._track_z_val() and self._val_z_kpi:
            zk = self._val_z_kpi
            zp = float(zk["tp"] / max(1, zk["pred"]))
            zr = float(zk["tp"] / max(1, zk["gt"]))
            zf1 = float(2.0 * zp * zr / max(1e-8, zp + zr)) if (zp + zr) > 0 else 0.0
            zdriv = float(zk["driv_inter"] / max(1, zk["driv_union"]))
            metrics.update(val_z_det_p3m=zp, val_z_det_r3m=zr, val_z_det_f1=zf1, val_z_drivable_iou=zdriv)
        self.log_dict(metrics, prog_bar=False, sync_dist=False)

    def configure_optimizers(self):
        base_lr = self.config["learning_rate"]
        if self.arch == "stage31c":
            backbone_params, head_params = [], []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith(("model.camera.stem.", "model.camera.layer")):
                    backbone_params.append(param)
                else:
                    head_params.append(param)
            param_groups = [
                dict(params=head_params, lr=base_lr),
                dict(params=backbone_params, lr=base_lr * 0.3),
            ]
            optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=.01, foreach=True)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=base_lr, weight_decay=.01, foreach=True)
        warmup, total = self.config["warmup_updates"], self.config["schedule_updates"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total))
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step"))

    def on_save_checkpoint(self, checkpoint):
        checkpoint["task_state"] = self.config

    def on_load_checkpoint(self, checkpoint):
        old = checkpoint["task_state"]
        for key in ("architecture", "config_sha256", "pack_sha256", "learning_rate", "clip_len", "physical_batch",
                    "gradient_accumulation", "lidar_dropout", "seed", "warmup_updates", "schedule_updates"):
            if old.get(key) != self.config[key]:
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
    cbs = [ValidationLog(out / "validation.jsonl", config), best, latest]
    if config.get("architecture") == "stage31c":
        best_det = ModelCheckpoint(dirpath=out, filename="best_det", monitor="val_det_f1", mode="max",
                                   save_top_k=1, enable_version_counter=False, save_on_train_epoch_end=False)
        cbs.append(best_det)
        if float(config.get("z_distill_weight", 0.0)) > 0.0 or bool(config.get("z_all_frames", False)):
            best_z = ModelCheckpoint(dirpath=out, filename="best_z_det", monitor="val_z_det_f1", mode="max",
                                     save_top_k=1, enable_version_counter=False, save_on_train_epoch_end=False)
            cbs.append(best_z)
    return cbs, best


def build_trainer(config, out, **overrides):
    cbs, best = callbacks(out, config)
    flags = dict(accelerator=config["accelerator"], devices=1, precision=config["precision"],
                 max_steps=config["max_updates"], accumulate_grad_batches=config["gradient_accumulation"],
                 default_root_dir=out, plugins=[VerifiedCheckpointIO()], callbacks=cbs,
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
               "-m v2_mapless_wa.phase26_train", "-m v3_world_model.train_stage31_colab",
               "-m v3_world_model.train_stage32")
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if proc.info["pid"] == os.getpid() or not (proc.info["name"] or "").lower().startswith("python"):
            continue
        command = " ".join(proc.info["cmdline"] or []).lower()
        if any(x in command for x in modules[1:]) or (modules[0] in command and "train" in command):
            # Windows venv launches a forwarding parent with the same command line.
            if proc.info["pid"] != os.getppid():
                raise RuntimeError(f"Another bev-wam-x training process is active (PID {proc.info['pid']})")


def config_for(args):
    saved = (json.loads((args.out / "training-config.json").read_text(encoding="utf-8"))
             if args.resume else {})
    total = args.schedule_updates
    if total is None:
        total = saved.get("schedule_updates", args.max_updates)
    warmup = args.warmup_updates
    if warmup is None:
        warmup = saved.get("warmup_updates", min(500, total // 20))
    if not 0 <= warmup < total:
        raise ValueError("Warmup updates must be nonnegative and below the schedule length")
    arch = getattr(args, "arch", None) or saved.get("architecture", "stage31b")
    cfg = dict(architecture=arch, config_sha256=CONFIG_SHA256, pack_sha256=sha256(args.dataset / "manifest.json"),
               physical_batch=args.batch, gradient_accumulation=args.accumulate, clip_len=CLIP_LEN,
               max_updates=args.max_updates, checkpoint_every=args.checkpoint_every,
               learning_rate=args.lr, warmup_updates=warmup, schedule_updates=total,
               lidar_dropout=args.lidar_dropout, seed=args.seed,
               accelerator=args.accelerator, precision=args.precision, sdpa=args.sdpa,
               num_workers=args.workers)
    if arch == "stage31c":
        cfg["balanced_maps"] = bool(getattr(args, "balanced_maps", False))
        cfg["no_pretrained"] = bool(getattr(args, "no_pretrained", False))
        z_aux_w = float(getattr(args, "z_aux_weight", saved.get("z_aux_weight", 0.15)))
        z_distill_w = float(getattr(args, "z_distill_weight", saved.get("z_distill_weight", 0.0)))
        z_all_frames = bool(getattr(args, "z_all_frames", saved.get("z_all_frames", False)))
        if z_aux_w != 0.15 or z_distill_w > 0.0 or z_all_frames:
            cfg["z_aux_weight"] = z_aux_w
            cfg["z_distill_weight"] = z_distill_w
            cfg["z_all_frames"] = z_all_frames
        init_from = getattr(args, "init_from", None)
        if init_from is not None:
            cfg["init_from"] = str(init_from)
    return cfg


def _balanced_sampler(dataset):
    counts = Counter(dataset.entries[ep]["map"] for ep, _ in dataset.starts)
    raw = {m: len(dataset.starts) / (len(counts) * c) for m, c in counts.items()}
    min_raw = min(raw.values())
    capped = {m: min(3.0 * min_raw, w) for m, w in raw.items()}
    weights = torch.tensor([capped[dataset.entries[ep]["map"]] for ep, _ in dataset.starts], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train(args):
    init_from = getattr(args, "init_from", None)
    if args.resume and init_from is not None:
        raise ValueError("Cannot combine --resume and --init-from")
    resume = args.out / "latest.ckpt" if args.resume else None
    if args.resume and not resume.is_file():
        raise FileNotFoundError(resume)
    if init_from is not None and not Path(init_from).is_file():
        raise FileNotFoundError(init_from)
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
    if not args.resume and any((args.out / f).exists() for f in ("latest.ckpt", "best.ckpt", "validation.jsonl")):
        raise ValueError("Use --resume or a fresh run directory")
    module = Stage31(cfg)
    if init_from is not None:
        src = VerifiedCheckpointIO().load_checkpoint(Path(init_from), map_location="cpu")
        src_ts = src.get("task_state", {})
        if src_ts.get("config_sha256") != CONFIG_SHA256:
            raise ValueError("init-from checkpoint sensor contract mismatch")
        if src_ts.get("architecture") != cfg["architecture"]:
            raise ValueError(f"init-from architecture mismatch: {src_ts.get('architecture')} != {cfg['architecture']}")
        module.load_state_dict(src["state_dict"], strict=True)
        cfg["init_from_step"] = int(src.get("global_step", 0))
        cfg["init_from_sha256"] = sha256(Path(init_from))
        del src
    cfg["parameter_count"] = sum(p.numel() for p in module.parameters())
    is_31c = cfg["architecture"] == "stage31c"
    train_data = Clips(args.dataset, "train", load_maptr=is_31c, strong_aug=is_31c)
    val_data = Clips(args.dataset, "val", load_maptr=is_31c, strong_aug=False)
    # Keep the last valid schedule on disk if an incompatible resume request fails.
    if not args.resume:
        atomic_json(args.out / "training-config.json", cfg)
    loaders = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=args.accelerator == "gpu",
                   persistent_workers=args.workers > 0)
    sampler = _balanced_sampler(train_data) if (is_31c and cfg.get("balanced_maps")) else None
    train_loader = DataLoader(train_data, shuffle=(sampler is None), sampler=sampler, **loaders)
    trainer, best = build_trainer(cfg, args.out)
    trainer.fit(module, train_loader, DataLoader(val_data, **loaders),
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
    atomic_json(args.out / "training-config.json", cfg)
    atomic_json(args.out / "training-result.json",
                dict(cfg, actual_updates=trainer.global_step,
                     learning_rate_after_update=trainer.optimizers[0].param_groups[0]["lr"],
                     peak_memory_bytes=torch.cuda.max_memory_allocated() if args.accelerator == "gpu" else 0,
                     peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.accelerator == "gpu" else 0,
                     device=torch.cuda.get_device_name(0) if args.accelerator == "gpu" else "CPU",
                     cuda=torch.version.cuda, rocm=torch.version.hip))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arch", choices=("stage31b", "stage31c"), default="stage31b")
    parser.add_argument("--balanced-maps", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="Warm-start model weights from a verified checkpoint into a fresh run/schedule")
    parser.add_argument("--z-aux-weight", type=float, default=0.15,
                        help="Weight on latent-only z_t decode supervision (stage31c)")
    parser.add_argument("--z-distill-weight", type=float, default=0.0,
                        help="Weight on bev_fine feature distillation from online branch to z_t-only branch (stage31c)")
    parser.add_argument("--z-all-frames", action="store_true",
                        help="Supervise z_t-only decoding on every frame in each 4-frame clip (stage31c)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--max-updates", type=int, default=10000)
    parser.add_argument("--warmup-updates", type=int, help="Default: 5%% of schedule length, capped at 500")
    parser.add_argument("--schedule-updates", type=int, help="Fixed cosine schedule length, defaults to first run's max-updates")
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
