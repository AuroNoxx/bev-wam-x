"""Training pipeline for Conditional Flow Matching trajectory generator on cached features."""
from dataclasses import asdict, is_dataclass
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import tarfile
import time
import traceback

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.flow_features import FlowFeaturesDataset
from mini_wa_jepa.hardware import detect_hardware
from mini_wa_jepa.normalization import Normalizer
from mini_wa_jepa.provenance import provenance, write_json
from models.flow_matcher import ConditionalFlowMatcher


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_model_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode('utf-8'))
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def get_all_rng_states(loader=None):
    states = dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
    )
    if torch.cuda.is_available():
        states['cuda'] = torch.cuda.get_rng_state()
    if loader is not None and getattr(loader, 'generator', None) is not None:
        states['loader_generator'] = loader.generator.get_state()
    return states


def set_all_rng_states(states, loader=None):
    random.setstate(states['python'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'])
    if torch.cuda.is_available() and 'cuda' in states:
        torch.cuda.set_rng_state(states['cuda'])
    if loader is not None and 'loader_generator' in states and getattr(loader, 'generator', None) is not None:
        loader.generator.set_state(states['loader_generator'])


def make_loader(dataset, batch_size=64, shuffle=True, seed=20261021):
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=gen,
        drop_last=False,
    )


def precompute_fixed_val_targets(dataset, seed=42):
    """Precompute fixed noise and time steps for deterministic validation comparison."""
    gen = torch.Generator().manual_seed(seed)
    n = len(dataset)
    fixed_noise = torch.randn(n, 10, 4, generator=gen, dtype=torch.float32)
    fixed_t = torch.rand(n, generator=gen, dtype=torch.float32)
    return fixed_noise, fixed_t


@torch.no_grad()
def evaluate_flow(model, dataset, fixed_noise, fixed_t, batch_size=64, device='cuda'):
    """Evaluate Flow Matching validation loss with strictly fixed targets and weighted average."""
    if len(dataset) == 0:
        raise ValueError("Validation dataset is empty")

    model.eval()
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
    total_loss_sum = 0.0
    total_denom_sum = 0.0
    offset = 0

    with sdpa_kernel(SDPBackend.MATH):
        for batch in loader:
            b = len(batch['target'])
            history = batch['history'].to(device)
            ego = batch['ego'].to(device)
            target = batch['target'].to(device)
            action_mask = batch['action_mask'].to(device)
            waypoint_valid = batch['waypoint_valid'].to(device)

            noise_batch = fixed_noise[offset:offset + b].to(device)
            t_batch = fixed_t[offset:offset + b].to(device)
            offset += b

            point_mask = (action_mask[:, None] & waypoint_valid)[..., None]
            denom = float(point_mask.float().sum() * 4.0)

            if hasattr(model, 'extract_context'):
                loss, _ = model.compute_loss(
                    history=history,
                    ego=ego,
                    target=target,
                    action_mask=action_mask,
                    waypoint_valid=waypoint_valid,
                    noise=noise_batch,
                    t=t_batch,
                )
            else:
                condition = model.extract_condition(history, ego)
                loss, _ = model.compute_loss(
                    x_1=target,
                    condition=condition,
                    action_mask=action_mask,
                    waypoint_valid=waypoint_valid,
                    noise=noise_batch,
                    t=t_batch,
                )
            total_loss_sum += loss.item() * denom
            total_denom_sum += denom

    if total_denom_sum == 0.0:
        raise ValueError("No valid trajectory points in validation dataset; cannot evaluate Flow Matching loss")

    return float(total_loss_sum / total_denom_sum)


@torch.no_grad()
def visualize_candidates(model, dataset, normalizer, output_dir, epoch, step, num_candidates=8, seed=42, writer=None, device='cuda'):
    """Visualize 8 generated candidate trajectories vs ground truth and save NPZ/PNG/RRD."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from mini_wa_jepa.visualization import RerunLogger

    model.eval()
    samples_dir = Path(output_dir) / 'samples' / f'epoch_{epoch:03d}'
    samples_dir.mkdir(parents=True, exist_ok=False)

    sample_indices = [0, min(16, len(dataset) - 1)]
    paths = []

    with sdpa_kernel(SDPBackend.MATH):
        for case_idx, data_idx in enumerate(sample_indices):
            item = dataset[data_idx]
            history = item['history'].unsqueeze(0).to(device)
            ego = item['ego'].unsqueeze(0).to(device)

            if hasattr(model, 'extract_context'):
                candidates_norm = model.sample(history, ego, num_steps=10, num_candidates=num_candidates, seed=seed + case_idx, device=device)
            else:
                condition = model.extract_condition(history, ego)
                candidates_norm = model.sample(condition, num_steps=10, num_candidates=num_candidates, seed=seed + case_idx, device=device)
            candidates_np = candidates_norm.squeeze(0).cpu().numpy()  # [8, 10, 4]
            gt_norm = item['target'].numpy()  # [10, 4]

            candidates_si = normalizer.inverse_trajectory(candidates_np)
            gt_si = normalizer.inverse_trajectory(gt_norm)
            mse_si = normalizer.inverse_trajectory(item['mse_prediction'].numpy())

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(0, 0, 'ks', markersize=6, label='Ego Anchor')

            colors = plt.cm.plasma(np.linspace(0.1, 0.9, num_candidates))
            for c_idx in range(num_candidates):
                c_traj = candidates_si[c_idx]
                ax.plot(c_traj[:, 0], c_traj[:, 1], '--', color=colors[c_idx], alpha=0.8, label=f'Cand {c_idx+1}')

            ax.plot(gt_si[:, 0], gt_si[:, 1], 'g-o', linewidth=2.5, markersize=4, label='Expert GT')
            ax.plot(mse_si[:, 0], mse_si[:, 1], 'c:', linewidth=2.0, label='Stage 4 MSE')
            ax.set_title(f'Diagnostic Trajectory Plan | Epoch {epoch} | Case {case_idx}')
            ax.set_xlabel('Ego X (forward, m)')
            ax.set_ylabel('Ego Y (left, m)')
            ax.axis('equal')
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.tight_layout()

            img_path = samples_dir / f'case_{case_idx}.png'
            fig.savefig(img_path)
            plt.close(fig)

            np_path = samples_dir / f'case_{case_idx}.npz'
            np.savez_compressed(
                np_path,
                candidates=candidates_si,
                ground_truth=gt_si,
                mse_prediction=mse_si,
                waypoint_valid=item['waypoint_valid'].numpy(),
                action_mask=bool(item['action_mask'].item()),
                history_tokens=item['history'].numpy(),
                ego=item['ego'].numpy(),
                anchor=item['anchor'],
                episode=item['episode'],
                step=item['step'],
            )

            rrd_path = samples_dir / f'case_{case_idx}.rrd'
            with RerunLogger(app_id='mini-wa-jepa-stage7-flow', spawn=False, save_path=rrd_path) as log:
                log.set_step(step, float(item['step'] * 0.1))
                log.log_camera(np.asarray(Image.open(img_path).convert('RGB')), name='camera/diagnostics_flow_candidates')
                log.log_trajectories(planned=candidates_si[0], ground_truth=gt_si)
                log.log_context(
                    epoch=epoch,
                    training_step=step,
                    case=case_idx,
                    anchor=item['anchor'],
                    episode=item['episode'],
                    step=item['step'],
                    num_candidates=num_candidates,
                    noise_seed=seed + case_idx,
                    future_prediction='None: Flow Matching branch does not generate world model latents',
                )
                candidate_colors = [
                    (40, 180, 255), (255, 100, 100), (100, 255, 100), (255, 200, 40),
                    (200, 100, 255), (255, 150, 50), (50, 255, 200), (200, 200, 200),
                ]
                for c_idx in range(num_candidates):
                    log.log_candidate(
                        f'cand_{c_idx+1}',
                        commanded=candidates_si[c_idx],
                        actual=np.empty((0, 4), dtype=np.float32),
                        predicted_latent=np.empty((0, 64, 256), dtype=np.float32),
                        target_latent=np.empty((0, 64, 256), dtype=np.float32),
                        color=candidate_colors[c_idx],
                    )

            if writer:
                writer.add_image(f'cases/case_{case_idx}', np.asarray(Image.open(img_path).convert('RGB')).transpose(2, 0, 1), step)

            paths.append(str(img_path))

    write_json(samples_dir / 'manifest.json', dict(epoch=epoch, step=step, paths=paths, selected=sample_indices))
    return paths


def calibrate_flow_step(model, dummy_batch, optimizer, loader=None, device='cuda'):
    """Calibration forward/backward to check time and peak VRAM with strict full state isolation."""
    was_training = model.training
    model_state_before = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    opt_state_before = {k: v for k, v in optimizer.state_dict().items()}
    rng_before = get_all_rng_states(loader)
    hash_before = compute_model_hash(model)

    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    model.train()
    history = dummy_batch['history'].to(device)
    ego = dummy_batch['ego'].to(device)
    target = dummy_batch['target'].to(device)
    action_mask = dummy_batch['action_mask'].to(device)
    waypoint_valid = dummy_batch['waypoint_valid'].to(device)

    optimizer.zero_grad()
    autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()
    with autocast_ctx:
        with sdpa_kernel(SDPBackend.MATH):
            if hasattr(model, 'extract_context'):
                loss, _ = model.compute_loss(
                    history=history,
                    ego=ego,
                    target=target,
                    action_mask=action_mask,
                    waypoint_valid=waypoint_valid,
                )
            else:
                condition = model.extract_condition(history, ego)
                loss, _ = model.compute_loss(
                    x_1=target,
                    condition=condition,
                    action_mask=action_mask,
                    waypoint_valid=waypoint_valid,
                )

    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
    if not np.isfinite(grad_norm):
        raise FloatingPointError(f"Non-finite gradient in calibration: {grad_norm}")
    optimizer.step()
    optimizer.zero_grad()

    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_gib = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
    else:
        peak_vram_gib = 0.0

    elapsed = time.perf_counter() - t0

    # Strictly restore all states
    model.load_state_dict(model_state_before)
    optimizer.load_state_dict(opt_state_before)
    model.train(was_training)
    set_all_rng_states(rng_before, loader=loader)

    hash_after = compute_model_hash(model)
    if hash_before != hash_after:
        raise RuntimeError(f"Calibration state corruption: before={hash_before}, after={hash_after}")

    return dict(
        step_seconds=elapsed,
        peak_vram_gib=peak_vram_gib,
        model_hash_before=hash_before,
        model_hash_after=hash_after,
        isolated=True,
    )


def train_flow(cache_dir, output_dir, config, seed=20261021, smoke=False, device='cuda'):
    if device == 'cuda' and not getattr(torch.version, 'hip', None):
        raise RuntimeError('Native ROCm HIP is required for Stage 7 Flow Matching')

    torch.set_num_threads(4)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    cache_dir = Path(cache_dir)
    cache_manifest = json.loads((cache_dir / 'manifest.json').read_text())
    if cache_manifest.get('smoke', False) and not smoke:
        raise ValueError('Cannot run formal training on smoke cache!')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    normalizer = Normalizer(**json.loads((cache_dir / 'normalization.json').read_text()))

    report = dict(
        **provenance(config),
        stage=7,
        flow_matching_training=True,
        completed=False,
        smoke=smoke,
        seed=seed,
        cache_manifest_hash=file_sha256(cache_dir / 'manifest.json'),
        normalization_hash=file_sha256(cache_dir / 'normalization.json'),
        backend='ROCm HIP SDPA MATH',
        history=[],
    )
    write_json(output / 'manifest.json', report)

    with tarfile.open(output / 'source_snapshot.tar.gz', 'w:gz') as archive:
        for source in report['source_files']:
            archive.add(source, arcname=source)
        for doc in ('GOAL.md', 'AGENTS.md', 'docs/STAGE7_PROTOCOL.md'):
            archive.add(doc, arcname=doc)

    writer = None
    try:
        train_data = FlowFeaturesDataset(cache_dir, split='train')
        val_data = FlowFeaturesDataset(cache_dir, split='val')

        if smoke:
            train_data.anchors = train_data.anchors[:128]
            val_data.anchors = val_data.anchors[:64]

        if getattr(config, 'flow_model_type', 'transformer_kinematic') == 'transformer_kinematic':
            from models.flow_matcher_transformer import TransformerKinematicFlowMatcher, kinematic_bicycle_rollout
            train_targets = train_data.targets
            train_ego = torch.from_numpy(train_data.ego)
            raw_prior = kinematic_bicycle_rollout(train_ego).numpy()
            norm_prior = normalizer.trajectory(raw_prior)
            kin_diff = train_targets - norm_prior
            residual_std_np = kin_diff.std(axis=0)
            residual_std = torch.tensor(residual_std_np, dtype=torch.float32)

            model = TransformerKinematicFlowMatcher(
                token_count=64,
                token_dim=256,
                d_model=getattr(config, 'flow_d_model', 384),
                n_heads=getattr(config, 'flow_num_heads', 6),
                n_layers=getattr(config, 'flow_num_layers', 5),
                dim_feedforward=getattr(config, 'flow_dim_feedforward', 1536),
                residual_std=residual_std,
                normalizer=normalizer,
            ).to(device)
        else:
            model = ConditionalFlowMatcher(token_count=64, token_dim=256, condition_dim=256, hidden_dim=config.flow_hidden_dim).to(device)

        head_params = sum(p.numel() for p in model.parameters())

        # Read base parameter count from Stage 6 checkpoint
        stage6_ckpt = torch.load(config.stage6_checkpoint, map_location='cpu', weights_only=False)
        active_keys = [k for k in stage6_ckpt['model'].keys() if k.startswith('world.') or k.startswith('base.encoder.')]
        if active_keys:
            stage6_base_params = sum(stage6_ckpt['model'][k].numel() for k in active_keys)
        else:
            stage6_base_params = int(stage6_ckpt.get('parameter_count', 80_926_272))
        if stage6_base_params < 15_000_000:
            raise ValueError(f"Stage 6 parameter count {stage6_base_params} is unexpectedly below 15M")

        reserved_risk_params = 4_000_000
        total_budgeted_params = stage6_base_params + head_params + reserved_risk_params
        if not (50_000_000 <= total_budgeted_params <= 100_000_000):
            raise ValueError(f'Total parameter count {total_budgeted_params} outside 50M-100M budget!')

        fixed_noise, fixed_t = precompute_fixed_val_targets(val_data, seed=42)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.flow_learning_rate, weight_decay=config.weight_decay)
        epochs = 2 if smoke else config.flow_epochs
        train_loader = make_loader(train_data, batch_size=config.batch_size, shuffle=True, seed=seed)
        total_steps = epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

        # Calibration inside try block
        dummy_batch = next(iter(train_loader))
        calib_result = calibrate_flow_step(model, dummy_batch, optimizer, loader=train_loader, device=device)

        hw = detect_hardware()
        hw_dict = asdict(hw) if is_dataclass(hw) else (hw.to_dict() if hasattr(hw, 'to_dict') else str(hw))

        report.update(dict(
            head_parameters=head_params,
            stage6_base_parameters=stage6_base_params,
            total_budgeted_parameters=total_budgeted_params,
            calibration=calib_result,
            hardware=hw_dict,
            train_samples=len(train_data),
            val_samples=len(val_data),
            epochs=epochs,
        ))
        write_json(output / 'manifest.json', report)
        write_json(output / 'calibration.json', calib_result)

        writer = SummaryWriter(log_dir=str(output / 'tensorboard'))
        best_val_loss = float('inf')
        global_step = 0
        start_time = time.perf_counter()

        # Initial visualization at epoch 0 inside try block
        visualize_candidates(model, val_data, normalizer, output, epoch=0, step=0, writer=writer, device=device)

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_losses = []
            epoch_t0 = time.perf_counter()

            for batch in train_loader:
                step_t0 = time.perf_counter()
                history = batch['history'].to(device)
                ego = batch['ego'].to(device)
                target = batch['target'].to(device)
                action_mask = batch['action_mask'].to(device)
                waypoint_valid = batch['waypoint_valid'].to(device)

                optimizer.zero_grad()
                autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()
                with autocast_ctx:
                    with sdpa_kernel(SDPBackend.MATH):
                        if hasattr(model, 'extract_context'):
                            loss, _ = model.compute_loss(
                                history=history,
                                ego=ego,
                                target=target,
                                action_mask=action_mask,
                                waypoint_valid=waypoint_valid,
                            )
                        else:
                            condition = model.extract_condition(history, ego)
                            loss, _ = model.compute_loss(
                                x_1=target,
                                condition=condition,
                                action_mask=action_mask,
                                waypoint_valid=waypoint_valid,
                            )

                loss_val = float(loss.item())
                if not np.isfinite(loss_val):
                    raise FloatingPointError(f"Non-finite loss encountered at epoch {epoch}, step {global_step}: {loss_val}")

                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip))
                if not np.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite gradient norm at epoch {epoch}, step {global_step}: {grad_norm}")

                optimizer.step()
                scheduler.step()

                step_time = time.perf_counter() - step_t0
                throughput = len(target) / max(step_time, 1e-5)
                vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3) if (device == 'cuda' and torch.cuda.is_available()) else 0.0

                epoch_losses.append(loss_val)
                writer.add_scalar('train/loss', loss_val, global_step)
                writer.add_scalar('train/grad_norm', grad_norm, global_step)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
                writer.add_scalar('train/throughput_fps', throughput, global_step)
                writer.add_scalar('train/vram_allocated_gib', vram_gib, global_step)
                global_step += 1

            train_loss = float(np.mean(epoch_losses))
            val_loss = evaluate_flow(model, val_data, fixed_noise, fixed_t, batch_size=config.batch_size, device=device)
            writer.add_scalar('val/flow_loss', val_loss, global_step)

            is_best = val_loss < best_val_loss
            ckpt_data = dict(
                model=model.state_dict(),
                model_arch=model.model_arch,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                epoch=epoch,
                step=global_step,
                val_loss=val_loss,
                rng_states=get_all_rng_states(train_loader),
                config=config.__dict__ if hasattr(config, '__dict__') else {},
            )

            if is_best:
                best_val_loss = val_loss
                torch.save(ckpt_data, output / 'best.pt')

            torch.save(ckpt_data, output / 'last.pt')

            if epoch % 5 == 0 or epoch == epochs:
                visualize_candidates(model, val_data, normalizer, output, epoch=epoch, step=global_step, writer=writer, device=device)

            epoch_record = dict(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                is_best=is_best,
                lr=float(scheduler.get_last_lr()[0]),
                epoch_seconds=round(time.perf_counter() - epoch_t0, 2),
            )
            report['history'].append(epoch_record)
            write_json(output / 'manifest.json', report)
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train: {train_loss:.5f} | Val: {val_loss:.5f} {'*' if is_best else ''}", flush=True)

        report['completed'] = True
        report['best_val_loss'] = best_val_loss
        report['elapsed_seconds'] = round(time.perf_counter() - start_time, 2)
        write_json(output / 'manifest.json', report)
        print("Flow Matching training completed successfully.", flush=True)
    except BaseException:
        report['error'] = traceback.format_exc()
        emergency_path = output / 'emergency_checkpoint.pt'
        try:
            ckpt_data = dict(
                epoch=epoch if 'epoch' in locals() else 0,
                step=global_step if 'global_step' in locals() else 0,
                rng_states=get_all_rng_states(train_loader if 'train_loader' in locals() else None),
            )
            if 'model' in locals():
                ckpt_data['model'] = model.state_dict()
                ckpt_data['model_arch'] = model.model_arch
            if 'optimizer' in locals():
                ckpt_data['optimizer'] = optimizer.state_dict()
            if 'scheduler' in locals():
                ckpt_data['scheduler'] = scheduler.state_dict()
            if 'batch' in locals():
                ckpt_data['failed_batch'] = {k: v.cpu() for k, v in batch.items()}
            torch.save(ckpt_data, emergency_path)
        except Exception:
            pass
        write_json(output / 'manifest.json', report)
        raise
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
