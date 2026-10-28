"""Stage 3 self-supervision, fixed validation and auditable latent artifacts."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.baseline import validate_manifest
from dataset.latent import LatentPairs
from evaluation.latent import LatentMoments, paired_episode_bootstrap
from evaluation.memory import memory_snapshot
from mini_wa_jepa.provenance import provenance, write_json
from models.latent import LatentModel
from training.vicreg import latent_loss


def make_loader(dataset, config, shuffle=False):
    options = dict(batch_size=config.batch_size, shuffle=shuffle, drop_last=shuffle,
        num_workers=config.workers, pin_memory=True,
        generator=torch.Generator().manual_seed(config.seed))
    if config.workers:
        options.update(multiprocessing_context='spawn', persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **options)


def inputs(batch, device):
    return [batch[name].to(device, non_blocking=True) for name in ('bev', 'ego', 'future')]


@torch.no_grad()
def evaluate(model, loader, config, device, max_batches=None):
    model.eval()
    moments = LatentMoments(model.encoder.token_count, config.encoder_dim, device)
    errors, copies, episodes, anchors, noisy, recovery = [], [], [], [], [], []
    reg_sum = np.zeros(3)
    for index, batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        prediction, target, current = model(*inputs(batch, device))  # FP32, no autocast
        _, terms = latent_loss(prediction, target, current, config)
        moments.update(target)
        errors.extend((prediction - target).square().mean((1, 2)).cpu().numpy())
        copies.extend((current - target).square().mean((1, 2)).cpu().numpy())
        episodes.extend(batch['episode'].numpy()); anchors.extend(batch['anchor'].numpy())
        noisy.extend(batch['noisy'].numpy()); recovery.extend(batch['recovery'].numpy())
        reg_sum += np.array([float(terms[k]) for k in ('prediction', 'variance', 'covariance')]) * len(target)
    health, arrays = moments.finish()
    prediction, copy = np.asarray(errors), np.asarray(copies)
    means = reg_sum / len(prediction)
    metrics = dict(latent_mse=float(prediction.mean()), copy_last_mse=float(copy.mean()),
        prediction_copy_ratio=float(prediction.mean() / max(copy.mean(), 1e-15)),
        normalized_latent_mse=float(prediction.mean() / max(health['covariance_trace'] / config.encoder_dim, 1e-15)),
        variance_loss=float(means[1]), covariance_loss=float(means[2]),
        loss=float(config.lambda_sim * means[0] + config.lambda_var * means[1] + config.lambda_cov * means[2]),
        **health)
    arrays.update(prediction_mse=prediction, copy_mse=copy, episodes=np.asarray(episodes),
                  anchors=np.asarray(anchors), noisy=np.asarray(noisy), recovery=np.asarray(recovery))
    return metrics, arrays


@torch.no_grad()
def visualize(model, dataset, config, device, output, step, epoch, health, health_arrays, writer=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from mini_wa_jepa.visualization import RerunLogger, colorize_bev_4ch
    model.eval()
    paths = []
    selected = [index for index in config.visualization_sample_ids if index < len(dataset)]
    if not selected:
        raise ValueError('No valid fixed visualization sample IDs')
    samples = [dataset[index] for index in selected]
    bev = torch.from_numpy(np.stack([x['bev'] for x in samples])).to(device)
    future = torch.from_numpy(np.stack([x['future'] for x in samples])).to(device)
    ego = torch.from_numpy(np.stack([x['ego'] for x in samples])).to(device)
    prediction, target, current = [x.float().cpu().numpy() for x in model(bev, ego, future)]
    position_mean = health_arrays['position_mean']
    basis = health_arrays['pca_basis'][:, :2]
    folder = output / 'samples' / f'step_{step:06d}'
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder / 'latents.npz', prediction=prediction, target=target, current=current,
        sample_ids=np.asarray(selected), anchors=np.asarray([s['anchor'] for s in samples]),
        current_bev=bev.cpu().numpy(), future_bev=future.cpu().numpy(),
        position_mean=position_mean, pca_basis=basis, global_step=step, epoch=epoch, horizon_s=dataset.dt_s)
    for slot, (index, sample) in enumerate(zip(selected, samples)):
        pred, truth = prediction[slot], target[slot]
        pmse = float(np.square(pred - truth).mean()); cmse = float(np.square(current[slot] - truth).mean())
        stem = f'sample_{index:05d}'
        with RerunLogger(app_id='mini-wa-jepa-stage3-latents', spawn=False,
                         save_path=folder / f'{stem}.rrd') as logger:
            logger.set_step(step, sample['step'] * dataset.dt_s)
            logger.log_context(global_step=step, epoch=epoch, sample_id=index, anchor=sample['anchor'],
                split=dataset.split, episode_id=sample['episode'], episode_step=sample['step'],
                future_horizon_s=dataset.dt_s, visualization_seed=config.visualization_seed,
                latent_semantics='encoder tokens; no pixel decoder or trajectory head')
            logger.log_bev(sample['bev']); logger.log_camera(None)
            logger.log_state(dataset.states[sample['anchor']])
            logger.log_latents(predicted=pred[None], ground_truth=truth[None])
            logger.log_metrics(dict(latent_mse=pmse, copy_last_mse=cmse,
                effective_rank_ratio=health['effective_rank_ratio'], std_mean=health['std_mean']))
        fig, axes = plt.subplots(2, 3, figsize=(12.8, 6.4), dpi=100)
        axes[0, 0].imshow(colorize_bev_4ch(sample['bev'])); axes[0, 0].set_title('Current BEV (input)')
        axes[0, 1].imshow(colorize_bev_4ch(sample['future'])); axes[0, 1].set_title('Next BEV (+0.1s, target only)')
        for value, label, marker in [(truth, 'target', '.'), (pred, 'prediction', 'x')]:
            pca = (value - position_mean) @ basis
            axes[0, 2].scatter(pca[:, 0], pca[:, 1], s=10, alpha=.7, marker=marker, label=label)
        axes[0, 2].set_title('Shared content PCA (tokens)'); axes[0, 2].legend(fontsize=7)
        bound = max(float(np.quantile(np.abs(np.concatenate([truth, pred])), .99)), .01)
        for ax, value, title in [(axes[1, 0], pred, 'Predicted latent'), (axes[1, 1], truth, 'True latent')]:
            image = ax.imshow(value.T, aspect='auto', cmap='coolwarm', vmin=-bound, vmax=bound)
            ax.set_title(title); ax.set_xlabel('token'); ax.set_ylabel('channel')
            fig.colorbar(image, ax=ax, fraction=.03)
        image = axes[1, 2].imshow(np.abs(pred - truth).T, aspect='auto', cmap='magma', vmin=0)
        axes[1, 2].set_title(f'Absolute error | MSE {pmse:.5f}'); axes[1, 2].set_xlabel('token')
        fig.colorbar(image, ax=axes[1, 2], fraction=.03)
        axes[0, 0].axis('off'); axes[0, 1].axis('off')
        fig.suptitle(f'Stage 3 | epoch {epoch} | train step {step} | {dataset.split} sample {index} | '
                     f'copy MSE {cmse:.5f} | content rank {health["effective_rank_ratio"]:.1%}\n'
                     'Latent heatmaps/PCA are features, not reconstructed images', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .92))
        path = folder / f'{stem}.png'; fig.savefig(path); plt.close(fig); paths.append(str(path))
        if writer:
            writer.add_image(f'latents/sample_{index}', np.asarray(Image.open(path).convert('RGB')).transpose(2, 0, 1), step)
    write_json(folder / 'manifest.json', dict(global_step=step, epoch=epoch,
        sample_ids=selected, sample_id_definition=f'index into full {dataset.split} LatentPairs', horizon_s=dataset.dt_s,
        npz='latents.npz', figures=paths))
    return paths


def checkpoint(model, optimizer, config, report, epoch, step, validation):
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), model_config=asdict(config),
        data_schema_version=2, epoch=epoch, global_step=step, validation=validation,
        parameter_count=report['parameter_count'],
        provenance={k:report[k] for k in ('git_commit', 'source_tree_sha256', 'goal_sha256')})


def train(root, output, config, max_batches=None):
    if not torch.version.hip or torch.version.cuda is not None or not torch.cuda.is_available():
        raise RuntimeError('Native PyTorch ROCm is required')
    manifest = validate_manifest(root)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.cpu_threads); torch.manual_seed(config.seed); np.random.seed(config.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    data = LatentPairs(root, 'train')
    val_full = LatentPairs(root, 'val')
    validation = LatentPairs(root, 'val', config.validation_pairs, config.seed)
    loader = make_loader(data, config, True); val_loader = make_loader(validation, config)
    model = LatentModel(config).to(device)
    encoder_count = sum(p.numel() for p in model.encoder.parameters())
    count = sum(p.numel() for p in model.parameters())
    if not 6_000_000 <= encoder_count <= 8_000_000 or not 15_000_000 <= count <= 30_000_000:
        raise ValueError(f'Parameter budget: encoder={encoder_count}, total={count}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay, fused=False, foreach=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    writer = SummaryWriter(output / 'tensorboard')
    report = dict(**provenance(config), stage=3, spec_version=2, completed=False, epochs=[],
        parameter_count=count, encoder_parameters=encoder_count, predictor_parameters=count-encoder_count,
        data_schema_version=2, dataset=str(Path(root).resolve()), dataset_source=manifest['source_tree_sha256'],
        torch_version=torch.__version__, hip=torch.version.hip, sdpa_backend='MATH',
        train_pairs=len(data), validation_pairs=len(validation), validation_full_pairs=len(val_full),
        horizon_s=data.dt_s, max_batches=max_batches,
        rank_definition='entropy of eigenvalues of position-centered encoder channel covariance / D; no whitening',
        checkpoint_selection='minimum validation prediction/copy ratio among rank>=0.75; else minimum VICReg loss',
        visualization_sample_ids=list(config.visualization_sample_ids), visualizations=[])
    write_json(output / 'report.json', report)
    step = 0; started = time.perf_counter(); best_ratio = best_fallback = float('inf')
    latest_batch = None
    try:
        with sdpa_kernel(SDPBackend.MATH):
            initial, health_arrays = evaluate(model, val_loader, config, device, max_batches)
            report['untrained_validation'] = initial
            report['visualizations'].append(dict(step=0, epoch=0, paths=visualize(model, val_full, config,
                device, output, 0, 0, initial, health_arrays, writer)))
            write_json(output / 'report.json', report)
            for epoch in range(1, config.epochs + 1):
                model.train(); loss_sum = np.zeros(4); seen = 0; tick = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                for batch_index, batch in enumerate(loader):
                    if max_batches and batch_index >= max_batches:
                        break
                    latest_batch = batch
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config.mixed_precision == 'bf16'):
                        predicted, target, current = model(*inputs(batch, device))
                    loss, terms = latent_loss(predicted, target, current, config)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss at step {step}')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                    if not torch.isfinite(norm):
                        raise FloatingPointError(f'Nonfinite gradients at step {step}')
                    optimizer.step(); step += 1
                    values = [float(loss.detach())] + [float(terms[k].detach()) for k in ('prediction', 'variance', 'covariance')]
                    loss_sum += np.asarray(values) * len(batch['bev']); seen += len(batch['bev'])
                    if step % 20 == 0:
                        for name, value in zip(('total', 'prediction', 'variance', 'covariance'), values):
                            writer.add_scalar(f'train/{name}', value, step)
                        writer.add_scalar('train/gradient_norm', float(norm), step)
                    if step % 100 == 0:
                        progress = dict(epoch=epoch, step=step, seen=seen, train_pairs=len(data), loss=values,
                                        elapsed_s=time.perf_counter()-started)
                        write_json(output / 'progress.json', progress); print(json.dumps(progress), flush=True)
                    if step % config.visualization_interval_steps == 0:
                        paths = visualize(model, val_full, config, device, output, step, epoch,
                                          report['epochs'][-1]['validation'] if report['epochs'] else initial,
                                          health_arrays, writer)
                        report['visualizations'].append(dict(step=step, epoch=epoch, paths=paths))
                        model.train()
                metrics, health_arrays = evaluate(model, val_loader, config, device, max_batches)
                row = dict(epoch=epoch, global_step=step, train_pairs_seen=seen,
                    train_loss=(loss_sum / max(seen, 1)).tolist(), validation=metrics,
                    seconds=time.perf_counter()-tick, peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3, memory=memory_snapshot())
                report['epochs'].append(row)
                np.savez_compressed(output / f'validation_epoch_{epoch:03d}.npz', **health_arrays)
                for key in ('latent_mse', 'copy_last_mse', 'prediction_copy_ratio', 'effective_rank_ratio',
                            'std_mean', 'std_p10', 'variance_loss', 'covariance_loss', 'pooled_effective_rank_ratio'):
                    writer.add_scalar(f'val/{key}', metrics[key], step)
                writer.add_histogram('val/channel_std', np.asarray(metrics['channel_std']), step)
                writer.add_histogram('val/covariance_eigenvalues', np.asarray(metrics['eigenvalues']), step)
                writer.add_scalar('memory/peak_allocated_gib', row['peak_allocated_gib'], step)
                state = checkpoint(model, optimizer, config, report, epoch, step, metrics)
                if metrics['loss'] < best_fallback:
                    best_fallback = metrics['loss']; torch.save(state, output / 'fallback.pt')
                    report['fallback_epoch'] = epoch
                if metrics['effective_rank_ratio'] >= config.rank_threshold and metrics['prediction_copy_ratio'] < best_ratio:
                    best_ratio = metrics['prediction_copy_ratio']; torch.save(state, output / 'best.pt')
                    report['best_epoch'] = epoch
                torch.save(dict(**state, scheduler=scheduler.state_dict()), output / 'last.pt')
                if not report['visualizations'] or report['visualizations'][-1]['step'] != step:
                    paths = visualize(model, val_full, config, device, output, step, epoch, metrics, health_arrays, writer)
                    report['visualizations'].append(dict(step=step, epoch=epoch, paths=paths))
                write_json(output / 'report.json', report); writer.flush()
                print(json.dumps(dict(epoch=epoch, step=step, rank=metrics['effective_rank_ratio'],
                    std=metrics['std_mean'], ratio=metrics['prediction_copy_ratio'], seconds=row['seconds'],
                    memory_gib=row['peak_allocated_gib'])), flush=True)
                if metrics['effective_rank'] < 2 or metrics['std_mean'] < .01:
                    raise RuntimeError('Severe latent degeneration; preserve MRE and stop for discussion')
                if row['peak_allocated_gib'] > 14:
                    raise RuntimeError('14 GiB training memory budget exceeded')
                scheduler.step()
            selected = output / ('best.pt' if (output / 'best.pt').exists() else 'fallback.pt')
            state = torch.load(selected, map_location=device, weights_only=False)
            model.load_state_dict(state['model']); report['selected_checkpoint'] = str(selected)
            report['selected_epoch'] = state['epoch']
            test = LatentPairs(root, 'test')
            test_metrics, test_arrays = evaluate(model, make_loader(test, config), config, device, max_batches)
            test_metrics['bootstrap'] = paired_episode_bootstrap(test_arrays['prediction_mse'],
                test_arrays['copy_mse'], test_arrays['episodes'], config.bootstrap_samples, config.seed)
            report['test'] = test_metrics
            np.savez_compressed(output / 'test_raw.npz', **test_arrays)
            tail = report['epochs'][-config.stable_epochs:]
            ci = test_metrics['bootstrap']['delta_ci95']
            report['gates'] = dict(full_run=max_batches is None and len(report['epochs']) == config.epochs,
                rank_stability=len(tail) == config.stable_epochs and all(
                    r['validation']['effective_rank_ratio'] >= config.rank_threshold for r in tail),
                test_rank=test_metrics['effective_rank_ratio'] >= config.rank_threshold,
                prediction_gain=test_metrics['prediction_copy_ratio'] <= 1-config.minimum_prediction_gain,
                significant_gain=ci is not None and ci[1] < 0,
                training_memory=max(r['peak_allocated_gib'] for r in report['epochs']) <= 14)
            report['completed'] = True
    except BaseException:
        report['error'] = traceback.format_exc()
        if latest_batch is not None:
            torch.save(dict(batch=latest_batch, model=model.state_dict(), config=asdict(config), global_step=step),
                       output / 'failure_mre.pt')
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        report['last_global_step'] = step
        write_json(output / 'report.json', report); writer.close()
    return report

