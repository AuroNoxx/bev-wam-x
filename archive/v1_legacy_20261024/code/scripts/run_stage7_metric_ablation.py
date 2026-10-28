"""Matched v2 loss experiment; train/validation only, immutable prior artifacts."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import tarfile
import time
import traceback

import numpy as np
import psutil
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from dataset.flow_features import FlowFeaturesDataset
from evaluation.memory import gpu_devices
from mini_wa_jepa.provenance import provenance, write_json
from mini_wa_jepa.visualization import RerunLogger
from models.flow_matcher_metric import MetricFlowMatcher
from models.joint import constant_turn_trajectory
from scripts.report_stage7 import kinematics
from training.flow_matching import compute_model_hash, file_sha256


@contextmanager
def isolated_rng():
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        yield


def resource_record():
    proc = psutil.Process()
    devices = []
    for dev in gpu_devices():
        row = dict(dev)
        path = Path(dev['path']) / 'mem_info_vram_used'
        row['device_used_bytes'] = int(path.read_text()) if path.exists() else None
        devices.append(row)
    return dict(time=time.time(), rss_bytes=proc.memory_info().rss,
                pss_bytes=proc.memory_full_info().pss,
                available_ram_bytes=psutil.virtual_memory().available,
                peak_allocator_bytes=torch.cuda.max_memory_allocated(), gpu=devices)


def append_json(path, data):
    with Path(path).open('a') as f:
        f.write(json.dumps(data, allow_nan=False) + '\n')


@torch.no_grad()
def sample_anchors(model, history, ego, anchors, temperature, candidates=8):
    """Batch integration using the SAME per-anchor HIP noise as the v2 sampler."""
    source = torch.stack([torch.randn(candidates, 10, 4, device=history.device,
                generator=torch.Generator(device=history.device).manual_seed(42 + int(anchor)))
                for anchor in anchors]) * temperature
    context = model.extract_context(history, ego).repeat_interleave(candidates, dim=0)
    z = source.flatten(0, 1)
    for step in range(10):
        t = torch.full((len(z),), step / 10, device=z.device, dtype=z.dtype)
        z = z + .1 * model.forward_vf(z, t, context)
    prior = model.compute_norm_prior(ego)[:, None]
    return prior + z.reshape(len(history), candidates, 10, 4) * model.residual_std


@torch.no_grad()
def sampled_evaluation(model, dataset, indices, temperatures, output, batch_size=32):
    """Evaluate literal candidate 1, all candidates and whole-candidate rejection."""
    output.mkdir(parents=True, exist_ok=False)
    normalizer = dataset.normalizer
    selected = Subset(dataset, indices.tolist())
    loader = DataLoader(selected, batch_size=batch_size, shuffle=False)
    per_temp = {str(t): [] for t in temperatures}
    truths = []; anchors = []; episodes = []; baselines = []
    model.eval()
    for batch in loader:
        h = batch['history'].to('cuda'); e = batch['ego'].to('cuda')
        truths.append(normalizer.inverse_trajectory(batch['target'].numpy()))
        baselines.append(normalizer.inverse_trajectory(batch['mse_prediction'].numpy()))
        anchors.append(batch['anchor'].numpy()); episodes.append(batch['episode'].numpy())
        for tau in temperatures:
            pred = sample_anchors(model, h, e, batch['anchor'].tolist(), tau)
            per_temp[str(tau)].append(normalizer.inverse_trajectory(pred.cpu().numpy()))
    gt = np.concatenate(truths); baseline = np.concatenate(baselines)
    anchor = np.concatenate(anchors); episode = np.concatenate(episodes)
    baseline_error = np.linalg.norm(baseline[..., :2] - gt[..., :2], axis=-1)
    reference = dict(ade_m=float(baseline_error.mean()), fde_m=float(baseline_error[:, -1].mean()))
    rows = {}
    for tau in temperatures:
        candidates = np.concatenate(per_temp[str(tau)])
        error = np.linalg.norm(candidates[..., :2] - gt[:, None, :, :2], axis=-1)
        upper = np.triu_indices(8, 1)
        pairwise = np.linalg.norm(candidates[:, upper[0], :, :2] - candidates[:, upper[1], :, :2], axis=-1).mean((1, 2))
        kin = kinematics(candidates)
        # Literal 0.1m/segment convention from the original deployed validator.
        row = dict(temperature=tau, ade_m=float(error[:, 0].mean()),
                   fde_m=float(error[:, 0, -1].mean()), diversity_m=float(pairwise.mean()),
                   all_candidate_ade_m=float(error.mean()),
                   oracle_min_ade_m=float(error.mean(-1).min(-1).mean()), **kin)
        row['gates'] = dict(ade=row['ade_m'] <= 1.25 * reference['ade_m'],
                            fde=row['fde_m'] <= 1.25 * reference['fde_m'],
                            diversity=row['diversity_m'] >= .1,
                            kinematics=kin['pointwise_valid_fraction'] >= .95)
        row['trajectory_gates_passed'] = all(row['gates'].values())
        rows[str(tau)] = row
        np.savez_compressed(output / f'tau_{tau:g}.npz', candidates=candidates, target=gt,
                            mse=baseline, anchors=anchor, episodes=episode,
                            sample1_ade_m=error[:, 0].mean(-1), sample1_fde_m=error[:, 0, -1],
                            diversity_m=pairwise)
    result = dict(split='val', anchors=len(indices), mse_reference=reference, temperatures=rows,
                  selection='Literal first random candidate; identical temperature for all 8 candidates',
                  kinematic_convention='Moving displacement >0.1m/segment; same as deployed planner',
                  latency_gate='Not measured by batched evaluation; separate final serial measurements')
    write_json(output / 'summary.json', result)
    return result


@torch.no_grad()
def serial_latency(model, dataset, indices, temperatures):
    rows = {}; model.eval()
    for tau in temperatures:
        durations = []
        for i in range(84):
            item = dataset[int(indices[i % len(indices)])]
            h = item['history'][None].to('cuda'); e = item['ego'][None].to('cuda')
            torch.cuda.synchronize(); started = time.perf_counter()
            model.sample(h, e, num_steps=10, num_candidates=8, temperature=tau, seed=42+item['anchor'])
            torch.cuda.synchronize()
            if i >= 20: durations.append(1000 * (time.perf_counter() - started))
        rows[str(tau)] = dict(p95_ms=float(np.percentile(durations, 95)),
                              p50_ms=float(np.median(durations)), samples_ms=durations,
                              passes_30ms=bool(np.percentile(durations, 95) <= 30))
    return rows


@torch.no_grad()
def save_predictions(model, dataset, folder, epoch, global_step):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True, exist_ok=False)
    model.eval()
    for index in (0, 16):
        item = dataset[index % len(dataset)]
        pred = model.sample(item['history'][None].to('cuda'), item['ego'][None].to('cuda'),
                            num_candidates=8, temperature=1., seed=42 + item['anchor'])
        pred = dataset.normalizer.inverse_trajectory(pred[0].cpu().numpy())
        gt = dataset.normalizer.inverse_trajectory(item['target'].numpy())
        stem = folder / f'anchor_{item["anchor"]}'
        np.savez_compressed(stem.with_suffix('.npz'), candidates=pred, ground_truth=gt,
                            history_tokens=item['history'].numpy(), observed_ego=item['ego'].numpy(),
                            anchor=item['anchor'], epoch=epoch, global_step=global_step)
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, trajectory in enumerate(pred):
            ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=.6, label='random candidate 1' if i == 0 else None)
        ax.plot(gt[:, 0], gt[:, 1], 'k.-', label='expert ground truth')
        ax.set(xlabel='Forward (m)', ylabel='Left (m)', title=f'Validation | epoch {epoch} | tau=1')
        ax.axis('equal'); ax.legend(); fig.tight_layout(); fig.savefig(stem.with_suffix('.png')); plt.close(fig)
        with RerunLogger(app_id='stage7-v2-loss-ablation', spawn=False, save_path=stem.with_suffix('.rrd')) as log:
            log.set_step(global_step, item['step'] * .1)
            log.log_trajectories(planned=pred[0], ground_truth=gt)
            log.log_context(epoch=epoch, anchor=item['anchor'], split='val', temperature=1.,
                            actual_execution='Missing: offline expert evaluation',
                            future_latents='Missing: trajectory generator does not predict world latents')
            for i, trajectory in enumerate(pred):
                log.log_candidate(str(i), commanded=trajectory, actual=np.empty((0, 4)),
                                  predicted_latent=np.empty((0, 64, 256)), target_latent=np.empty((0, 64, 256)),
                                  color=(40 + i*25, 160, 240 - i*25))


@torch.no_grad()
def fixed_validation_loss(model, dataset, loader, noise, times, weights):
    total = {k: 0. for k in ('flow', 'reconstructed_ade_m', 'reconstructed_fde_m', 'total')}
    offset = 0; model.eval()
    for b in loader:
        size = len(b['target'])
        _, _, parts = model.compute_loss(b['history'].to('cuda'), b['ego'].to('cuda'),
            b['target'].to('cuda'), b['action_mask'].to('cuda'), b['waypoint_valid'].to('cuda'),
            noise[offset:offset+size].to('cuda'), times[offset:offset+size].to('cuda'),
            ade_weight=weights[0], fde_weight=weights[1], return_parts=True)
        for k, v in parts.items(): total[k] += v.item() * size
        offset += size
    return {k: v / offset for k, v in total.items()}


def train_run(root, train, val, config, seed, arm, weights, evaluation_indices):
    out = root / f'{arm}_{seed}'; out.mkdir(exist_ok=False)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    prior = constant_turn_trajectory(torch.from_numpy(train.ego)).numpy()
    residual_std = (train.targets - train.normalizer.trajectory(prior)).std(axis=0)
    model = MetricFlowMatcher(residual_std=residual_std, normalizer=train.normalizer).to('cuda')
    parameters = sum(p.numel() for p in model.parameters())
    frozen_world_parameters = 22535784  # Measured v2 world module, including frozen Joint.
    if not 15e6 <= parameters + frozen_world_parameters + 473094 <= 30e6:
        raise ValueError('Full policy parameter budget exceeded')
    initial_hash = compute_model_hash(model)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, generator=generator)
    val_loader = DataLoader(val, batch_size=config['batch_size'], shuffle=False)
    g = torch.Generator().manual_seed(42)
    # Reproduce v2 validation RNG ordering: all noise, then all times.
    noise = torch.randn(len(val), 10, 4, generator=g); times = torch.rand(len(val), generator=g)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs']*len(loader))
    writer = SummaryWriter(str(out / 'tensorboard'))
    best = float('inf'); global_step = 0; started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    report = dict(completed=False, seed=seed, arm=arm, weights=weights, initial_model_sha256=initial_hash,
                  head_parameters=parameters, full_policy_parameters=parameters+frozen_world_parameters+473094,
                  config=config, source_tree_sha256=config['source_tree_sha256'])
    write_json(out / 'report.json', report)
    for epoch in range(1, config['epochs'] + 1):
        model.train(); train_losses = {k: 0. for k in ('flow', 'reconstructed_ade_m', 'reconstructed_fde_m', 'total')}
        count = 0
        for b in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, parts = model.compute_loss(b['history'].to('cuda'), b['ego'].to('cuda'),
                    b['target'].to('cuda'), b['action_mask'].to('cuda'), b['waypoint_valid'].to('cuda'),
                    ade_weight=weights[0], fde_weight=weights[1], return_parts=True)
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at {global_step}')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); scheduler.step(); global_step += 1
            for k, v in parts.items(): train_losses[k] += v.item() * len(b['target'])
            count += len(b['target'])
        with isolated_rng():
            val_losses = fixed_validation_loss(model, val, val_loader, noise, times, weights)
            if val_losses['flow'] < best:
                best = val_losses['flow']
                torch.save(dict(model=model.state_dict(), model_arch=model.model_arch, epoch=epoch,
                    val_flow=best, config=config, seed=seed, arm=arm, weights=weights,
                    initial_model_sha256=initial_hash), out / 'best.pt')
                report['selected_epoch'] = epoch
            row = dict(epoch=epoch, global_step=global_step,
                       train={k:v/count for k,v in train_losses.items()}, val=val_losses,
                       elapsed_s=time.perf_counter()-started)
            for k in train_losses: writer.add_scalar('train/'+k, row['train'][k], global_step)
            for k, v in val_losses.items(): writer.add_scalar('val/'+k, v, global_step)
            if epoch == 1 or epoch % 5 == 0 or epoch == config['epochs']:
                save_predictions(model, val, out / 'samples' / f'epoch_{epoch:03}', epoch, global_step)
                probe = sampled_evaluation(model, val, evaluation_indices[:64], [1.],
                                            out / 'probes' / f'epoch_{epoch:03}')
                row['probe_tau1'] = probe['temperatures']['1.0']
                for k in ('ade_m', 'fde_m', 'diversity_m'):
                    writer.add_scalar('sampled_probe_tau1/'+k, row['probe_tau1'][k], global_step)
        append_json(out / 'history.jsonl', row)
        append_json(out / 'resources.jsonl', dict(epoch=epoch, **resource_record()))
        writer.flush()
        print(json.dumps(dict(run=out.name, epoch=epoch, train=row['train']['total'],
                              val_flow=val_losses['flow'], seconds=row['elapsed_s'],
                              probe=row.get('probe_tau1')), ensure_ascii=False), flush=True)
    model.load_state_dict(torch.load(out / 'best.pt', map_location='cuda', weights_only=False)['model'])
    result = sampled_evaluation(model, val, evaluation_indices, config['temperatures'], out / 'validation')
    timing = serial_latency(model, val, evaluation_indices, config['temperatures'])
    write_json(out / 'serial_latency.json', timing)
    for tau, row in result['temperatures'].items():
        row['head_p95_ms'] = timing[tau]['p95_ms']
        row['gates']['latency'] = timing[tau]['passes_30ms']
        row['all_gates_passed'] = all(row['gates'].values())
    write_json(out / 'validation' / 'summary.json', result)
    report.update(completed=True, validation=result, training_seconds=time.perf_counter()-started,
                  best_checkpoint_sha256=file_sha256(out / 'best.pt'), resource=resource_record())
    write_json(out / 'report.json', report); writer.close()
    print(json.dumps(dict(run=out.name, completed=True, validation=result)), flush=True)
    return report


def checkpoint_inventory():
    return {str(p): file_sha256(p) for p in Path('artifacts').rglob('best.pt')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args(); root = args.output; root.mkdir(parents=True, exist_ok=False)
    state = dict(completed=False, phase='preflight')
    try:
        if not torch.version.hip or torch.version.cuda:
            raise RuntimeError('Native ROCm PyTorch required')
        torch.set_num_threads(4)
        config = dict(epochs=1 if args.smoke else 25, batch_size=64,
            training_seeds=[20261021] if args.smoke else [20261021,20261022],
            temperatures=[1.] if args.smoke else [.4,.6,.8,1.],
            arms={'control':[0.,0.], 'metric':[.25,.5]},
            loss='flow + 0.25/m * reconstructed_ADE_m + 0.5/m * reconstructed_FDE_m',
            reconstruction='z_t+(1-t)*v; physical linear-path endpoint estimator, not sampled rollout',
            checkpoint_selection='Minimum fixed-target validation FLOW loss, identical in both arms',
            sampled_metric_selection='No sampled metrics used for checkpoint selection',
            candidates=8, euler_steps=10, precision='float32 (matched to v2 runner)',
            test_access=False, mixed_candidate_temperatures=False,
            normalization='Train split residual std and existing train-only global normalization',
            kinematic_convention='Original strict moving displacement >0.1m/segment',
            smoke=args.smoke)
        prov = provenance(config); config['source_tree_sha256'] = prov['source_tree_sha256']
        write_json(root / 'provenance.json', prov); write_json(root / 'config.json', config)
        with tarfile.open(root / 'source.tar.gz', 'w:gz') as archive:
            for file in prov['source_files']:
                archive.add(file, arcname=file)
            for file in ('GOAL.md','AGENTS.md','docs/STAGE7_METRIC_ABLATION.md'):
                archive.add(file, arcname=file)
        before = checkpoint_inventory(); write_json(root / 'protected_checkpoints.json', before)
        train = FlowFeaturesDataset('artifacts/restart_stage7/flow_features_001', 'train')
        val = FlowFeaturesDataset('artifacts/restart_stage7/flow_features_001', 'val')
        if not (np.all(train.action_mask) and np.all(train.waypoint_valid)
                and np.all(val.action_mask) and np.all(val.waypoint_valid)):
            raise ValueError('This experiment requires the original fully-valid expert cohort')
        if args.smoke:
            # Views retain metadata used for train-only residual statistics.
            for dataset, n in ((train,128),(val,64)):
                for name in ('history_indices','ego','targets','waypoint_valid','action_mask','anchors','episodes','steps','mse_predictions'):
                    setattr(dataset,name,getattr(dataset,name)[:n])
        indices = np.random.default_rng(20261023).permutation(len(val))
        write_json(root / 'cohort.json', dict(train_anchors=train.anchors.tolist(), val_anchors=val.anchors.tolist(),
            validation_order=indices.tolist(), feature_manifest_sha256=file_sha256(train.root / 'manifest.json'),
            normalization_sha256=file_sha256(train.root / 'normalization.json')))
        reports = {}
        with sdpa_kernel(SDPBackend.MATH):
            for seed in config['training_seeds']:
                for arm, weights in config['arms'].items():
                    state.update(phase='training', seed=seed, arm=arm)
                    write_json(root / 'status.json', state)
                    reports[f'{arm}_{seed}'] = train_run(root, train, val, config, seed, arm, weights, indices)
        for path, digest in before.items():
            if file_sha256(path) != digest: raise AssertionError(f'Prior checkpoint changed: {path}')
        for seed in config['training_seeds']:
            if reports[f'control_{seed}']['initial_model_sha256'] != reports[f'metric_{seed}']['initial_model_sha256']:
                raise AssertionError('Control and metric initialization differ')
        state.update(completed=True, phase='experiments_complete', protected_checkpoints_unchanged=True)
        write_json(root / 'summary.json', dict(config=config, runs=reports, **state))
        write_json(root / 'status.json', state)
    except BaseException:
        state.update(error=traceback.format_exc()); write_json(root / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
