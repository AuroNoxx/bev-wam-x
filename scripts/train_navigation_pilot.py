"""Frozen-encoder schema3 navigation-head pilot; train/validation only."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time
import traceback
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from configs.integrated import IntegratedConfig
from dataset.navigation import eligible_navigation_anchors
from mini_wa_jepa.provenance import provenance, write_json
from mini_wa_jepa.visualization import RerunLogger
from models.navigation_head import NavigationTrajectoryHead
from models.navigation_encoder import load_navigation_encoder
from scripts.collect_integrated_navigation import digest
from evaluation.memory import memory_snapshot


def cache_split(root, output, encoder, checkpoint_sha, token_shape=(64, 256)):
    manifest = json.loads((root/'manifest.json').read_text())
    if not manifest.get('control_verified') or (root/'INVALID_CONTROL_WIRING.json').exists():
        raise ValueError('Collection lacks verified requested/applied external-control alignment')
    if not manifest['completed'] or manifest['schema_version'] != 3 or manifest['split'] not in ('train', 'validation'):
        raise ValueError('Only completed schema3 train/validation collection is allowed')
    output.mkdir()
    frames = sum(row['steps'] for row in manifest['episodes'])
    tokens = np.lib.format.open_memmap(output/'tokens.npy', mode='w+', dtype=np.float16,
                                      shape=(frames, *token_shape))
    values = {k: [] for k in ('history_index', 'ego', 'route_points', 'route_valid', 'vehicle', 'trajectory', 'valid')}
    if manifest.get('input_profile') == 'schema3.1_static_maneuver_permissions':
        values['maneuver_allowed'] = []
    anchors = []; episode_ids = []; episode_local_steps = []; raw_paths = []; offset = 0
    for episode_index, episode in enumerate(manifest['episodes']):
        path = root/episode['raw']
        if digest(path) != episode['raw_sha256']:
            raise ValueError('Raw episode checksum mismatch')
        with np.load(path, allow_pickle=False) as raw:
            np.testing.assert_allclose(raw['action'], raw['applied_action'], atol=1e-7, rtol=0)
            bev = raw['bev']; n = len(bev)
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                for start in range(0, n, 128):
                    encoded = encoder(torch.from_numpy(bev[start:start+128]).to('cuda'))
                    tokens[offset+start:offset+start+len(encoded)] = encoded.half().cpu().numpy()
            for key in values:
                value = raw[key]
                values[key].append(value+offset if key == 'history_index' else value)
            # Every intermediate transition matters, not only 0.2s endpoints.
            steps = np.arange(n)
            keep = eligible_navigation_anchors(raw['valid'], raw['route_valid'], raw['events'],
                                               raw['noise_active'] if 'noise_active' in raw else None)
            anchors.extend((steps[keep]+offset).tolist())
            episode_ids.extend([episode_index]*int(keep.sum()))
            episode_local_steps.extend(steps[keep].tolist())
            raw_paths.append(str(path.resolve()))
            offset += n
    tokens.flush(); del tokens
    for key, arrays in values.items():
        np.save(output/(key+'.npy'), np.concatenate(arrays))
    np.save(output/'anchors.npy', np.asarray(anchors, np.int64))
    np.save(output/'episode_ids.npy', np.asarray(episode_ids, np.int32))
    np.save(output/'episode_steps.npy', np.asarray(episode_local_steps, np.int32))
    info = dict(completed=True, split=manifest['split'], frames=frames, anchors=len(anchors),
                source_manifest_sha256=digest(root/'manifest.json'), encoder_checkpoint_sha256=checkpoint_sha,
                raw_paths=raw_paths, token_shape=list(token_shape), input_profile=manifest.get('input_profile', 'schema3_navigation'),
                navigation_permission=manifest['navigation_permission'])
    write_json(output/'manifest.json', info)
    print(json.dumps(info), flush=True)
    if not anchors:
        raise ValueError('No eligible navigation demonstration anchors')


class CachedNavigation(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        self.arrays = {k: np.load(self.root/(k+'.npy'), mmap_mode='r') for k in
                       ('tokens', 'history_index', 'ego', 'route_points', 'route_valid',
                        'vehicle', 'trajectory', 'anchors', 'episode_ids', 'episode_steps')}
        if self.manifest.get('input_profile') == 'schema3.1_static_maneuver_permissions':
            self.arrays['maneuver_allowed'] = np.load(self.root/'maneuver_allowed.npy', mmap_mode='r')
    def __len__(self):
        return len(self.arrays['anchors'])
    def __getitem__(self, index):
        a = int(self.arrays['anchors'][index]); h = self.arrays['history_index'][a]
        result = dict(history=np.array(self.arrays['tokens'][h], dtype=np.float32),
                      ego=np.array(self.arrays['ego'][h]), sample=index)
        for key in ('route_points', 'route_valid', 'vehicle', 'trajectory'):
            result[key] = np.array(self.arrays[key][a])
        if 'maneuver_allowed' in self.arrays:
            result['maneuver_allowed'] = np.array(self.arrays['maneuver_allowed'][a])
        return result


def predict(model, batch, zero_navigation=False):
    inputs = [batch[k].to('cuda') for k in ('history', 'ego', 'route_points', 'route_valid', 'vehicle')]
    if zero_navigation:
        inputs[2] = torch.zeros_like(inputs[2]); inputs[3] = torch.zeros_like(inputs[3])
    kwargs = dict(maneuver_allowed=batch['maneuver_allowed'].to('cuda')) if model.arch.get('permission_dim', 0) else {}
    return model(*inputs, **kwargs)


@torch.no_grad()
def evaluate(model, loader, zero_navigation=False):
    model.eval(); plans = []; targets = []
    for batch in loader:
        plans.append(predict(model, batch, zero_navigation).cpu().numpy())
        targets.append(batch['trajectory'].numpy())
    plans, targets = np.concatenate(plans), np.concatenate(targets)
    distances = np.linalg.norm(plans[..., :2]-targets[..., :2], axis=-1)
    metrics = dict(ade_m=float(distances.mean()), fde_m=float(distances[:, -1].mean()),
                   normalized_mse=float(np.mean(((plans-targets)/np.array([20., 5., 10., 1.]))**2)))
    return metrics, plans, targets


def visualize(output, data, plans, targets, epoch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from mini_wa_jepa.visualization import colorize_bev_4ch
    from mini_wa_jepa.geometry import ego_to_pixels
    path = output/f'epoch_{epoch:03d}'; path.mkdir()
    chosen = np.linspace(0, len(data)-1, min(4, len(data)), dtype=int)
    np.savez_compressed(path/'predictions.npz', samples=chosen, prediction=plans[chosen], target=targets[chosen])
    figure, axes = plt.subplots(1, len(chosen), figsize=(4*len(chosen), 4), squeeze=False)
    with RerunLogger(app_id='navigation-head-training', spawn=False, save_path=path/'predictions.rrd') as log:
        for i in chosen:
            ep = int(data.arrays['episode_ids'][i]); step = int(data.arrays['episode_steps'][i])
            with np.load(data.manifest['raw_paths'][ep], allow_pickle=False) as raw:
                log.set_step(int(i), float(i)*.1); log.log_bev(raw['bev'][step])
                log.log_navigation(raw['route_points'][step], raw['route_valid'][step], source='map_assisted_route_lane_centerline')
                log.log_trajectories(planned=plans[i], ground_truth=targets[i])
                log.log_context(epoch=epoch, original_episode=ep, original_step=step,
                                tensor_semantics='actual_two_second_expert_trajectory_target')
                ax = axes[0, list(chosen).index(i)]
                ax.imshow(colorize_bev_4ch(raw['bev'][step]))
                for points, color, label in ((raw['route_points'][step][raw['route_valid'][step]], 'magenta', 'route'),
                                              (plans[i, :, :2], 'cyan', 'prediction'),
                                              (targets[i, :, :2], 'gold', 'actual')):
                    pixels = ego_to_pixels(points, 128, 64)
                    ax.plot(pixels[:, 0], pixels[:, 1], color=color, label=label)
                ax.set_xlim(0, 127); ax.set_ylim(127, 0)
                ax.set_title(f'epoch {epoch} | episode {ep} step {step}')
                ax.legend(fontsize=7)
        figure.tight_layout(); figure.savefig(path/'prediction_truth.png', dpi=120)
        plt.close(figure)
        with Image.open(path/'prediction_truth.png') as im:
            log.log_camera(np.asarray(im.convert('RGB')), name='diagnostics/prediction_truth')


def run(args):
    if not torch.version.hip or torch.version.cuda:
        raise RuntimeError('Native ROCm required')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = dict(epochs=25, batch=128, learning_rate=3e-4, weight_decay=1e-4,
               seeds=[20261023, 20261024], encoder_kind=args.encoder,
               checkpoint=('artifacts/restart_stage2/baseline/best.pt' if args.encoder == 'resnet'
                           else IntegratedConfig().encoder_checkpoint),
               train=str(args.train), validation=str(args.validation), reference_prior=args.reference_prior)
    report = dict(**provenance(cfg), completed=False, pilot=True, models=[], stage=7, control_verified=True,
                  test_access=False, source_protocol_sha256=digest('docs/STAGE7_INTEGRATED_PROTOCOL.md'))
    write_json(args.output/'report.json', report)
    with tarfile.open(args.output/'source_snapshot.tar.gz', 'w:gz') as archive:
        for path in report['source_files']:
            archive.add(path, arcname=path)
        archive.add('docs/STAGE7_INTEGRATED_PROTOCOL.md')
    started = time.monotonic()
    try:
        encoder, token_shape = load_navigation_encoder(args.encoder, cfg['checkpoint'], 'cuda')
        encoder_parameters = sum(p.numel() for p in encoder.parameters())
        checkpoint_sha = digest(cfg['checkpoint'])
        for split, source in [('train', args.train), ('validation', args.validation)]:
            cache_split(source, args.output/split, encoder, checkpoint_sha, token_shape)
        del encoder; torch.cuda.empty_cache()
        train = CachedNavigation(args.output/'train'); val = CachedNavigation(args.output/'validation')
        permission_dim = 2 if 'maneuver_allowed' in train.arrays else 0
        if ('maneuver_allowed' in val.arrays) != bool(permission_dim):
            raise ValueError('Training/validation permission input profiles differ')
        validation = DataLoader(val, batch_size=128, shuffle=False, num_workers=0)
        for seed in cfg['seeds']:
            torch.manual_seed(seed); np.random.seed(seed)
            head = NavigationTrajectoryHead(tokens=token_shape[0], token_dim=token_shape[1],
                                            reference_prior=args.reference_prior, permission_dim=permission_dim).to('cuda')
            optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(train, batch_size=128, shuffle=True, generator=generator, num_workers=0)
            folder = args.output/f'head_{seed}'; folder.mkdir()
            writer = SummaryWriter(str(folder/'tensorboard'))
            best = float('inf'); epochs = []; global_step = 0
            try:
                for epoch in range(1, 26):
                    torch.cuda.reset_peak_memory_stats()
                    head.train(); losses = []
                    for batch in loader:
                        target = batch['trajectory'].to('cuda')
                        plan = predict(head, batch)
                        loss = ((plan-target)/head.scale).square().mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite navigation training loss')
                        optimizer.zero_grad(set_to_none=True); loss.backward()
                        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
                        optimizer.step(); losses.append(float(loss.detach())); global_step += 1
                        writer.add_scalar('train/loss', float(loss.detach()), global_step)
                    metrics, plans, targets = evaluate(head, validation)
                    row = dict(epoch=epoch, train_loss=float(np.mean(losses)), **metrics,
                               memory=memory_snapshot(), peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
                    epochs.append(row)
                    for key, value in metrics.items():
                        writer.add_scalar('validation/'+key, value, epoch)
                    if metrics['normalized_mse'] < best:
                        best = metrics['normalized_mse']
                        torch.save(dict(model=head.state_dict(), arch=head.arch, seed=seed, epoch=epoch,
                                        encoder_checkpoint=cfg['checkpoint'], encoder_sha256=checkpoint_sha,
                                        encoder_kind=args.encoder, encoder_parameters=encoder_parameters,
                                        schema_version=3, source_tree_sha256=report['source_tree_sha256'],
                                        goal_sha256=report['goal_sha256']), folder/'best.pt')
                    if epoch % 5 == 0:
                        visualize(folder, val, plans, targets, epoch)
                    write_json(folder/'epochs.json', epochs)
                    print(json.dumps(dict(seed=seed, **row)), flush=True)
                selected = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
                head.load_state_dict(selected['model'])
                metrics, plans, targets = evaluate(head, validation)
                zero, _, _ = evaluate(head, validation, True)
                np.savez_compressed(folder/'validation.npz', prediction=plans, target=targets,
                                    episode_ids=val.arrays['episode_ids'], episode_steps=val.arrays['episode_steps'])
                result = dict(seed=seed, selected_epoch=selected['epoch'], validation=metrics,
                              navigation_zeroed_diagnostic=zero, checkpoint_sha256=digest(folder/'best.pt'),
                              head_parameters=sum(p.numel() for p in head.parameters()), completed=True)
                result['encoder_parameters'] = encoder_parameters
                result['total_navigation_parameters'] = encoder_parameters+result['head_parameters']
                write_json(folder/'report.json', result); report['models'].append(result)
                write_json(args.output/'report.json', report)
            finally:
                writer.close()
        report['completed'] = True
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['elapsed_s'] = time.monotonic()-started
        write_json(args.output/'report.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference-prior', action='store_true', help='Explicit declared-map physical route prior instead of constant observed motion')
    p.add_argument('--encoder', choices=('joint', 'resnet'), default='joint')
    run(p.parse_args())


if __name__ == '__main__':
    main()
