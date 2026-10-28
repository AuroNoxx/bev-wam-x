"""Train/validation-only navigation Flow pilot over audited frozen observations."""
import argparse
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
from mini_wa_jepa.provenance import provenance, write_json
from mini_wa_jepa.visualization import RerunLogger
from models.navigation_flow import NavigationFlow
from planning.navigation_reference import navigation_reference
from scripts.train_navigation_pilot import CachedNavigation, visualize
from scripts.collect_integrated_navigation import digest
from evaluation.memory import memory_snapshot


class FlowData(Dataset):
    def __init__(self, cache, output):
        self.base = CachedNavigation(cache)
        self.manifest = self.base.manifest; self.arrays = self.base.arrays
        refs = []
        for start in range(0, len(self.base), 128):
            items = [self.base[i] for i in range(start, min(start+128, len(self.base)))]
            values = [np.stack([x[k] for x in items]) for k in ('ego', 'route_points', 'route_valid', 'vehicle')]
            refs.append(navigation_reference(*values))
        self.reference = np.concatenate(refs)
        np.save(output/'reference.npy', self.reference)
        write_json(output/'source.json', dict(cache=str(cache), manifest_sha256=digest(cache/'manifest.json'),
                   reference_sha256=digest(output/'reference.npy'), samples=len(self.base)))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return dict(**self.base[index], reference=self.reference[index].copy())


def inputs(batch):
    return [batch[k].to('cuda') for k in ('history', 'ego', 'route_points', 'route_valid', 'vehicle', 'reference')]


def permission_inputs(model, batch):
    return dict(maneuver_allowed=batch['maneuver_allowed'].to('cuda')) if model.arch.get('permission_dim') else {}


@torch.no_grad()
def validation_loss(model, loader, noise, times):
    model.eval(); values = []; offset = 0
    for batch in loader:
        n = len(batch['history'])
        loss = model.loss(*inputs(batch), batch['trajectory'].to('cuda'),
                          noise[offset:offset+n].to('cuda'), times[offset:offset+n].to('cuda'),
                          **permission_inputs(model, batch))
        values.append(float(loss)*n); offset += n
    return sum(values)/offset


@torch.no_grad()
def nominal_predictions(model, loader, steps=10, solver='euler'):
    model.eval(); predictions = []; candidates = []
    for batch in loader:
        out = model.sample(*inputs(batch), candidates=8, steps=steps, solver=solver, seed=20261025,
                           **permission_inputs(model, batch))
        predictions.append(out[:, 0].cpu().numpy()); candidates.append(out.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(candidates)


def candidate_metrics(candidates, target):
    distance = np.linalg.norm(candidates[..., :2]-target[:, None, :, :2], axis=-1)
    pairs = [np.linalg.norm(candidates[:, i, :, :2]-candidates[:, j, :, :2], axis=-1).mean()
             for i in range(candidates.shape[1]) for j in range(i)]
    return dict(nominal_ade_m=float(distance[:, 0].mean()), nominal_fde_m=float(distance[:, 0, -1].mean()),
                oracle_min_ade_m=float(distance.mean(-1).min(1).mean()),
                pairwise_diversity_m=float(np.mean(pairs)))


def run(args):
    if not torch.version.hip or torch.version.cuda:
        raise RuntimeError('Native ROCm required')
    torch.set_num_threads(4); args.output.mkdir(parents=True, exist_ok=False)
    parent = json.loads((args.cache/'report.json').read_text())
    if not parent['completed'] or not parent['control_verified'] or parent['config'].get('encoder_kind', 'joint') != 'joint':
        raise ValueError('Only verified Joint navigation observation caches allowed')
    config = dict(cache=str(args.cache), seeds=[20261023, 20261024], epochs=25, batch=128,
                  learning_rate=3e-4, weight_decay=1e-4, steps=10, solver='euler', temperature=.85,
                  candidates=8, prior='identical_declared_map_reference_train_and_inference',
                  residual_floor=[.05, .05, .1, .02], selection='minimum_fixed_validation_flow_loss')
    report = dict(**provenance(config), completed=False, stage=7, development_only=True,
                  test_access=False, control_verified=True, stage7_acceptance=False, models=[],
                  source_protocol_sha256=digest('docs/STAGE7_INTEGRATED_PROTOCOL.md'),
                  input_cache_report_sha256=digest(args.cache/'report.json'))
    write_json(args.output/'report.json', report)
    with tarfile.open(args.output/'source_snapshot.tar.gz', 'w:gz') as archive:
        for p in report['source_files']:
            archive.add(p, arcname=p)
        archive.add('docs/STAGE7_INTEGRATED_PROTOCOL.md')
    started = time.monotonic()
    try:
        for split in ('train', 'validation'):
            (args.output/split).mkdir()
        train = FlowData(args.cache/'train', args.output/'train')
        val = FlowData(args.cache/'validation', args.output/'validation')
        permission_dim = 2 if 'maneuver_allowed' in train.arrays else 0
        if ('maneuver_allowed' in val.arrays) != bool(permission_dim):
            raise ValueError('Training/validation static permission profiles differ')
        # Only training residuals determine the output scales.
        target_train = train.arrays['trajectory'][train.arrays['anchors']]
        residual_std = np.maximum((target_train-train.reference).std(0), np.array(config['residual_floor']))
        np.save(args.output/'train_residual_std.npy', residual_std)
        validation = DataLoader(val, batch_size=128, shuffle=False)
        fixed = torch.Generator().manual_seed(20261025)
        val_noise = torch.randn(len(val), 10, 4, generator=fixed); val_times = torch.rand(len(val), generator=fixed)
        np.savez_compressed(args.output/'fixed_validation.npz', noise=val_noise.numpy(), time=val_times.numpy())
        with sdpa_kernel(SDPBackend.MATH):
            for seed in config['seeds']:
                torch.manual_seed(seed); np.random.seed(seed)
                model = NavigationFlow(residual_std, permission_dim=permission_dim).to('cuda')
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
                loader = DataLoader(train, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed))
                folder = args.output/f'flow_{seed}'; folder.mkdir(); epochs = []; best = float('inf')
                writer = SummaryWriter(str(folder/'tensorboard'))
                try:
                    for epoch in range(1, 26):
                        torch.cuda.reset_peak_memory_stats()
                        model.train(); losses = []; seen = 0
                        for batch in loader:
                            loss = model.loss(*inputs(batch), batch['trajectory'].to('cuda'), **permission_inputs(model, batch))
                            if not torch.isfinite(loss):
                                raise FloatingPointError('Nonfinite Flow loss')
                            optimizer.zero_grad(set_to_none=True); loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
                            n = len(batch['history']); seen += n; losses.append(float(loss.detach())*n)
                        val_loss = validation_loss(model, validation, val_noise, val_times)
                        row = dict(epoch=epoch, train_loss=sum(losses)/seen, validation_flow_loss=val_loss,
                                   memory=memory_snapshot(), peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
                        writer.add_scalar('train/flow_loss', row['train_loss'], epoch)
                        writer.add_scalar('validation/flow_loss', val_loss, epoch)
                        if val_loss < best:
                            best = val_loss
                            torch.save(dict(model=model.state_dict(), arch=model.arch, epoch=epoch, seed=seed,
                                residual_std=residual_std, schema_version=3, cache_report_sha256=report['input_cache_report_sha256'],
                                encoder_checkpoint=parent['config']['checkpoint'],
                                encoder_sha256=digest(parent['config']['checkpoint']),
                                goal_sha256=report['goal_sha256'], source_tree_sha256=report['source_tree_sha256']), folder/'best.pt')
                        if epoch % 5 == 0:
                            # Same four real validation frames/targets as MSE diagnostics.
                            chosen = np.linspace(0, len(val)-1, min(4, len(val)), dtype=int)
                            batch = next(iter(DataLoader(torch.utils.data.Subset(val, chosen), batch_size=4)))
                            plan = model.sample(*inputs(batch), candidates=1, steps=10, **permission_inputs(model, batch))
                            plans = np.full((len(val), 10, 4), np.nan, np.float32)
                            plans[chosen] = plan[:, 0].cpu().numpy()
                            targets = val.arrays['trajectory'][val.arrays['anchors']]
                            visualize(folder, val, plans, targets, epoch)
                        epochs.append(row); write_json(folder/'epochs.json', epochs)
                        print(json.dumps(dict(seed=seed, **row)), flush=True)
                    selected = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
                    model.load_state_dict(selected['model']); model.eval()
                    nominal, candidates = nominal_predictions(model, validation)
                    targets = val.arrays['trajectory'][val.arrays['anchors']]
                    metrics = candidate_metrics(candidates, targets)
                    np.savez_compressed(folder/'validation.npz', candidates=candidates, target=targets,
                                        episode_ids=val.arrays['episode_ids'], episode_steps=val.arrays['episode_steps'])
                    result = dict(seed=seed, selected_epoch=selected['epoch'], validation=metrics,
                                  checkpoint_sha256=digest(folder/'best.pt'),
                                  parameters=sum(p.numel() for p in model.parameters()), completed=True)
                    write_json(folder/'report.json', result); report['models'].append(result)
                    write_json(args.output/'report.json', report)
                finally:
                    writer.close()
        report['completed'] = True
    except BaseException:
        report['error'] = traceback.format_exc(); raise
    finally:
        report['elapsed_s'] = time.monotonic()-started; write_json(args.output/'report.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
