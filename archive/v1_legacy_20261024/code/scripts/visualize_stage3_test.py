"""Save frozen-checkpoint latent diagnostics for one fixed midpoint per test map."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from configs.stage3 import Stage3Config
from dataset.latent import LatentPairs
from mini_wa_jepa.provenance import provenance, write_json
from models.latent import LatentModel
from training.latent import visualize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('artifacts/restart_stage3/vicreg_001'))
    args = p.parse_args()
    report = json.loads((args.run/'report.json').read_text())
    if not report.get('completed') or report.get('max_batches') is not None:
        raise ValueError('Requires a completed formal run with a frozen selected checkpoint')
    cfg = Stage3Config(**report['config']); torch.set_num_threads(cfg.cpu_threads)
    if not torch.version.hip or torch.version.cuda is not None:
        raise RuntimeError('ROCm required')
    data = LatentPairs(report['dataset'], 'test')
    manifest = json.loads((data.root/'manifest.json').read_text())
    episodes = [row for row in manifest['episodes'] if row['split'] == 'test']
    selected, context = [], []
    for map_spec in sorted({row['map_spec'] for row in episodes}):
        ids = [row['episode_id'] for row in episodes if row['map_spec'] == map_spec]
        eligible = np.flatnonzero(np.isin(data.episodes[data.anchors], ids))
        sample_id = int(eligible[len(eligible)//2])
        episode = next(row for row in episodes if row['episode_id'] == data[sample_id]['episode'])
        selected.append(sample_id)
        context.append(dict(sample_id=sample_id, anchor=int(data.anchors[sample_id]),
                            map_spec=map_spec, seed=episode['seed'], episode=episode['episode_id']))
    output = args.run/'selected_test_diagnostics'
    output.mkdir(exist_ok=False)
    state = torch.load(report['selected_checkpoint'], map_location='cpu', weights_only=False)
    model = LatentModel(cfg).to('cuda'); model.load_state_dict(state['model'])
    del state
    arrays = np.load(args.run/'test_raw.npz')
    with sdpa_kernel(SDPBackend.MATH):
        paths = visualize(model, data, replace(cfg, visualization_sample_ids=tuple(selected)),
            torch.device('cuda'), output, report['epochs'][report['selected_epoch']-1]['global_step'],
            report['selected_epoch'], report['test'], arrays)
    write_json(output/'manifest.json', dict(**provenance(cfg), selected_checkpoint=report['selected_checkpoint'],
        selection='fixed midpoint by pair index within each test map; no error-based sample selection',
        samples=context, paths=paths, affects_acceptance_metrics=False))


if __name__ == '__main__':
    main()
