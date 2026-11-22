"""Validated, memory-mapped cache and balanced draw for the frozen 20k dataset."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


CAMERAS = ('front', 'front_left', 'front_right')
COUNTS = {('Town12', 'train'): 10000, ('Town10HD_Opt', 'train'): 8000, ('Town12', 'val'): 2000}
C_COUNTS = {('Town12', 'train'): 22000, ('Town10HD_Opt', 'train'): 16000, ('Town12', 'val'): 2000}
LAYOUT = {
    'rgb': ((3, 3, 224, 384), np.uint8),
    'lidar': ((6, 128, 128), np.float16),
    'nav': ((3, 2), np.float32),
    'cmd': ((3,), np.uint8),
    'lines': ((7, 20, 2), np.float32),
    'valid': ((7, 20), np.bool_),
    'exists': ((7,), np.bool_),
    'raster_bits': ((9, 128, 16), np.uint8),
    'path': ((40, 2), np.float32),
}
C_LAYOUT = {k:v for k,v in LAYOUT.items() if k!='rgb'}
C_LAYOUT.update(front=((3,448,768),np.uint8),sides=((2,3,224,384),np.uint8))


def manifest(root):
    content = (root / 'manifest.jsonl').read_bytes()
    rows = [json.loads(line) for line in content.splitlines()]
    counts = Counter((r['map'], r['split']) for r in rows)
    if counts not in (COUNTS,C_COUNTS) or len({(r['episode'], r['frame']) for r in rows}) != len(rows):
        raise ValueError(f'Unexpected training corpus: {counts}')
    episodes = {}
    for row in rows:
        episodes.setdefault(row['episode'], []).append(row)
    for key, ep in episodes.items():
        if len({(r['map'], r['split']) for r in ep}) != 1 or len(ep) != 2000:
            raise ValueError(f'Mixed or incomplete episode {key}')
        if np.any(np.diff([r['frame'] for r in ep]) != 1):
            raise ValueError(f'Noncontinuous episode {key}')
    return rows, hashlib.sha256(content).hexdigest()


def decode(root, row, native=None):
    frame = root / 'episodes' / row['episode'] / f'{row["frame"]:07d}'
    views = []
    for name in CAMERAS:
        with Image.open(frame / (name + '.png')) as im:
            views.append(np.asarray(im.convert('RGB')).transpose(2, 0, 1))
    with np.load(frame / 'lidar.npz') as lidar, np.load(frame / 'input.npz') as inp, \
            np.load(frame / 'labels.npz') as labels, np.load(frame / 'index.npz') as index:
        if int(index['frame']) != row['frame'] or set(inp.files) != {'sd_nav', 'cmd_one_hot'}:
            raise ValueError(f'Observation contract failed for {frame}')
        maps = np.concatenate((labels['masks'], labels['p_occ']))
        if np.any(maps > 1):
            raise ValueError(f'Invalid binary masks at {frame}')
        if native is None:
            native=views[0].shape==(3,448,768)
        result = dict(lidar=lidar['bev'], nav=inp['sd_nav'], cmd=inp['cmd_one_hot'],
                      lines=labels['map_lines'], valid=labels['point_valid'], exists=labels['exists'],
                      raster_bits=np.packbits(maps, axis=-1), path=labels['path_gt'])
        if not np.array_equal(result['exists'], result['valid'].any(1)) or result['cmd'].sum() != 1:
            raise ValueError(f'Invalid point masks or tactical intent at {frame}')
        if native:
            result.update(front=views[0],sides=np.stack(views[1:]))
        else:
            result['rgb']=np.stack(views)
        for name, (shape, _) in (C_LAYOUT if native else LAYOUT).items():
            if result[name].shape != shape or not np.isfinite(result[name]).all():
                raise ValueError(f'Invalid {name} at {frame}')
    return result


def prepare(root, cache):
    rows, digest = manifest(root)
    native=len(rows)==sum(C_COUNTS.values())
    layout=C_LAYOUT if native else LAYOUT
    if cache.exists():
        meta = json.loads((cache / 'index.json').read_text())
        if meta != {'manifest_sha256': digest, 'rows': len(rows)}:
            raise ValueError(f'Cache does not match the current manifest: {cache}')
        for name, (shape, dtype) in layout.items():
            arr = np.load(cache / f'{name}.npy', mmap_mode='r')
            if arr.shape != (len(rows), *shape) or arr.dtype != dtype:
                raise ValueError(f'Invalid cached array: {name}')
        return
    building = cache.with_name(cache.name + '.building')
    building.mkdir(parents=True)
    arrays = {k: np.lib.format.open_memmap(building / f'{k}.npy', 'w+', dtype=d, shape=(len(rows), *s))
              for k, (s, d) in layout.items()}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for first in range(0, len(rows), 120):
            batch = pool.map(lambda row: decode(root, row, native), rows[first:first + 120])
            for i, sample in enumerate(batch, first):
                for field, array in arrays.items():
                    array[i] = sample[field]
            if (first + 120) % 1200 == 0 or first + 120 >= len(rows):
                print('cache', min(first + 120, len(rows)), '/', len(rows), flush=True)
    for array in arrays.values():
        array.flush()
    del array, arrays
    (building / 'index.json').write_text(json.dumps({'manifest_sha256': digest, 'rows': len(rows)}))
    building.rename(cache)


def bucket(row):
    if row['lane_change'] or row['recovery']:
        return 'rare'
    if row['sharp_curve'] or row['junction']:
        return 'turn'
    return 'cruise'


class Corpus(Dataset):
    def __init__(self, root, cache, split):
        rows, self.digest = manifest(root)
        self.layout=C_LAYOUT if len(rows)==sum(C_COUNTS.values()) else LAYOUT
        self.cache = cache
        self.indices = [i for i, row in enumerate(rows) if row['split'] == split]
        self.rows = [rows[i] for i in self.indices]
        self.arrays = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self.arrays is None:
            self.arrays = {k: np.load(self.cache / f'{k}.npy', mmap_mode='r') for k in self.layout}
        index = self.indices[i]
        return {k: v[index].copy() for k, v in self.arrays.items()}

    def sampler(self, seed):
        groups = [bucket(r) for r in self.rows]
        counts = Counter(groups)
        fractions = {'cruise': .50, 'turn': .38, 'rare': .12}
        if counts.keys() != fractions.keys():
            raise ValueError(f'Missing sampling group: {counts}')
        weights = [fractions[name] / counts[name] for name in groups]
        return WeightedRandomSampler(weights, len(groups), replacement=True,
                                     generator=torch.Generator().manual_seed(seed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('cache', type=Path)
    args = parser.parse_args()
    prepare(args.dataset, args.cache)
