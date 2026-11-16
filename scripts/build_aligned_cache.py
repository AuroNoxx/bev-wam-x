"""Build version 3 alongside v2, on exactly the frozen train/validation windows.

Only OCC file pairing changes: anno i uses OCC i-1 and pose i.
No test samples, downloads, model training, or old-cache writes.
"""
from pathlib import Path
import argparse, gzip, json, os, time
import numpy as np
from dataset_v1 import ROOT, CFG, compact, warp, digest

OLD = ROOT / 'data/occ_v1'
OUT = ROOT / 'data/occ_v3'
DIAG = ROOT / 'artifacts/diagnostics/alignment-v3'
KEEP = ['state', 'trajectory', 'frame', 'time_s', 'world2ego', 'source_sha']


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending.json')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    json.loads(tmp.read_text(encoding='utf-8'))
    os.replace(tmp, path)


def frame_ids(frame):
    history = np.array(CFG['history_offsets'], np.int32) + frame
    future = np.array(CFG['future_occ_offsets'], np.int32) + frame
    return history, future


def check_sample(sample, old, old_sha):
    i = int(old['frame'])
    h, f = frame_ids(i)
    assert int(sample['preprocess_version']) == 3
    assert int(sample['occ_file_offset']) == -1
    assert str(sample['parent_cache_sha256']) == old_sha
    assert str(sample['config_sha256']) == digest(ROOT / 'configs/train_v1.json')
    for k in KEEP:
        assert np.array_equal(sample[k], old[k]), k
    for prefix, ids in [('history', h), ('future', f)]:
        assert np.array_equal(sample[prefix + '_pose_ids'], ids)
        assert np.array_equal(sample[prefix + '_occ_ids'], ids - 1)
        assert sample[prefix].shape == (len(ids), *CFG['grid']['shape_zyx'])
        assert sample[prefix].dtype == np.uint8
        assert np.isin(sample[prefix], [0, 1, 2, 3, 4, 5, 6, 7, 255]).all()
    assert (sample['history_pose_ids'] <= i).all()
    assert (sample['future_pose_ids'] > i).all()


def build_route(row, sample_names):
    clip = ROOT / 'data/public/bench2drive/clips' / row['route']
    needed = sorted({int(Path(n).stem) + h for n in sample_names
                     for h in CFG['history_offsets'] + CFG['future_occ_offsets']})
    poses, grids, sources = {}, {}, []
    for i in needed:
        anno = clip / 'anno' / f'{i:05d}.json.gz'
        occ = clip / 'occ' / f'{i-1:05d}.npz'
        with gzip.open(anno, 'rt') as stream:
            a = json.load(stream)
        poses[i] = np.asarray(next(b for b in a['bounding_boxes']
                                  if b['class'] == 'ego_vehicle')['world2ego'], np.float64)
        with np.load(occ, allow_pickle=False) as raw:
            grids[i] = compact(raw['occ'], raw['seg'])
        sources.append({'annotation_id': i, 'pose_id': i, 'occ_file_id': i-1,
                        'anno_sha256': digest(anno), 'occ_sha256': digest(occ)})
    records = []
    for name in sample_names:
        source = OLD / name
        old_sha = digest(source)
        with np.load(source, allow_pickle=False) as a:
            old = {k: a[k].copy() for k in a.files}
        i = int(old['frame'])
        assert i == int(Path(name).stem)
        assert np.array_equal(poses[i], old['world2ego'])
        current = np.linalg.inv(poses[i])
        h, f = frame_ids(i)
        sample = {k: old[k] for k in KEEP}
        sample.update(history=np.stack([warp(grids[j], poses[j], current) for j in h]),
                      future=np.stack([warp(grids[j], poses[j], current) for j in f]),
                      preprocess_version=np.int32(3), occ_file_offset=np.int32(-1),
                      history_pose_ids=h, future_pose_ids=f,
                      history_occ_ids=h-1, future_occ_ids=f-1,
                      parent_cache_sha256=np.array(old_sha),
                      config_sha256=np.array(digest(ROOT / 'configs/train_v1.json')))
        path = OUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.pending.npz')
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, **sample)
            stream.flush()
            os.fsync(stream.fileno())
        with np.load(tmp, allow_pickle=False) as check:
            check_sample(check, old, old_sha)
        os.replace(tmp, path)
        assert digest(source) == old_sha
        records.append({'sample': name, 'old_sha256': old_sha, 'new_sha256': digest(path),
                        'history_changed_fraction': float((sample['history'] != old['history']).mean()),
                        'future_changed_fraction': float((sample['future'] != old['future']).mean())})
    atomic_json(DIAG / 'routes' / (row['route'] + '.json'),
                {'route': row['route'], 'split': row['split'], 'sources': sources, 'samples': records})
    print(json.dumps({'route': row['route'], 'windows': len(records)}), flush=True)
    return {**row, 'samples': sample_names, 'windows': len(sample_names),
            'occ_file_offset': -1, 'paired_frames': None,
            'pairing_note': 'Window-preserving rebuild; v2 unpaired-frame statistics not reused',
            'unpaired_annotation_ids': None, 'unpaired_occ_ids': None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='One old window per train/validation route')
    args = parser.parse_args()
    started = time.monotonic()
    parent = OLD / 'manifest.json'
    parent_sha = digest(parent)
    old = json.loads(parent.read_text())
    assert old['preprocess_version'] == 2 and old['revision'] == 'd84ae18cf20a88b9220df1bc15c073d8824b7ac1'
    assert old['config_sha256'] == digest(ROOT / 'configs/train_v1.json')
    # Both modes write correct v3 samples; smoke never publishes the formal manifest.
    rows = []
    for row in old['routes']:
        if row['split'] not in ['train', 'validation']:
            continue
        names = row['samples'][:1] if args.smoke else row['samples']
        rows.append(build_route(row, names))
    splits = {s: [p for r in rows if r['split'] == s for p in r['samples']]
              for s in ['train', 'validation', 'test']}
    if not args.smoke:
        for split in ['train', 'validation']:
            assert splits[split] == old['splits'][split]
    report = {**old, 'preprocess_version': 3, 'cache_root': 'data/occ_v3',
              'parent_manifest_sha256': parent_sha, 'parent_manifest': 'data/occ_v1/manifest.json',
              'occ_file_offset': -1, 'pairing': 'OCC filename = annotation/pose frame - 1',
              'evidence': 'docs/FRAME_ALIGNMENT_20261115.md', 'pilot': args.smoke,
              'routes': rows, 'splits': splits, 'counts': {s: len(v) for s, v in splits.items()},
              'reserved_test': {'status': 'unread_unmodified_not_prepared',
                                'parent_count': old['counts']['test']},
              'only_change': 'OCC source pairing; fixed windows/grid/state/navigation/trajectory',
              'preprocess_seconds': time.monotonic() - started}
    assert digest(parent) == parent_sha
    atomic_json(OUT / ('smoke.json' if args.smoke else 'manifest.json'), report)
    print(json.dumps({'pass': True, 'counts': report['counts'], 'seconds': report['preprocess_seconds'],
                      'smoke': args.smoke}), flush=True)


if __name__ == '__main__':
    main()
