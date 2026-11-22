"""Publish only completed, replayable episodes to an append-only clean manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from .bulk import sync_manifest
from .sd_policy import route_vertices
from .sd_replay import policy_digest


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def public_routes(plan):
    fields = ('name','xy','laps','events','georef','sd_accuracy','source')
    return {town: {'missions': [{k: spec[k] for k in fields if k in spec}
                               for spec in section['missions']]}
            for town, section in plan.items()}


def prepare_episode(folder, meta, spec):
    public = dict(xy=route_vertices(spec).tolist(), georef=spec['georef'], events=spec['events'],
                  sd_accuracy=spec['sd_accuracy'], drift_phase=meta['drift_phase'],
                  retained_frames=len(meta['frames']))
    if (public['xy'] != meta['plan_world_xy'] or
            any(public[k] != meta[k] for k in ('georef','events','sd_accuracy'))):
        raise ValueError(f'Published SD route differs from the observation source for {folder.name}')
    nav_path = folder / 'navigation.json'
    if nav_path.exists():
        existing = json.loads(nav_path.read_text())
        if any(existing.get(k) != v for k,v in public.items()):
            raise ValueError(f'Navigation metadata changed for {folder.name}')
        return
    with np.load(folder / 'drive-log.npz') as pack:
        log = {k: pack[k] for k in pack}
    poses = log.pop('expert_pose', None)
    if poses is not None:
        n = public['retained_frames']
        tmp = folder / 'teacher-tail.tmp.npz'
        np.savez_compressed(tmp, frame=log['frame'][n:], world_from_ego=poses[n:])
        tmp.replace(folder / 'teacher-tail.npz')
        tmp = folder / 'drive-log.tmp.npz'
        np.savez_compressed(tmp, **log)
        tmp.replace(folder / 'drive-log.npz')
    digest = policy_digest(public, log['fix'], float(log['dt']))
    observed = hashlib.sha256()
    for row in meta['frames']:
        with np.load(folder / f'{row["frame"]:07d}' / 'input.npz') as inputs:
            observed.update(inputs['sd_nav'].astype('<f4', copy=False).tobytes())
    if digest != observed.hexdigest():
        raise ValueError(f'Isolated policy replay differs from saved navigation for {folder.name}')
    public['sd_nav_sha256'] = digest
    atomic_json(nav_path, public)


def release(dest, plan):
    atlas = public_routes(plan)
    ready = []
    for folder in sorted((dest/'episodes').glob('ep_*')):
        if not folder.name[3:].isdigit():
            continue
        meta = json.loads((folder/'episode.json').read_text())
        if 'verified_change_frames' not in meta:
            raise ValueError(f'Uncertified physical maneuver labels for {folder.name}')
        if any(meta['qc'][key] for key in ('ego_opposite_frames','atlas_opposite_frames')):
            print(f'Pending direction check for {folder.name}', flush=True)
            continue
        town = meta['frames'][0]['map']
        spec = next(m for m in atlas[town]['missions'] if m['name'] == meta['mission'])
        prepare_episode(folder,meta,spec)
        ready.append(meta)
    rows = sync_manifest(dest,ready)
    atomic_json(dest/'sd-navigation.json',atlas)
    print(f'Published {len(rows)} samples in {len(ready)} episodes',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('dest',type=Path)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    release(args.dest,json.loads(args.plan.read_text()))
