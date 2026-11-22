"""Reproduce the policy navigation input offline without CARLA or any teacher data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .sd_policy import RoadNavigator


def policy_digest(spec,fixes,dt):
    """Only these arguments are visible to this actor-side replay."""
    nav=RoadNavigator(spec['xy'],spec['georef'],spec['drift_phase'],spec['events'],spec['sd_accuracy'])
    digest=hashlib.sha256()
    for i,(lat,lon,_,compass,*rest) in enumerate(fixes[:spec['retained_frames']]):
        digest.update(nav.update(lat,lon,compass,i*dt).astype('<f4',copy=False).tobytes())
    return digest.hexdigest()


def verify(folder):
    meta=json.loads((folder/'episode.json').read_text())
    public_path=folder/'navigation.json'
    public=json.loads(public_path.read_text()) if public_path.is_file() else {
        'xy':meta['plan_world_xy'],'georef':meta['georef'],'drift_phase':meta['drift_phase'],
        'events':meta['events'],'sd_accuracy':meta.get('sd_accuracy')}
    policy=RoadNavigator(public['xy'],public['georef'],public['drift_phase'],public['events'],public['sd_accuracy'])
    digest=hashlib.sha256()
    with np.load(folder/'drive-log.npz') as log:
        frames=log['frame']
        if not np.all(np.diff(frames)==1) or not np.array_equal(frames[1:],log['effective_frame']):
            raise ValueError('Action time alignment failed')
        err=np.max(abs(log['requested'][:-1]-log['applied'][1:]))
        if err>.0011:raise ValueError(f'Applied action mismatch {err}')
        fixes=log['fix']
        commands=log['tactical']
        one_hot=np.eye(3,dtype=np.uint8)
        if np.any((commands<0)|(commands>=3)):raise ValueError('Unknown tactical request')
        if not np.isfinite(fixes).all():raise ValueError('Invalid localization')
        for i,row in enumerate(meta['frames']):
            fid=row['frame']
            if fid!=frames[i] or row['sensor_frames']!=[fid]*7:raise ValueError('Sensor frame mismatch')
            lat,lon,_,compass=fixes[i,:4]
            nav=policy.update(lat,lon,compass,i*float(log['dt']))
            with np.load(folder/f'{fid:07d}'/'input.npz') as pack:
                if set(pack.files)!={'sd_nav','cmd_one_hot'} or not np.isfinite(nav).all():
                    raise ValueError(f'Invalid actor fields at {fid}')
                cmd=pack['cmd_one_hot']
                if cmd.shape!=(3,) or not np.array_equal(cmd,one_hot[commands[i]]):
                    raise ValueError(f'Invalid tactical request at {fid}')
                if not np.array_equal(pack['sd_nav'],nav):raise ValueError(f'Navigation replay mismatch at {fid}')
                digest.update(nav.astype('<f4',copy=False).tobytes())
    if public_path.is_file() and digest.hexdigest()!=public['sd_nav_sha256']:
        raise ValueError(f'Navigation SHA mismatch for {folder}')
    return dict(frames=len(meta['frames']),max_actuation_error=float(err))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('episode',type=Path)
    result=verify(p.parse_args().episode)
    assert 'carla' not in sys.modules
    print('Offline replay',result,flush=True)
