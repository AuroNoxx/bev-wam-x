"""Prepare real expert labels and camera records for the newly mined route pairs."""
import hashlib
import json
import tarfile
from pathlib import Path
import numpy as np
from .mine_launch_pairs import DATASET, SOURCE, PAIR, load
from .build_real_manifest import read_rows, actor, ROOT
from .real_data_geometry import labels


def bench():
    records = []
    for p in load(DATASET,'mined.json')['pairs']:
        route = p['route']
        rows = read_rows(route,'bench2drive')
        additions = {}
        for branch in ('hold','launch'):
            row=p[branch]
            i=row['index']
            a=actor(rows,i,route,'bench2drive')
            folder=DATASET/'bench2drive'/route/branch
            folder.mkdir(parents=True,exist_ok=True)
            for view,paths in a['images'].items():
                for j,s in enumerate(paths):
                    if (ROOT/s).is_file():continue
                    frame=Path(s)
                    key=f'{route}/camera/{frame.parent.name}/{frame.name}'
                    dest=DATASET/'rgb'/route/frame.parent.name/frame.name
                    additions[key]=dest
                    paths[j]=str(dest.relative_to(ROOT))
            (folder/'actor.json').write_text(json.dumps(a))
            y=labels(rows,i,'bench2drive')
            np.savez_compressed(folder/'labels.npz',trajectory=y['trajectory'],controls=y['controls'])
            cams=[rows[t]['sensors']['CAM_FRONT'] for t in (i-10,i-5,i)]
            records.append(dict(**row,route=route,branch=branch,source='bench2drive',
                                actor_path=str((folder/'actor.json').relative_to(ROOT)),
                                label_path=str((folder/'labels.npz').relative_to(ROOT)),paths=a['images']['<FRONT VIEW>'][-3:],
                                cameras=cams,ego=a['world_from_ego']))
        if additions:
            archive=ROOT/'data/public/bench2drive/archives'/f'{route}.tar.gz'
            with tarfile.open(archive,'r|gz') as tar:
                for item in tar:
                    key=item.name.removeprefix('./')
                    if key in additions and item.isfile():
                        target=additions.pop(key)
                        target.parent.mkdir(parents=True,exist_ok=True)
                        target.write_bytes(tar.extractfile(item).read())
                        if not additions:break
            assert not additions
    (DATASET/'bench.json').write_text(json.dumps(records))
    print('Bench2Drive windows',len(records),flush=True)


def manifest():
    records=load(DATASET,'carla.json') + load(DATASET,'bench.json')
    if (DATASET/'carla-fast.json').exists():records.extend(load(DATASET,'carla-fast.json'))
    holdout={sorted({r['route'] for r in records if r['route'].startswith(town+'-')})[-1]
             for town in ('Town01','Town04','Town05') if any(r['route'].startswith(town+'-') for r in records)}
    for r in records:
        r['split']='paired_holdout' if r['route'] in holdout else 'fit'
        a=json.loads((ROOT/r['actor_path']).read_text())
        assert all((ROOT/p).is_file() for paths in a['images'].values() for p in paths)
        assert a['target_speed_kmh']==0 and a['speed_kmh']<=3
    for route in {r['route'] for r in records}:
        pair=[r for r in records if r['route']==route]
        assert len(pair)==2 and {r['branch'] for r in pair}=={'hold','launch'}
        if pair[0]['source']=='carla':
            poses=[json.loads((ROOT/r['actor_path']).with_name('physics.json').read_text())[0] for r in pair]
            for kind in ('ego','lead'):
                assert np.max(np.abs(np.asarray(poses[0][kind])-poses[1][kind]))<.02
    PAIR.mkdir(exist_ok=True)
    (PAIR/'cohort.json').write_text(json.dumps(records,allow_nan=False))
    print('new pairs',len(records)//2,'held out road names',sorted(holdout),flush=True)


if __name__=='__main__':
    import sys
    {'bench':bench,'manifest':manifest}[sys.argv[1]]()
