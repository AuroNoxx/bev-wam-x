"""Assemble exposure-specific front camera poses and actor inputs for matched planning replay."""
import gzip
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from .scale_geometry import DATA, ROOT, RUN

DEST = RUN / 'motion-profile'


@lru_cache(maxsize=12)
def highway_poses(route):
    poses = {}
    with (DATA / 'highway' / route / 'frames.jsonl').open() as stream:
        for line in stream:
            frame = json.loads(line)
            poses[frame['index']] = frame['current_world_from_ego']
    return poses


def actor_window(record):
    actor = json.loads((ROOT / record['actor_path']).read_text())
    paths = actor['images']['<FRONT VIEW>'][-3:]
    indices = [int(Path(p).stem) for p in paths]
    assert indices == [record['index'] + offset for offset in (-10, -5, 0)]
    cameras = []
    for i in indices:
        if record['source'] == 'bench2drive':
            folder = DATA.parent / 'public/bench2drive/clips' / record['route'] / 'anno'
            with gzip.open(folder / f'{i:05d}.json.gz', 'rt') as stream:
                sensor = json.load(stream)['sensors']['CAM_FRONT']
            cameras.append({k: sensor[k] for k in ('intrinsic', 'world2cam')})
        else:
            sensor = json.loads((ROOT / actor['camera_calibration']).read_text())['CAM_FRONT']
            pose = highway_poses(record['route'])[i]
            cameras.append(dict(intrinsic=sensor['intrinsics'],
                                world2cam=np.linalg.inv(np.asarray(pose) @ sensor['ego_from_sensor']).tolist()))
    return dict(actor_path=record['actor_path'], paths=paths, cameras=cameras, ego=actor['world_from_ego'])


def main():
    from .geometry_repair_features import split_records
    from .real_dataset import RealData
    DEST.mkdir(exist_ok=True)
    sets = {k: [actor_window(r) for r in rows] for k, rows in split_records(RealData('subset2000')).items()}
    (DEST / 'geometry.json').write_text(json.dumps(dict(splits=sets)))


if __name__ == '__main__':
    main()
