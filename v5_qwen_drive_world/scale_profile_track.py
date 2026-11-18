"""Compute lead observations using only past/current RGB, rig and ego poses."""
import hashlib
import json

import cv2
from .scale_geometry import RUN, displacement
from .scale_profile_inputs import DEST
from .scale_track import track


def main():
    cv2.setNumThreads(3)
    cv2.setRNGSeed(9147)
    geometry_file = DEST / 'geometry.json'
    splits = json.loads(geometry_file.read_text())['splits']
    detector = json.loads((DEST / 'rgb-boxes.json').read_text())
    calibration = json.loads((RUN / 'scale-results.json').read_text())
    width = calibration['width_m']
    output = {}
    for name, records in splits.items():
        rows = []
        for r in records:
            b = detector['predictions'][r['paths'][-1]]
            row = dict(actor_path=r['actor_path'], depth=None, displacement=None)
            if b is not None:
                box = b['box']
                prev = track(r['paths'], box)
                if prev is not None:
                    depth = width * r['cameras'][-1]['intrinsic'][0][0] / (box[2] - box[0])
                    row.update(depth=depth, displacement=displacement(prev[0], box, depth, r))
            rows.append(row)
        output[name] = rows
        print(name, sum(r['depth'] is not None for r in rows), '/', len(rows), flush=True)
    (DEST / 'observations.json').write_text(json.dumps(dict(
        geometry_sha256=hashlib.sha256(geometry_file.read_bytes()).hexdigest(),
        detector_sha256=detector['sha256'], width_m=width, splits=output), allow_nan=False))


if __name__ == '__main__':
    main()
