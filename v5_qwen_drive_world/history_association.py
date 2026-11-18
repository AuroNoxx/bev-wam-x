"""Associate historical detections through scale-invariant features on the vehicle."""
import json
from functools import lru_cache
import cv2
import numpy as np
from .scale_geometry import RUN, displacement
from .scale_profile_inputs import DEST
from .scale_track import image, metrics, iou

OUT = DEST / 'history-association'


def read(folder, name):
    return json.loads((folder / name).read_text())


@lru_cache(maxsize=24)
def descriptors(path, box):
    frame = image(path)
    height, width = frame.shape
    left, top = np.maximum(0, np.floor(box[:2]).astype(int))
    right, bottom = np.minimum([width, height], np.ceil(box[2:]).astype(int))
    if min(right-left, bottom-top) < 12:
        return np.empty((0,2)), None
    crop = frame[top:bottom, left:right]
    points, desc = cv2.SIFT_create(nfeatures=400).detectAndCompute(crop, None)
    if desc is None:
        return np.empty((0,2)), None
    xy = np.float32([k.pt for k in points])
    # Discard boundary pixels so static background has less influence on scale.
    inset = .08 * np.array([right-left, bottom-top])
    keep = ((xy >= inset) & (xy < [right-left, bottom-top]-inset)).all(axis=1)
    return xy[keep] + [left, top], desc[keep]


def match_pair(path, anchor, earlier_path, candidates):
    src = np.asarray(anchor)
    center = (src[:2] + src[2:]) / 2
    size = src[2:] - src[:2]
    xy, desc = descriptors(path, tuple(anchor))
    if desc is None or len(desc) < 6:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    best = None
    for candidate in candidates:
        b = np.asarray(candidate['box'])
        other_size = b[2:] - b[:2]
        ratio = other_size[0] / size[0]
        offset = abs((b[:2]+b[2:])/2 - center) / np.maximum(size, other_size)
        if not .25 <= ratio <= 4 or max(offset) > .85:
            continue
        prev, prev_desc = descriptors(earlier_path, tuple(b))
        if prev_desc is None or len(prev_desc) < 6:
            continue
        pairs = matcher.knnMatch(desc, prev_desc, k=2)
        reverse = matcher.match(prev_desc, desc)
        mutual = [a for a, alt in pairs if a.distance < .8 * alt.distance
                  and reverse[a.trainIdx].trainIdx == a.queryIdx]
        if len(mutual) < 6:
            continue
        a = np.float32([xy[m.queryIdx] for m in mutual])
        z = np.float32([prev[m.trainIdx] for m in mutual])
        transform, mask = cv2.estimateAffinePartial2D(a, z, method=cv2.RANSAC,
                                                       ransacReprojThreshold=2.5)
        if transform is None:
            continue
        keep = mask.ravel().astype(bool)
        count = int(keep.sum())
        scale = np.linalg.norm(transform[:,0])
        if count < 6 or not .65 <= scale / ratio <= 1.55:
            continue
        # Multiple keypoint orientations at one corner cannot establish scale.
        spread = np.ptp(a[keep], axis=0) / size
        if spread[0] < .18 or spread[1] < .08:
            continue
        rank = count * min(1., spread[0]) * min(1., spread[1])
        if best is None or rank > best[0]:
            best = (rank, candidate['box'], transform, count)
    return best


def association(r, anchor, historical, width, matcher=match_pair):
    paths = r['paths']
    box = anchor
    combined = np.eye(3)
    evidence = []
    for index in (2, 1):
        match = matcher(paths[index], box, paths[index-1], historical[paths[index-1]])
        if match is None:
            return None
        _, box, affine, count = match
        combined = np.vstack([affine, [0, 0, 1]]) @ combined
        evidence.append(dict(matches=count, box=box))
    a = np.asarray(anchor)
    center = (a[:2] + a[2:]) / 2
    transformed = (combined @ [*center, 1])[:2]
    half = (a[2:] - a[:2]) * np.linalg.norm(combined[:2,0]) / 2
    previous = np.r_[transformed-half, transformed+half]
    depth = width * r['cameras'][-1]['intrinsic'][0][0] / (a[2]-a[0])
    return dict(displacement=displacement(previous, anchor, depth, r), depth=depth, evidence=evidence)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('fit','score'))
    parser.add_argument('--seeded', action='store_true')
    args = parser.parse_args()
    phase = args.phase
    output = OUT / 'seeded' if args.seeded else OUT
    output.mkdir(exist_ok=True)
    matcher = match_pair
    if args.seeded:
        from .box_seeded_flow import match_pair as matcher
    cv2.setNumThreads(3)
    cv2.setRNGSeed(9147)
    geom = read(RUN, 'geometry.json')['splits']
    old = read(RUN, 'scale-results.json')
    current = read(DEST, 'rgb-boxes.json')['predictions']
    historical = read(OUT, 'historical-boxes.json')['predictions']
    sets = ('fit',) if phase == 'fit' else ('acceptance','regression','internal')
    results = {}
    for name in sets:
        rows = []
        for r in geom[name]:
            anchor = current[r['paths'][-1]]
            measured = association(r, anchor['box'], historical, old['width_m'], matcher) if anchor else None
            identity_match = None
            if measured:
                identity_match = [iou(e['box'], lab['box']) for e,lab in
                                  zip(measured['evidence'], reversed(r['labels'][:-1]), strict=True)]
            rows.append(dict(route=r['route'], index=r['index'], branch=r['branch'], target=r['target'],
                             rgb=measured['displacement'] if measured else None, historical_iou=identity_match))
        results[name] = dict(metrics=metrics(rows,'rgb'), rows=rows)
        print(name, results[name]['metrics'], flush=True)
    (output / f'{phase}-scores.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    if phase == 'fit':
        return
    score = results['acceptance']['metrics']
    if min(score['precision'], score['recall']) < .75 or score['mae'] >= score['zero_same_coverage']:
        print('Association failed independent geometry criteria', flush=True)
        return
    profile = read(DEST,'geometry.json')['splits']
    old_profile = read(OUT if args.seeded else DEST, 'observations.json')
    for split, records in profile.items():
        recovered = 0
        for r, row in zip(records, old_profile['splits'][split], strict=True):
            assert r['actor_path'] == row['actor_path']
            anchor = current[r['paths'][-1]]
            if row['displacement'] is not None or not anchor:
                continue
            result = association(r, anchor['box'], historical, old['width_m'], matcher)
            if result:
                row.update(result, association_method='box_seeded_flow' if args.seeded else 'sift')
                recovered += 1
        print(split, 'recovered', recovered, flush=True)
    old_profile['method'] = 'Retain previous accepted tracks; recover missing observations by historical detection association.'
    (output / 'observations.json').write_text(json.dumps(old_profile, allow_nan=False))


if __name__ == '__main__':
    main()
