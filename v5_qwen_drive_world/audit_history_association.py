"""Label-side audit of newly available Bench2Drive tracks and matched planning outcomes."""
import gzip
import json
from pathlib import Path
import numpy as np
from .history_association import DEST, OUT, read
from .scale_geometry import DATA, ROOT, project
from .scale_track import iou


def lead_truth(actor, anchors):
    route, index = actor['route'], actor['index']
    folder = DATA.parent / 'public/bench2drive/clips' / route / 'anno'
    frames = []
    for offset in (-10, -5, 0):
        with gzip.open(folder / f'{index+offset:05d}.json.gz', 'rt') as f:
            frames.append(json.load(f))
    inv = np.linalg.inv(actor['world_from_ego'])
    leads = []
    for obj in frames[-1]['bounding_boxes']:
        if obj['class'] == 'vehicle':
            x, y = (inv @ np.r_[obj['location'], 1])[:2]
            if 0 < x < 35 and abs(y) < 2:
                leads.append((x, obj['id']))
    if not leads:
        return dict(status='no_lane_lead')
    lead = min(leads)[1]
    matched = [next((o for o in f['bounding_boxes'] if o.get('id') == lead and o['class']=='vehicle'), None)
               for f in frames]
    if any(o is None for o in matched):
        return dict(status='missing_identity')
    delta = np.asarray(matched[-1]['location']) - matched[0]['location']
    true_move = float((inv @ np.r_[delta, 0])[0])
    overlap = []
    for f, obj, anchor in zip(frames, matched, anchors, strict=True):
        cam = f['sensors']['CAM_FRONT']
        vertices = np.array(obj['world_cord'])
        # Clip the projected label to the RGB field of view, as the detector does.
        if np.min((np.c_[vertices, np.ones(8)] @ np.asarray(cam['world2cam']).T)[:,0]) <= .1:
            overlap.append(None)
            continue
        box = project(vertices, obj['location'], cam)['box']
        visible = np.clip(box, 0, [cam['image_size_x'], cam['image_size_y']]*2)
        overlap.append(iou(visible, anchor))
    return dict(status='scored', true_move=true_move, label_box_iou=overlap)


def summarize(rows):
    scored = [r for r in rows if r['status']=='scored']
    identified = [r for r in scored if all(v is not None and v >= .5 for v in r['label_box_iou'])]
    return dict(new_bench_tracks=len(rows), with_lane_lead=len(scored), all_three_iou_ge_half=len(identified),
                mae=float(np.mean([abs(r['true_move']-r['displacement']) for r in scored])) if scored else None,
                zero_mae=float(np.mean([abs(r['true_move']) for r in scored])) if scored else None)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeded', action='store_true')
    seeded = parser.parse_args().seeded
    folder = OUT / 'seeded' if seeded else OUT
    before = read(OUT if seeded else DEST, 'observations.json')['splits']
    updated = read(folder, 'observations.json')['splits']
    current = read(DEST, 'rgb-boxes.json')['predictions']
    audit = {}
    for split in before:
        rows = []
        diff = 0
        for a, b in zip(before[split], updated[split], strict=True):
            assert a['actor_path'] == b['actor_path']
            if a['displacement'] is not None:
                assert a == b
            elif b['displacement'] is not None:
                diff += 1
                actor = json.loads((ROOT / a['actor_path']).read_text())
                if actor['source'] != 'bench2drive':
                    continue
                paths = actor['images']['<FRONT VIEW>']
                anchors = [e['box'] for e in reversed(b['evidence'])] + [current[paths[-1]]['box']]
                truth = lead_truth(actor, anchors)
                rows.append(dict(route=actor['route'], index=actor['index'], displacement=b['displacement'], **truth))
        audit[split] = dict(recovered=diff, **summarize(rows), rows=rows)
        print(split, {k:v for k,v in audit[split].items() if k!='rows'}, flush=True)
    (folder / 'lead-identity-audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
