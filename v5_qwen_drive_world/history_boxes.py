"""Collect vehicle candidates at the two earlier RGB exposures for association replay."""
import json
import hashlib
from PIL import Image
from .runtime import configure
configure()
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms.functional import pil_to_tensor
from .scale_geometry import ROOT, RUN
from .scale_profile_inputs import DEST

OUT = DEST / 'history-association'


def read(folder, name):
    return json.loads((folder / name).read_text())


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    weights = ROOT / 'models/v5-qwen-drive-world/scale-replay/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth'
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    boxes = read(DEST, 'rgb-boxes.json')
    assert sha == boxes['sha256'] and sha.startswith('fb6a3cc7')
    # Fixed current anchors, all historical candidates. Replay accepted tracks unchanged.
    current = boxes['predictions']
    geom = read(RUN, 'geometry.json')['splits']
    profile = read(DEST, 'geometry.json')['splits']
    observed = read(DEST, 'observations.json')['splits']
    paths = {}
    for split in ('fit', 'acceptance', 'regression', 'internal'):
        for rec in geom[split]:
            if current[rec['paths'][-1]]:
                paths.update(dict.fromkeys(rec['paths'][:-1]))
    for split, records in profile.items():
        for rec, obs in zip(records, observed[split], strict=True):
            if current[rec['paths'][-1]] and obs['displacement'] is None:
                paths.update(dict.fromkeys(rec['paths'][:-1]))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--paired-launch', action='store_true')
    parser.add_argument('--independent',action='store_true')
    parser.add_argument('--real-road',action='store_true')
    opts=parser.parse_args()
    if opts.real_road:
        from .prepare_real_road_replay import OUT as folder
        paths = dict.fromkeys(p for r in read(folder,'cohort.json') for p in r['paths'])
    elif opts.paired_launch or opts.independent:
        folder = OUT / 'paired-launch'
        if opts.independent:folder=folder/'independent-roads'
        paths = dict.fromkeys(p for r in read(folder,'cohort.json') for p in r['paths'])
    else:
        folder = OUT
    target = folder / 'historical-boxes.json'
    saved = read(folder, target.name) if target.exists() else dict(sha256=sha, predictions={})
    assert saved['sha256'] == sha
    preds = saved['predictions']
    net = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
    net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    net.eval()
    print('unique historical images', len(paths), flush=True)
    for path in paths:
        if path in preds:
            continue
        with Image.open(ROOT / path) as im:
            w, h = im.size
            net.transform.min_size, net.transform.max_size = (min(w,h),), max(w,h)
            tensor = pil_to_tensor(im.convert('RGB')).float() / 255
        result = net([tensor])[0]
        preds[path] = [dict(box=box, label=label, score=score)
                       for box, label, score in zip(result['boxes'].tolist(), result['labels'].tolist(),
                                                    result['scores'].tolist(), strict=True)
                       if label in (3,6,8) and score >= .5]
        if len(preds) % 30 == 0 or len(preds) == len(paths):
            tmp = target.with_suffix('.pending')
            tmp.write_text(json.dumps(saved))
            tmp.replace(target)
            print(len(preds), '/', len(paths), flush=True)


if __name__ == '__main__':
    main()
