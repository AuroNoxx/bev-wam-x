"""Independent RGB-box diagnostic. Fixed COCO weights; no downstream policy changes."""
import hashlib
import json
from PIL import Image
from .runtime import configure
configure()
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms.functional import pil_to_tensor
from .scale_geometry import RUN,ROOT


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    weights=ROOT/'models/v5-qwen-drive-world/scale-replay/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth'
    fingerprint=hashlib.sha256(weights.read_bytes()).hexdigest()
    assert fingerprint.startswith('fb6a3cc7')
    net=fasterrcnn_mobilenet_v3_large_fpn(weights=None,weights_backbone=None,min_size=900,max_size=1600)
    net.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    net.eval()
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',action='store_true')
    folder=RUN/'motion-profile' if parser.parse_args().profile else RUN
    data=json.loads((folder/'geometry.json').read_text())
    predictions={}
    for cache in (RUN/'rgb-boxes.json',folder/'rgb-boxes.json'):
        if cache.exists():
            saved=json.loads(cache.read_text())
            assert saved['sha256']==fingerprint
            predictions.update(saved['predictions'])
    def save():
        output=folder/'rgb-boxes.json'
        temp=output.with_suffix('.pending')
        temp.write_text(json.dumps(dict(sha256=fingerprint,predictions=predictions)))
        temp.replace(output)
    for rows in data['splits'].values():
        for r in rows:
            path=r['paths'][-1]
            if path in predictions:continue
            with Image.open(ROOT/path) as im:
                w,h=im.size
                net.transform.min_size=(min(w,h),)
                net.transform.max_size=max(w,h)
                x=pil_to_tensor(im.convert('RGB')).float()/255
            det=net([x])[0]
            boxes=[]
            for xy,label,score in zip(det['boxes'].tolist(),det['labels'].tolist(),det['scores'].tolist()):
                if label in (3,6,8) and score>=.5 and w/3<=(xy[0]+xy[2])/2<=2*w/3:
                    boxes.append(dict(box=xy,score=score))
            # Prefer the forward vehicle straddling the optical axis, then the
            # lowest image bottom among eligible candidates.
            predictions[path]=max(boxes,key=lambda b:(b['box'][0]<=w/2<=b['box'][2],b['box'][3]),default=None)
            if len(predictions)%25==0:
                save()
                print(len(predictions),flush=True)
    save()


if __name__=='__main__':main()
