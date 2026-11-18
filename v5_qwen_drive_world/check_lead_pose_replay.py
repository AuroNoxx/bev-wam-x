"""Verify paired outputs and test a zero-motion image replay with the native processor."""
import json
from unittest.mock import patch

import numpy as np
from . import lead_motion
from .train_lead_motion import LeadReadout,metrics
from .lead_motion import DEST,RealData,load_base,load_actor,observation,torch


@torch.no_grad()
def main():
    folder=DEST/'full-pose'
    old=json.loads((DEST/'result.json').read_text())
    new=json.loads((folder/'result.json').read_text())
    meta=json.loads((folder/'labels-and-split.json').read_text())
    assert meta['windows']==json.loads((DEST/'labels-and-split.json').read_text())['windows']
    assert old['step']==new['step']==700
    data=RealData('subset2000')
    base=load_base(data)
    x=torch.load(folder/'observations.pt',weights_only=True)
    net=LeadReadout(x,{},[]).cuda().eval()
    checkpoint=torch.load(folder/'latest.ckpt',map_location='cpu',weights_only=False)
    assert checkpoint['global_step']==new['step']
    net.load_state_dict(checkpoint['state_dict'])
    entries=[]
    indices={r['actor_path']:i for i,r in enumerate(meta['windows'])}
    for a,b in zip(old['windows'],new['windows'],strict=True):
        assert (a['actor_path'],a['target'])==(b['actor_path'],b['target'])
        scene,actor=load_actor(b['actor_path'])
        cam=actor['camera_calibration']['CAM_FRONT']
        front=scene.views['<FRONT VIEW>']
        front[:]=[front[-1]]*4
        with patch.object(lead_motion,'load_actor',return_value=(scene,actor)):
            feat=observation(base,b['actor_path'],[cam]*4)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            static=float(net(feat[None].cuda()).item())
            original=float(net(x[indices[b['actor_path']]][None].cuda()).item())
        assert abs(original-b['prediction'])<1e-5
        z=feat.reshape(4,-1).mean(1)
        entries.append(dict(route=b['route'],index=b['index'],target=b['target'],before=a['prediction'],
                           after=b['prediction'],static_prediction=static,same_budget=float(z[:2].mean()),
                           mixed_budget=float(z[2:].mean())))
    replay=dict(n=len(entries),called_moving=sum(r['static_prediction']>.3 for r in entries),
                mean_same_budget=float(np.mean([r['same_budget'] for r in entries])),
                mean_mixed_budget=float(np.mean([r['mixed_budget'] for r in entries])),
                old_false_positives_still_positive=sum(r['before']>.3 and r['after']>.3 and r['target']<=.3 for r in entries),
                processor_pixels=[base.model.processor.config.history_image_pixels,base.model.processor.config.current_image_pixels])
    (folder/'paired-replay.json').write_text(json.dumps(dict(static=replay,windows=entries),indent=2))
    print(replay,flush=True)


if __name__=='__main__':
    main()
