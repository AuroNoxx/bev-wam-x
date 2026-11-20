"""Audit held-out KEEP tracking and exact LEFT/RIGHT reflection of the selected checkpoint."""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import NavigationData
from .deployment import load_driver
from .sensors import ROOT, write_json


def main(spatial=False,film=False):
    spatial=spatial or film
    folder=ROOT/"artifacts/v1-dual-system"/("fork-film" if film else "spatial" if spatial else "navigation")
    run=json.loads((folder/"completed.json").read_bytes())
    if run["epochs"]!=(10 if film else 12 if spatial else 30):raise ValueError("Incomplete navigation training")
    model,step,_=load_driver(folder/"best.ckpt")
    model.to("cuda",memory_format=torch.channels_last)
    torch.backends.mha.set_fastpath_enabled(False)
    dataset=NavigationData("validation",spatial=spatial,film=film)
    counts=np.zeros(3,dtype=int)
    sums=np.zeros(3)
    mirror=0.
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for batch in DataLoader(dataset,batch_size=16,num_workers=2):
            rgb,bev,state,cmd,truth=batch[:5]
            rgb,bev,state,cmd,truth=(v.cuda() for v in (rgb,bev,state,cmd,truth))
            xy,_=model(rgb,bev,state,cmd)
            errors=(xy[:,:15,1]-truth[:,:15,1]).abs().mean(1).cpu().numpy()
            labels=cmd[:,:3].argmax(1)
            label=labels.cpu().numpy()
            counts+=np.bincount(label,minlength=3)
            sums+=np.bincount(label,weights=errors,minlength=3)
            # Keep the same batch layout for both passes, then compare paired turn rows.
            reflected=state.clone()
            reflected[:,2::2]*=-1
            other,_=model(rgb.flip((-1,)),bev.flip((-2,)),reflected,cmd[:,[0,2,1,3]])
            diff=(xy-other*xy.new_tensor([1,-1])).abs().amax((1,2))
            mirror=max(mirror,float(torch.where(labels!=0,diff,0.).amax()))
    scores=(sums/counts).tolist()
    result=dict(step=step,epochs=run["epochs"],windows=counts.tolist(),
                near_mae_m=scores,mirror_max_m=mirror,passed=bool(scores[0]<(.14 if spatial else .12) and mirror==0.))
    if film:
        from .fork_probe import probe
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            contrast=probe(model,dataset)
        result["probe"]=contrast
        result["passed"]=bool(scores[0]<=.14 and contrast["one_second_passed"] and contrast["four_seconds_passed"])
    write_json(folder/"open-loop.json",result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--spatial",action="store_true")
    p.add_argument("--film",action="store_true")
    args=p.parse_args()
    main(args.spatial,args.film)
