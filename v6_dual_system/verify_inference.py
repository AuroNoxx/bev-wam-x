"""Time and numerically check BN folding on the selected checkpoint and native held-out inputs."""
import copy
import argparse
import json
import time

import numpy as np
import torch

from .dataset import FastData, NavigationData
from .deployment import fold_batch_norm, load_driver
from .sensors import ROOT, write_json


def main(corrective,lane_supervised=False,road_roi=False,dagger=False,navigation=False,spatial=False,film=False):
    navigation=navigation or spatial or film
    torch.backends.cudnn.benchmark=False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_num_threads(3)
    torch.cuda.set_per_process_memory_fraction(.15)
    folder=ROOT/"artifacts/v6-dual-system"
    if film:folder=folder/"fork-film"
    elif spatial:folder=folder/"spatial"
    elif navigation:folder=folder/"navigation"
    elif dagger:folder=folder/"dagger"
    elif road_roi:folder=folder/"road-roi"
    elif lane_supervised:folder=folder/"lane-supervised"
    elif corrective:folder=folder/"corrective"
    baseline,step,identity=load_driver(folder/"best.ckpt")
    folded=copy.deepcopy(baseline)
    pairs=fold_batch_norm(folded)
    models=[m.to("cuda",memory_format=torch.channels_last) for m in (baseline,folded)]
    ds=NavigationData("validation",film=film) if navigation else FastData("validation",corrective or lane_supervised or road_roi or dagger)
    times=[[],[]]
    peaks=[0.,0.,0.,0.]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for j,i in enumerate(np.linspace(0,len(ds)-1,68,dtype=int)):
            obs=tuple(torch.as_tensor(a,device="cuda").unsqueeze(0) for a in ds[i][:4])
            outputs={}
            for k in (j%2,1-j%2):
                torch.cuda.synchronize()
                start=time.perf_counter()
                output=models[k](*obs,return_lane=navigation)
                xy,logits=output[:2]
                torch.cuda.synchronize()
                if j>=4:times[k].append((time.perf_counter()-start)*1000)
                outputs[k]=(xy,logits.float().sigmoid(),output[2] if navigation else None)
            diff=(outputs[0][0]-outputs[1][0]).abs()
            values=[diff[:,:10].max(),diff.max(),(outputs[0][1]-outputs[1][1]).abs().max()]
            values.append((outputs[0][2]-outputs[1][2]).abs().max() if navigation else 0.)
            peaks=[max(old,float(val)) for old,val in zip(peaks,values)]
    med=[float(np.median(t)) for t in times]
    # Only deploy a measured improvement that retains centimetre-scale near-field geometry.
    use=med[1]<med[0] and peaks[0]<.05 and peaks[1]<.2 and peaks[2]<.05 and peaks[3]<.05
    result=dict(step=step,identity=identity,torch=torch.__version__,device=torch.cuda.get_device_name(),pairs=pairs,
                check_windows=68,timed_windows=64,eager_median_ms=med[0],folded_median_ms=med[1],
                eager_p95_ms=float(np.percentile(times[0],95)),folded_p95_ms=float(np.percentile(times[1],95)),
                max_near_xy_difference_m=peaks[0],max_xy_difference_m=peaks[1],
                max_occ_probability_difference=peaks[2],max_lane_difference=peaks[3],deploy_folded=use)
    write_json(folder/"inference-operators.json",result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--corrective",action="store_true")
    p.add_argument("--lane-supervised",action="store_true")
    p.add_argument("--road-roi",action="store_true")
    p.add_argument("--dagger",action="store_true")
    p.add_argument("--navigation",action="store_true")
    p.add_argument("--spatial",action="store_true")
    p.add_argument("--film",action="store_true")
    args=p.parse_args()
    main(args.corrective,args.lane_supervised,args.road_roi,args.dagger,args.navigation,args.spatial,args.film)
