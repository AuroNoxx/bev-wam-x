"""Verify frozen, native-size graph replay on real frames before the live pilot can select it."""
import argparse
import json
import time

import numpy as np
import torch

from .dataset import FastData, NavigationData
from .deployment import GraphDriver, fold_batch_norm, load_driver
from .sensors import ROOT, write_json


def main(corrective,lane_supervised=False,road_roi=False,dagger=False,navigation=False,spatial=False,film=False):
    navigation=navigation or spatial or film
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.set_per_process_memory_fraction(.15)
    folder=ROOT/"artifacts/v6-dual-system"
    if film:folder=folder/"fork-film"
    elif spatial:folder=folder/"spatial"
    elif navigation:folder=folder/"navigation"
    elif dagger:folder=folder/"dagger"
    elif road_roi:folder=folder/"road-roi"
    elif lane_supervised:folder=folder/"lane-supervised"
    elif corrective:folder=folder/"corrective"
    m,step,identity=load_driver(folder/"best.ckpt")
    config=json.loads((folder/"inference-operators.json").read_bytes())
    if config["deploy_folded"]:fold_batch_norm(m)
    m.to("cuda",memory_format=torch.channels_last)
    ds=NavigationData("validation",film=film) if navigation else FastData("validation",corrective or lane_supervised or road_roi or dagger)
    result=dict(step=step,identity=identity,torch=torch.__version__,device=torch.cuda.get_device_name(),
                deploy_graph=False,folded=config["deploy_folded"])
    write_json(folder/"graph.json",result)
    try:
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            def forward(*obs):return m(*obs,return_lane=navigation)
            graph=GraphDriver(forward,len(m.state_scale))
            maxima=[0.]*(3 if navigation else 2)
            times=[[],[]]
            engines=(forward,graph)
            for j,i in enumerate(np.linspace(0,len(ds)-1,68,dtype=int)):
                args=tuple(torch.as_tensor(x,device="cuda").unsqueeze(0) for x in ds[i][:4])
                out={}
                for k in (j%2,1-j%2):
                    torch.cuda.synchronize()
                    start=time.perf_counter()
                    output=engines[k](*args)
                    torch.cuda.synchronize()
                    if j>=4:times[k].append((time.perf_counter()-start)*1000)
                    out[k]=output
                for k in range(len(maxima)):
                    maxima[k]=max(maxima[k],float((out[0][k].float()-out[1][k].float()).abs().max()))
            med=[float(np.median(t)) for t in times]
            result.update(check_windows=68,timed_windows=64,eager_ms=med[0],graph_ms=med[1],
                          eager_p95_ms=float(np.percentile(times[0],95)),graph_p95_ms=float(np.percentile(times[1],95)),
                          max_xy_delta_m=maxima[0],max_occ_logit_delta=maxima[1],
                          max_lane_delta=maxima[-1] if navigation else None,
                          deploy_graph=bool((np.percentile(times[1],95)<20 if navigation else med[1]<med[0]) and max(maxima)<1e-4))
    except RuntimeError as exc:
        result["error"]=str(exc)
    write_json(folder/"graph.json",result)
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
