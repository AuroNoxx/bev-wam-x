"""Independent Stage 2 gates for the selected unified checkpoint."""
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from .unified_data import ART,BASE
from .unified_dataset import UnifiedData
from .unified_grid import HORIZONS, corridor
from .fast_model import UnifiedDriver
from .sensors import write_json


def load(checkpoint):
    ck=torch.load(checkpoint,weights_only=False,map_location="cpu")
    if ck["task_state"]["architecture"]!="unified-ipm":raise ValueError("Wrong checkpoint")
    model=UnifiedDriver()
    model.load_state_dict({k[7:]:v for k,v in ck["state_dict"].items() if k.startswith("driver.")})
    return model.eval(),ck["global_step"]


def choose_pairs(data):
    buckets=[[],[]]
    for idx,(ep,i) in enumerate(data.index):
        folder,meta,lab=data.episodes[ep]
        if not meta["fork"] or meta.get("case")!="fork-left":continue
        if i!=0:continue
        with np.load(folder/"labels.npz") as full:
            for j,station in enumerate(full["station"]):
                if not -30<=station<0:continue
                # S starts at -18 m with 2 m spacing; compare true 50-m branch headings.
                a,b=full["nav"][j],full["alt"][j]
                if a[34,1]-b[34,1]<=0:continue
                buckets[int(a[34,1]*b[34,1]<0)].append((idx+j,meta["group"],float(station)))
    result=[]
    for pairs,n in zip(buckets,(8,7)):
        if len(pairs)<n:raise ValueError(f"Need {n} same/opposite-side probes; found {len(pairs)}")
        result.extend([pairs[j] for j in np.linspace(0,len(pairs)-1,n,dtype=int)])
    return result


def gate():
    completed=json.loads((ART/"completed.json").read_bytes())
    assert completed["epochs"]==30
    model,step=load(ART/"best.ckpt")
    model.cuda()
    ds=UnifiedData("validation")
    errors=[]; lengths=[]; congestion_52=[]; cases=[]
    occupancy_counts = torch.zeros((3,5), dtype=torch.int64, device="cuda")
    occupancy_windows = 0
    position=0
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for batch in DataLoader(ds,batch_size=40,num_workers=2):
            rgb,lidar,nav,v,cmd,truth=batch[:6]
            xy,logits=model(*(a.cuda() for a in (rgb,lidar,nav,v,cmd,batch[-1])))
            xy=xy.cpu()
            valid=batch[7]
            if valid.any():
                predicted=logits[valid.to(logits.device)]>0.
                truth_occ=batch[6][valid].to(device="cuda",dtype=torch.bool)
                for totals,mask in zip(occupancy_counts,(predicted & truth_occ,predicted,truth_occ)):
                    totals+=mask.sum((0,2,3))
                occupancy_windows+=int(valid.sum())
            near=(xy[:,:15,1]-truth[:,:15,1]).abs().mean(1)
            errors.extend(near[cmd[:,0]==1].tolist())
            dist=torch.diff(xy,dim=1,prepend=torch.zeros_like(xy[:,:1])).norm(dim=-1).sum(1)
            for j,speed in enumerate(v):
                ep,_=ds.index[position+j]
                if ds.episodes[ep][1].get("case")=="congestion" and 50<=speed*3.6<=54:
                    lengths.append(float(dist[j]))
                    if 51.5<=speed*3.6<52.5:
                        congestion_52.append(float(dist[j]))
            position+=len(v)
        for idx,group,station in choose_pairs(ds):
            sample=ds[idx]
            rgb,lidar,nav,v,cmd=sample[:5]
            ep,i=ds.index[idx]
            folder=ds.episodes[ep][0]
            with np.load(folder/"labels.npz") as full:
                other=corridor(full["alt"][i])[None]
            args=[torch.as_tensor(a,device="cuda")[None] for a in (rgb,lidar,nav,v,cmd,sample[-1])]
            left=model(*args)[0][0,-1,1]
            args[2]=torch.as_tensor(other,device="cuda")[None]
            right=model(*args)[0][0,-1,1]
            cases.append(dict(group=group,station_m=station,delta_m=float(left-right)))
    median=float(np.median([c["delta_m"] for c in cases]))
    keep=float(np.mean(errors))
    checks=dict(keep=keep<=.14,fork=median>=3.5,
                congestion=bool(congestion_52) and all(52<=l<=65 for l in congestion_52))
    tp,positive,occupied=occupancy_counts.cpu().numpy()
    iou=tp/np.maximum(1,positive+occupied-tp)
    result=dict(step=step,passed=all(checks.values()),checks=checks,keep_mae=keep,keep_windows=len(errors),
                checkpoint_sha256=hashlib.sha256((ART/"best.ckpt").read_bytes()).hexdigest(),
                dynamic_occupancy=dict(windows=occupancy_windows,horizons_s=list(HORIZONS),iou=iou.tolist(),
                    recall=(tp/np.maximum(1,occupied)).tolist(),precision=(tp/np.maximum(1,positive)).tolist()),
                fork_4s_delta_m=median,probes=cases,congestion_windows=len(lengths),
                required_52_kmh=dict(speed_interval_kmh=[51.5,52.5],windows=len(congestion_52),
                    range_m=[min(congestion_52),max(congestion_52)] if congestion_52 else None,
                    within_band=sum(52<=l<=65 for l in congestion_52)),
                extended_50_54_all_in_band=bool(lengths) and all(52<=l<=65 for l in lengths),
                congestion_within_band=sum(52<=length<=65 for length in lengths),
                congestion_median_m=float(np.median(lengths)) if lengths else None,
                congestion_range_m=[min(lengths),max(lengths)] if lengths else None)
    write_json(ART/"open-loop.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":gate()
