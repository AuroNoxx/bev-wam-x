"""Paired true-route probe on the original 15 held-out Town04 approach frames."""
import json

import numpy as np
import torch

from .dataset import FastData
from .navigation import model_navigation_state
from .sensors import DATA


def probe(model, dataset):
    episode="Town04-fork-50-6-left"
    root=DATA/"spatial-navigation"
    split=json.loads((root/"manifest.json").read_bytes())
    if episode not in split["holdout"] or episode in split["episodes"]:
        raise ValueError("Probe route is not held out")
    data=FastData("validation",True)
    with np.load(dataset.metadata_folder/"probe.npz") as evidence:
        left,right,indices,station=(evidence[k] for k in ("left","right","indices","station"))
    if len(indices)!=15 or len(set(indices))!=15 or not np.all((-30<station)&(station<0)):
        raise ValueError("Changed probe population")
    data.entries=[("spatial-navigation/"+episode,int(idx)) for idx in indices]
    deltas=[]
    for i in range(len(indices)):
        rgb,bev,state,cmd,*_=data[i]
        if cmd[0]!=1:raise ValueError("Expected a KEEP observation")
        obs=[torch.as_tensor(a,device="cuda").unsqueeze(0) for a in (rgb,bev,cmd)]
        prediction=[]
        for anchors in (left[i],right[i]):
            nav=torch.as_tensor(model_navigation_state(state[0],anchors),device="cuda")[None]
            xy,_=model(*obs[:2],nav,obs[2])
            prediction.append(xy[0,[9,19,39],1].float().cpu().numpy())
        deltas.append(prediction[0]-prediction[1])
    deltas=np.stack(deltas)
    nav_delta=float(np.median(left[:,3]-right[:,3]))
    if abs(nav_delta-7.2049284)>.015:raise ValueError("Changed 60 m route contrast")
    medians=np.median(deltas,axis=0)
    return dict(heldout_frames=len(indices),indices=indices.tolist(),station_m=station.tolist(),
                median_nav60_y_change_m=nav_delta,signed_median_1_2_4s_m=medians.tolist(),
                median_left_nav50_y_m=float(np.median(left[:,-1])),
                median_right_nav50_y_m=float(np.median(right[:,-1])),
                median_right_vs_reflected_left_nav50_gap_m=float(np.median(abs(right[:,-1]+left[:,-1]))),
                signed_deltas_m=deltas.tolist(),positive_left_difference_counts=(deltas>0).sum(0).tolist(),
                one_second_passed=bool(medians[0]>=.45),four_seconds_passed=bool(medians[2]>=3.5))
