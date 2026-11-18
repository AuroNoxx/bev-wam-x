"""Offline control-target consistency checks. Source labels remain read-only."""
import json
from collections import defaultdict

import numpy as np
import torch

from .kinematics import integrate
from .prepare_real_road_replay import ROOT, OUT, write


def main():
    manifest=json.loads((ROOT/"data/v5-qwen-drive-world/manifest.json").read_text())["splits"]
    trace=json.loads((OUT/"straight-control-trace.json").read_text())["windows"]
    summary={}
    for split in ("train","validation"):
        groups={r["key"]:r["group"] for r in trace[split]}
        labels,raw,speeds,truth=[],[],[],[]
        by_group=defaultdict(list)
        for i,r in enumerate(manifest[split]):
            key=f"{r['source']}:{r['route']}:{r['index']}"
            with np.load(ROOT/r["label_path"]) as f:
                labels.append(f["controls"])
                raw.append(f["raw_controls"])
                speeds.append(float(f["v0"]))
                truth.append(f["trajectory"])
            by_group["all"].append(i)
            by_group[groups.get(key,"other")].append(i)
        ctrl=torch.tensor(np.stack(labels))
        original=torch.tensor(np.stack(raw))
        v=torch.tensor(speeds)
        xy=torch.tensor(np.stack(truth))
        avg=original[:,:,1].reshape(len(ctrl),10,5).mean(-1).repeat_interleave(5,dim=1)
        def with_ay(ay):return torch.stack((ctrl[:,:,0],ay),dim=-1)
        variants={"stored":ctrl,"only_ax_bounded":with_ay(original[:,:,1]),"raw":original,
                  "half_second_lateral":with_ay(avg.clamp(-3.5,3.5))}
        stats={g:{} for g in by_group}
        for name,a in variants.items():
            delta=integrate(a,v)[:,:,:2]-xy
            for g,ids in by_group.items():
                d=delta[ids]
                lateral=d[:,-1,1].abs()
                stats[g][name]=dict(n=len(ids),dy5_abs_median=float(torch.quantile(lateral,.5)),
                    dy5_abs_p95=float(torch.quantile(lateral,.95)),ade=float(d.norm(dim=2).mean()),
                    fde=float(d[:,-1].norm(dim=1).mean()))
        summary[split]=stats
    write(OUT/"control-target-roundtrip.json",summary)


if __name__=="__main__":
    torch.set_num_threads(2)
    main()

