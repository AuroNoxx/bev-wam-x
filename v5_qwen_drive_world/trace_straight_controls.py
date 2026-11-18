"""Trace signed acceleration from straight-road labels through the frozen planner."""
from collections import Counter
from functools import lru_cache
import gzip
import json

import numpy as np
import torch

from .kinematics import integrate
from .prepare_real_road_replay import OUT, ROOT, write
from .real_data_geometry import ego

DATA = ROOT / "data/v5-qwen-drive-world"
ART = ROOT / "artifacts/v5-qwen-drive-world"


def draw_counts():
    slots = {}
    for line in (ART/"subset2000/samples.jsonl").read_text().splitlines():
        r = json.loads(line)
        slot = r["epoch"],r["batch_index"]
        value = r["id"],r["step"]
        if slot in slots and slots[slot]!=value:
            raise ValueError(f"Ambiguous resumed microbatch {slot}")
        slots[slot]=value
    assert max(step for _,step in slots.values())==750 and len(slots)==750*8
    return Counter(key for key,_ in slots.values())


@lru_cache(maxsize=None)
def highway_poses(route):
    with (DATA/"highway"/route/"frames.jsonl").open() as f:
        frames = (json.loads(line) for line in f)
        return {r["index"]:np.array(r["current_world_from_ego"]) for r in frames}


@lru_cache(maxsize=4096)
def pose(source,route,index):
    if source=="highway":
        return highway_poses(route)[index]
    path = ROOT/"data/public/bench2drive/clips"/route/"anno"/f"{index:05d}.json.gz"
    with gzip.open(path,"rt") as f:
        return np.linalg.inv(ego(json.load(f))["world2ego"])


def observations(record, draws, cache):
    key = f"{record['source']}:{record['route']}:{record['index']}"
    with np.load(ROOT/record["label_path"]) as lab:
        xy,raw,ctrl,v0 = (lab[k].copy() for k in ("trajectory","raw_controls","controls","v0"))
    v = v0+np.cumsum(raw[:,0].astype(np.float64))*.1
    ref = pose(record["source"],record["route"],record["index"])
    last = pose(record["source"],record["route"],record["index"]+50)
    rel = np.linalg.solve(ref,last)
    assert np.linalg.norm(xy[-1]-rel[:2,3]*[1,-1])<.001
    heading = abs(np.arctan2(rel[1,0],rel[0,0]))
    if v0>=3 and v0-v[-1]>=3 and heading<.18:
        group = "brake"
    elif v0>=20/3.6 and max(abs(v[4::5]-v0))<=1 and heading<.08 and abs(xy[-1,1])<=1.5:
        group = "cruise"
    else:
        return None
    route = record["route"]
    row = dict(group=group,key=key,route=route,source=record["source"],draws=draws[key])
    def add(name,a):
        row[f"{name}_ax"],row[f"{name}_ay"]=a.mean(axis=0,dtype=np.float64).tolist()
        tr = integrate(torch.from_numpy(a[None]),torch.tensor([float(v0)]))[0]
        row[f"{name}_dy5"]=float(tr[-1,1])-float(xy[-1,1])
    add("label",ctrl)
    add("raw",raw)
    row["clipped_ay_fraction"]=float(np.mean(np.abs(raw[:,1]-ctrl[:,1])>1e-5))
    if record["actor_path"] in cache:
        add("pred",cache[record["actor_path"]])
    return row


def summary(rows,weighted=False):
    if weighted:rows=[r for r in rows if r["draws"]]
    if not rows:return {"windows":0}
    weights=np.array([r["draws"] if weighted else 1 for r in rows])
    keys=[k for k in rows[0] if k.startswith(("label_","raw_","pred_")) and all(k in r for r in rows)]
    stats={k:float(np.average([r[k] for r in rows],weights=weights)) for k in keys}
    if "pred_ay" in stats:
        stats["pred_left_count"]=sum(r["pred_ay"]>0 for r in rows)
    return dict(windows=len(rows),routes=len({r["route"] for r in rows}),weight=int(weights.sum()),**stats)


def main():
    manifest=json.loads((DATA/"manifest.json").read_text())["splits"]
    draws=draw_counts()
    groups={}
    details={}
    scope={}
    for split in ("train","validation"):
        cache={}
        for p in (ART/"geometry-repair/features"/split).glob("*.pt"):
            f=torch.load(p,weights_only=True,map_location="cpu")
            cache[f["actor_path"]]=f["ctrl"].numpy()
        records=manifest[split]
        ids={r["actor_path"] for r in records}
        assert set(cache)<=ids
        if split=="train":
            assert set(draws)<={f"{r['source']}:{r['route']}:{r['index']}" for r in records}
        rows=[result for r in records if (result:=observations(r,draws,cache)) is not None]
        scope[split]=dict(manifest=len(records),cached=len(cache))
        details[split]=rows
        for g in ("cruise","brake"):
            selected=[r for r in rows if r["group"]==g]
            pred=[r for r in selected if "pred_ay" in r]
            groups[f"{split}/{g}"]=dict(labels=summary(selected),same_windows=summary(pred),
                sources={s:summary([r for r in selected if r["source"]==s]) for s in sorted({r["source"] for r in selected})})
            if split=="train":groups[f"{split}/{g}"]["sampled_labels"]=summary(selected,weighted=True)
    write(OUT/"straight-control-trace.json",dict(scope=scope,groups=groups,windows=details))


if __name__=="__main__":
    main()

