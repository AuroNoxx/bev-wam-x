"""Open-loop replay of the frozen step-750 planner on its own SFT data."""
import json

import numpy as np
from .geometry_repair_features import BASE, RUN, ROOT, RealData, load_base, split_records, math_attention, torch
from .kinematics import integrate
from .native_scene import encode_native_scene
from .real_dataset import inputs_for
from .trace_straight_controls import draw_counts
from .train_scale_profile import fingerprint

OUT = BASE / "training-replay"


def score(ctrl, row):
    with np.load(ROOT/row["label_path"]) as labels:
        truth=torch.from_numpy(labels["trajectory"])
        v=torch.tensor([float(labels["v0"])])
        assert abs(v.item()*3.6-row["speed_kmh"])<.001
        bounded=torch.from_numpy(labels["controls"])
    tr=integrate(torch.stack([ctrl.cpu(),bounded]),v.expand(2))[:,:,:2]
    cv=torch.stack([torch.arange(1,51)*.1*v.item(),torch.zeros(50)],1)
    err=torch.linalg.vector_norm(torch.cat([tr,cv[None]])-truth,dim=-1)
    return dict(ade=float(err[0].mean()),fde5=float(err[0,-1]),label_ade=float(err[1].mean()),
                cv_ade=float(err[2].mean()),model_to_label=float(torch.linalg.vector_norm(tr[0]-tr[1],dim=-1).mean()))


def sample(task,row,seed):
    path=row["actor_path"]
    if task.prefix.lookup(path,True) is None:
        inp=inputs_for(task.model.processor,path,task.device,task.token_ids)
        native=encode_native_scene(task.model,inp)
    else:
        native,inp=task.encode_path(path)
    context=task.planner.context(native,inp)
    speed=torch.tensor([row["speed_kmh"]/3.6],device="cuda")
    with torch.random.fork_rng(devices=[task.device.index]):
        torch.manual_seed(seed)
        noise=torch.randn(1,50,2,device="cuda")
    return task.planner.sample(context,speed,noise,steps=10)["controls"][0].float().cpu()


def commit(path, data):
    tmp=path.with_suffix(".pending")
    tmp.write_text(json.dumps(data,allow_nan=False),encoding="utf-8")
    tmp.replace(path)


def main():
    torch.set_num_threads(3)
    torch.manual_seed(71)
    torch.cuda.set_per_process_memory_fraction(.8)
    torch.backends.cudnn.benchmark=False
    data=RealData("subset2000")
    ck_sha=fingerprint(BASE/"latest.ckpt")
    cached=json.loads((RUN/"feature-identity.json").read_text())
    assert ck_sha==cached["base_sha256"] and cached["manifest"]==data.identity
    subset=split_records(data)["train"]
    cached_index={r["actor_path"]:i for i,r in enumerate(subset)}
    overfit={r["actor_path"]:i for i,r in enumerate(data.manifest["overfit64"])}
    draws=draw_counts()
    records=data.records
    assert len({r["actor_path"] for r in records})==len(records)==2000
    assert set(overfit)<={r["actor_path"] for r in records}
    assert not {r["route"] for r in records}&{r["route"] for r in data.validation}
    identity=dict(base_sha256=ck_sha,manifest=data.identity,protocol=1)
    OUT.mkdir(exist_ok=True)
    path=OUT/"windows.json"
    result=json.loads(path.read_text()) if path.exists() else dict(identity=identity,rows={})
    assert result["identity"]==identity
    task=None
    # Return inexpensive cached training-window results before filling the rest.
    order=sorted(enumerate(records),key=lambda pair:pair[1]["actor_path"] not in cached_index or pair[1]["actor_path"] in overfit)
    with torch.inference_mode(),math_attention(),torch.autocast("cuda",dtype=torch.bfloat16):
        for i,r in order:
            actor=r["actor_path"]
            if actor in result["rows"]:continue
            j=cached_index.get(actor)
            if actor in overfit:
                seed=1907+overfit[actor]
            elif j is not None:
                seed=43001+j
            else:
                seed=85001+i
            if j is not None and actor not in overfit:
                feat=torch.load(RUN/f"features/train/{j:04d}.pt",weights_only=True)
                assert feat["actor_path"]==actor
                ctrl=feat["ctrl"]
            else:
                if task is None:task=load_base(data)
                ctrl=sample(task,r,seed)
            key=f"{r['source']}:{r['route']}:{r['index']}"
            result["rows"][actor]=dict(**score(ctrl,r),source=r["source"],route=r["route"],speed=r["speed_kmh"],
                draws=draws[key],overfit64=actor in overfit,seed=seed)
            n=len(result["rows"])
            if n%100==0:
                commit(path,result)
                print(n,"/ 2000",flush=True)
    assert len(result["rows"])==2000 and sum(r["draws"] for r in result["rows"].values())==6000
    commit(path,result)


if __name__=="__main__":
    main()

