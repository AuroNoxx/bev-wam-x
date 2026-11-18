"""Frozen step-750 paired diagnosis of missing route inputs and noise sensitivity."""
import json
import argparse
import io
import tarfile
from collections import defaultdict
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
from .geometry_repair_features import BASE, RUN, RealData, load_base, math_attention, torch
from .prepare_real_road_replay import OUT, ROOT, SOURCE, MetadataReader, interp, write
from .native_scene import encode_native_scene
from .real_dataset import inputs_for
from .train_scale_profile import fingerprint


def evaluate(task, actor_path, seed, reference, mirror=False, direct=False, trajectory=False):
    if direct:
        inputs=inputs_for(task.model.processor,actor_path,task.device,task.token_ids)
        native=encode_native_scene(task.model,inputs)
    else:
        native,inputs=task.encode_path(actor_path)
    context=task.planner.context(native,inputs)
    actor=json.loads((ROOT/actor_path).read_text())
    speed=torch.tensor([actor["speed_kmh"]/3.6],device="cuda",dtype=torch.float32)
    with torch.random.fork_rng(devices=[task.device.index]):
        torch.manual_seed(seed)
        noise=torch.randn(1,50,2,device="cuda")
    def sample(z):
        data=task.planner.sample(context,speed,z,steps=10)
        ctrl=data["controls"][0].float().cpu()
        xy=data["trajectory"][0,-1,:2].float().cpu()
        stats=dict(ax=float(ctrl[:,0].mean()),ay=float(ctrl[:,1].mean()),xy5=xy.tolist())
        if trajectory:
            stats["xy"]=data["trajectory"][0,4::5,:2].float().cpu().tolist()
        return stats,ctrl
    stats,ctrl=sample(noise)
    if reference is not None:
        torch.testing.assert_close(ctrl,reference,rtol=0.,atol=.0001)
    return (stats,sample(-noise)[0]) if mirror else stats


def pose_history_check(task,base_hash):
    cohort=json.loads((OUT/"cohort.json").read_text())
    inputs=json.loads((OUT/"straight-control-inputs.json").read_text())
    assert inputs["base_sha256"]==base_hash
    prior={r["token"]:r["original"] for r in inputs["real"]}
    predictions={r["token"]:r for r in json.loads((OUT/"predictions.json").read_text())}
    routes=defaultdict(list)
    for i,r in enumerate(cohort):
        if r["token"] in prior:routes[r["route"]].append((i,r))
    rows=[]
    with tarfile.open(SOURCE/"openscene_metadata_mini.tgz","r|gz") as archive, TemporaryDirectory(dir=OUT) as tmp:
        for entry in archive:
            route=Path(entry.name).stem
            if route not in routes:continue
            frames=sorted(MetadataReader(io.BytesIO(archive.extractfile(entry).read())).load(),key=lambda f:f["timestamp"])
            for i,r in routes[route]:
                j=r["frame_index"]
                hist=frames[j-4:j+1]
                assert hist[-1]["token"]==r["token"]
                ts=(np.array([f["timestamp"] for f in hist])-hist[-1]["timestamp"])/1e6
                assert np.all((np.diff(ts)>.43)&(np.diff(ts)<.57)) and ts[0]<-1.5
                poses=np.stack([f["ego2global"] for f in hist])
                xy=(np.linalg.inv(poses[-1])@poses)[:,:2,3]
                velocity=np.gradient(xy,ts,axis=0)
                accel=np.gradient(velocity,ts,axis=0)
                actor=json.loads((ROOT/r["actor_path"]).read_text())
                grid=np.arange(-15,1)*.1
                actor.update(history_velocity=interp(grid,ts,velocity).tolist(),history_acceleration=interp(grid,ts,accel).tolist(),
                             ego_velocity=[actor["speed_kmh"]/3.6,0.],ego_acceleration=accel[-1].tolist())
                path=Path(tmp)/f"{i:04d}.json"
                write(path,actor)
                new=evaluate(task,str(path.relative_to(ROOT)),75001+i,None,direct=True,trajectory=True)
                new_xy=np.asarray(new.pop("xy"))
                truth=np.asarray(r["truth_xy"])
                original=prior[r["token"]].copy()
                original["ade"]=predictions[r["token"]]["scores"]["original750"]["ADE"]
                new["ade"]=float(np.linalg.norm(new_xy-truth,axis=1).mean())
                for stats in (original,new):stats["dy5"]=stats["xy5"][1]-float(truth[-1,1])
                rows.append(dict(token=r["token"],group=r["group"],original=original,pose_derived=new))
    assert len(rows)==len(prior)==26
    summary={}
    for g in ("cruise","brake"):
        items=[r for r in rows if r["group"]==g]
        summary[g]={}
        for key in ("original","pose_derived"):
            summary[g][key]={k:float(np.mean([r[key][k] for r in items])) for k in ("ax","ay","ade","dy5")}
            summary[g][key]["left"]=sum(r[key]["ay"]>0 for r in items)
    write(OUT/"straight-control-history.json",dict(base_sha256=base_hash,summary=summary,windows=rows))
    print(json.dumps(summary),flush=True)


def main(history=False):
    torch.manual_seed(71)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.8)
    torch.backends.cudnn.benchmark=False
    base_hash=fingerprint(BASE/"latest.ckpt")
    assert base_hash==json.loads((RUN/"feature-identity.json").read_text())["base_sha256"]
    data=RealData("subset2000")
    task=load_base(data)
    if history:
        with torch.inference_mode(),math_attention(),torch.autocast("cuda",dtype=torch.bfloat16):
            pose_history_check(task,base_hash)
        return
    audit=json.loads((OUT/"straight-control-trace.json").read_text())
    sim_keys={r["key"] for r in audit["windows"]["validation"] if r["group"]=="cruise"}
    sim=[]
    real=[]
    with torch.inference_mode(),math_attention(),torch.autocast("cuda",dtype=torch.bfloat16),TemporaryDirectory(dir=OUT) as tmp:
        for i,r in enumerate(data.validation):
            key=f"{r['source']}:{r['route']}:{r['index']}"
            if key not in sim_keys:continue
            actor_path=r["actor_path"]
            actor=json.loads((ROOT/actor_path).read_text())
            old_inputs={k:actor[k] for k in ("nav_command","navigation_flu","target_speed_kmh","driving_command")}
            ref=torch.load(RUN/f"features/validation/{i:04d}.pt",weights_only=True)
            assert ref["actor_path"]==actor_path
            original=evaluate(task,actor_path,1907+i,ref["ctrl"])
            actor.update(nav_command=0,navigation_flu=[0.,0.],target_speed_kmh=actor["speed_kmh"],driving_command=[0,0,0,0])
            path=Path(tmp)/f"{i:04d}.json"
            write(path,actor)
            modified=evaluate(task,str(path.relative_to(ROOT)),1907+i,None,direct=True)
            sim.append(dict(key=key,original=original,modified=modified,original_route_inputs=old_inputs))
        print("matched simulated cruise",len(sim),flush=True)
        pack=torch.load(OUT/"features.pt",map_location="cpu",weights_only=True)
        assert pack["base_sha256"]==base_hash
        rows=json.loads((OUT/"cohort.json").read_text())
        for i,r in enumerate(rows):
            if r["group"] not in ("cruise","brake"):continue
            a,b=evaluate(task,r["actor_path"],75001+i,pack["features"][r["token"]]["ctrl"],mirror=True)
            real.append(dict(token=r["token"],group=r["group"],route=r["route"],original=a,mirrored_noise=b))
        print("matched real windows",len(real),flush=True)
    def summary(items,field):
        ax,ay=np.array([[r[field]["ax"],r[field]["ay"]] for r in items]).T
        return dict(n=len(items),positive_ay=int((ay>0).sum()),ax=float(ax.mean()),ay=float(ay.mean()))
    summaries={"simulation":{k:summary(sim,k) for k in ("original","modified")}}
    for g in ("cruise","brake"):
        sub=[r for r in real if r["group"]==g]
        summaries[g]={k:summary(sub,k) for k in ("original","mirrored_noise")}
        summaries[g]["same_sign"]=sum((r["original"]["ay"]>0)==(r["mirrored_noise"]["ay"]>0) for r in sub)
    write(OUT/"straight-control-inputs.json",dict(base_sha256=base_hash,summary=summaries,simulation=sim,real=real))
    print(json.dumps(summaries),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--history",action="store_true")
    main(parser.parse_args().history)

