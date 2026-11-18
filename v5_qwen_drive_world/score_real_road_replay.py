"""Score fixed planning checkpoints on six-view recorded road windows."""
import argparse
import json
from collections import defaultdict

import numpy as np
from .prepare_real_road_replay import OUT, ROOT, TIMES, write


def read(name):
    return json.loads((OUT / name).read_text())


def observe_roads(rows):
    import cv2
    from .paired_launch_observations import observe
    from .scale_geometry import RUN
    cv2.setNumThreads(3)
    cv2.setRNGSeed(9147)
    width = json.loads((RUN/"scale-results.json").read_text())["width_m"]
    boxes = read("historical-boxes.json")["predictions"]
    observations = {}
    for row in rows:
        result,_ = observe(row,boxes,width)
        observations[row["token"]] = {k:result[k] if result else None for k in ("displacement","depth")}
    write(OUT/"observations.json",observations)


def extract(rows):
    from .geometry_repair_features import actor_features, load_base, RealData, BASE, RUN, math_attention, torch
    from .train_scale_profile import fingerprint
    torch.set_num_threads(4)
    torch.manual_seed(71)
    torch.cuda.set_per_process_memory_fraction(.8)
    torch.backends.cudnn.benchmark=False
    sha = fingerprint(BASE/"latest.ckpt")
    assert sha==json.loads((RUN/"feature-identity.json").read_text())["base_sha256"]
    task=load_base(RealData("subset2000"))
    features = {}
    with torch.inference_mode(), math_attention(), torch.autocast("cuda",dtype=torch.bfloat16):
        for i,row in enumerate(rows):
            features[row["token"]] = actor_features(task,{"actor_path":row["actor_path"]},75001+i)
            if (i+1)%5==0:print("actor",i+1,"/",len(rows),flush=True)
    torch.save(dict(base_sha256=sha,features=features),OUT/"features.pt")


def report(rows):
    from .geometry_repair_features import torch, math_attention
    from .kinematics import integrate
    from .motion_profile import profile_features
    from .mine_launch_pairs import PAIR
    from .train_scale_profile import ScaleMotion, extend_observation, fingerprint
    pack=torch.load(OUT/"features.pt",map_location="cpu",weights_only=True)
    ck=torch.load(PAIR/"best.ckpt",map_location="cpu",weights_only=False)
    sha=fingerprint(PAIR/"best.ckpt")
    assert sha==json.loads((PAIR/"validation.json").read_text())["checkpoint_sha256"] and ck["global_step"]==162
    model=ScaleMotion(ck["task_state"]["identity"]).cuda().eval()
    model.load_state_dict(ck["state_dict"])
    cues=read("observations.json")
    results=[]
    with torch.inference_mode(),math_attention(),torch.autocast("cuda",dtype=torch.bfloat16):
        for r in rows:
            feature={k:v[None].cuda() for k,v in pack["features"][r["token"]].items()}
            obs,base=profile_features(feature)
            v0=feature["v0"].flatten()
            _,candidate,_=model.profile(extend_observation(obs[0],cues[r["token"]])[None],base,feature["ctrl"],v0)
            original=integrate(feature["ctrl"],v0)
            speed=r["speed_kmh"]/3.6
            cv=np.c_[TIMES*speed,np.zeros(10),np.full(10,speed)]
            out=dict(token=r["token"],group=r["group"],route=r["route"],city=r["city"],speed_kmh=r["speed_kmh"],
                     cue=cues[r["token"]],scores={},predictions={})
            for name,tr in (("candidate162",candidate),("original750",original),("constant_velocity",None)):
                pred=cv if tr is None else tr[0,4::5][:,[0,1,3]].float().cpu().numpy()
                error=pred[:,:2]-r["truth_xy"]
                dist=np.linalg.norm(error,axis=1)
                out["scores"][name]=dict(ADE=float(dist.mean()),FDE1=float(dist[1]),FDE3=float(dist[5]),FDE5=float(dist[9]),
                    longitudinal_absolute_5s=float(abs(error[-1,0])),lateral_absolute_5s=float(abs(error[-1,1])),
                    speed_error_5s_mps=float(pred[-1,2]-r["truth_speed"][-1]),
                    longitudinal_jerk_mean_abs_mps3=float(np.abs(np.diff(np.r_[speed,pred[:,2]],n=2)).mean()/.25))
                out["predictions"][name]=pred.tolist()
            results.append(out)
    write(OUT/"predictions.json",results)
    groups=defaultdict(list)
    for r in results:
        for k in ("all",r["group"],"speed_0_40" if r["speed_kmh"]<40 else "speed_40_80"):
            groups[k].append(r)
    summary={}
    for k,items in groups.items():
        expert_jerk=[np.abs(np.diff(np.r_[r["speed_kmh"]/3.6,r["truth_speed"]],n=2)).mean()/.25
                     for r in rows if r["token"] in {item["token"] for item in items}]
        summary[k]=dict(n=len(items),logs=len(set(r["route"] for r in items)),expert_jerk_mean_abs_mps3=float(np.mean(expert_jerk)),
            metrics={name:{m:float(np.mean([r["scores"][name][m] for r in items])) for m in scores}
                     for name,scores in items[0]["scores"].items()})
    write(OUT/"metrics.json",dict(groups=summary,base_sha256=pack["base_sha256"],candidate_sha256=sha,
         cue_available=sum(r["cue"]["displacement"] is not None for r in results)))
    print(json.dumps(summary["all"]),flush=True)


def render(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from .prepare_real_road_replay import VIEWS
    results = {r["token"]:r for r in read("predictions.json")}
    colours = {"Expert":"#131D27","Step 162":"#A43E21","Step 750":"#277CA9","Constant speed":"#777B7C"}
    fig, axes = plt.subplots(2,3,figsize=(13,7.5),layout="constrained",height_ratios=[1,2])
    for j,group in enumerate(("cruise","brake","turn")):
        items=sorted((r for r in rows if r["group"]==group),
                     key=lambda r:results[r["token"]]["scores"]["candidate162"]["ADE"])
        r=items[(len(items)-1)//2]
        pred=results[r["token"]]
        actor=json.loads((ROOT/r["actor_path"]).read_text())
        with Image.open(ROOT/actor["images"][VIEWS[0]][-1]) as im:
            axes[0,j].imshow(im)
        axes[0,j].axis("off")
        axes[0,j].set_title(f"{group.capitalize()}   {r['speed_kmh']:.1f} km/h")
        paths=dict(Expert=r["truth_xy"])
        for label,name in (("Step 162","candidate162"),("Step 750","original750"),("Constant speed","constant_velocity")):
            paths[label]=pred["predictions"][name]
        ax=axes[1,j]
        for label,points in paths.items():
            xy=np.vstack(([0,0],np.asarray(points)[:,:2]))
            ax.plot(xy[:,1],xy[:,0],color=colours[label],label=label,
                    linestyle="--" if label=="Constant speed" else "-",marker=".",markersize=5)
        ax.invert_xaxis()
        ax.set_aspect("equal",adjustable="datalim")
        ax.grid(alpha=.2)
        ax.set_xlabel("Ego left (m)")
        ax.set_ylabel("Forward (m)")
    axes[1,-1].legend(fontsize=8)
    fig.suptitle("Real-road replay | lower median step-162 ADE window per maneuver")
    fig.savefig(OUT/"examples.png",dpi=150)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["observations","features","score","plot"])
    args=parser.parse_args()
    {"observations":observe_roads,"features":extract,"score":report,"plot":render}[args.phase](read("cohort.json"))

