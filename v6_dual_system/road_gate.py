"""Plot the measured offline steering response and bind the decision to a checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .sensors import write_json


def digest(path):
    with path.open("rb") as source:
        checksum=hashlib.sha256()
        for data in iter(lambda:source.read(4*1024*1024),b""):
            checksum.update(data)
    return checksum.hexdigest()


def check_gate(ckpt):
    gate=json.loads((ckpt.parent/"open-loop-gate.json").read_bytes())
    authorization=ckpt.parent/"closed-loop-authorization.json"
    approved=(authorization.is_file() and
              json.loads(authorization.read_bytes()).get("checkpoint_sha256")==gate["checkpoint_sha256"])
    if gate["checkpoint_sha256"]!=digest(ckpt) or not (gate["passed"] or approved):
        raise ValueError("Checkpoint has not passed the recorded open-loop evaluation")


def evaluate(ckpt):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"]=False

    folder=ckpt.parent
    metrics=json.loads((folder/"lateral-audit.json").read_bytes())
    probe=json.loads((folder/"same-observation.json").read_bytes())
    if (metrics["step"],metrics["manifest"])!=(probe["step"],probe["manifest"]):
        raise ValueError("Mismatched validation and command probe records")
    observations=sorted(probe["observations"],key=lambda o:o["key"])
    if [o["key"] for o in observations]!=["a","d"]:
        raise ValueError("One real frame per key is required")
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.0),layout="constrained")
    moves={}
    mismatches=[]
    colors={"keep":"#636363","left":"#007a61","right":"#c04b18"}
    for col,obs in enumerate(observations):
        plans=obs["predictions"]
        ax=axes[0,col]
        for cmd,plan in plans.items():
            ax.plot([1,2,3,4],plan["signed_error_at_1_2_3_4s_m"]["keep"],marker="o",
                    label=cmd.upper(),color=colors[cmd])
        ax.set_title(f'{obs["key"].upper()} 键输入帧　{obs["speed_kmh"]:.1f} km/h')
        ax.axhline(0,color="black",linewidth=.8,linestyle=":")
        ax.set(xlabel="预测时距（秒）",ylabel="相对原车道中心的横移（米，左为正）",xticks=[1,2,3,4])
        ax.grid(alpha=.2)
        ax.legend()
        shifts={c:float(v["signed_error_at_1_2_3_4s_m"]["keep"][2]) for c,v in plans.items()}
        moves[obs["key"]]={c:shifts[c]-shifts["keep"] for c in ("left","right")}
        ax=axes[1,col]
        right=np.asarray(plans["right"]["xy_flu_m"])
        left=np.asarray(obs["reflected_left_xy_flu_m"])
        mismatches.append(float(np.abs(right-left*[1,-1]).max()))
        ax.plot(right[:,0],right[:,1],color=colors["right"],label="原场景 RIGHT")
        ax.plot(left[:,0],left[:,1],"--",color=colors["left"],label="镜像场景 LEFT")
        ax.axhline(0,color="black",linewidth=.8)
        ax.set(xlabel="车体前向距离（米）",ylabel="车体横向距离（米，左为正）")
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(folder/"command-trajectories.png",dpi=160)
    plt.close(fig)
    # Symmetric curves can still ignore the command. Verify movement on the recorded
    # A/D inputs relative to KEEP, in the same road-centred coordinates as the plot.
    direction_ok=moves["a"]["left"]>0 and moves["d"]["right"]<0
    passed=metrics["near_pass"] and metrics["mirror_equal"] and max(mismatches)==0
    result=dict(passed=bool(passed),near_mae_m=metrics["all"]["near_m"],near_threshold_m=.15,
                heldout_mirror_max_xy_m=metrics["mirror_turns"]["max_xy_m"],
                probe_mirror_max_xy_m=max(mismatches),turn_response_3s_vs_keep_m=moves,
                recorded_commands_have_requested_sign=direction_ok,
                checkpoint_sha256=digest(ckpt),step=metrics["step"],manifest=metrics["manifest"])
    write_json(folder/"open-loop-gate.json",result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    (check_gate if args.check else evaluate)(args.checkpoint)
