"""Evaluate the spatial checkpoint on the registered route suite from the produced telemetry."""
import json

from .sensors import ROOT, write_json



STANDARD=("straight","gentle","lane","curve","cruise","braking","blind")


def main(film=False):
    art=ROOT/"artifacts/v6-dual-system"/("fork-film" if film else "spatial")
    paths=json.loads((art/"closed-loop-runs.json").read_text(encoding="utf-8-sig"))
    results={name:json.loads((ROOT/path).read_bytes()) for name,path in paths.items()}
    if not all(j["spatial"] and j["hip_graph"] and j["checkpoint"]==str(art/"best.ckpt") for j in results.values()):
        raise ValueError("Mixed checkpoints or deployment paths")
    val=json.loads((art/"open-loop.json").read_bytes())
    straight=results["straight"]["designated_lane"]
    rolling=results["straight-rolling"]["designated_lane"]
    forks={}
    for name in ("fork-left","fork-right"):
        j=results[name]
        window=j["fork_window"]
        frac=window["target_frames"]/max(window["frames"],1)
        forks[name]=dict(**window,target_fraction=frac,all_wrong_frames=j["wrong_branch_frames"],
            off_driving_lane_frames=j.get("off_driving_lane_frames"),
            passed=window["complete"] and (frac>=.90 if film else frac>.95) and (film or j["wrong_branch_frames"]==0)
                   and (not film or j["off_driving_lane_frames"]==0))
    handoffs=results["lane"]["lane_handoffs"]
    standard=[results[k] for k in STANDARD]
    peak=max(j["wddm_peak_dedicated_mib"] for j in results.values())*2**20
    p95=max(j["inference_p95_ms"] for j in results.values())
    checks=dict(heldout_keep=val["near_mae_m"][0]<.14,
                left_branch=forks["fork-left"]["passed"],right_branch=forks["fork-right"]["passed"],
                straight_from_rest_p95=straight["all_p95_m"]<.35,
                sustained_high_speed=rolling["longest_high_speed_centered_s"]>=10 and rolling["all_p95_m"]<.35,
                handoffs=len(handoffs)==2 and [e["direction"] for e in handoffs]==["a","d"] and all(e["within_2p5s"] for e in handoffs),
                collision_free=all(not j["collision_events"] for j in results.values()),latency=p95<20,memory=peak<1.5e9)
    if film:checks["heldout_keep"]=val["passed"]
    out=dict(passed=all(checks.values()),checks=checks,training=json.loads((art/"completed.json").read_bytes()),
             independent_validation=val,standard_frames=sum(j["frames"] for j in standard),
             standard_starts=len(set((j["town"],j["route_start"]) for j in standard)),
             standard_collisions=sum(len(j["collision_events"]) for j in standard),
             straight=straight,rolling=rolling,forks=forks,handoffs=handoffs,max_inference_p95_ms=p95,
             max_wddm_dedicated_bytes=round(peak),runs=paths)
    write_json(art/"acceptance.json",out)
    print(json.dumps(out),flush=True)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--film",action="store_true")
    main(parser.parse_args().film)
