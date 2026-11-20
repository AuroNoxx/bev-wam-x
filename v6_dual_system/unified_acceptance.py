"""Score synchronized CARLA telemetry for the unified driver."""
import json
from pathlib import Path

import numpy as np

from .sensors import ROOT, write_json
from .spatial_acceptance import STANDARD
from .unified_data import ART


def main():
    paths = json.loads((ART / "closed-loop-runs.json").read_text(encoding="utf-8-sig"))
    reports = {case: json.loads((ROOT / path).read_bytes()) for case, path in paths.items()}
    if not all(r["unified"] and r["hip_graph"] and Path(r["checkpoint"]) == ART / "best.ckpt"
               for r in reports.values()):
        raise ValueError("Results contain another model or inference backend")

    def rows(case):
        file = (ROOT / paths[case]).with_name("frames.jsonl")
        return [json.loads(line) for line in file.read_bytes().splitlines()]

    checks = {"offline": json.loads((ART / "open-loop.json").read_bytes())["passed"]}
    forks = {}
    for case in ("fork-left", "fork-right"):
        r = reports[case]
        w = r["fork_window"]
        forks[case] = w["target_frames"] / max(1, w["frames"])
        checks[case] = w["complete"] and forks[case] >= .9 and r["off_driving_lane_frames"] == 0
    straight = reports["straight"]["designated_lane"]
    rolling = reports["straight-rolling"]["designated_lane"]
    checks["straight"] = (straight["all_p95_m"] < .35 and rolling["all_p95_m"] < .35
                          and rolling["longest_high_speed_centered_s"] >= 10.)
    handoffs = reports["lane"]["lane_handoffs"]
    checks["mouse_changes"] = ([r["direction"] for r in handoffs] == ["a", "d"]
                               and all(r["within_2p5s"] for r in handoffs))

    sharp = [r for r in rows("curve") if r["map_curvature"] > .008 and r["kmh"] > 10.]
    errors = np.array([r["route_offset"] for r in sharp])
    # Count alternating excursions of more than half a metre across the centre.
    excursion = np.sign(errors[abs(errors) > .5])
    reversals = int(np.count_nonzero(np.diff(excursion)))
    checks["sharp_curve"] = (bool(sharp) and reversals < 2
                             and all(abs(r["route_offset"]) < r["route_width_m"] / 2 for r in sharp))

    follow = rows("follow50")
    steady = [r for r in follow if 12. <= r["sim_s"] < 22. and 45. <= r["npc"]["kmh"] <= 55.]
    after = [r for r in follow if r["sim_s"] >= 27. and r["npc"]["kmh"] >= 65.]
    follow_speed = float(np.median([r["kmh"] for r in steady])) if steady else None
    peak_after = max((r["kmh"] for r in after), default=None)
    extremes = sum((a["throttle"] >= .95 and b["brake"] >= .95)
                   or (a["brake"] >= .95 and b["throttle"] >= .95) for a,b in zip(follow,follow[1:]))
    checks["following"] = bool(steady and after and 45. <= follow_speed <= 55.
                               and peak_after > follow_speed + 5. and extremes == 0
                               and all(r["target"] == 100 and r["npc"]["ahead_m"] > 0
                                       and abs(r["npc"]["left_m"]) < 2. for r in steady))
    for case, mode in (("gap-accel","ACCEL_GAP"), ("gap-yield","YIELD_GAP")):
        frames = rows(case)
        start = next((i for i,r in enumerate(frames) if r["key"] == "a"), len(frames))
        finish = next((i for i,r in enumerate(frames[start:],start)
                       if r["target_lane_shift"] > 2.5 and r["maneuver"] == "KEEP"), None)
        active = frames[start:finish] if finish is not None else []
        adjusting = [r for r in active if r["maneuver"] == mode]
        checks[case] = (bool(adjusting) and all(r["pending"] == 1 for r in active)
                        and any(r["maneuver"] == "EXECUTE" for r in active)
                        and abs(frames[finish]["route_offset"]-frames[finish]["target_lane_shift"]) < .5
                        and any(r["accel"] > 0 if case == "gap-accel" else r["accel"] < 0 for r in adjusting))
    checks["collision_free"] = all(r["frames"] > 0 and not r["collision_events"] and
                                   (r["end"] == "duration" or r["end"].endswith("_segment_complete"))
                                   for r in reports.values())
    checks["lead_brake_exercised"] = reports["braking"]["lead_brake_event"]["triggered_at_s"] is not None
    latency = max(r["inference_p95_ms"] for r in reports.values())
    peak = max(r["wddm_peak_dedicated_mib"] for r in reports.values()) * 2**20
    checks.update(latency=latency < 20., memory=peak < 1.5e9)
    result = dict(passed=all(checks.values()), checks=checks, forks=forks, straight=straight,
                  rolling=rolling, handoffs=handoffs, standard_frames=sum(reports[c]["frames"] for c in STANDARD),
                  standard_routes=len({(reports[c]["town"],reports[c]["route_start"]) for c in STANDARD}),
                  standard_collisions=sum(len(reports[c]["collision_events"]) for c in STANDARD),
                  all_runs=len(reports), all_frames=sum(r["frames"] for r in reports.values()),
                  collisions={case:r["collision_events"] for case,r in reports.items() if r["collision_events"]},
                  maximum_inference_p95_ms=latency, maximum_dedicated_bytes=round(peak),
                  following_median_kmh=follow_speed, after_gap_peak_kmh=peak_after,
                  extreme_transitions=extremes, curve_frames=len(sharp), curve_reversals=reversals,
                  curve_max_abs_offset_m=float(np.max(np.abs(errors))) if sharp else None, runs=paths)
    write_json(ART / "acceptance.json",result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
