"""Diagnostic reproduction of Stage 7 representative failure & timeout cases.

Instruments step-by-step:
1. Blocked lane rear-end collision (Seed 4300302, Arm D Seed 20261022)
2. Single-lane off-road violation (Seed 4300303, Arm D Seed 20261022)
3. Safe timeout / stagnation case (Seed 4300306, Arm D Seed 20261023)

Saves full telemetry and diagnosis to artifacts/stage7_diagnostic/.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from configs.stage3 import Stage3Config
from configs.stage2 import BEVConfig
from configs.integrated import IntegratedConfig
from models.bev_wa_joint_flow import ArmDJointFlowModel
from env.data_v2 import configure_egl, NativeBEVProvider
from env.integrated import make_external_env, checked_step
from env.integrated_scenarios import setup_driving_scene
from env.navigation import NavigationProvider, vehicle_parameters
from env.lane_permissions import maneuver_permissions
from dataset.recording import vehicle_state
from controller.trajectory_tracker import TrajectoryTracker
from mini_wa_jepa.geometry import world_to_ego, ego_to_world, wrap_angle

OUT_DIR = WORKSPACE / "artifacts/stage7_diagnostic"
VAL_MANIFEST = WORKSPACE / "artifacts/stage7_integrated/mixed_validation_010/manifest.json"
STAGE4_DIR = WORKSPACE / "artifacts/bev_wa_alignment/stage4_001"


def run_case_diagnosis(case_name, scene_seed, train_seed, max_steps=600):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"Running Diagnosis: {case_name}")
    print(f"Scene Seed: {scene_seed}, Model Arm D Seed: {train_seed}")
    print(f"=======================================================")

    # Load manifest scene config
    with open(VAL_MANIFEST) as f:
        val_manifest = json.load(f)
    scene = None
    for sc in val_manifest["scenes"]:
        if sc.get("seed") == scene_seed:
            scene = sc
            break
    if scene is None:
        raise ValueError(f"Scene seed {scene_seed} not found in manifest")

    # Load model
    cfg = Stage3Config()
    model = ArmDJointFlowModel(cfg).to(device).eval()
    ckpt_path = STAGE4_DIR / f"arm_d_(joint_wa-flow)_seed_{train_seed}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # Setup environment
    sim, provider, npc_ids, overrides = setup_driving_scene(scene)
    nav = NavigationProvider(sim, IntegratedConfig())
    tracker = TrajectoryTracker(sim.agent, 0.1)

    history_bev = []
    history_ego = []
    last_heading = None

    records = []
    crashed = False
    offroad = False
    arrived = False
    termination_reason = "horizon_reached"

    # Fix inference random stream for reproducibility
    torch.manual_seed(train_seed + scene_seed)

    for step in range(max_steps):
        # 1. State extraction
        state = vehicle_state(sim.agent)
        ego_pos = np.array([float(state[0]), float(state[1])], dtype=np.float32)
        ego_heading = float(state[2])
        ego_speed = float(state[3])
        ego_lane = sim.agent.lane
        ego_s, ego_lat = ego_lane.local_coordinates(ego_pos)

        curr_h = ego_heading
        yaw_rate = 0.0 if last_heading is None else float(wrap_angle(curr_h - last_heading)) / 0.1
        last_heading = curr_h

        # 2. BEV capture
        clean_bev = provider.capture().raster.copy()
        history_bev.append(clean_bev)
        history_ego.append(np.array([ego_speed / 10.0, yaw_rate], dtype=np.float32))
        if len(history_bev) > 4:
            history_bev.pop(0)
            history_ego.pop(0)

        h_bev = np.stack([history_bev[0]] * (4 - len(history_bev)) + history_bev, axis=0)
        h_ego = np.stack([history_ego[0]] * (4 - len(history_ego)) + history_ego, axis=0)

        nav_frame = nav.capture()
        perm = maneuver_permissions(sim)
        veh = vehicle_parameters(sim.agent)

        # 3. Model forward
        h_bev_t = torch.from_numpy(h_bev[None]).to(device)
        h_ego_t = torch.from_numpy(h_ego[None]).to(device)
        rp_t = torch.from_numpy(nav_frame.points[None].astype(np.float32)).to(device)
        rv_t = torch.from_numpy(nav_frame.valid[None]).to(device)
        veh_t = torch.from_numpy(veh[None].astype(np.float32)).to(device)
        perm_t = torch.from_numpy(perm[None]).to(device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                plan_t, scene_latent = model.solve_euler(
                    h_bev_t, h_ego_t, rp_t, rv_t, veh_t, perm_t, num_steps=5
                )

        plan = plan_t[0].cpu().float().numpy()  # [10, 4]: x, y, v, heading in ego frame
        scene_lat = scene_latent[0].cpu().float().numpy()  # [4, 64, 256]

        # 4. Controller action
        action, ctrl_detail = tracker.act(plan, ego_speed)
        steer_cmd, throttle_cmd = float(action[0]), float(action[1])

        # 5. Extract NPC details
        actors = list(sim.engine.get_objects(lambda o: getattr(o, "metadrive_type", None) == "VEHICLE").values())
        lead_npcs = []
        for a in actors:
            if a is sim.agent:
                continue
            ns, nlat = ego_lane.local_coordinates(a.position)
            rel_s = ns - ego_s
            eucl_d = float(np.linalg.norm(a.position - ego_pos))
            lead_npcs.append({
                "id": str(getattr(a, "id", "npc")),
                "pos": [round(float(a.position[0]), 3), round(float(a.position[1]), 3)],
                "speed_m_s": round(float(a.speed), 3),
                "longitudinal_gap_m": round(float(rel_s - (sim.agent.LENGTH + a.LENGTH) * 0.5), 3),
                "euclidean_dist_m": round(eucl_d, 3),
                "lateral_offset_in_ego_lane": round(float(nlat), 3),
                "length": float(a.LENGTH),
                "width": float(a.WIDTH),
            })
        # Sort by distance
        lead_npcs.sort(key=lambda x: x["euclidean_dist_m"])
        closest_npc = lead_npcs[0] if lead_npcs else None

        # 6. Analyze trajectory intersection with lead NPC
        plan_intersects_npc = False
        plan_min_dist_to_npc = float("inf")
        if closest_npc is not None:
            lead_pos_world = np.array(closest_npc["pos"], dtype=np.float32)
            # Transform plan waypoints from ego to world
            for wp in plan:
                wp_ego = wp[:2]
                wp_world = ego_to_world(wp_ego, ego_pos, ego_heading)
                d_wp = float(np.linalg.norm(wp_world - lead_pos_world))
                if d_wp < plan_min_dist_to_npc:
                    plan_min_dist_to_npc = d_wp
                # Bounding box collision threshold
                if d_wp < (closest_npc["length"] * 0.5 + 1.0):
                    plan_intersects_npc = True

        # Check trajectory lane boundary violation
        lane_width = ego_lane.width_at(ego_s) if hasattr(ego_lane, "width_at") else 3.5
        plan_max_lateral_m = float(np.max(np.abs(plan[:, 1])))
        plan_exceeds_lane = bool(plan_max_lateral_m > (lane_width * 0.5 + 0.5))

        # 7. Step simulation
        obs, r, tm, tc, info = checked_step(sim, action)
        provider.update(obs)

        # Record step
        record = {
            "step": step,
            "ego": {
                "pos": [round(float(ego_pos[0]), 3), round(float(ego_pos[1]), 3)],
                "heading_rad": round(float(ego_heading), 4),
                "speed_m_s": round(float(ego_speed), 3),
                "lane_s": round(float(ego_s), 3),
                "lane_lat": round(float(ego_lat), 3),
            },
            "maneuver_allowed": [bool(perm[0]), bool(perm[1])],
            "closest_npc": closest_npc,
            "plan_analysis": {
                "plan_speeds": [round(float(v), 3) for v in plan[:, 2]],
                "plan_x": [round(float(x), 3) for x in plan[:, 0]],
                "plan_y": [round(float(y), 3) for y in plan[:, 1]],
                "plan_headings": [round(float(h), 4) for h in plan[:, 3]],
                "target_speed_m_s": round(float(ctrl_detail.get("target_speed_m_s", 0.0)), 3),
                "target_point": [round(float(plan[0, 0]), 3), round(float(plan[0, 1]), 3)],
                "plan_intersects_npc": plan_intersects_npc,
                "plan_min_dist_to_npc_m": round(float(plan_min_dist_to_npc), 3) if np.isfinite(plan_min_dist_to_npc) else None,
                "plan_exceeds_lane": plan_exceeds_lane,
                "plan_max_lateral_m": round(plan_max_lateral_m, 3),
            },
            "control": {
                "steer": round(steer_cmd, 4),
                "throttle": round(throttle_cmd, 4),
                "feedforward": round(float(ctrl_detail.get("feedforward", 0.0)), 4),
            },
            "events": {
                "crash": bool(info.get("crash")),
                "out_of_road": bool(info.get("out_of_road")),
                "arrive_dest": bool(info.get("arrive_dest")),
                "route_completion": round(float(info.get("route_completion", 0.0)), 4),
            }
        }
        records.append(record)

        if info.get("crash"):
            crashed = True
            termination_reason = "crash"
            break
        if info.get("out_of_road"):
            offroad = True
            termination_reason = "out_of_road"
            break
        if info.get("arrive_dest"):
            arrived = True
            termination_reason = "arrived"
            break
        if tm or tc:
            termination_reason = "truncated"
            break

    sim.close()

    # Formulate diagnosis conclusion
    last_rec = records[-1]
    last_plan = last_rec["plan_analysis"]
    last_closest = last_rec["closest_npc"]

    print(f"Terminated at step {len(records)}: {termination_reason}")
    print(f"Final ego speed: {last_rec['ego']['speed_m_s']} m/s, Route Completion: {last_rec['events']['route_completion']}")
    if last_closest:
        print(f"Final closest NPC: gap={last_closest['longitudinal_gap_m']}m, speed={last_closest['speed_m_s']}m/s")
    print(f"Plan analysis at termination: intersects_npc={last_plan['plan_intersects_npc']}, exceeds_lane={last_plan['plan_exceeds_lane']}")
    print(f"Plan speeds (last step): {last_plan['plan_speeds']}")
    print(f"Plan x (last step): {last_plan['plan_x']}")
    print(f"Plan y (last step): {last_plan['plan_y']}")

    return {
        "case_name": case_name,
        "scene_seed": scene_seed,
        "train_seed": train_seed,
        "termination_reason": termination_reason,
        "total_steps": len(records),
        "arrived": arrived,
        "crashed": crashed,
        "offroad": offroad,
        "final_route_completion": last_rec["events"]["route_completion"],
        "records": records,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        configure_egl(1)
    except Exception:
        pass

    results = {}

    # Case 1: Blocked lane rear-end collision
    results["case1_blocked_rear_end"] = run_case_diagnosis(
        case_name="Blocked Lane Rear-End Collision",
        scene_seed=4300302,
        train_seed=20261022,
        max_steps=600,
    )

    # Case 2: Single-lane off-road violation
    results["case2_single_lane_offroad"] = run_case_diagnosis(
        case_name="Single Lane Off-Road Violation",
        scene_seed=4300303,
        train_seed=20261022,
        max_steps=600,
    )

    # Case 3: The unique safe timeout case
    results["case3_safe_timeout"] = run_case_diagnosis(
        case_name="Safe Car-Following Timeout",
        scene_seed=4300306,
        train_seed=20261023,
        max_steps=600,
    )

    # Save detailed JSON summary
    summary = {}
    for k, v in results.items():
        summary[k] = {
            "case_name": v["case_name"],
            "scene_seed": v["scene_seed"],
            "train_seed": v["train_seed"],
            "termination_reason": v["termination_reason"],
            "total_steps": v["total_steps"],
            "arrived": v["arrived"],
            "crashed": v["crashed"],
            "offroad": v["offroad"],
            "final_route_completion": v["final_route_completion"],
            "first_step": v["records"][0],
            "last_step": v["records"][-1],
            "mid_step": v["records"][len(v["records"]) // 2],
        }

    summary_file = OUT_DIR / "diagnosis_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Save full telemetry
    full_file = OUT_DIR / "diagnosis_full_telemetry.json"
    with open(full_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDiagnosis complete! Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
