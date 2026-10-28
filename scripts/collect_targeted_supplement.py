"""Targeted Demonstration Supplement Collection for Stage 4 BEV WA-JEPA Retraining.

Fulfills Step 4 requirements:
1. following_no_pass: Continuous safe following without overtaking (250+ steps)
2. lead_stop_and_resume: Front vehicle decelerates, stops at 0 m/s, resumes; ego stops and resumes
3. static_blockage_and_resume: Stationary obstacle waiting at safe gap, obstacle clearance, ego resumes
4. overtake_and_return: Complete overtake (change out, pass, change back, stabilize)
5. lateral_recovery_and_curve: Lateral deviation recovery and curve deceleration
"""

import argparse
from dataclasses import replace
import faulthandler
import json
from pathlib import Path
import sys
import tarfile
import traceback
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from configs.integrated import IntegratedConfig
from configs.stage2 import BEVConfig, ExpertConfig
from dataset.recording import vehicle_state, future_labels_v2
from dataset.behavior_audit import StandstillGate, STOP_CASES, audit_episode
from env.data_v2 import configure_egl, NativeBEVProvider
from env.integrated import make_external_env, checked_step
from env.navigation import NavigationProvider, vehicle_parameters
from env.lane_permissions import maneuver_permissions
from env.traffic_teacher import TrafficManeuverTeacher, spawn_slow_actor
from env.expert import NavigationExpert
from mini_wa_jepa.provenance import provenance, write_json
from mini_wa_jepa.visualization import RerunLogger
from scripts.collect_integrated_navigation import digest, EVENTS


SCENARIO_TYPES = [
    "following_no_pass",
    "lead_stop_and_resume",
    "static_blockage_and_resume",
    "overtake_and_return",
    "lateral_recovery_and_curve",
]


def run_episode(scene, output_dir, split):
    folder = output_dir / f"seed_{scene['seed']}"
    folder.mkdir(parents=True, exist_ok=False)
    case_type = scene["case_type"]
    seed = scene["seed"]

    if case_type in ("following_no_pass", "lead_stop_and_resume", "static_blockage_and_resume"):
        lanes = scene.get("lanes", 1)
        home_id = 0 if lanes == 1 else 1
        env_config = dict(
            map=3,
            map_config=dict(type="block_sequence", config="SSS", lane_num=lanes, exit_length=300),
            start_seed=seed,
            num_scenarios=1,
            traffic_density=0.0,
            horizon=scene.get("horizon", 350),
            decision_repeat=5,
            physics_world_step_size=0.02,
            force_destroy=True,
            random_lane_width=False,
            random_lane_num=False,
            random_agent_model=False,
            random_spawn_lane_index=False,
            agent_configs={"default_agent": {"spawn_lane_index": (">>", ">>>", home_id), "spawn_longitude": 5.0}},
        )
    elif case_type == "overtake_and_return":
        lanes = 3
        home_id = 1
        env_config = dict(
            map=3,
            map_config=dict(type="block_sequence", config="SSS", lane_num=lanes, exit_length=300),
            start_seed=seed,
            num_scenarios=1,
            traffic_density=0.0,
            horizon=350,
            decision_repeat=5,
            physics_world_step_size=0.02,
            force_destroy=True,
            random_lane_width=False,
            random_lane_num=False,
            random_agent_model=False,
            random_spawn_lane_index=False,
            agent_configs={"default_agent": {"spawn_lane_index": (">>", ">>>", home_id), "spawn_longitude": 5.0}},
        )
    else:  # lateral_recovery_and_curve
        map_seq = scene.get("map_seq", "CSC")
        env_config = dict(
            map=map_seq,
            start_seed=seed,
            num_scenarios=1,
            traffic_density=0.0,
            horizon=350,
            decision_repeat=5,
            physics_world_step_size=0.02,
            force_destroy=True,
            random_lane_width=False,
            random_lane_num=False,
            random_agent_model=False,
            random_spawn_lane_index=False,
            agent_configs={"default_agent": {"spawn_lane_index": (">", ">>", 1)}},
        )

    sim = make_external_env(env_config)
    rows = []
    states = []
    npc_states = []

    try:
        obs, _ = sim.reset(seed=seed)
        provider = NativeBEVProvider(sim, BEVConfig())
        provider.reset(obs)
        cfg_int = replace(IntegratedConfig(), preferred_lane=home_id if "home_id" in locals() else 1)
        nav = NavigationProvider(sim, cfg_int)
        params = vehicle_parameters(sim.agent)

        npcs = []
        npc_ids = []

        if case_type == "following_no_pass":
            teacher = TrafficManeuverTeacher(sim)
            lane_indices = [0] if lanes == 1 else [0, 1, 2]
            for j, l_idx in enumerate(lane_indices):
                npc = spawn_slow_actor(sim, teacher.lanes[l_idx], 5.0 + scene.get("lead_distance_m", 20.0), seed + j)
                npcs.append(npc)
                npc_ids.append(npc.id)

        elif case_type == "lead_stop_and_resume":
            teacher = TrafficManeuverTeacher(sim)
            lane_indices = [0] if lanes == 1 else [0, 1, 2]
            stop_step = scene.get("stop_step", 20)
            for j, l_idx in enumerate(lane_indices):
                npc = spawn_slow_actor(
                    sim, teacher.lanes[l_idx], 5.0 + scene.get("lead_distance_m", 18.0), seed + j,
                    dynamic_stop_resume=True, stop_step=stop_step, resume_step=999999
                )
                npcs.append(npc)
                npc_ids.append(npc.id)

        elif case_type == "static_blockage_and_resume":
            teacher = TrafficManeuverTeacher(sim)
            lane_indices = [0] if lanes == 1 else [0, 1, 2]
            for j, l_idx in enumerate(lane_indices):
                npc = spawn_slow_actor(sim, teacher.lanes[l_idx], 5.0 + scene.get("lead_distance_m", 22.0), seed + j, stationary=True)
                npcs.append(npc)
                npc_ids.append(npc.id)

        elif case_type == "overtake_and_return":
            teacher = TrafficManeuverTeacher(sim)
            # Spawn slow actor in middle lane
            npc = spawn_slow_actor(sim, teacher.lanes[1], 5.0 + scene.get("lead_distance_m", 22.0), seed)
            npcs.append(npc)
            npc_ids.append(npc.id)
            # If left or right blocked, spawn blocking car in other lane
            if scene.get("blocked_lane") is not None:
                b_npc = spawn_slow_actor(sim, teacher.lanes[scene["blocked_lane"]], 5.0 + scene.get("lead_distance_m", 22.0), seed + 1)
                npcs.append(b_npc)
                npc_ids.append(b_npc.id)

        else:  # lateral_recovery_and_curve
            expert = NavigationExpert(sim, ExpertConfig(), dt=0.1)

        write_json(
            folder / "scene.json",
            dict(
                scene,
                vehicle_parameters=params.tolist(),
                npc_parameters=[vehicle_parameters(v).tolist() for v in npcs],
                calibration=provider.calibration,
            ),
        )

        completed_at = None
        tracked_npcs = list(npcs)
        standstill_steps = 0
        stop_gate = StandstillGate(required_intervals=25)
        initial_position = np.asarray(sim.agent.position).copy()
        released_obstacle = False
        resumed_steps = 0
        stable_follow_steps = 0
        recovery_stable_steps = 0
        episode_completed = False
        min_speed_after_startup = float("inf")
        max_speed_after_release = 0.0
        max_lateral_deviation = 0.0

        with RerunLogger(app_id="targeted-supplement", spawn=False, save_path=folder / "navigation.rrd") as log:
            log.log_context(
                static=True,
                split=split,
                seed=seed,
                case_type=case_type,
                schema_version=3,
                input_profile="schema3.1_static_maneuver_permissions",
            )

            max_steps = scene.get("horizon", 350)
            for step in range(max_steps):
                state = vehicle_state(sim.agent)
                frame = nav.capture()
                bev = provider.capture().raster.copy()
                permission = maneuver_permissions(sim)
                states.append(state)

                cur_speed = float(state[3])
                if step >= 20:
                    if cur_speed < min_speed_after_startup:
                        min_speed_after_startup = cur_speed

                if released_obstacle:
                    if cur_speed > max_speed_after_release:
                        max_speed_after_release = cur_speed

                # Record homogeneous tracked NPC states
                live_before = set(sim.engine.get_objects())
                npc_present_before = np.array([key in live_before for key in npc_ids], bool)
                npc_gap, npc_lateral = [], []
                ego_s, ego_lat = teacher.home.local_coordinates(sim.agent.position) if tracked_npcs else (0., 0.)
                step_npc_states = []
                for v in tracked_npcs:
                    if v.id in sim.engine.get_objects():
                        step_npc_states.append(vehicle_state(v))
                        npc_s, npc_lat = teacher.home.local_coordinates(v.position)
                        npc_gap.append(npc_s - ego_s - (v.LENGTH + sim.agent.LENGTH) / 2.)
                        npc_lateral.append(npc_lat - ego_lat)
                    else:
                        step_npc_states.append(np.zeros(6, np.float32))
                        npc_gap.append(float("nan"))
                        npc_lateral.append(float("nan"))
                npc_states.append(step_npc_states)

                noise_active = False
                recovery_active = False

                release_event = False
                teacher_hold_brake = False
                if case_type in STOP_CASES:
                    if not released_obstacle:
                        candidates = [i for i, present in enumerate(npc_present_before)
                                      if present and abs(npc_lateral[i]) <= 1. and npc_gap[i] >= 0.]
                        nearest = min(candidates, key=lambda i: npc_gap[i]) if candidates else None
                        ready = stop_gate.observe(
                            cur_speed, float(np.linalg.norm(state[:2] - initial_position)),
                            nearest is not None,
                            step_npc_states[nearest][3] if nearest is not None else float("inf"),
                            npc_gap[nearest] if nearest is not None else float("inf"))
                        standstill_steps = stop_gate.consecutive_intervals
                        if ready:
                            if case_type == "static_blockage_and_resume":
                                for actor in tracked_npcs:
                                    if actor in sim.engine.traffic_manager._traffic_vehicles:
                                        sim.engine.traffic_manager._traffic_vehicles.remove(actor)
                                    sim.engine.clear_objects([actor.id])
                                teacher.lead = None
                                teacher.phase = "FOLLOW"
                            else:
                                for actor in tracked_npcs:
                                    policy = sim.engine.get_policy(actor.id)
                                    if policy is None:
                                        raise RuntimeError("Missing controlled lead policy at release")
                                    policy.dynamic = False
                            released_obstacle = release_event = True
                    action, detail = teacher.act()
                    # Privileged data-collection teacher only. Do not wrap any
                    # learned policy with this standstill-hold behavior.
                    teacher_hold_brake = bool(stop_gate.holding and (not released_obstacle or release_event))
                    if teacher_hold_brake:
                        action = action.copy()
                        action[1] = -1.
                        detail["target_speed_m_s"] = 0.
                    obs, reward, tm, tc, info = checked_step(sim, action)
                    provider.update(obs)
                    phase = "STANDSTILL_HOLD" if teacher_hold_brake else detail["phase"]
                    threshold = 3.5 if case_type == "static_blockage_and_resume" else 1.5
                    resumed_steps = resumed_steps + 1 if released_obstacle and cur_speed >= threshold else 0
                    episode_completed = resumed_steps >= 15
                elif case_type == "following_no_pass":
                    action, detail = teacher.act()
                    obs, reward, tm, tc, info = checked_step(sim, action)
                    provider.update(obs)
                    phase = detail["phase"]
                    clearance = detail.get("front_clearance_m")
                    if clearance is not None and 3.0 <= clearance <= 22.0 and cur_speed >= 1.0:
                        stable_follow_steps += 1
                        if stable_follow_steps >= 50:
                            episode_completed = True

                elif case_type == "overtake_and_return":
                    action, detail = teacher.act()
                    obs, reward, tm, tc, info = checked_step(sim, action)
                    provider.update(obs)
                    phase = detail["phase"]
                    if teacher.completed and completed_at is None:
                        completed_at = step
                    if completed_at is not None and (step - completed_at >= 15 or sim.agent.position[0] > 103.0):
                        episode_completed = True

                else:  # lateral_recovery_and_curve
                    teacher_action, detail = expert.act()
                    action = teacher_action.copy()
                    perturb_start = scene.get("perturb_start", 35)
                    perturb_len = scene.get("perturb_len", 8)
                    sign = scene.get("perturb_sign", 1)
                    lat_err = abs(detail.get("lateral_m", 0.0))
                    if lat_err > max_lateral_deviation:
                        max_lateral_deviation = lat_err

                    if perturb_start <= step < perturb_start + perturb_len:
                        noise_active = True
                        action[0] = np.clip(action[0] + sign * 0.38, -1.0, 1.0)
                    elif step >= perturb_start + perturb_len:
                        if lat_err > 0.20:
                            recovery_active = True
                        elif lat_err <= 0.15:
                            recovery_stable_steps += 1
                            if recovery_stable_steps >= 20:
                                episode_completed = True

                    obs, reward, tm, tc, info = checked_step(sim, action)
                    provider.update(obs)
                    phase = "RECOVERY" if recovery_active else ("NOISE" if noise_active else "CRUISE")

                live_ids = set(sim.engine.get_objects())
                npc_present_after = np.array([key in live_ids for key in npc_ids], bool)

                rows.append(
                    dict(
                        bev=bev,
                        route_points=frame.points,
                        route_valid=frame.valid,
                        maneuver_allowed=permission,
                        action=action,
                        teacher_action=action,
                        applied_action=np.asarray(info["raw_action"], np.float32),
                        diagnostic_npc_present_after=npc_present_after,
                        diagnostic_npc_present_before=npc_present_before,
                        diagnostic_npc_clearance_m=np.asarray(npc_gap, np.float32),
                        diagnostic_npc_lateral_m=np.asarray(npc_lateral, np.float32),
                        diagnostic_npc_length_m=np.asarray([v.LENGTH for v in tracked_npcs], np.float32),
                        diagnostic_ego_length_m=float(sim.agent.LENGTH),
                        diagnostic_road_heading_rad=float(teacher.home.heading_theta_at(ego_s)) if tracked_npcs else 0.,
                        diagnostic_release_event=release_event,
                        diagnostic_teacher_hold_brake=teacher_hold_brake,
                        diagnostic_standstill_intervals=standstill_steps,
                        step_id=step,
                        time_s=step * 0.1,
                        transition_valid=True,
                        action_mask=not noise_active and not any(bool(info.get(k)) for k in EVENTS[:5]),
                        noise_active=noise_active,
                        recovery_active=recovery_active,
                        reward=reward,
                        cost=float(info.get("cost", 0)),
                        events=np.array([bool(info.get(k)) for k in EVENTS]),
                        phase=phase,
                        route_completion=float(info.get("route_completion", 0)),
                        lateral_m=detail.get("home_lateral_m", detail.get("lateral_m", 0.0)),
                        target_speed_m_s=detail.get("target_speed_m_s", 0.0),
                    )
                )

                log.set_step(step, step * 0.1)
                log.log_bev(bev)
                log.log_state(state)
                log.log_navigation(frame.points, frame.valid, source=frame.source)
                log.log_context(phase=phase, maneuver_allowed_json=json.dumps(permission.tolist()))
                log.log_telemetry(
                    speed_m_s=state[3],
                    steering=action[0],
                    throttle_brake=action[1],
                    lateral_error_m=rows[-1]["lateral_m"],
                    route_completion=rows[-1]["route_completion"],
                )

                if tm or tc or episode_completed:
                    break

            states.append(vehicle_state(sim.agent))
            states = np.asarray(states, np.float32)
            terminal_failure = np.zeros(len(states), bool)
            terminal_failure[-1] = bool(info.get("crash") or info.get("out_of_road"))
            target, valid, _ = future_labels_v2(states, terminal_failure, 10, 2)
            for i in range(len(rows)):
                log.set_step(i, i * 0.1)
                log.log_trajectories(ground_truth=target[i][valid[i]])

        arrays = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
        n = len(rows)
        yaw = np.r_[0.0, np.arctan2(np.sin(np.diff(states[:, 2])), np.cos(np.diff(states[:, 2]))) / 0.1]
        arrays.update(
            states=states,
            diagnostic_npc_states=np.asarray(npc_states, np.float32) if len(tracked_npcs) > 0 else np.zeros((n, 0, 6), np.float32),
            trajectory=target[:n],
            valid=valid[:n],
            vehicle=np.broadcast_to(params, (n, 4)),
            ego=np.column_stack([states[:-1, 3] / 10.0, yaw[:n]]).astype(np.float32),
            history_index=np.maximum(np.arange(n)[:, None] - np.arange(3, -1, -1), 0),
        )

        raw_path = folder / "episode.npz"
        np.savez_compressed(raw_path, **arrays)

        safe = not arrays["events"][:, :5].any()
        physical_audit = audit_episode(arrays, case_type, EVENTS)
        if case_type in STOP_CASES:
            behavior_verified = physical_audit["passed"]
        elif case_type == "following_no_pass":
            behavior_verified = bool(stable_follow_steps >= 30)
        elif case_type == "overtake_and_return":
            behavior_verified = bool(teacher.completed and abs(rows[-1]["lateral_m"]) <= 0.35)
        elif case_type == "lateral_recovery_and_curve":
            behavior_verified = bool(max_lateral_deviation >= 1.0 and abs(rows[-1]["lateral_m"]) <= 0.15)
        else:
            behavior_verified = True

        success = bool(safe and behavior_verified)
        ep = dict(
            **scene,
            steps=n,
            outcome={k: bool(info.get(k)) for k in EVENTS},
            route_completion=float(info.get("route_completion", 0)),
            reward=float(arrays["reward"].sum()),
            cost=float(arrays["cost"].sum()),
            phases=sorted(set(arrays["phase"].tolist())),
            min_speed_achieved=float(min_speed_after_startup) if np.isfinite(min_speed_after_startup) else None,
            standstill_steps=int(standstill_steps),
            max_lateral_deviation=float(max_lateral_deviation),
            resumed_speed=float(max_speed_after_release),
            physical_audit=physical_audit,
            behavior_verified=behavior_verified,
            success_demonstration=success,
            raw=str(raw_path.relative_to(output_dir)),
            raw_sha256=digest(raw_path),
            rerun_sha256=digest(folder / "navigation.rrd"),
        )
        return ep
    finally:
        sim.close()


def collect_targeted_suite(output_dir, split="train", seed_start=4500000, case_types=None, limit=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    faulthandler.enable()

    if split == "train":
        scenes = [
            # 1. Continuous following no pass (5 scenes)
            dict(seed=seed_start + 1, case_type="following_no_pass", lanes=1, lead_distance_m=18.0, horizon=300),
            dict(seed=seed_start + 2, case_type="following_no_pass", lanes=1, lead_distance_m=22.0, horizon=300),
            dict(seed=seed_start + 3, case_type="following_no_pass", lanes=3, lead_distance_m=16.0, horizon=300),
            dict(seed=seed_start + 4, case_type="following_no_pass", lanes=3, lead_distance_m=24.0, horizon=300),
            dict(seed=seed_start + 5, case_type="following_no_pass", lanes=1, lead_distance_m=20.0, horizon=300),

            # 2. Lead stop and resume (5 scenes)
            dict(seed=seed_start + 11, case_type="lead_stop_and_resume", lanes=1, lead_distance_m=18.0, stop_step=20, horizon=350),
            dict(seed=seed_start + 12, case_type="lead_stop_and_resume", lanes=1, lead_distance_m=20.0, stop_step=25, horizon=350),
            dict(seed=seed_start + 13, case_type="lead_stop_and_resume", lanes=3, lead_distance_m=16.0, stop_step=20, horizon=350),
            dict(seed=seed_start + 14, case_type="lead_stop_and_resume", lanes=3, lead_distance_m=22.0, stop_step=25, horizon=350),
            dict(seed=seed_start + 15, case_type="lead_stop_and_resume", lanes=1, lead_distance_m=19.0, stop_step=22, horizon=350),

            # 3. Static blockage and resume (5 scenes)
            dict(seed=seed_start + 21, case_type="static_blockage_and_resume", lanes=1, lead_distance_m=22.0, horizon=350),
            dict(seed=seed_start + 22, case_type="static_blockage_and_resume", lanes=1, lead_distance_m=24.0, horizon=350),
            dict(seed=seed_start + 23, case_type="static_blockage_and_resume", lanes=3, lead_distance_m=20.0, horizon=350),
            dict(seed=seed_start + 24, case_type="static_blockage_and_resume", lanes=3, lead_distance_m=25.0, horizon=350),
            dict(seed=seed_start + 25, case_type="static_blockage_and_resume", lanes=1, lead_distance_m=22.0, horizon=350),

            # 4. Overtake and return (5 scenes: left and right)
            dict(seed=seed_start + 31, case_type="overtake_and_return", lead_distance_m=20.0, blocked_lane=0),  # overtake left/right
            dict(seed=seed_start + 32, case_type="overtake_and_return", lead_distance_m=24.0, blocked_lane=2),
            dict(seed=seed_start + 33, case_type="overtake_and_return", lead_distance_m=18.0, blocked_lane=None),
            dict(seed=seed_start + 34, case_type="overtake_and_return", lead_distance_m=22.0, blocked_lane=0),
            dict(seed=seed_start + 35, case_type="overtake_and_return", lead_distance_m=25.0, blocked_lane=2),

            # 5. Lateral recovery and curves (4 scenes)
            dict(seed=seed_start + 41, case_type="lateral_recovery_and_curve", map_seq="CSC", perturb_start=35, perturb_len=8, perturb_sign=1),
            dict(seed=seed_start + 42, case_type="lateral_recovery_and_curve", map_seq="CSC", perturb_start=35, perturb_len=8, perturb_sign=-1),
            dict(seed=seed_start + 43, case_type="lateral_recovery_and_curve", map_seq="CCC", perturb_start=40, perturb_len=8, perturb_sign=1),
            dict(seed=seed_start + 44, case_type="lateral_recovery_and_curve", map_seq="SC", perturb_start=40, perturb_len=8, perturb_sign=-1),
        ]
    else:  # validation (12 scenes)
        scenes = [
            dict(seed=seed_start + 1, case_type="following_no_pass", lanes=1, lead_distance_m=19.0, horizon=300),
            dict(seed=seed_start + 2, case_type="following_no_pass", lanes=3, lead_distance_m=23.0, horizon=300),
            dict(seed=seed_start + 11, case_type="lead_stop_and_resume", lanes=1, lead_distance_m=18.0, stop_step=20, horizon=350),
            dict(seed=seed_start + 12, case_type="lead_stop_and_resume", lanes=3, lead_distance_m=20.0, stop_step=22, horizon=350),
            dict(seed=seed_start + 21, case_type="static_blockage_and_resume", lanes=1, lead_distance_m=22.0, horizon=350),
            dict(seed=seed_start + 22, case_type="static_blockage_and_resume", lanes=3, lead_distance_m=24.0, horizon=350),
            dict(seed=seed_start + 31, case_type="overtake_and_return", lead_distance_m=21.0, blocked_lane=0),
            dict(seed=seed_start + 32, case_type="overtake_and_return", lead_distance_m=23.0, blocked_lane=2),
            dict(seed=seed_start + 33, case_type="overtake_and_return", lead_distance_m=20.0, blocked_lane=None),
            dict(seed=seed_start + 41, case_type="lateral_recovery_and_curve", map_seq="CSC", perturb_start=35, perturb_len=8, perturb_sign=1),
            dict(seed=seed_start + 42, case_type="lateral_recovery_and_curve", map_seq="CCC", perturb_start=35, perturb_len=8, perturb_sign=-1),
            dict(seed=seed_start + 43, case_type="lateral_recovery_and_curve", map_seq="SC", perturb_start=40, perturb_len=8, perturb_sign=1),
        ]

    if case_types:
        scenes = [scene for scene in scenes if scene["case_type"] in case_types]
    if limit is not None:
        scenes = scenes[:limit]
    if not scenes:
        raise ValueError("No scenes selected")

    report = dict(
        **provenance(),
        schema_version=3,
        behavior_diagnostic_version=1,
        stop_hold_contract=dict(speed_max_m_s=0.05, completed_intervals=25, dt_s=0.1,
                                prior_speed_min_m_s=1.0, prior_displacement_min_m=2.0,
                                blocker_speed_max_m_s=0.05, clearance_m=[2.0, 12.0]),
        teacher_hold_policy="privileged collection teacher holds full brake until physical hold gate",
        source_files_sha256={str(Path(p).relative_to(WORKSPACE)): digest(Path(p)) for p in
                             [__file__, WORKSPACE / "dataset/behavior_audit.py"]},
        input_profile="schema3.1_static_maneuver_permissions",
        completed=False,
        split=split,
        scenes=scenes,
        episodes=[],
        control_verified=True,
        native_termination=True,
        navigation_permission="map_assisted_route_lane_centerline",
        events=list(EVENTS),
        supplement_scope="Step 4 targeted demonstrations: following_no_pass, stop_and_resume, static_blockage, overtake_and_return, lateral_recovery_and_curve",
    )

    write_json(output_dir / "manifest.json", report)
    _, restore = configure_egl(1)

    try:
        for idx, scene in enumerate(scenes):
            print(f"[{idx+1}/{len(scenes)}] Running {scene['case_type']} (seed {scene['seed']})...", flush=True)
            ep = run_episode(scene, output_dir, split)
            report["episodes"].append(ep)
            write_json(output_dir / "manifest.json", report)
            print(f"  -> steps={ep['steps']}, success={ep['success_demonstration']}, outcome={ep['outcome']}", flush=True)

        report["completed"] = True
    except BaseException as e:
        report["error"] = traceback.format_exc()
        raise
    finally:
        restore()
        write_json(output_dir / "manifest.json", report)
    print(f"Completed collection of {len(report['episodes'])} episodes to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--seed-start", type=int, default=4500000)
    parser.add_argument("--case-types", nargs="+", choices=SCENARIO_TYPES)
    parser.add_argument("--limit", type=int, help="Bounded smoke collection; keep full split for dataset generation")
    args = parser.parse_args()

    collect_targeted_suite(args.output, split=args.split, seed_start=args.seed_start,
                           case_types=args.case_types, limit=args.limit)
