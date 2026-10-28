"""Train-only EGL preflight experiment for Stage 7 risk data collection.

Verifies that real simulator crash and out-of-road events can be reliably captured
under controlled candidate commands without hidden controller perturbations.
Strictly checks same-prefix replay determinism and logs termination censoring masks.
"""
import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import tarfile
import time
import traceback

import numpy as np

from configs.stage2 import Stage2Config
from configs.stage7 import Stage7Config, STAGE7_SPLITS, STAGE7_RISK_CANDIDATE_SPECS
from controller.trajectory_tracker import TrajectoryTracker
from dataset.action_branches import remaining_command, realized_waypoints
from dataset.recording import vehicle_state
from env.data_v2 import NativeBEVProvider, configure_egl, make_env
from env.expert import NavigationExpert
from mini_wa_jepa.geometry import world_to_ego, wrap_angle
from mini_wa_jepa.provenance import provenance, write_json


TYPED_CRASH_KEYS = ('crash_vehicle', 'crash_object', 'crash_building', 'crash_human', 'crash_sidewalk')


def scene_state(sim):
    """Capture sorted state array of all vehicles in simulation."""
    objects = sim.engine.get_objects(lambda obj: getattr(obj, 'metadrive_type', None) == 'VEHICLE').values()
    values = [vehicle_state(obj) for obj in objects]
    values.sort(key=lambda row: tuple(np.round(row[:2], 3)))
    return np.asarray(values, np.float32)


def compute_observed_lateral_left(expert, car_position):
    """Compute vehicle's current lateral offset relative to route centerline (positive to the left).

    MetaDrive's lane.local_coordinates uses lateral positive to the right.
    Our project's ego coordinates use y positive to the left.
    To eliminate sign ambiguity, we explicitly project via the road's left-normal:
    left_normal = [-sin(road_heading), cos(road_heading)].
    """
    route_point, _, _, road_heading = expert.route(0.0)
    left_normal = np.array([-np.sin(road_heading), np.cos(road_heading)], dtype=np.float32)
    delta = np.asarray(car_position[:2], dtype=np.float32) - np.asarray(route_point[:2], dtype=np.float32)
    return float(np.dot(delta, left_normal))


def make_risk_candidates(expert, candidate_specs=STAGE7_RISK_CANDIDATE_SPECS, observed_lateral_left=0.0):
    """Construct 10x4 candidate trajectory commands in vehicle coordinate frame.

    Displacement is relative to current roadside position: lateral(t) = observed_lateral_left + offset * h(t).
    No hidden actions or execution-time overrides; candidate is executed strictly by the tracker.
    """
    car = expert.sim.agent
    times = np.arange(1, 11) * 0.2
    commands = {}

    for name, offset, target_speed, shift_time_s in candidate_specs:
        rows = []
        speed = float(target_speed)
        shift_t = max(float(shift_time_s), 0.2)
        for t in times:
            distance = speed * t
            point, _, _, angle = expert.route(float(distance))
            left_normal = np.array([-np.sin(angle), np.cos(angle)], dtype=np.float32)

            # Smooth cubic polynomial transition to offset
            q = min(1.0, t / shift_t)
            h = 3 * q * q - 2 * q * q * q
            h_prime = (6 * q - 6 * q * q) / shift_t if t < shift_t else 0.0

            # Lateral position is relative to observed roadside offset
            lateral = observed_lateral_left + offset * h
            slope = (offset * h_prime) / max(speed, 1.0)
            shifted_point = np.asarray(point) + lateral * left_normal
            xy = world_to_ego(shifted_point, car.position, car.heading_theta)
            waypoint_heading = float(wrap_angle(angle + np.arctan(slope) - car.heading_theta))
            rows.append([xy[0], xy[1], speed, waypoint_heading])
        commands[name] = np.asarray(rows, np.float32)

    return commands


def compute_risk_targets(flags_array):
    """Compute risk target labels and censoring validity masks.

    Rule: label_valid = event_seen OR full_2s_completed.
    If rollout terminated prematurely without the event, the event is censored.
    """
    flags_arr = np.asarray(flags_array, dtype=bool)
    crash_seen = bool(np.any(flags_arr[:, 0])) if len(flags_arr) > 0 else False
    offroad_seen = bool(np.any(flags_arr[:, 1])) if len(flags_arr) > 0 else False
    full_2s = bool(len(flags_arr) == 20)

    crash_valid = bool(crash_seen or full_2s)
    offroad_valid = bool(offroad_seen or full_2s)

    crash_target = float(crash_seen)
    offroad_target = float(offroad_seen)

    risk_target = np.array([crash_target, offroad_target], dtype=np.float32)
    risk_mask = np.array([crash_valid, offroad_valid], dtype=bool)
    return risk_target, risk_mask, crash_seen, offroad_seen, full_2s


def collect_preflight(output_dir, config, map_names=('S', 'SC'), seeds_per_map=1, prefix_steps=(50,), seed_offset=0):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    report = dict(
        **provenance(config),
        stage=7,
        preflight=True,
        completed=False,
        split='train',
        requested_maps=list(map_names),
        seeds_per_map=seeds_per_map,
        prefix_steps=list(prefix_steps),
        seed_offset=seed_offset,
        groups=[],
        summary_counts=dict(
            total_branches=0,
            crash_positives=0,
            offroad_positives=0,
            crash_valid=0,
            offroad_valid=0,
            full_2s_completed=0,
        ),
        failures=[],
    )
    write_json(output / 'manifest.json', report)

    with tarfile.open(output / 'source_snapshot.tar.gz', 'w:gz') as archive:
        for source in report['source_files']:
            archive.add(source, arcname=source)
        for doc in ('GOAL.md', 'AGENTS.md', 'docs/STAGE7_PROTOCOL.md'):
            archive.add(doc, arcname=doc)

    _, restore = configure_egl(1)
    cfg = Stage2Config()
    started = time.perf_counter()

    train_spec = STAGE7_SPLITS['train']
    candidate_specs = STAGE7_RISK_CANDIDATE_SPECS

    try:
        maps_with_idx = [(i, m) for i, m in enumerate(train_spec['maps']) if m in map_names]
        if not maps_with_idx:
            raise ValueError(f'No matching train maps in requested: {map_names}')

        for map_i, map_spec in maps_with_idx:
            base_seed = train_spec['seed_start'] + map_i * 10000
            sim = make_env(
                map_spec,
                base_seed,
                cfg,
                num_scenarios=1000,
                control_dt_s=config.control_dt_s,
                environment_overrides={'force_destroy': config.force_destroy_between_resets},
            )
            try:
                for s_idx in range(seeds_per_map):
                    seed_idx = s_idx + seed_offset
                    seed = base_seed + seed_idx
                    is_right_turn_context = (seed_idx % 2 == 1)

                    for anchor_step in prefix_steps:
                        group_id = f'train_{map_spec}_{seed}_step{anchor_step}'
                        folder = output / group_id
                        folder.mkdir(parents=True, exist_ok=False)

                        prefix_actions = None
                        prefix_flags = None
                        prefix_typed = None
                        reference_history = None
                        reference_states = None
                        reference_scene = None
                        commands = None

                        group_row = dict(
                            id=group_id,
                            split='train',
                            map_spec=map_spec,
                            seed=seed,
                            seed_index=seed_idx,
                            context_type='right_turn_roadside' if is_right_turn_context else 'center_expert',
                            anchor_step=anchor_step,
                            branches=[],
                        )

                        for ci, (cand_name, offset_m, speed_ms, shift_s) in enumerate(candidate_specs):
                            obs, _ = sim.reset(seed=seed)
                            provider = NativeBEVProvider(sim, cfg.bev)
                            provider.reset(obs)
                            expert = NavigationExpert(sim, cfg.expert, config.control_dt_s)

                            states = [vehicle_state(sim.agent)]
                            bevs = [provider.capture().raster.copy()]
                            actions = []
                            flags_prefix = []
                            typed_prefix = []
                            tm = tc = False

                            if prefix_actions is None:
                                # First candidate in context: generate prefix
                                if not is_right_turn_context:
                                    # Even seed: standard expert prefix up to anchor_step
                                    for step in range(anchor_step):
                                        action = expert.act()[0]
                                        actions.append(action.copy())
                                        obs, _, tm, tc, info = sim.step(action)
                                        provider.update(obs)
                                        states.append(vehicle_state(sim.agent))
                                        bevs.append(provider.capture().raster.copy())
                                        flags_prefix.append([info['crash'], info['out_of_road'], tm, tc])
                                        typed_prefix.append([info[k] for k in TYPED_CRASH_KEYS])
                                        if tm or tc:
                                            raise RuntimeError(f'Prefix terminated early: {group_id} at step {step} ({info})')
                                else:
                                    # Odd seed: expert up to anchor_step - 10, then 10 steps roadside maneuver
                                    expert_steps = max(0, anchor_step - config.roadside_prefix_steps)
                                    for step in range(expert_steps):
                                        action = expert.act()[0]
                                        actions.append(action.copy())
                                        obs, _, tm, tc, info = sim.step(action)
                                        provider.update(obs)
                                        states.append(vehicle_state(sim.agent))
                                        bevs.append(provider.capture().raster.copy())
                                        flags_prefix.append([info['crash'], info['out_of_road'], tm, tc])
                                        typed_prefix.append([info[k] for k in TYPED_CRASH_KEYS])
                                        if tm or tc:
                                            raise RuntimeError(f'Prefix terminated early during expert: {group_id} at step {step} ({info})')

                                    # Roadside maneuver for last 10 steps BEFORE anchor
                                    roadside_cmd = make_risk_candidates(
                                        expert,
                                        [('roadside', config.roadside_prefix_offset_m, config.roadside_prefix_speed_ms, config.roadside_prefix_shift_s)],
                                        observed_lateral_left=compute_observed_lateral_left(expert, sim.agent.position),
                                    )['roadside']
                                    origin_state = vehicle_state(sim.agent)
                                    p_tracker = TrajectoryTracker(sim.agent, config.control_dt_s)

                                    for step in range(config.roadside_prefix_steps):
                                        rem = remaining_command(roadside_cmd, origin_state, vehicle_state(sim.agent), step * config.control_dt_s)
                                        action, _ = p_tracker.act(rem, sim.agent.speed)
                                        actions.append(action.copy())
                                        obs, _, tm, tc, info = sim.step(action)
                                        provider.update(obs)
                                        states.append(vehicle_state(sim.agent))
                                        bevs.append(provider.capture().raster.copy())
                                        flags_prefix.append([info['crash'], info['out_of_road'], tm, tc])
                                        typed_prefix.append([info[k] for k in TYPED_CRASH_KEYS])
                                        if tm or tc:
                                            raise RuntimeError(f'Roadside prefix terminated early: {group_id} at step {step} ({info})')

                                prefix_actions = actions
                                prefix_flags = flags_prefix
                                prefix_typed = typed_prefix
                                reference_history = np.stack(bevs[-4:])
                                reference_states = np.stack(states[-4:])
                                reference_scene = scene_state(sim)

                                # Compute observed lateral left at anchor
                                obs_lat_left = compute_observed_lateral_left(expert, sim.agent.position)
                                commands = make_risk_candidates(expert, candidate_specs, observed_lateral_left=obs_lat_left)

                                np.savez_compressed(
                                    folder / 'context.npz',
                                    history=reference_history,
                                    ego=np.zeros((4, 2), np.float32),
                                    states=reference_states,
                                    commands=np.stack(list(commands.values())),
                                    candidate_names=list(commands.keys()),
                                    prefix_actions=np.asarray(prefix_actions),
                                    prefix_flags=np.asarray(prefix_flags),
                                    prefix_typed=np.asarray(prefix_typed),
                                    scene_state=reference_scene,
                                    observed_lateral_left=obs_lat_left,
                                )
                            else:
                                # Subsequent candidates: replay exact saved prefix actions
                                for step in range(anchor_step):
                                    action = prefix_actions[step]
                                    obs, _, tm, tc, info = sim.step(action)
                                    provider.update(obs)
                                    states.append(vehicle_state(sim.agent))
                                    bevs.append(provider.capture().raster.copy())
                                    if tm or tc:
                                        raise RuntimeError(f'Prefix replay terminated early: {group_id} at step {step}')

                            history = np.stack(bevs[-4:])
                            past = np.stack(states[-4:])
                            ego = np.zeros((4, 2), np.float32)
                            ego[:, 0] = past[:, 3] / 10.0
                            ego[:, 1] = wrap_angle(np.stack(states[-4:])[:, 2] - np.stack(states[-5:-1])[:, 2]) / 0.1

                            # Replay determinism verification
                            state_error = float(np.max(np.abs(reference_states - past)))
                            current_scene = scene_state(sim)
                            scene_bounds = np.array(
                                [config.replay_scene_position_m] * 2
                                + [config.replay_scene_heading_rad, config.replay_scene_speed_m_s]
                                + [config.replay_scene_velocity_m_s] * 2
                            )
                            scene_ok = reference_scene.shape == current_scene.shape and bool(
                                np.all(np.abs(reference_scene - current_scene) <= scene_bounds)
                            )
                            scene_error = (
                                float(np.max(np.abs(reference_scene - current_scene)))
                                if reference_scene.shape == current_scene.shape
                                else float('inf')
                            )
                            bev_diff = float(np.mean(reference_history != history))

                            if state_error > config.replay_state_atol or not scene_ok or bev_diff > config.replay_bev_different_fraction:
                                np.savez_compressed(
                                    folder / f'replay_mismatch_{cand_name}.npz',
                                    history=history,
                                    states=past,
                                    reference_history=reference_history,
                                    reference_states=reference_states,
                                    scene=current_scene,
                                    reference_scene=reference_scene,
                                )
                                raise RuntimeError(
                                    f'Replay outside tolerance for {group_id} cand {cand_name}: '
                                    f'state_error={state_error}, scene_error={scene_error}, bev_diff={bev_diff}'
                                )

                            # Branch rollout execution
                            anchor_state = states[-1].copy()
                            tracker = TrajectoryTracker(sim.agent, config.control_dt_s)
                            future = []
                            executed = []
                            flags = []
                            typed = []
                            branch_states = [anchor_state]
                            cmd = commands[cand_name]

                            for step in range(20):
                                remaining = remaining_command(cmd, anchor_state, vehicle_state(sim.agent), step * 0.1)
                                action, _ = tracker.act(remaining, sim.agent.speed)
                                obs, _, tm, tc, info = sim.step(action)
                                provider.update(obs)
                                executed.append(action)
                                branch_states.append(vehicle_state(sim.agent))
                                future.append(provider.capture().raster.copy())
                                flags.append([bool(info['crash']), bool(info['out_of_road']), bool(tm), bool(tc)])
                                typed.append([bool(info[k]) for k in TYPED_CRASH_KEYS])
                                if tm or tc:
                                    break

                            length = len(future)
                            future_steps = np.rint(np.asarray(config.future_times_s) / 0.1).astype(int)
                            valid = future_steps <= length
                            targets = np.zeros((4, 4, 128, 128), np.uint8)
                            targets[valid] = np.asarray(future)[future_steps[valid] - 1]

                            realized = np.zeros((10, 4), np.float32)
                            way_valid = np.arange(2, 21, 2) <= length
                            realized[way_valid] = realized_waypoints(branch_states, 0, np.arange(2, 21, 2)[way_valid])

                            risk_target, risk_mask, crash_seen, offroad_seen, full_2s = compute_risk_targets(flags)

                            branch_file = folder / f'{cand_name}.npz'
                            np.savez_compressed(
                                branch_file,
                                future=targets,
                                valid=valid,
                                command=cmd,
                                states=np.asarray(branch_states),
                                executed_actions=np.asarray(executed),
                                outcome_flags=np.asarray(flags),
                                typed_events=np.asarray(typed),
                                realized=realized,
                                waypoint_valid=way_valid,
                                observed_history=history,
                                observed_states=past,
                                scene_state=current_scene,
                                risk_target=risk_target,
                                risk_mask=risk_mask,
                            )

                            branch_meta = dict(
                                id=cand_name,
                                path=str(branch_file.relative_to(output)),
                                sha256=hashlib.sha256(branch_file.read_bytes()).hexdigest(),
                                steps=length,
                                crash_seen=crash_seen,
                                offroad_seen=offroad_seen,
                                full_2s=full_2s,
                                risk_target=risk_target.tolist(),
                                risk_mask=risk_mask.tolist(),
                                typed_events_seen=np.asarray(typed, dtype=bool).any(axis=0).tolist(),
                            )
                            group_row['branches'].append(branch_meta)

                            report['summary_counts']['total_branches'] += 1
                            if crash_seen:
                                report['summary_counts']['crash_positives'] += 1
                            if offroad_seen:
                                report['summary_counts']['offroad_positives'] += 1
                            if risk_mask[0]:
                                report['summary_counts']['crash_valid'] += 1
                            if risk_mask[1]:
                                report['summary_counts']['offroad_valid'] += 1
                            if full_2s:
                                report['summary_counts']['full_2s_completed'] += 1

                        group_row['context'] = str((folder / 'context.npz').relative_to(output))
                        report['groups'].append(group_row)
                        report['elapsed_seconds'] = time.perf_counter() - started
                        write_json(output / 'manifest.json', report)
                        print(
                            json.dumps(
                                dict(
                                    group=group_id,
                                    groups=len(report['groups']),
                                    summary=report['summary_counts'],
                                    elapsed_s=round(report['elapsed_seconds'], 2),
                                )
                            ),
                            flush=True,
                        )
            finally:
                sim.close()

        report['completed'] = True
    except BaseException:
        report['error'] = traceback.format_exc()
        write_json(output / 'manifest.json', report)
        raise
    finally:
        restore()
        report['elapsed_seconds'] = time.perf_counter() - started
        write_json(output / 'manifest.json', report)


def main():
    faulthandler.enable()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True, help='Output directory (must not exist)')
    p.add_argument('--maps', nargs='+', default=['S', 'SC'], help='Train maps to collect on')
    p.add_argument('--seeds-per-map', type=int, default=1, help='Number of seeds per map')
    p.add_argument('--prefix-steps', nargs='+', type=int, default=[50], help='Anchor prefix steps')
    p.add_argument('--seed-offset', type=int, default=0, help='Seed offset (e.g. 1 for right-turn context)')
    args = p.parse_args()

    cfg = Stage7Config()
    collect_preflight(
        output_dir=args.output,
        config=cfg,
        map_names=args.maps,
        seeds_per_map=args.seeds_per_map,
        prefix_steps=args.prefix_steps,
        seed_offset=args.seed_offset,
    )


if __name__ == '__main__':
    main()
