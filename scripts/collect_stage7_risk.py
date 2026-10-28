"""Formal data collector for Stage 7 risk prediction with controlled candidate rollouts."""
import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import shutil
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
from scripts.preflight_risk_collection import (
    TYPED_CRASH_KEYS,
    scene_state,
    compute_observed_lateral_left,
    make_risk_candidates,
    compute_risk_targets,
)


def collect_risk_data(output_dir, config, split='train', map_names=None, seeds_per_map=None,
                      prefix_steps=None, preflight=False, seed_offset=0, resume_from=None, ordered_semantics=False):
    spec = STAGE7_SPLITS[split]
    if preflight and split != 'train':
        raise ValueError('Preflight is train-only')
    if map_names is not None and not set(map_names) <= set(spec['maps']):
        raise ValueError('Requested map outside the frozen split')
    if not preflight and (seed_offset != 0 or
            seeds_per_map not in (None, spec['seeds_per_map']) or
            (prefix_steps is not None and tuple(prefix_steps) != config.prefix_steps)):
        raise ValueError('Formal seed/anchor overrides are forbidden; partition maps only')
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Preserve existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)

    split_spec = STAGE7_SPLITS[split]
    selected_maps = list(map_names) if map_names is not None else list(split_spec['maps'])
    count_per_map = seeds_per_map if seeds_per_map is not None else (1 if preflight else split_spec['seeds_per_map'])
    steps_list = list(prefix_steps) if prefix_steps is not None else ([50] if preflight else list(config.prefix_steps))

    if preflight and split != 'train':
        raise ValueError("Preflight collection is strictly train-only.")

    report = dict(
        **provenance(config),
        stage=7,
        preflight=preflight,
        completed=False,
        split=split,
        requested_maps=selected_maps,
        seeds_per_map=count_per_map,
        prefix_steps=steps_list,
        seed_offset=seed_offset,
        groups=[],
        summary_counts=dict(
            total_branches=0,
            crash_positives=0,
            offroad_positives=0,
            crash_valid=0,
            offroad_valid=0,
            full_2s_completed=0,
            typed_events_seen={k: 0 for k in TYPED_CRASH_KEYS},
        ),
        failures=[],
        ordered_semantics=ordered_semantics,
    )
    reused=set()
    if resume_from is not None:
        origin=Path(resume_from);prior=json.loads((origin/'manifest.json').read_text())
        if prior['split']!=split or prior['preflight'] or preflight:
            raise ValueError('Continuation must keep the same formal split')
        if json.dumps(prior['config'],sort_keys=True)!=json.dumps(report['config'],sort_keys=True):
            raise ValueError('Continuation cannot change the frozen experiment config')
        report['continuation_from']=dict(path=str(origin),manifest_sha256=hashlib.sha256((origin/'manifest.json').read_bytes()).hexdigest())
        for old in prior['groups']:
            group=dict(old);group['semantic_order']=bool(old.get('semantic_order',prior.get('ordered_semantics',False)))
            if group['map_spec'] not in selected_maps:raise ValueError('Continuation map cohort changed')
            source=origin/group['context'];destination=output/group['context']
            shutil.copytree(source.parent,destination.parent)
            group['original_partition']=str(origin);report['groups'].append(group);reused.add(group['id'])
            for b in group['branches']:
                counts=report['summary_counts'];counts['total_branches']+=1
                counts['crash_positives']+=int(b['crash_seen']);counts['offroad_positives']+=int(b['offroad_seen'])
                counts['crash_valid']+=int(b['risk_mask'][0]);counts['offroad_valid']+=int(b['risk_mask'][1])
                counts['full_2s_completed']+=int(b['full_2s'])
                for name,event in zip(TYPED_CRASH_KEYS,b['typed_events_seen']):counts['typed_events_seen'][name]+=int(event)
        report['reused_complete_contexts']=len(reused)
    write_json(output / 'manifest.json', report)

    with tarfile.open(output / 'source_snapshot.tar.gz', 'w:gz') as archive:
        for source in report['source_files']:
            archive.add(source, arcname=source)
        for doc in ('GOAL.md', 'AGENTS.md', 'docs/STAGE7_PROTOCOL.md'):
            archive.add(doc, arcname=doc)

    _, restore = configure_egl(1)
    cfg = Stage2Config()
    started = time.perf_counter()

    candidate_specs = STAGE7_RISK_CANDIDATE_SPECS

    try:
        maps_with_idx = [(i, m) for i, m in enumerate(split_spec['maps']) if m in selected_maps]
        if not maps_with_idx:
            raise ValueError(f"No matching maps for split {split} in requested: {selected_maps}")

        for map_i, map_spec in maps_with_idx:
            base_seed = split_spec['seed_start'] + map_i * 10000
            sim = make_env(
                map_spec,
                base_seed,
                cfg,
                num_scenarios=1000,
                control_dt_s=config.control_dt_s,
                environment_overrides={
                    'force_destroy': config.force_destroy_between_resets,
                    'traffic_density': getattr(config, 'traffic_density', 0.3),
                },
                semantic_order=ordered_semantics,
            )
            try:
                for s_idx in range(count_per_map):
                    seed_idx = s_idx + seed_offset
                    seed = base_seed + seed_idx
                    is_right_turn_context = (seed_idx % 2 == 1)

                    for anchor_step in steps_list:
                        group_id = f'{split}_{map_spec}_{seed}_step{anchor_step}'
                        if group_id in reused:continue
                        folder = output / split / group_id
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
                            split=split,
                            map_spec=map_spec,
                            seed=seed,
                            seed_index=seed_idx,
                            context_type='right_turn_roadside' if is_right_turn_context else 'center_expert',
                            anchor_step=anchor_step,
                            branches=[],
                            semantic_order=ordered_semantics,
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
                                # First candidate: construct prefix
                                if not is_right_turn_context:
                                    # Even seed: standard expert prefix
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
                                            raise RuntimeError(f"Prefix terminated early: {group_id} at step {step} ({info})")
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
                                            raise RuntimeError(f"Prefix terminated early in expert: {group_id} at step {step} ({info})")

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
                                            raise RuntimeError(f"Roadside prefix terminated early: {group_id} at step {step} ({info})")

                                prefix_actions = actions
                                prefix_flags = flags_prefix
                                prefix_typed = typed_prefix
                                reference_history = np.stack(bevs[-4:])
                                reference_states = np.stack(states[-4:])
                                reference_scene = scene_state(sim)

                                obs_lat_left = compute_observed_lateral_left(expert, sim.agent.position)
                                commands = make_risk_candidates(expert, candidate_specs, observed_lateral_left=obs_lat_left)
                                reference_ego=np.zeros((4,2),np.float32)
                                reference_ego[:,0]=reference_states[:,3]/10
                                reference_ego[:,1]=wrap_angle(np.stack(states[-4:])[:,2]-np.stack(states[-5:-1])[:,2])/.1

                                np.savez_compressed(
                                    folder / 'context.npz',
                                    history=reference_history,
                                    ego=reference_ego,
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
                                # Replay exact saved prefix actions
                                for step in range(anchor_step):
                                    action = prefix_actions[step]
                                    obs, _, tm, tc, info = sim.step(action)
                                    provider.update(obs)
                                    states.append(vehicle_state(sim.agent))
                                    bevs.append(provider.capture().raster.copy())
                                    if tm or tc:
                                        raise RuntimeError(f"Prefix replay terminated early: {group_id} at step {step}")

                            history = np.stack(bevs[-4:])
                            past = np.stack(states[-4:])
                            ego = np.zeros((4, 2), np.float32)
                            ego[:, 0] = past[:, 3] / 10.0
                            ego[:, 1] = wrap_angle(np.stack(states[-4:])[:, 2] - np.stack(states[-5:-1])[:, 2]) / 0.1

                            # Strict determinism validation
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
                                    f"Replay mismatch for {group_id} cand {cand_name}: "
                                    f"state_error={state_error}, scene_error={scene_error}, bev_diff={bev_diff}"
                                )

                            # Execute branch rollout
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
                                prefix_max_state_error=state_error,
                                prefix_scene_max_state_error=scene_error,
                                prefix_bev_different_fraction=bev_diff,
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

                            for k_i, k in enumerate(TYPED_CRASH_KEYS):
                                if np.any(np.asarray(typed, dtype=bool)[:, k_i]):
                                    report['summary_counts']['typed_events_seen'][k] += 1

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
        # Preserve partially observed prefixes too, including pre-anchor terminations.
        if 'folder' in locals():
            np.savez_compressed(folder / 'failure_mre.npz',
                states=np.asarray(states), history=np.asarray(bevs),
                actions=np.asarray(actions), prefix_flags=np.asarray(flags_prefix),
                prefix_typed=np.asarray(typed_prefix))
            report['failures'].append(dict(group=group_id, candidate=cand_name,
                mre=str((folder / 'failure_mre.npz').relative_to(output))))
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
    p.add_argument('--split', choices=list(STAGE7_SPLITS), default='train', help='Dataset partition to collect')
    p.add_argument('--maps', nargs='+', default=None, help='Subset of maps to collect')
    p.add_argument('--seeds-per-map', type=int, default=None, help='Seeds per map override')
    p.add_argument('--prefix-steps', nargs='+', type=int, default=None, help='Prefix steps override')
    p.add_argument('--preflight', action='store_true', help='Preflight mode (train-only, small scale)')
    p.add_argument('--seed-offset', type=int, default=0, help='Seed offset (e.g. 1 for right-turn context)')
    p.add_argument('--resume-from',type=Path,help='Preserve source partition and reuse only complete contexts into a new output')
    p.add_argument('--ordered-semantics',action='store_true',help='Stable camera-only semantic tie order; native depth/geometry retained')
    args = p.parse_args()

    cfg = Stage7Config()
    collect_risk_data(
        output_dir=args.output,
        config=cfg,
        split=args.split,
        map_names=args.maps,
        seeds_per_map=args.seeds_per_map,
        prefix_steps=args.prefix_steps,
        preflight=args.preflight,
        seed_offset=args.seed_offset,
        resume_from=args.resume_from,
        ordered_semantics=args.ordered_semantics,
    )


if __name__ == '__main__':
    main()
