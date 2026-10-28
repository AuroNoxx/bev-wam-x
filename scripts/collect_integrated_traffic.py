"""Actual expert maneuver trajectories, static permissions and diagnostic NPCs.

Schema3.1 traffic supplement: controlled straight roads; never a benchmark.
"""
import argparse
from dataclasses import replace
import faulthandler
import json
from pathlib import Path
import tarfile
import traceback
import numpy as np
from configs.integrated import IntegratedConfig
from configs.stage2 import BEVConfig
from dataset.recording import vehicle_state, future_labels_v2
from env.data_v2 import configure_egl, NativeBEVProvider
from env.integrated import make_external_env, checked_step
from env.navigation import NavigationProvider, vehicle_parameters
from env.lane_permissions import maneuver_permissions
from env.traffic_teacher import TrafficManeuverTeacher, spawn_slow_actor
from mini_wa_jepa.provenance import provenance, write_json
from mini_wa_jepa.visualization import RerunLogger
from scripts.collect_integrated_navigation import digest, EVENTS


def collect(args):
    faulthandler.enable(); args.output.mkdir(parents=True, exist_ok=False)
    start = dict(train=4200300, validation=4300300)[args.split]
    types = [('allowed_left', 3, [1]), ('allowed_right', 3, [1, 0]),
             ('blocked', 3, [1, 0, 2]), ('no_adjacent_lane', 1, [0])]
    scenes = [dict(seed=start+i, case=types[i%4][0], lanes=types[i%4][1],
                  actor_lanes=types[i%4][2], lead_distance_m=(18., 22., 26.)[(i//4)%3])
              for i in range(args.seed_offset, args.seed_offset+args.episodes)]
    report = dict(**provenance(), schema_version=3, input_profile='schema3.1_static_maneuver_permissions',
                  completed=False, split=args.split, scenes=scenes, episodes=[],
                  control_verified=True, native_termination=True,
                  navigation_permission='map_assisted_route_lane_centerline',
                  maneuver_permission_source='static_map_adjacent_broken_straight_lane_only',
                  diagnostic_actor_truth_for_learner=False, teacher='TrafficManeuverTeacher_privileged',
                  events=list(EVENTS), source_protocol_sha256=digest('docs/STAGE7_INTEGRATED_PROTOCOL.md'))
    write_json(args.output/'manifest.json', report)
    with tarfile.open(args.output/'source_snapshot.tar.gz', 'w:gz') as archive:
        for p in report['source_files']:
            archive.add(p, arcname=p)
        archive.add('docs/STAGE7_INTEGRATED_PROTOCOL.md')
    _, restore = configure_egl(1)
    try:
        for scene in scenes:
            folder = args.output/f"seed_{scene['seed']}"; folder.mkdir()
            rows = []; states = []; npc_states = []; home_id = min(1, scene['lanes']-1)
            sim = make_external_env(dict(map=3, map_config=dict(type='block_sequence', config='S',
                lane_num=scene['lanes'], exit_length=250), start_seed=scene['seed'], num_scenarios=1,
                traffic_density=0., horizon=600, decision_repeat=5, physics_world_step_size=.02,
                force_destroy=True, random_lane_width=False, random_lane_num=False, random_agent_model=False,
                random_spawn_lane_index=False,
                agent_configs={'default_agent': {'spawn_lane_index': ('>>', '>>>', home_id), 'spawn_longitude':5.}}))
            try:
                obs, _ = sim.reset(seed=scene['seed']); teacher = TrafficManeuverTeacher(sim)
                npcs = [spawn_slow_actor(sim, teacher.lanes[i], 5.+scene['lead_distance_m'], scene['seed']+j)
                        for j, i in enumerate(scene['actor_lanes'])]
                npc_ids = [v.id for v in npcs]
                before = np.array([vehicle_state(v) for v in [sim.agent, *npcs]])
                sim.engine.taskMgr.step(); sim.engine.taskMgr.step()
                np.testing.assert_array_equal(before, [vehicle_state(v) for v in [sim.agent, *npcs]])
                provider = NativeBEVProvider(sim, BEVConfig()); provider.reset()
                if not provider.capture().raster[2].any():
                    raise RuntimeError('No visible initial NPC')
                nav = NavigationProvider(sim, replace(IntegratedConfig(), preferred_lane=home_id))
                params = vehicle_parameters(sim.agent)
                write_json(folder/'scene.json', dict(scene, vehicle_parameters=params.tolist(),
                    npc_parameters=[vehicle_parameters(v).tolist() for v in npcs],
                    home_lane=teacher.home.index, calibration=provider.calibration,
                    lane_types=[list(l.line_types) for l in teacher.lanes]))
                completed_at = None; stable_follow = 0; info = {}; removed_ids = []
                with RerunLogger(app_id='integrated-traffic-data', spawn=False,
                                 save_path=folder/'navigation.rrd') as log:
                    log.log_context(static=True, split=args.split, seed=scene['seed'], teacher_privileged=True,
                                    actor_truth_for_learner=False, input_profile=report['input_profile'])
                    for step in range(600):
                        state = vehicle_state(sim.agent); frame = nav.capture()
                        bev = provider.capture().raster.copy(); permission = maneuver_permissions(sim)
                        states.append(state); npc_states.append([vehicle_state(v) for v in npcs])
                        action, detail = teacher.act()
                        obs, reward, tm, tc, info = checked_step(sim, action); provider.update(obs)
                        live_ids = set(sim.engine.get_objects())
                        npc_present_after = np.array([key in live_ids for key in npc_ids], bool)
                        removed_ids = [key for key in npc_ids if key not in live_ids]
                        clearance = detail['front_clearance_m']
                        if scene['case'] in ('blocked', 'no_adjacent_lane'):
                            stable_follow = (stable_follow+1 if detail['phase']=='FOLLOW' and clearance is not None
                                             and .5<=state[3]<=3 and 3<=clearance<=18 else 0)
                        if teacher.completed and completed_at is None:
                            completed_at = step
                        rows.append(dict(bev=bev, route_points=frame.points, route_valid=frame.valid,
                            maneuver_allowed=permission, action=action, teacher_action=action, applied_action=np.asarray(info['raw_action'], np.float32),
                            diagnostic_npc_present_after=npc_present_after,
                            noise_active=False, recovery_active=False, reward=reward, cost=float(info.get('cost', 0)),
                            events=np.array([bool(info.get(k)) for k in EVENTS]), phase=detail['phase'],
                            route_completion=float(info.get('route_completion', 0)),
                            lateral_m=detail['home_lateral_m'], target_speed_m_s=detail['target_speed_m_s']))
                        log.set_step(step, step*.1); log.log_bev(bev); log.log_state(state)
                        log.log_navigation(frame.points, frame.valid, source=frame.source)
                        log.log_context(phase=detail['phase'], maneuver_allowed_json=json.dumps(permission.tolist()))
                        log.log_telemetry(speed_m_s=state[3], steering=action[0], throttle_brake=action[1],
                                          lateral_error_m=detail['home_lateral_m'], route_completion=rows[-1]['route_completion'])
                        if tm or tc or removed_ids or (completed_at is not None and step-completed_at>=30) or stable_follow>=80:
                            break
                    states.append(vehicle_state(sim.agent)); states = np.asarray(states, np.float32)
                    terminal_failure = np.zeros(len(states), bool)
                    terminal_failure[-1] = bool(info.get('crash') or info.get('out_of_road'))
                    target, valid, _ = future_labels_v2(states, terminal_failure, 10, 2)
                    for i in range(len(rows)):
                        log.set_step(i, i*.1); log.log_trajectories(ground_truth=target[i][valid[i]])
                arrays = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}; n = len(rows)
                yaw = np.r_[0., np.diff(np.unwrap(states[:, 2]))/.1]
                arrays.update(states=states, diagnostic_npc_states=np.asarray(npc_states, np.float32),
                    trajectory=target[:n], valid=valid[:n], vehicle=np.broadcast_to(params, (n, 4)),
                    ego=np.column_stack([states[:-1, 3]/10., yaw[:n]]).astype(np.float32),
                    history_index=np.maximum(np.arange(n)[:, None]-np.arange(3, -1, -1), 0))
                raw = folder/'episode.npz'; np.savez_compressed(raw, **arrays)
                safe = not arrays['events'][:, [0, 4]].any()
                success = bool(safe and not removed_ids and (teacher.completed if scene['case'].startswith('allowed') else stable_follow>=80))
                ep = dict(**scene, steps=n, outcome={k:bool(info.get(k)) for k in EVENTS},
                    route_completion=float(info.get('route_completion', 0)),
                    reward=float(arrays['reward'].sum()), cost=float(arrays['cost'].sum()),
                    phases=sorted(set(arrays['phase'].tolist())), maneuver_demonstration_success=success,
                    success_demonstration=success, actual_arrival=bool(info.get('arrive_dest')),
                    stop_reason='controlled_actor_removed' if removed_ids else ('native' if tm or tc else 'bounded_maneuver_demonstration'),
                    removed_npc_ids=removed_ids,
                    raw=str(raw.relative_to(args.output)), raw_sha256=digest(raw),
                    rerun_sha256=digest(folder/'navigation.rrd'))
                report['episodes'].append(ep); write_json(args.output/'manifest.json', report)
                print(json.dumps(ep), flush=True)
            except BaseException:
                write_json(folder/'failure_mre.json', dict(error=traceback.format_exc(), observed_rows=len(rows)))
                if rows:
                    np.savez_compressed(folder/'partial.npz', states=np.asarray(states),
                                        action=np.asarray([r['action'] for r in rows]))
                raise
            finally:
                sim.close()
        report['completed'] = True
    except BaseException:
        report['error'] = traceback.format_exc(); raise
    finally:
        restore(); write_json(args.output/'manifest.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=('train', 'validation'), required=True)
    p.add_argument('--episodes', type=int, required=True)
    p.add_argument('--seed-offset', type=int, default=0, help='Replay a registered four-case group')
    args = p.parse_args()
    if not 4 <= args.episodes <= 48 or args.episodes%4 or args.seed_offset not in (0, 4, 8):
        p.error('Bounded collection needs4–48 episodes in full four-case groups')
    collect(args)
