"""Schema3 navigation probes/data with native termination; no hidden repairs."""
import argparse
from dataclasses import asdict
import faulthandler
import hashlib
import json
from pathlib import Path
import tarfile
import time
import traceback
import numpy as np

from configs.integrated import IntegratedConfig
from configs.stage2 import BEVConfig, ExpertConfig
from dataset.recording import vehicle_state, future_labels_v2
from env.data_v2 import configure_egl, NativeBEVProvider
from env.semantic_bev import make_semantic_env
from env.integrated import make_external_env, checked_step
from env.navigation import NavigationProvider, vehicle_parameters
from env.lane_permissions import maneuver_permissions
from env.expert import NavigationExpert
from evaluation.memory import memory_snapshot
from mini_wa_jepa.provenance import provenance, write_json
from mini_wa_jepa.visualization import RerunLogger

EVENTS = ('crash', 'crash_vehicle', 'crash_object', 'crash_sidewalk',
          'out_of_road', 'arrive_dest', 'max_step')
MAPS = ('S', 'CCC', 'TXC', 'SRS', 'SC', 'CSC')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(args):
    faulthandler.enable()
    cfg = IntegratedConfig()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    start_seed = dict(probe=cfg.probe_seed_start, train=cfg.train_seed_start,
                      validation=cfg.validation_seed_start)[args.split]+args.seed_offset
    scenes = [dict(seed=start_seed+i, map=MAPS[i % len(MAPS)],
                   density=(.1 if (i//len(MAPS)) % 2 == 0 else .3),
                   recovery=bool(args.recovery and i >= args.episodes//2)) for i in range(args.episodes)]
    report = dict(**provenance(cfg), schema_version=3, completed=False, split=args.split,
                  input_profile='schema3.1_static_maneuver_permissions' if args.maneuver_permissions else 'schema3_navigation',
                  scenes=scenes, max_steps=args.max_steps, teacher='NavigationExpert_no_overtaking',
                  navigation_permission='map_assisted_route_lane_centerline', native_termination=True,
                  events=list(EVENTS), episodes=[], control_verified=True,
                  source_protocol_sha256=digest('docs/STAGE7_INTEGRATED_PROTOCOL.md'),
                  memory_before=memory_snapshot())
    write_json(output/'manifest.json', report)
    with tarfile.open(output/'source_snapshot.tar.gz', 'w:gz') as archive:
        for path in report['source_files']:
            archive.add(path, arcname=path)
        for path in ('AGENTS.md', 'GOAL.md', 'docs/STAGE7_INTEGRATED_PROTOCOL.md'):
            archive.add(path, arcname=path)
    _, restore = configure_egl(1)
    started = time.monotonic()
    try:
        for scene in scenes:
            folder = output/f"seed_{scene['seed']}"; folder.mkdir()
            overrides = dict(map=scene['map'], start_seed=scene['seed'], num_scenarios=1,
                             traffic_density=scene['density'], horizon=args.max_steps,
                             decision_repeat=5, physics_world_step_size=.02, force_destroy=True,
                             random_lane_width=False, random_lane_num=False, random_agent_model=False,
                             random_spawn_lane_index=False,
                             agent_configs={'default_agent': {'spawn_lane_index': ('>', '>>', 1)}})
            sim = make_external_env(overrides)
            rows = []; states = []; info = {}
            try:
                observation, _ = sim.reset(seed=scene['seed'])
                provider = NativeBEVProvider(sim, BEVConfig()); provider.reset(observation)
                navigation = NavigationProvider(sim, cfg)
                expert = NavigationExpert(sim, ExpertConfig(), .1)
                params = vehicle_parameters(sim.agent)
                write_json(folder/'scene.json', dict(scene, vehicle_parameters=params.tolist(),
                                                     calibration=provider.calibration,
                                                     map_hash=provider.geometry_hash,
                                                     dynamics=sim.agent.get_dynamics_parameters()))
                with RerunLogger(app_id='integrated-navigation', spawn=False,
                                 save_path=folder/'navigation.rrd') as log:
                    log.log_context(static=True, split=args.split, seed=scene['seed'],
                                    schema_version=3, navigation_source='map_assisted_route_lane_centerline',
                                    teacher='NavigationExpert_no_overtaking')
                    for step in range(args.max_steps):
                        frame = navigation.capture(); state = vehicle_state(sim.agent)
                        teacher_action, detail = expert.act()
                        action = teacher_action.copy()
                        noise_active = bool(scene['recovery'] and step >= 30 and (step-30) % 160 < 6)
                        if noise_active:
                            sign = 1 if (scene['seed']+((scene['seed']-start_seed)//6)) % 2 == 0 else -1
                            action[0] = np.clip(action[0]+sign*.25, -1., 1.)
                        bev = provider.capture().raster.copy()
                        permission = maneuver_permissions(sim) if args.maneuver_permissions else None
                        observation, reward, terminated, truncated, info = checked_step(sim, action)
                        provider.update(observation)
                        row = dict(bev=bev, route_points=frame.points, route_valid=frame.valid,
                                   action=action, teacher_action=teacher_action, noise_active=noise_active,
                                   recovery_active=bool(not noise_active and abs(detail['lateral_m']) > .3),
                                   applied_action=np.asarray(info['raw_action'], np.float32),
                                   reward=reward, cost=float(info.get('cost', 0)),
                                   events=np.array([bool(info.get(k)) for k in EVENTS]),
                                   route_completion=float(info.get('route_completion', 0)),
                                   lateral_m=detail['lateral_m'], target_speed_m_s=detail['target_speed_m_s'])
                        if permission is not None:
                            row['maneuver_allowed'] = permission
                        rows.append(row); states.append(state)
                        log.set_step(step, step*.1); log.log_bev(bev); log.log_state(state)
                        log.log_navigation(frame.points, frame.valid, source=frame.source)
                        log.log_context(valid_route_points=int(frame.valid.sum()))
                        log.log_telemetry(speed_m_s=state[3], steering=action[0], throttle_brake=action[1],
                                          lateral_error_m=detail['lateral_m'], route_completion=row['route_completion'])
                        if terminated or truncated:
                            break
                    states.append(vehicle_state(sim.agent)); states = np.asarray(states, np.float32)
                    terminal_failure = np.zeros(len(states), bool)
                    terminal_failure[-1] = bool(info.get('crash') or info.get('out_of_road'))
                    targets, valid, _ = future_labels_v2(states, terminal_failure, 10, 2)
                    for step in range(len(rows)):
                        log.set_step(step, step*.1)
                        log.log_trajectories(ground_truth=targets[step][valid[step]])
                arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
                n = len(rows)
                yaw = np.r_[0., np.arctan2(np.sin(np.diff(states[:, 2])), np.cos(np.diff(states[:, 2])))/.1]
                arrays.update(states=states, ego=np.column_stack([states[:-1, 3]/10, yaw[:n]]).astype(np.float32),
                              vehicle=np.broadcast_to(params, (n, 4)),
                              trajectory=targets[:n], valid=valid[:n],
                              history_index=np.maximum(np.arange(n)[:, None]-np.arange(3, -1, -1), 0))
                raw = folder/'episode.npz'; np.savez_compressed(raw, **arrays)
                outcome = {key: bool(info.get(key)) for key in EVENTS}
                episode = dict(scene, steps=n, outcome=outcome,
                               route_completion=float(info.get('route_completion', 0)),
                               reward=float(arrays['reward'].sum()), cost=float(arrays['cost'].sum()),
                               full_horizon_anchors=int(valid[:n].all(1).sum()),
                               success_demonstration=bool(info.get('arrive_dest')),
                               raw=str(raw.relative_to(output)), raw_sha256=digest(raw),
                               rerun_sha256=digest(folder/'navigation.rrd'), memory=memory_snapshot())
                report['episodes'].append(episode)
                print(json.dumps(episode, ensure_ascii=False), flush=True)
                write_json(output/'manifest.json', report)
            except BaseException:
                write_json(folder/'failure_mre.json', dict(scene=scene, traceback=traceback.format_exc(),
                                                         observed_rows=len(rows)))
                if rows:
                    np.savez_compressed(folder/'partial.npz', states=np.asarray(states),
                                        actions=np.asarray([r['action'] for r in rows]),
                                        route_points=np.asarray([r['route_points'] for r in rows]))
                raise
            finally:
                sim.close()
        report['completed'] = True
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        restore()
        report.update(elapsed_s=time.monotonic()-started, memory_after=memory_snapshot())
        write_json(output/'manifest.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=('probe', 'train', 'validation'), default='probe')
    p.add_argument('--episodes', type=int, default=6)
    p.add_argument('--max-steps', type=int, default=300)
    p.add_argument('--seed-offset', type=int, default=0)
    p.add_argument('--recovery', action='store_true', help='Half scenes use declared steering pulses; never imitate noisy transitions')
    p.add_argument('--maneuver-permissions', action='store_true', help='Record declared conservative static-map lane-change permissions')
    args = p.parse_args()
    if not (1 <= args.episodes <= 10000 and 1 <= args.max_steps <= 1000 and
            0 <= args.seed_offset < 90000):
        p.error('Invalid bounded collection budget')
    collect(args)


if __name__ == '__main__':
    main()
