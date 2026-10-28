"""Native no-traffic braking controller probes with actual yaw and strict reporting."""
import argparse
from pathlib import Path
import json
import traceback
import numpy as np
from configs.stage2 import BEVConfig
from configs.planner7 import Planner7Config
from controller.trajectory_tracker import TrajectoryTracker
from dataset.recording import vehicle_state, future_labels_v2
from env.data_v2 import configure_egl, NativeBEVProvider
from env.semantic_bev import make_semantic_env
from env.integrated import make_external_env, checked_step
from env.expert import NavigationExpert
from mini_wa_jepa.geometry import wrap_angle
from mini_wa_jepa.provenance import provenance, write_json
from mini_wa_jepa.visualization import RerunLogger
from planning.integrated import curvature_braking
from planning.candidates import static_cost, dynamic_cost, actor_tracks
from scripts.collect_integrated_navigation import digest, EVENTS


def run(output, adaptive_curve_anchor=False):
    output.mkdir(parents=True, exist_ok=False)
    specs = [(4100100, 'S', 80), (4100101, 'CCC', 150), (4100102, 'SRS', 250)]
    report = dict(**provenance(), completed=False, no_traffic=True, cases=[], control_verified=True,
                  adaptive_curve_anchor=adaptive_curve_anchor,
                  protocol_sha256=digest('docs/STAGE7_INTEGRATED_PROTOCOL.md'),
                  threshold=dict(final_speed_m_s=.5, minimum_curve_yaw_rate=.05), specs=specs)
    write_json(output/'report.json', report)
    _, restore = configure_egl(1)
    try:
        for seed, map_spec, prefix in specs:
            folder = output/str(seed); folder.mkdir()
            sim = make_external_env(dict(map=map_spec, start_seed=seed, num_scenarios=1,
                                                   horizon=(640 if adaptive_curve_anchor and map_spec == 'CCC' else prefix+40),
                                                   traffic_density=0., force_destroy=True,
                                                   decision_repeat=5, physics_world_step_size=.02))
            try:
                obs, _ = sim.reset(seed=seed)
                provider = NativeBEVProvider(sim, BEVConfig()); provider.reset(obs)
                teacher = NavigationExpert(sim); previous = vehicle_state(sim.agent)
                for step in range(prefix):
                    previous = vehicle_state(sim.agent)
                    obs, _, tm, tc, info = checked_step(sim, teacher.act()[0]); provider.update(obs)
                    if tm or tc:
                        raise RuntimeError(f'Prefix ended before braking at {step}: {info}')
                if adaptive_curve_anchor and map_spec == 'CCC':
                    while prefix < 600:
                        current = vehicle_state(sim.agent)
                        observed_yaw = float(wrap_angle(current[2]-previous[2]))/.1
                        if abs(observed_yaw) >= .05:
                            break
                        previous = current
                        obs, _, tm, tc, info = checked_step(sim, teacher.act()[0]); provider.update(obs)
                        prefix += 1
                        if tm or tc:
                            raise RuntimeError('Episode ended before finding an actual curved anchor')
                tracker = TrajectoryTracker(sim.agent, .1)
                rows = []; states = []; bevs = []
                for step in range(30):
                    state = vehicle_state(sim.agent); bev = provider.capture().raster.copy()
                    yaw = float(wrap_angle(state[2]-previous[2]))/.1
                    plan = curvature_braking(float(state[3]), yaw)
                    states.append(state); bevs.append(bev)
                    tracks = actor_tracks(bevs[-4:], states[-4:])
                    geometric = static_cost(bev, plan, sim.agent.LENGTH, sim.agent.WIDTH)
                    dynamic = dynamic_cost(plan, tracks, sim.agent.LENGTH, sim.agent.WIDTH, Planner7Config())
                    action, _ = tracker.act(plan, state[3]); previous = state
                    obs, _, tm, tc, info = checked_step(sim, action); provider.update(obs)
                    rows.append(dict(bev=bev, plan=plan, action=action,
                                     applied_action=np.asarray(info['raw_action'], np.float32), yaw=yaw,
                                     static_cost=geometric, dynamic_cost=dynamic,
                                     events=np.array([bool(info.get(k)) for k in EVENTS])))
                    if tm or tc:
                        break
                states.append(vehicle_state(sim.agent)); states = np.asarray(states, np.float32)
                data = {k:np.asarray([r[k] for r in rows]) for k in rows[0]}
                data['states'] = states; np.savez_compressed(folder/'actual.npz', **data)
                target, valid, _ = future_labels_v2(states, np.zeros(len(states), bool), 10, 2)
                with RerunLogger(app_id='integrated-braking-probe', spawn=False, save_path=folder/'braking.rrd') as log:
                    log.log_context(static=True, seed=seed, mode='measured_curvature_braking_no_traffic')
                    for i, row in enumerate(rows):
                        log.set_step(i, i*.1); log.log_bev(row['bev']); log.log_state(states[i])
                        log.log_planning(candidates=row['plan'][None], selected_id='STOP',
                                         selected_trajectory=row['plan'], selection_source='measured_curvature_braking',
                                         executed_action=row['action'])
                        log.log_trajectories(ground_truth=target[i][valid[i]])
                        log.log_metrics(dict(static_cost=row['static_cost'], dynamic_cost=row['dynamic_cost']))
                unsafe = any(info.get(k) for k in ('crash', 'out_of_road'))
                case = dict(seed=seed, map=map_spec, prefix=prefix, initial_speed=float(states[0, 3]),
                            initial_yaw_rate=rows[0]['yaw'], final_speed=float(states[-1, 3]),
                            outcome={k:bool(info.get(k)) for k in EVENTS},
                            max_static_cost=float(data['static_cost'].max()),
                            passed=bool(states[-1, 3] <= .5 and not unsafe),
                            raw_sha256=digest(folder/'actual.npz'))
                report['cases'].append(case); write_json(output/'report.json', report)
                print(json.dumps(case), flush=True)
            finally:
                sim.close()
        report['completed'] = True
        report['curved_coverage'] = any(abs(c['initial_yaw_rate']) >= .05 for c in report['cases'])
        report['probe_passed'] = all(c['passed'] for c in report['cases']) and report['curved_coverage']
    except BaseException:
        report['error'] = traceback.format_exc(); raise
    finally:
        restore(); write_json(output/'report.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--adaptive-curve-anchor', action='store_true')
    args = p.parse_args(); run(args.output, args.adaptive_curve_anchor)
