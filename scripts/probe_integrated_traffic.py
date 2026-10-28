"""Actual-control straight passing/following teacher probe, not acceptance."""
import argparse
import faulthandler
import json
from pathlib import Path
import tarfile
import traceback
import numpy as np
from configs.stage2 import BEVConfig
from dataset.recording import vehicle_state, future_labels_v2
from env.data_v2 import configure_egl, NativeBEVProvider
from env.integrated import make_external_env, checked_step
from env.navigation import NavigationProvider, vehicle_parameters
from env.traffic_teacher import TrafficManeuverTeacher, spawn_slow_actor
from mini_wa_jepa.provenance import provenance, write_json
from mini_wa_jepa.visualization import RerunLogger
from scripts.collect_integrated_navigation import digest, EVENTS


def run(output):
    faulthandler.enable(); output.mkdir(parents=True, exist_ok=False)
    scenes = [dict(seed=4100300, case='allowed', lanes=3, actor_lanes=[1]),
              dict(seed=4100301, case='no_adjacent_lane', lanes=1, actor_lanes=[0]),
              dict(seed=4100302, case='blocked', lanes=3, actor_lanes=[1, 0, 2])]
    report = dict(**provenance(), completed=False, scenes=scenes, episodes=[],
                  control_verified=True, teacher_only=True, background_density=0., stage7_acceptance=False,
                  protocol_sha256=digest('docs/STAGE7_INTEGRATED_PROTOCOL.md'),
                  native_traffic_source_sha256=digest('.deps/metadrive/metadrive/manager/traffic_manager.py'))
    write_json(output/'report.json', report)
    with tarfile.open(output/'source_snapshot.tar.gz', 'w:gz') as archive:
        for file in report['source_files']:
            archive.add(file, arcname=file)
        archive.add('docs/STAGE7_INTEGRATED_PROTOCOL.md')
    _, restore = configure_egl(1)
    try:
        for scene in scenes:
            folder = output/scene['case']; folder.mkdir(); rows = []; states = []; npc_states = []
            home_id = min(1, scene['lanes']-1)
            sim = make_external_env(dict(map=3, map_config=dict(type='block_sequence', config='S',
                lane_num=scene['lanes'], exit_length=250),
                start_seed=scene['seed'], num_scenarios=1, traffic_density=0., horizon=600,
                decision_repeat=5, physics_world_step_size=.02, force_destroy=True,
                random_lane_width=False, random_lane_num=False, random_agent_model=False,
                random_spawn_lane_index=False,
                agent_configs={'default_agent': {'spawn_lane_index': ('>>', '>>>', home_id), 'spawn_longitude':5.}}))
            try:
                obs, _ = sim.reset(seed=scene['seed'])
                teacher = TrafficManeuverTeacher(sim)
                npcs = [spawn_slow_actor(sim, teacher.lanes[i], 27., scene['seed']+j)
                        for j, i in enumerate(scene['actor_lanes'])]
                before_render = np.array([vehicle_state(v) for v in [sim.agent, *npcs]])
                # reset() image predates NPC spawning. Render the new scene without
                # advancing physics, as used by native camera pose/reset rendering.
                sim.engine.taskMgr.step(); sim.engine.taskMgr.step()
                np.testing.assert_array_equal(before_render, [vehicle_state(v) for v in [sim.agent, *npcs]])
                provider = NativeBEVProvider(sim, BEVConfig()); provider.reset()
                if not provider.capture().raster[2].any():
                    raise RuntimeError('Spawned lead actor absent from the initial semantic BEV')
                nav = NavigationProvider(sim)
                from dataclasses import replace
                nav.config = replace(nav.config, preferred_lane=home_id)
                write_json(folder/'scene.json', dict(scene, home_lane=teacher.home.index,
                    lane_types=[list(l.line_types) for l in teacher.lanes],
                    lane_lengths=[float(l.length) for l in teacher.lanes],
                    vehicle_parameters=vehicle_parameters(sim.agent).tolist(),
                    calibration=provider.calibration, npc_ids=[v.id for v in npcs]))
                stable_follow = 0; info = {}; probe_complete = False
                with RerunLogger(app_id='integrated-traffic-teacher', spawn=False,
                                 save_path=folder/'traffic.rrd') as log:
                    log.log_context(static=True, seed=scene['seed'], case=scene['case'],
                                    teacher_privileged=True, learner_traffic_truth_access=False)
                    for step in range(600):
                        state = vehicle_state(sim.agent); states.append(state)
                        npc_states.append(np.array([vehicle_state(v) for v in npcs], np.float32))
                        frame = nav.capture(); bev = provider.capture().raster.copy()
                        action, detail = teacher.act()
                        obs, reward, tm, tc, info = checked_step(sim, action); provider.update(obs)
                        clearance = detail['front_clearance_m']
                        if scene['case'] != 'allowed' and detail['phase'] == 'FOLLOW' and clearance is not None:
                            if .5 <= state[3] <= 3 and 3 <= clearance <= 18:
                                stable_follow += 1
                            else:
                                stable_follow = 0
                        probe_complete = bool(teacher.completed or stable_follow >= 20)
                        rows.append(dict(bev=bev, route_points=frame.points, route_valid=frame.valid,
                            action=action, applied_action=np.asarray(info['raw_action'], np.float32),
                            reward=reward, cost=float(info.get('cost', 0)),
                            events=np.array([bool(info.get(k)) for k in EVENTS]),
                            route_completion=float(info.get('route_completion', 0)),
                            phase=detail['phase'], home_lateral_m=detail['home_lateral_m'],
                            heading_error_rad=detail['heading_error_rad'],
                            target_lane=detail['target_lane'], target_speed_m_s=detail['target_speed_m_s'],
                            lead_rear_clearance_m=np.nan if detail['lead_rear_clearance_m'] is None else detail['lead_rear_clearance_m'],
                            front_clearance_m=np.nan if clearance is None else clearance))
                        log.set_step(step, step*.1); log.log_bev(bev); log.log_state(state)
                        log.log_navigation(frame.points, frame.valid, source=frame.source)
                        log.log_context(phase=detail['phase'], teacher_privileged=True,
                                        executed_action_json=json.dumps(action.tolist()))
                        log.log_telemetry(speed_m_s=state[3], steering=action[0], throttle_brake=action[1],
                                          route_completion=rows[-1]['route_completion'],
                                          lateral_error_m=detail['home_lateral_m'])
                        if tm or tc or probe_complete:
                            break
                    states.append(vehicle_state(sim.agent)); states = np.asarray(states, np.float32)
                    target, valid, _ = future_labels_v2(states, np.zeros(len(states), bool), 10, 2)
                    for i in range(len(rows)):
                        log.set_step(i, i*.1); log.log_trajectories(ground_truth=target[i][valid[i]])
                arrays = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
                arrays.update(states=states, diagnostic_npc_states=np.asarray(npc_states),
                              target=target[:-1], valid=valid[:-1])
                np.savez_compressed(folder/'actual.npz', **arrays)
                phases = sorted(set(arrays['phase'].tolist()))
                safe = not bool(arrays['events'][:, [0, 4]].any())
                passed = bool(probe_complete and safe and (scene['case'] != 'allowed' or
                              {'CHANGE_OUT', 'PASS', 'CHANGE_BACK', 'COMPLETE'}.issubset(phases)))
                ep = dict(**scene, steps=len(rows), phases=phases, probe_complete=probe_complete,
                          probe_passed=passed, stable_follow_steps=stable_follow,
                          outcome={k: bool(info.get(k)) for k in EVENTS},
                          raw_sha256=digest(folder/'actual.npz'), rerun_sha256=digest(folder/'traffic.rrd'))
                report['episodes'].append(ep); write_json(output/'report.json', report)
                print(json.dumps(ep), flush=True)
            except BaseException:
                write_json(folder/'failure_mre.json', dict(error=traceback.format_exc(), steps=len(rows)))
                if rows:
                    np.savez_compressed(folder/'partial.npz', states=np.asarray(states),
                                        action=np.asarray([r['action'] for r in rows]))
                raise
            finally:
                sim.close()
        report['completed'] = True
        report['probe_passed'] = all(ep['probe_passed'] for ep in report['episodes'])
    except BaseException:
        report['error'] = traceback.format_exc(); raise
    finally:
        restore(); write_json(output/'report.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
