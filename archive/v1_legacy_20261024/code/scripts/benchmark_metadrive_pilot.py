"""Small PG-generalization benchmark using native MetaDrive reward/done/cost.

This is an adapted pilot, not a reproduction of the original paper's RL trials.
The neural planner/scorer are loaded from the preserved Stage-7 main snapshot.
"""
import argparse
from collections import deque
import faulthandler
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import time
import traceback

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.stage2 import BEVConfig
from controller.trajectory_tracker import TrajectoryTracker
from dataset.recording import future_labels_v2, vehicle_state
from env.data_v2 import NativeBEVProvider, configure_egl
from env.semantic_bev import SemanticBEVSpec, make_semantic_env
from evaluation.history import ObservedHistory
from mini_wa_jepa.provenance import provenance, write_json
from mini_wa_jepa.visualization import RerunLogger
from scripts.preflight_risk_collection import scene_state

SNAPSHOT = ROOT / 'artifacts/restart_stage7/closed_full_20261021/source_snapshot.tar.gz'
EVENTS = ('crash', 'crash_vehicle', 'crash_object', 'crash_building',
          'crash_human', 'crash_sidewalk', 'out_of_road', 'arrive_dest', 'max_step')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_policy(arm, out):
    """Preserve and execute the original eight-candidate implementation."""
    sources = out / 'main_policy_source'
    sources.mkdir()
    with tarfile.open(SNAPSHOT) as archive:
        for name in ('planning/candidates.py', 'planning/policy7.py'):
            (sources / Path(name).name).write_bytes(archive.extractfile(name).read())
        # These imported components must match the same main-run implementation.
        for name in ('configs/stage7.py', 'configs/stage6.py', 'configs/planner7.py',
                     'models/flow_matcher_transformer.py', 'models/action_conditioned.py',
                     'models/risk_head.py', 'controller/trajectory_tracker.py',
                     'controller/pure_pursuit.py', 'models/joint.py'):
            if archive.extractfile(name).read() != (ROOT / name).read_bytes():
                raise RuntimeError(f'Frozen policy dependency changed: {name}')
    candidates = load_module('_pilot_main_candidates', sources / 'candidates.py')
    policy = load_module('_pilot_main_policy', sources / 'policy7.py')
    policy.score_candidates = candidates.score_candidates
    policy.emergency_trajectory = candidates.emergency_trajectory
    return policy.Stage7Policy(arm, 20261021)


def stats(values):
    if not values:
        return None
    return dict(p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)),
                p99=float(np.percentile(values, 99)))


def save_episode(folder, records, states, encoded, futures, arm):
    folder.mkdir()
    states = np.asarray(states, np.float32)
    arrays = dict(states=states, bev=np.stack([r['bev'] for r in records]),
                  actions=np.stack([r['action'] for r in records]),
                  rewards=np.array([r['reward'] for r in records]),
                  costs=np.array([r['cost'] for r in records]),
                  route=np.array([r['route'] for r in records]),
                  events=np.array([r['events'] for r in records], bool),
                  decision_ms=np.array([r['decision_ms'] for r in records]),
                  pipeline_ms=np.array([r['pipeline_ms'] for r in records]))
    target, valid, _ = future_labels_v2(states, np.zeros(len(states), bool), 10, 2)
    arrays.update(trajectory_target=target[:len(records)], trajectory_valid=valid[:len(records)])
    if arm != 'idm':
        for key in ('candidates', 'plan', 'scores', 'terms', 'probability', 'selected_index'):
            arrays[key] = np.asarray([r['decision'][key] for r in records])
        arrays['selected_ids'] = np.asarray([r['decision']['selected_id'] for r in records])
        arrays['rejections'] = np.asarray([[';'.join(x) for x in r['decision']['rejection_reasons']]
                                           for r in records])
    slots = sorted(futures)
    horizons = [.1, .2, .3, .4] if arm == 'mse' else [.5, 1., 1.5, 2.]
    observed = np.zeros((len(slots), 4, 64, 256), np.float16)
    latent_valid = np.zeros((len(slots), 4), bool)
    for i, step in enumerate(slots):
        for j, seconds in enumerate(horizons):
            index = step + round(seconds / .1)
            if index < len(encoded):
                observed[i, j] = encoded[index]
                latent_valid[i, j] = True
    if slots:
        arrays.update(latent_steps=np.asarray(slots), latent_prediction=np.stack([futures[s] for s in slots]),
                      actual_future_under_replanning=observed, latent_valid=latent_valid)
    raw = folder / 'actual.npz'
    np.savez_compressed(raw, **arrays)
    # All multimodal records go through the project public Rerun bridge.
    rrd = folder / 'planning.rrd'
    with RerunLogger(app_id='metadrive-pg-pilot', spawn=False, save_path=rrd) as log:
        log.log_context(static=True, arm=arm, event_columns=list(EVENTS),
                        future_semantics='observed subsequent replanning, not fixed-command counterfactual')
        lookup = {s: i for i, s in enumerate(slots)}
        for i, row in enumerate(records):
            log.set_step(i, i * .1)
            log.log_bev(row['bev'])
            log.log_state(states[i])
            log.log_telemetry(speed_m_s=float(states[i, 3]), steering=float(row['action'][0]),
                              throttle_brake=float(row['action'][1]), route_completion=row['route'])
            log.log_metrics(dict(native_reward=row['reward'], native_cost=row['cost']))
            if arm == 'idm':
                log.log_context(selected_id='IDM', selection_source='native_IDM_privileged_reference')
                log.log_trajectories(ground_truth=target[i][valid[i]])
                continue
            d = row['decision']
            log.log_planning(candidates=d['candidates'], selected_id=d['selected_id'],
                             selected_trajectory=d['plan'], selection_source=d['selection_source'],
                             scores=None if arm == 'mse' else d['scores'],
                             terms=None if arm == 'mse' else d['terms'],
                             rejection_reasons=d['rejection_reasons'], executed_action=row['action'])
            log.log_trajectories(ground_truth=target[i][valid[i]])
            if i in lookup:
                j = lookup[i]
                mask = latent_valid[j]
                if mask.any():
                    log.log_latents(predicted=futures[i][mask], ground_truth=observed[j][mask])
    return dict(raw=str(raw.relative_to(ROOT)), raw_sha256=sha(raw),
                rerun=str(rrd.relative_to(ROOT)), rerun_sha256=sha(rrd))


def evaluate(args):
    faulthandler.enable()
    if not torch.version.hip or torch.version.cuda:
        raise RuntimeError('Native ROCm required')
    torch.set_num_threads(4)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    seeds = list(range(args.first_seed, args.first_seed + args.episodes))
    if not all(0 <= s < 200 for s in seeds):
        raise ValueError('Official example test range is 0..199')
    config = dict(map=3, start_seed=0, num_scenarios=200, random_lane_width=True,
                  random_lane_num=True, random_agent_model=True, traffic_density=.1,
                  horizon=args.max_steps, decision_repeat=5, physics_world_step_size=.02,
                  force_destroy=True)
    report = dict(**provenance(config), benchmark='MetaDrive PG-generalization adapted pilot',
                  official_example='https://metadrive-simulator.readthedocs.io/en/latest/rl_environments.html#generalization-environment',
                  official_paper_reproduction=False, completed=False, smoke=args.smoke, arm=args.arm,
                  seeds=seeds, policy_training_seed=None if args.arm == 'idm' else 20261021,
                  policy_snapshot_sha256=sha(SNAPSHOT), episodes=[], event_columns=list(EVENTS),
                  adaptations=['four-channel semantic BEV sensor and EGL fixed semantic draw order',
                               'frozen main Stage7 neural models, not retrained under the original RL benchmark protocol',
                               'only prefix of the 200 official-example test seeds',
                               'IDM has privileged navigation/state access; a reference, not observation-matched baseline'],
                  native_termination=True, project_FullOffroadRule=False,
                  control_dt_s=.1, torch_version=torch.__version__, hip=torch.version.hip,
                  device=torch.cuda.get_device_name(), latency_definition='BEV ready to neural control output; excludes sim.step and logging. IDM decision occurs inside native sim.step; its decision latency is missing.')
    write_json(out / 'report.json', report)
    with tarfile.open(out / 'runner_source.tar.gz', 'w:gz') as archive:
        for f in report['source_files']:
            archive.add(ROOT / f, arcname=f)
    policy = None if args.arm == 'idm' else frozen_policy(args.arm, out)
    if policy is not None:
        report.update(checkpoints=policy.provenance, parameter_count=policy.parameter_count)
    from metadrive.policy.env_input_policy import EnvInputPolicy
    from metadrive.policy.idm_policy import IDMPolicy
    from metadrive.envs.metadrive_env import METADRIVE_DEFAULT_CONFIG
    report['metadrive_native_source_sha256'] = sha(ROOT / '.deps/metadrive/metadrive/envs/metadrive_env.py')
    report['native_defaults'] = {k: METADRIVE_DEFAULT_CONFIG[k] for k in (
        'success_reward', 'driving_reward', 'speed_reward', 'out_of_road_penalty',
        'crash_vehicle_penalty', 'crash_object_penalty', 'crash_sidewalk_penalty',
        'out_of_road_cost', 'crash_vehicle_cost', 'crash_object_cost',
        'on_continuous_line_done', 'on_broken_line_done', 'out_of_road_done',
        'crash_vehicle_done', 'crash_object_done', 'traffic_mode')}
    write_json(out / 'report.json', report)
    _, restore = configure_egl(1)
    sim = None
    began = time.perf_counter()
    try:
        env_config = dict(config, agent_policy=IDMPolicy if args.arm == 'idm' else EnvInputPolicy)
        sim = make_semantic_env(SemanticBEVSpec(128, 64., 50.), episodes=200,
                                start_seed=0, horizon=args.max_steps, overrides=env_config,
                                road_termination=None, semantic_order=True)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for seed in seeds:
                records = []
                states = []
                encoded = []
                futures = {}
                obs, _ = sim.reset(seed=seed)
                provider = NativeBEVProvider(sim, BEVConfig())
                provider.reset(obs)
                history = ObservedHistory(.1, .1)
                bevs, past = deque(maxlen=4), deque(maxlen=4)
                tracker = None if policy is None else TrajectoryTracker(sim.agent, .1)
                initial_scene = scene_state(sim)
                start = dict(initial_state=vehicle_state(sim.agent).tolist(),
                             initial_scene=initial_scene.tolist(),
                             initial_bev_sha256=hashlib.sha256(provider.capture().raster.tobytes()).hexdigest(),
                             vehicle_model=type(sim.agent).__name__,
                             vehicle_dimensions_m=[float(sim.agent.LENGTH), float(sim.agent.WIDTH)],
                             lane_num=int(sim.current_map.config['lane_num']),
                             lane_width_m=float(sim.current_map.config['lane_width']),
                             map_blocks=[b.ID for b in sim.current_map.blocks])
                episode_tick = time.perf_counter()
                for step in range(args.max_steps):
                    state = vehicle_state(sim.agent)
                    bev = provider.capture().raster
                    tick = time.perf_counter()
                    decision = None
                    if policy is None:
                        action = np.zeros(2, np.float32)
                        decision_ms = float('nan')
                    else:
                        z = policy.encoder(torch.from_numpy(bev[None]).to('cuda'))
                        observed, motion, _ = history.append(step, z, float(state[3]), float(state[2]))
                        bevs.append(bev)
                        past.append(state)
                        decision = policy.decide(observed, motion, bev, list(bevs), list(past),
                                                 sim.agent.LENGTH, sim.agent.WIDTH, 42 + seed * 10000 + step)
                        action, _ = tracker.act(decision['plan'], sim.agent.speed)
                        torch.cuda.synchronize()
                        decision_ms = (time.perf_counter() - tick) * 1000
                        predicted = decision.pop('future')
                        index = decision['selected_index']
                        if predicted is not None and step % 50 == 0 and index >= 0:
                            futures[step] = predicted[index].astype(np.float16)
                        encoded.append(z[0].cpu().numpy().astype(np.float16))
                        k = len(decision['candidates'])
                        for key, shape in [('scores', (k,)), ('terms', (k, 8)), ('probability', (k, 2))]:
                            if decision.get(key) is None:
                                decision[key] = np.full(shape, np.nan)
                    obs, reward, tm, tc, info = sim.step(action)
                    provider.update(obs)
                    pipeline_ms = (time.perf_counter() - tick) * 1000
                    actual_action = np.asarray(info['raw_action'], np.float32)
                    states.append(state)
                    records.append(dict(bev=bev.copy(), action=actual_action, reward=float(reward),
                                        cost=float(info['cost']), route=float(info['route_completion']),
                                        events=[bool(info[k]) for k in EVENTS], decision=decision,
                                        decision_ms=decision_ms, pipeline_ms=pipeline_ms))
                    if step % 100 == 0:
                        write_json(out / 'progress.json', dict(arm=args.arm, seed=seed, step=step,
                                                             route_completion=float(info['route_completion'])))
                    if tm or tc:
                        break
                states.append(vehicle_state(sim.agent))
                elapsed = time.perf_counter() - episode_tick
                evidence = save_episode(out / f'seed_{seed:03d}', records, states, encoded, futures, args.arm)
                flags = np.array([r['events'] for r in records]).any(0)
                events = dict(zip(EVENTS, map(bool, flags)))
                reward_sum = sum(r['reward'] for r in records)
                if not np.isclose(reward_sum, info['episode_reward']):
                    raise AssertionError('Native reward accumulation mismatch')
                row = dict(seed=seed, steps=len(records), **start, **events,
                           route_completion=float(info['route_completion']),
                           episode_reward=float(info['episode_reward']), total_cost=sum(r['cost'] for r in records),
                           terminated=bool(tm), truncated=bool(tc), rollout_elapsed_s=elapsed,
                           decision_latency_ms=None if policy is None else stats([r['decision_ms'] for r in records[20:]]),
                           pipeline_latency_ms=stats([r['pipeline_ms'] for r in records[20:]]),
                           travelled_distance_m=float(np.linalg.norm(np.diff(np.asarray(states)[:, :2], axis=0), axis=1).sum()),
                           **evidence)
                report['episodes'].append(row)
                write_json(out / 'report.json', report)
                print(json.dumps({k: row[k] for k in ('seed', 'steps', 'arrive_dest', 'crash', 'out_of_road', 'route_completion', 'episode_reward')}), flush=True)
        report['completed'] = True
    except BaseException:
        report['error'] = traceback.format_exc()
        if 'records' in locals() and records:
            torch.save(dict(records=records, states=states), out / 'failure_mre.pt')
        raise
    finally:
        if sim is not None:
            sim.close()
        restore()
        report['elapsed_s'] = time.perf_counter() - began
        write_json(out / 'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['idm', 'mse', 'geometry', 'full'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--first-seed', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--smoke', action='store_true')
    evaluate(parser.parse_args())
