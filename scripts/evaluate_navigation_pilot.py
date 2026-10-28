"""Single-arm navigation-head validation rollout; not Stage8/scientific acceptance."""
import argparse
from pathlib import Path
import json
import tarfile
import time
import traceback
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from configs.integrated import IntegratedConfig
from configs.stage2 import BEVConfig
from controller.trajectory_tracker import TrajectoryTracker
from dataset.recording import vehicle_state, future_labels_v2
from env.data_v2 import configure_egl, NativeBEVProvider
from env.semantic_bev import make_semantic_env
from env.integrated import make_external_env, checked_step
from env.navigation import NavigationProvider, vehicle_parameters
from env.lane_permissions import maneuver_permissions
from evaluation.history import ObservedHistory
from mini_wa_jepa.provenance import provenance, write_json
from mini_wa_jepa.visualization import RerunLogger
from models.navigation_head import NavigationTrajectoryHead
from models.navigation_encoder import load_navigation_encoder
from scripts.collect_integrated_navigation import EVENTS, digest


def evaluate(args):
    if not torch.version.hip or torch.version.cuda:
        raise RuntimeError('Native ROCm required')
    torch.set_num_threads(4)
    if (args.checkpoint.parent.parent/'INVALID_CONTROL_WIRING.json').exists():
        raise ValueError('Invalid-control training checkpoint cannot be evaluated as a trained navigation policy')
    selected = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if selected['schema_version'] != 3 or selected['seed'] != 20261023:
        raise ValueError('Protocol selects first navigation-head seed only')
    if digest(selected['encoder_checkpoint']) != selected['encoder_sha256']:
        raise ValueError('Encoder checkpoint checksum mismatch')
    source = json.loads((args.validation/'manifest.json').read_text())
    if not source.get('control_verified') or (args.validation/'INVALID_CONTROL_WIRING.json').exists():
        raise ValueError('Validation scene source lacks verified external control')
    if not source['completed'] or source['split'] != 'validation':
        raise ValueError('Only completed validation scenes permitted')
    scenes = source['scenes'][:6]
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(**provenance(), completed=False, stage=7, validation_only=True,
                  checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
                  source_protocol_sha256=digest('docs/STAGE7_INTEGRATED_PROTOCOL.md'),
                  scenes=scenes, episodes=[], scientific_acceptance=False, control_verified=True,
                  reference_prior=selected['arch'].get('reference_prior', False), prior_only=args.prior_only,
                  encoder_kind=selected.get('encoder_kind', 'joint'),
                  decision_latency='BEV-ready to control, includes navigation/encoder/history/head/tracker; first20 excluded')
    write_json(args.output/'report.json', report)
    with tarfile.open(args.output/'source_snapshot.tar.gz', 'w:gz') as archive:
        for file in report['source_files']:
            archive.add(file, arcname=file)
        archive.add('docs/STAGE7_INTEGRATED_PROTOCOL.md')
    encoder, _ = load_navigation_encoder(selected.get('encoder_kind', 'joint'), selected['encoder_checkpoint'], 'cuda')
    head = NavigationTrajectoryHead(**selected['arch']).to('cuda').eval()
    head.load_state_dict(selected['model'])
    report['encoder_parameters'] = sum(p.numel() for p in encoder.parameters())
    report['head_parameters'] = sum(p.numel() for p in head.parameters())
    report['loaded_navigation_parameters'] = report['encoder_parameters']+report['head_parameters']
    _, restore = configure_egl(1)
    try:
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for scene in scenes:
                folder = args.output/f"seed_{scene['seed']}"; folder.mkdir()
                overrides = dict(map=scene['map'], start_seed=scene['seed'], num_scenarios=1,
                                 traffic_density=scene['density'], horizon=800,
                                 decision_repeat=5, physics_world_step_size=.02, force_destroy=True,
                                 random_lane_width=False, random_lane_num=False, random_agent_model=False,
                                 random_spawn_lane_index=False,
                                 agent_configs={'default_agent': {'spawn_lane_index': ('>', '>>', 1)}})
                sim = make_external_env(overrides)
                rows = []; states = []; info = {}
                try:
                    obs, _ = sim.reset(seed=scene['seed'])
                    provider = NativeBEVProvider(sim, BEVConfig()); provider.reset(obs)
                    nav = NavigationProvider(sim); tracker = TrajectoryTracker(sim.agent, .1)
                    history = ObservedHistory(.1, .1)
                    vehicle = torch.from_numpy(vehicle_parameters(sim.agent)[None]).to('cuda')
                    for step in range(800):
                        bev = provider.capture().raster.copy(); state = vehicle_state(sim.agent)
                        torch.cuda.synchronize(); started = time.perf_counter()
                        route = nav.capture()
                        encoded = encoder(torch.from_numpy(bev[None]).to('cuda'))
                        hist, ego, _ = history.append(step, encoded, state[3], state[2])
                        reference = None
                        if selected['arch'].get('reference_prior') or args.prior_only:
                            from planning.navigation_reference import navigation_reference
                            reference = navigation_reference(ego, route.points[None], route.valid[None], vehicle.cpu().numpy())[0]
                        if args.prior_only:
                            plan = reference
                        else:
                            permission_args = (dict(maneuver_allowed=torch.from_numpy(maneuver_permissions(sim)[None]).to('cuda'))
                                               if selected['arch'].get('permission_dim', 0) else {})
                            plan = head(hist, torch.from_numpy(ego).to('cuda'),
                                        torch.from_numpy(route.points[None]).to('cuda'),
                                        torch.from_numpy(route.valid[None]).to('cuda'), vehicle,
                                        reference=None if reference is None else torch.from_numpy(reference[None]).to('cuda'),
                                        **permission_args)[0].cpu().numpy()
                        action, _ = tracker.act(plan, state[3])
                        torch.cuda.synchronize(); elapsed = (time.perf_counter()-started)*1000
                        obs, reward, tm, tc, info = checked_step(sim, action); provider.update(obs)
                        states.append(state)
                        rows.append(dict(bev=bev, plan=plan, action=action,
                                         applied_action=np.asarray(info['raw_action'], np.float32), route_points=route.points,
                                         route_valid=route.valid, decision_ms=elapsed, reward=reward,
                                         cost=float(info.get('cost', 0)),
                                         events=np.array([bool(info.get(k)) for k in EVENTS]),
                                         route_completion=float(info.get('route_completion', 0))))
                        if reference is not None:
                            rows[-1]['reference'] = reference
                        if tm or tc:
                            break
                    states.append(vehicle_state(sim.agent)); states = np.asarray(states, np.float32)
                    targets, valid, _ = future_labels_v2(states, np.zeros(len(states), bool), 10, 2)
                    arrays = {k:np.asarray([row[k] for row in rows]) for k in rows[0]}
                    arrays.update(states=states, target=targets[:-1], valid=valid[:-1])
                    np.savez_compressed(folder/'actual.npz', **arrays)
                    with RerunLogger(app_id='navigation-validation', spawn=False, save_path=folder/'rollout.rrd') as log:
                        log.log_context(static=True, seed=scene['seed'], source='navigation_MSE_validation',
                                        no_teacher_control=True, future_semantics='actual_future_under_replanning')
                        for i, row in enumerate(rows):
                            log.set_step(i, i*.1); log.log_bev(row['bev']); log.log_state(states[i])
                            log.log_navigation(row['route_points'], row['route_valid'], source='map_assisted_route_lane_centerline')
                            if not args.prior_only:
                                log.log_planning(candidates=row['plan'][None], selected_id='MSE',
                                                 selected_trajectory=row['plan'], selection_source='navigation_MSE_head',
                                                 executed_action=row['action'])
                            else:
                                log.log_trajectories(planned=row['plan'])
                                log.log_context(selected_id='NAV_PRIOR', selection_source='declared_navigation_prior_only',
                                                executed_action_json=json.dumps(row['action'].tolist()))
                            if 'reference' in row:
                                log.log_candidate('NAV_PRIOR', commanded=row['reference'], actual=np.empty((0, 4)),
                                                  predicted_latent=np.empty((0, 64, 256)), target_latent=None)
                                log.log_context(selection_source='declared_navigation_prior_only' if args.prior_only
                                                else 'navigation_physical_prior_plus_learned_residual',
                                                prior_is_rule=True)
                            log.log_trajectories(ground_truth=targets[i][valid[i]])
                            log.log_telemetry(speed_m_s=states[i, 3], steering=row['action'][0],
                                              throttle_brake=row['action'][1], route_completion=row['route_completion'])
                    timing = arrays['decision_ms'][20:]
                    ep = dict(scene, steps=len(rows), outcome={k: bool(info.get(k)) for k in EVENTS},
                              route_completion=float(info.get('route_completion', 0)),
                              decision_p95_ms=float(np.percentile(timing, 95)) if len(timing) else None,
                              raw_sha256=digest(folder/'actual.npz'), rerun_sha256=digest(folder/'rollout.rrd'))
                    report['episodes'].append(ep); write_json(args.output/'report.json', report)
                    print(json.dumps(ep), flush=True)
                except BaseException:
                    write_json(folder/'failure_mre.json', dict(scene=scene, observed_steps=len(rows),
                                                             traceback=traceback.format_exc()))
                    if rows:
                        np.savez_compressed(folder/'partial.npz', states=np.asarray(states),
                                            actions=np.asarray([r['action'] for r in rows]))
                    raise
                finally:
                    sim.close()
        report['completed'] = True
    except BaseException:
        report['error'] = traceback.format_exc(); raise
    finally:
        restore(); write_json(args.output/'report.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prior-only', action='store_true', help='Diagnostic of the declared map/controller prior without learned residual')
    evaluate(p.parse_args())
