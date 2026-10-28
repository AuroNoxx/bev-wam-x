"""Matched-input development driving with native outcomes and explicit sources."""
import argparse
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import tarfile
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from configs.integrated import IntegratedConfig
from controller.trajectory_tracker import TrajectoryTracker
from dataset.recording import vehicle_state, future_labels_v2
from env.data_v2 import configure_egl
from env.integrated import checked_step
from env.integrated_scenarios import setup_driving_scene
from env.navigation import NavigationProvider, vehicle_parameters
from env.lane_permissions import maneuver_permissions
from evaluation.history import ObservedHistory
from mini_wa_jepa.geometry import wrap_angle
from mini_wa_jepa.provenance import provenance, write_json
from mini_wa_jepa.visualization import RerunLogger
from models.navigation_encoder import load_navigation_encoder
from models.navigation_head import NavigationTrajectoryHead
from models.navigation_flow import NavigationFlow
from models.integrated_world import load_integrated_world
from models.risk_head import ConsequenceRisk
from planning.integrated_policy import IntegratedPolicy
from planning.navigation_reference import navigation_reference
from scripts.collect_integrated_navigation import digest, EVENTS
from scripts.preflight_risk_collection import scene_state


class ZeroWorld(nn.Module):
    def forward(self, history, ego, command):
        return torch.zeros_like(history)


def evaluate(args):
    if not torch.version.hip or torch.version.cuda:
        raise RuntimeError('Native ROCm required')
    torch.set_num_threads(4); args.output.mkdir(parents=True, exist_ok=False)
    selected = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if selected['schema_version'] != 3 or selected['seed'] not in (20261023, 20261024) or selected['arch'].get('permission_dim')!=2:
        raise ValueError('Registered mixed schema3.1 checkpoint required')
    source = json.loads((args.validation/'manifest.json').read_text())
    assert source['completed'] and source['split']=='validation' and source['input_profile']=='schema3.1_static_maneuver_permissions'
    if digest(selected['encoder_checkpoint']) != selected['encoder_sha256']:
        raise ValueError('Encoding checkpoint changed')
    ordinary = [s for s in source['scenes'] if 'case' not in s]
    traffic = [s for s in source['scenes'] if 'case' in s]
    scenes = ordinary[:6]+traffic[:4] if args.scope=='pilot10' else ordinary+traffic
    report = dict(**provenance(), completed=False, arm=args.arm, training_seed=selected['seed'],
        checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint), scenes=scenes, episodes=[],
        control_verified=True, development_only=True, stage7_acceptance=False, test_access=False,
        scope=args.scope, recovery_pulses=False, native_termination=True,
        source_protocol_sha256=digest('docs/STAGE7_INTEGRATED_PROTOCOL.md'))
    write_json(args.output/'report.json', report)
    with tarfile.open(args.output/'source_snapshot.tar.gz', 'w:gz') as arc:
        for f in report['source_files']:
            arc.add(f, arcname=f)
        arc.add('docs/STAGE7_INTEGRATED_PROTOCOL.md')
    encoder, _ = load_navigation_encoder(selected.get('encoder_kind','joint'), selected['encoder_checkpoint'], 'cuda')
    policy = None; head = None
    if args.arm=='mse':
        head = NavigationTrajectoryHead(**selected['arch']).to('cuda').eval(); head.load_state_dict(selected['model'])
    else:
        flow = NavigationFlow(**selected['arch'], residual_std=selected['residual_std']).to('cuda').eval()
        flow.load_state_dict(selected['model']); world = risk = thresholds = None
        if args.arm in ('full','zero'):
            if args.risk is None:
                raise ValueError('Calibrated matched risk checkpoint required')
            risk_saved = torch.load(args.risk, map_location='cpu', weights_only=False)
            risk = ConsequenceRisk(**risk_saved['arch']).to('cuda').eval(); risk.load_state_dict(risk_saved['model'])
            assert risk.use_world == (args.arm=='full')
            thresholds = risk_saved['thresholds']
            if args.arm=='full':
                bundle = load_integrated_world('cuda'); encoder = bundle.encoder; world = bundle.world
            else:
                world = ZeroWorld()
            report['risk_checkpoint_sha256'] = digest(args.risk)
        policy = IntegratedPolicy(encoder, flow, world, risk, thresholds).eval()
    report['loaded_parameters'] = sum(p.numel() for p in (policy.parameters() if policy is not None else list(encoder.parameters())+list(head.parameters())))
    _, restore = configure_egl(1)
    try:
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for scene in scenes:
                folder = args.output/f"seed_{scene['seed']}"; folder.mkdir(); sim = None; rows = []; states = []
                try:
                    sim, provider, npc_ids, overrides = setup_driving_scene(scene)
                    params = vehicle_parameters(sim.agent); home_lane = sim.agent.lane
                    home_id = min(1, scene.get('lanes',3)-1)
                    nav = NavigationProvider(sim, replace(IntegratedConfig(), preferred_lane=home_id))
                    write_json(folder/'initial_scene.json', dict(overrides=overrides, actor_states=scene_state(sim).tolist(),
                        bev_sha256=__import__('hashlib').sha256(provider.capture().raster.tobytes()).hexdigest(),
                        vehicle_parameters=params.tolist(), map_hash=provider.geometry_hash, calibration=provider.calibration))
                    tracker = TrajectoryTracker(sim.agent, .1); history = ObservedHistory(.1,.1)
                    observed_bev = deque(maxlen=4); observed_states = deque(maxlen=4)
                    controlled = 'case' in scene; info = {}; departed = False; stable = 0; goal_complete = False; removed_ids = []
                    with RerunLogger(app_id='integrated-driving-development', spawn=False, save_path=folder/'driving.rrd') as log:
                        log.log_context(static=True, seed=scene['seed'], arm=args.arm, no_teacher_control=True,
                                        npc_truth_for_model=False, input_profile=source['input_profile'])
                        for step in range(600 if controlled else 800):
                            bev = provider.capture().raster.copy(); state = vehicle_state(sim.agent)
                            states.append(state); observed_bev.append(bev); observed_states.append(state)
                            torch.cuda.synchronize(); tick = time.perf_counter()
                            route = nav.capture(); permission = maneuver_permissions(sim)
                            encoded = encoder(torch.from_numpy(bev[None]).to('cuda'))
                            h, ego, _ = history.append(step, encoded, float(state[3]), float(state[2]))
                            if policy is None:
                                ref = navigation_reference(ego, route.points[None], route.valid[None], params[None])
                                plan = head(h, torch.from_numpy(ego).to('cuda'), torch.from_numpy(route.points[None]).to('cuda'),
                                    torch.from_numpy(route.valid[None]).to('cuda'), torch.from_numpy(params[None]).to('cuda'),
                                    reference=torch.from_numpy(ref).to('cuda'), maneuver_allowed=torch.from_numpy(permission[None]).to('cuda'))[0].cpu().numpy()
                                decision = dict(plan=plan, candidates=plan[None], reference=ref[0], selected_index=0,
                                    selected_id='MSE', selection_source='navigation_MSE_residual', fallback=False,
                                    fallback_geometry_safe=None, scores=None, terms=None, rejection_reasons=None, predicted=None)
                            else:
                                decision = policy.decide(h, ego, bev, list(observed_bev), list(observed_states), route,
                                                         params, permission, 20261025+scene['seed']*1000+step)
                            action, _ = tracker.act(decision['plan'], float(state[3]))
                            torch.cuda.synchronize(); elapsed = (time.perf_counter()-tick)*1000
                            obs, reward, tm, tc, info = checked_step(sim, action); provider.update(obs)
                            # Diagnostic NPC truth below is used solely for evaluation/stopping,
                            # never passed to the encoder, generator, scorer or controller.
                            lat = float('nan'); lead_gap = float('nan')
                            if controlled:
                                live = sim.engine.get_objects(); removed_ids = [key for key in npc_ids if key not in live]
                                own_s, lat = home_lane.local_coordinates(sim.agent.position)
                                departed = departed or abs(lat)>2.5
                                if not removed_ids:
                                    lead = live[npc_ids[0]]; lead_s, _ = home_lane.local_coordinates(lead.position)
                                    lead_gap = own_s-lead_s-(sim.agent.LENGTH+lead.LENGTH)*.5
                                    angle = abs(float(wrap_angle(sim.agent.heading_theta-home_lane.heading_theta_at(own_s))))
                                    if scene['case'].startswith('allowed'):
                                        condition = departed and lead_gap>10 and abs(lat)<.35 and angle<.08
                                        stable = stable+1 if condition else 0; goal_complete = stable>=10
                                    else:
                                        gap = lead_s-own_s-(sim.agent.LENGTH+lead.LENGTH)*.5
                                        condition = not departed and abs(lat)<.35 and .5<=sim.agent.speed<=3 and 3<=gap<=18
                                        stable = stable+1 if condition else 0; goal_complete = stable>=80
                            row = dict(bev=bev, route_points=route.points, route_valid=route.valid, maneuver_allowed=permission,
                                plan=decision['plan'], reference=decision['reference'], candidates=decision['candidates'],
                                selected_index=decision['selected_index'], fallback=decision['fallback'],
                                fallback_geometry_safe=-1 if decision['fallback_geometry_safe'] is None else int(decision['fallback_geometry_safe']),
                                action=action, applied_action=np.asarray(info['raw_action'],np.float32), decision_ms=elapsed,
                                reward=reward, cost=float(info.get('cost',0)), events=np.array([bool(info.get(k)) for k in EVENTS]),
                                route_completion=float(info.get('route_completion',0)), diagnostic_home_lateral_m=lat,
                                diagnostic_lead_rear_clearance_m=lead_gap, diagnostic_goal_complete=goal_complete)
                            if policy is not None:
                                row.update(scores=decision['scores'], terms=decision['terms'],
                                    rejection_json=json.dumps(decision['rejection_reasons']),
                                    risk_probability=decision['probability'], risk_assessed=decision['risk_assessed'])
                            rows.append(row)
                            log.set_step(step,step*.1); log.log_bev(bev); log.log_state(state)
                            log.log_navigation(route.points,route.valid,source=route.source)
                            log.log_planning(candidates=decision['candidates'],selected_id=decision['selected_id'],
                                selected_trajectory=decision['plan'],selection_source=decision['selection_source'],
                                scores=decision['scores'],terms=decision['terms'],rejection_reasons=decision['rejection_reasons'],
                                predicted=decision['predicted'] if step%100==0 else None,executed_action=action)
                            log.log_context(fallback_geometry_safe=str(decision['fallback_geometry_safe']),
                                risk_assessed_json=json.dumps(decision.get('risk_assessed',np.zeros(len(decision['candidates']),bool)).tolist()))
                            log.log_telemetry(speed_m_s=state[3],steering=action[0],throttle_brake=action[1],route_completion=row['route_completion'])
                            if tm or tc or (controlled and (goal_complete or removed_ids)):
                                break
                        states.append(vehicle_state(sim.agent)); states=np.asarray(states,np.float32)
                        target, valid, _ = future_labels_v2(states,np.zeros(len(states),bool),10,2)
                        for i in range(len(rows)):
                            log.set_step(i,i*.1);log.log_trajectories(ground_truth=target[i][valid[i]])
                    arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
                    arrays.update(states=states,target=target[:-1],valid=valid[:-1]);np.savez_compressed(folder/'actual.npz',**arrays)
                    timing=arrays['decision_ms'][20:]
                    ep=dict(**scene,steps=len(rows),outcome={k:bool(info.get(k)) for k in EVENTS},
                        route_completion=float(info.get('route_completion',0)),reward=float(arrays['reward'].sum()),cost=float(arrays['cost'].sum()),
                        maneuver_complete=bool(controlled and goal_complete and not arrays['events'][:,[0,4]].any()),
                        removed_npc_ids=removed_ids,probe_stop='native' if tm or tc else ('maneuver_goal' if goal_complete else 'controlled_actor_removed'),
                        fallback_fraction=float(arrays['fallback'].mean()),decision_p95_ms=float(np.percentile(timing,95)) if len(timing) else None,
                        raw_sha256=digest(folder/'actual.npz'),rrd_sha256=digest(folder/'driving.rrd'))
                    report['episodes'].append(ep);write_json(args.output/'report.json',report);print(json.dumps(ep),flush=True)
                except BaseException:
                    write_json(folder/'failure_mre.json',dict(error=traceback.format_exc(),observed_steps=len(rows)))
                    if rows:
                        np.savez_compressed(folder/'partial.npz',states=np.asarray(states),action=np.asarray([r['action'] for r in rows]))
                    raise
                finally:
                    if sim is not None:
                        sim.close()
        report['completed']=True
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        restore();write_json(args.output/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--arm',choices=('mse','geometry','full','zero'),required=True);p.add_argument('--risk',type=Path)
    p.add_argument('--scope',choices=('pilot10','all24'),default='pilot10');p.add_argument('--output',type=Path,required=True)
    evaluate(p.parse_args())
