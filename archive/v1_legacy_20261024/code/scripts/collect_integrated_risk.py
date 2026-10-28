"""Actual candidate consequences with native termination and exact prefix replay."""
import argparse
from collections import deque
import json
from pathlib import Path
import tarfile
import traceback
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from configs.stage2 import BEVConfig
from controller.trajectory_tracker import TrajectoryTracker
from dataset.action_branches import remaining_command, realized_waypoints
from dataset.recording import vehicle_state
from env.data_v2 import configure_egl, NativeBEVProvider
from env.integrated import make_external_env, checked_step
from env.expert import NavigationExpert
from env.navigation import NavigationProvider, vehicle_parameters
from env.lane_permissions import maneuver_permissions
from env.traffic_teacher import spawn_slow_actor
from mini_wa_jepa.geometry import wrap_angle
from mini_wa_jepa.provenance import provenance, write_json
from mini_wa_jepa.visualization import RerunLogger
from models.navigation_encoder import load_navigation_encoder
from models.navigation_flow import NavigationFlow
from planning.integrated import curvature_braking
from planning.navigation_reference import navigation_reference
from scripts.collect_integrated_navigation import digest, EVENTS
from scripts.preflight_risk_collection import scene_state, compute_risk_targets, TYPED_CRASH_KEYS


def scenes_for(split, profile='wide'):
    base, count = {'probe':(4101000, 2), 'train':(4201000, 16), 'validation':(4301000, 8)}[split]
    if profile == 'balanced_restart':
        scenes = []
        for i in range(count):
            if i % 2 == 0:
                # 50% normal open road scenes: static launch (anchor 0) and stop-and-restart (anchor 50)
                scenes.append(dict(
                    seed=base+i, kind='open_road', controlled=True, has_npc=False,
                    density=0., map='S', lanes=3, lead_distance_m=None,
                    anchors=[0, 50] if split != 'probe' else [0]
                ))
            else:
                # 50% hazardous traffic scenes: close stationary lead, moving lead, CCC, TXC
                modes = ('stationary_lead', 'moving_lead', 'CCC', 'TXC')
                h_idx = (i // 2) % 4
                mode = modes[h_idx]
                scenes.append(dict(
                    seed=base+i, kind=mode, controlled=h_idx < 2, has_npc=True,
                    density=0. if h_idx < 2 else .3, map='S' if h_idx < 2 else mode,
                    lanes=1 if h_idx == 0 else 3,
                    lead_distance_m=7.5 if h_idx == 0 else 22.,
                    anchors=[70, 110] if split != 'probe' else [70]
                ))
        return scenes
    modes = ('stationary_lead', 'moving_lead', 'CCC', 'TXC')
    return [dict(seed=base+i, kind=modes[i%4], controlled=i%4<2,
                 density=0. if i%4<2 else .3, map='S' if i%4<2 else modes[i%4],
                 lanes=1 if profile=='near_single' and i%4==0 else 3,
                 lead_distance_m=7.5 if profile=='near_single' and i%4==0 else 22.) for i in range(count)]


def setup(scene, existing=None):
    overrides = dict(map=scene['map'], start_seed=scene['seed'], num_scenarios=1,
        traffic_density=scene['density'], horizon=400, decision_repeat=5, physics_world_step_size=.02,
        force_destroy=True, random_lane_width=False, random_lane_num=False, random_agent_model=False,
        random_spawn_lane_index=False,
        agent_configs={'default_agent':{'spawn_lane_index':('>', '>>', 1)}})
    if scene['controlled']:
        home=min(1,scene['lanes']-1)
        overrides.update(map=3, map_config=dict(type='block_sequence', config='S', lane_num=scene['lanes'], exit_length=250),
            agent_configs={'default_agent':{'spawn_lane_index':('>>', '>>>', home), 'spawn_longitude':5.}})
    sim = make_external_env(overrides) if existing is None else existing
    try:
        obs, _ = sim.reset(seed=scene['seed'])
        npc_ids = []
        if scene['controlled'] and scene.get('has_npc', True):
            npc = spawn_slow_actor(sim, sim.agent.lane, 5.+scene['lead_distance_m'], scene['seed'],
                                  stationary=scene['kind']=='stationary_lead'); npc_ids.append(npc.id)
            before = scene_state(sim)
            sim.engine.taskMgr.step(); sim.engine.taskMgr.step()
            np.testing.assert_array_equal(before, scene_state(sim))
        provider = NativeBEVProvider(sim, BEVConfig()); provider.reset(None if (scene['controlled'] and scene.get('has_npc', True)) else obs)
        return sim, provider, npc_ids
    except BaseException:
        sim.close(); raise


def candidate_commands(flow, encoder, history_bev, observed_states, ego, navigation, vehicle, permission, seed):
    encoded = encoder(torch.from_numpy(history_bev).to('cuda'))[None]
    reference = navigation_reference(ego[None], navigation.points[None], navigation.valid[None], vehicle[None])
    args = [encoded, torch.from_numpy(ego[None]).to('cuda'), torch.from_numpy(navigation.points[None]).to('cuda'),
            torch.from_numpy(navigation.valid[None]).to('cuda'), torch.from_numpy(vehicle[None]).to('cuda'),
            torch.from_numpy(reference).to('cuda')]
    candidates = flow.sample(*args, candidates=8, steps=10, seed=seed,
                             maneuver_allowed=torch.from_numpy(permission[None]).to('cuda'))[0].cpu().numpy()
    # Four separately identified diagnostic commands, never expert imitation labels.
    t = np.arange(1, 11)*.2; sentinels = []
    for lateral, speed in [(0., 15.), (12., 10.), (-12., 10.)]:
        q = t/2.; y = lateral*(3*q*q-2*q*q*q)
        dy = lateral*(6*q-6*q*q)/2.
        sentinels.append(np.column_stack([speed*t, y, np.full(10, speed), np.arctan2(dy, speed)]).astype(np.float32))
    sentinels.append(curvature_braking(float(observed_states[-1, 3]), float(ego[-1, 1])))
    return np.concatenate([candidates, np.stack(sentinels)]), ego


def collect(args):
    if not torch.version.hip or torch.version.cuda:
        raise RuntimeError('Native ROCm required')
    torch.set_num_threads(4); args.output.mkdir(parents=True, exist_ok=False)
    selected = torch.load(args.flow, map_location='cpu', weights_only=False)
    if selected['seed'] != 20261023 or selected['arch'].get('permission_dim') != 2:
        raise ValueError('Pre-registered first mixed schema3.1 Flow seed required')
    flow = NavigationFlow(**selected['arch'], residual_std=selected['residual_std']).to('cuda').eval()
    flow.load_state_dict(selected['model'])
    if digest(selected['encoder_checkpoint']) != selected['encoder_sha256']:
        raise ValueError('Encoder changed')
    encoder, _ = load_navigation_encoder('joint', selected['encoder_checkpoint'], 'cuda')
    scenes = scenes_for(args.split,args.context_profile); default_anchors = [70] if args.split=='probe' else [70, 110]
    names = [f'flow_{i}' for i in range(8)]+['diagnostic_fast', 'diagnostic_left12m', 'diagnostic_right12m', 'diagnostic_brake']
    report = dict(**provenance(), completed=False, split=args.split, scenes=scenes, anchors=default_anchors,
        context_profile=args.context_profile,
        reuse_engine=args.reuse_engine,force_destroy_between_resets=True,
        groups=[], prefix_failures=[], candidate_names=names, control_verified=True, native_termination=True,
        flow_checkpoint=str(args.flow), flow_sha256=digest(args.flow), development_only=True,
        stage7_acceptance=False, protocol_sha256=digest('docs/STAGE7_INTEGRATED_PROTOCOL.md'))
    write_json(args.output/'manifest.json', report)
    with tarfile.open(args.output/'source_snapshot.tar.gz', 'w:gz') as arc:
        for file in report['source_files']:
            arc.add(file, arcname=file)
        arc.add('docs/STAGE7_INTEGRATED_PROTOCOL.md')
    _, restore = configure_egl(1)
    try:
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for scene in scenes:
                anchors = scene.get('anchors', default_anchors)
                for anchor in anchors:
                    folder = args.output/f"seed_{scene['seed']}_step_{anchor}"; folder.mkdir()
                    reference = None; commands = None; group = dict(**scene, anchor=anchor, branches=[])
                    reused_sim = None
                    for ci, name in enumerate(names):
                        sim = None; branch_complete = False
                        try:
                            sim, provider, npc_ids = setup(scene,reused_sim)
                            if args.reuse_engine:
                                reused_sim = sim
                            initial_scene = scene_state(sim)
                            expert = NavigationExpert(sim); past_bev = []; past_state = []; prefix_actions = []; prefix_applied = []
                            previous_state = vehicle_state(sim.agent)
                            brake_tracker = TrajectoryTracker(sim.agent, 0.1) if (anchor == 50 and scene['kind'] == 'open_road') else None
                            for step in range(anchor):
                                current_state = vehicle_state(sim.agent)
                                past_bev.append(provider.capture().raster.copy()); past_state.append(current_state)
                                if anchor == 50 and scene['kind'] == 'open_road' and step >= 20:
                                    yaw = float(wrap_angle(current_state[2] - previous_state[2])) / 0.1
                                    plan = curvature_braking(float(current_state[3]), yaw)
                                    action, _ = brake_tracker.act(plan, current_state[3])
                                else:
                                    action = expert.act()[0]
                                previous_state = current_state
                                obs, _, tm, tc, info = checked_step(sim, action); provider.update(obs)
                                prefix_actions.append(action); prefix_applied.append(info['raw_action'])
                                if tm or tc:
                                    raise RuntimeError(f'Pre-registered prefix unavailable at {step}: native termination')
                                if any(key not in sim.engine.get_objects() for key in npc_ids):
                                    raise RuntimeError('Controlled NPC removed during prefix')
                            past_bev.append(provider.capture().raster.copy()); past_state.append(vehicle_state(sim.agent))
                            while len(past_bev) < 4:
                                past_bev.insert(0, past_bev[0])
                                past_state.insert(0, past_state[0])
                            past_state = np.asarray(past_state, np.float32)
                            history_bev = np.asarray(past_bev[-4:]); observed = past_state[-4:]
                            yaw = np.r_[0., np.diff(np.unwrap(past_state[:, 2]))/.1]
                            ego = np.column_stack([past_state[-4:,3]/10., yaw[-4:]]).astype(np.float32)
                            nav = NavigationProvider(sim).capture(); permission = maneuver_permissions(sim)
                            if reference is None:
                                reference = dict(history=history_bev.copy(), states=observed.copy(),
                                    scene=scene_state(sim), initial_scene=initial_scene, ego=ego,
                                    prefix_actions=np.asarray(prefix_actions), prefix_applied=np.asarray(prefix_applied))
                                # Same causal observation convention used by the training cache.
                                commands, _ = candidate_commands(flow, encoder, history_bev, observed, ego, nav,
                                    vehicle_parameters(sim.agent), permission, 20261025+scene['seed']+anchor)
                                np.savez_compressed(folder/'context.npz', **reference, command=commands,
                                    route_points=nav.points, route_valid=nav.valid, maneuver_allowed=permission)
                            else:
                                np.testing.assert_array_equal(initial_scene, reference['initial_scene'])
                                np.testing.assert_array_equal(history_bev, reference['history'])
                                np.testing.assert_allclose(observed, reference['states'], atol=1e-5, rtol=0)
                                np.testing.assert_allclose(scene_state(sim), reference['scene'], atol=1e-5, rtol=0)
                                np.testing.assert_allclose(ego, reference['ego'], atol=1e-5, rtol=0)
                                np.testing.assert_array_equal(prefix_actions, reference['prefix_actions'])
                            command = commands[ci]; tracker = TrajectoryTracker(sim.agent, .1)
                            states = [vehicle_state(sim.agent)]; bevs = []; actions = []; applied = []; events = []; rewards = []; costs = []; typed = []
                            for step in range(20):
                                plan = remaining_command(command, reference['states'][-1], states[-1], step*.1)
                                action, _ = tracker.act(plan, float(states[-1][3]))
                                obs, reward, tm, tc, info = checked_step(sim, action); provider.update(obs)
                                states.append(vehicle_state(sim.agent)); bevs.append(provider.capture().raster.copy())
                                actions.append(action); applied.append(info['raw_action']); rewards.append(reward); costs.append(float(info.get('cost', 0)))
                                events.append([bool(info.get(k)) for k in EVENTS])
                                typed.append([bool(info.get(k)) for k in TYPED_CRASH_KEYS])
                                if tm or tc:
                                    break
                            states = np.asarray(states, np.float32); events = np.asarray(events, bool); n = len(actions)
                            risk, risk_mask, *_ = compute_risk_targets(events[:, [0, 4]])
                            path = folder/f'{name}.npz'
                            np.savez_compressed(path, command=command, states=states, future_bev=np.asarray(bevs),
                                action=np.asarray(actions), applied_action=np.asarray(applied), events=events,
                                typed_events=np.asarray(typed, bool),
                                reward=rewards, cost=costs, risk=risk, risk_mask=risk_mask,
                                observed_history=history_bev, observed_states=observed)
                            with RerunLogger(app_id='integrated-risk-branch', spawn=False, save_path=folder/f'{name}.rrd') as log:
                                log.set_step(0,0.)
                                log.log_context(static=True, seed=scene['seed'], anchor=anchor, candidate=name,
                                                source='learned_flow' if ci<8 else 'diagnostic_rule_not_expert')
                                log.log_candidate(name, commanded=command,
                                    actual=realized_waypoints(states, 0, np.arange(2, n+1, 2)),
                                    predicted_latent=np.empty((0,64,256)), target_latent=None)
                                for step in range(n):
                                    log.set_step(step, step*.1); log.log_state(states[step])
                                    log.log_bev(history_bev[-1] if step==0 else bevs[step-1])
                                    log.log_context(executed_action_json=json.dumps(np.asarray(actions[step]).tolist()))
                            branch = dict(name=name, steps=n, risk=risk.tolist(), risk_mask=risk_mask.tolist(),
                                path=str(path.relative_to(args.output)), sha256=digest(path),
                                rrd_sha256=digest(folder/f'{name}.rrd'))
                            group['branches'].append(branch)
                            print(json.dumps(dict(seed=scene['seed'], anchor=anchor, **branch)), flush=True)
                            branch_complete = True
                        except BaseException:
                            write_json(folder/f'{name}_failure_mre.json', dict(error=traceback.format_exc(),
                                branch=name, scene=scene, anchor=anchor))
                            raise
                        finally:
                            if sim is not None and (not args.reuse_engine or not branch_complete or ci==len(names)-1):
                                sim.close()
                                reused_sim = None
                    group.update(context=str((folder/'context.npz').relative_to(args.output)),
                                 context_sha256=digest(folder/'context.npz'))
                    report['groups'].append(group); write_json(args.output/'manifest.json', report)
        report['completed'] = True
    except BaseException:
        report['error'] = traceback.format_exc(); raise
    finally:
        restore(); write_json(args.output/'manifest.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--flow', type=Path, required=True)
    p.add_argument('--split', choices=('probe', 'train', 'validation'), required=True)
    p.add_argument('--context-profile',choices=('wide','near_single','balanced_restart'),default='wide')
    p.add_argument('--reuse-engine',action='store_true',help='Native force_destroy resets within one context; strict replay checks remain')
    p.add_argument('--output', type=Path, required=True); collect(p.parse_args())
