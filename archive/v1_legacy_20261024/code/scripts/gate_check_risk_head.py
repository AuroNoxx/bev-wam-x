"""Targeted gate check for the risk head before full closed-loop evaluations.

Verifies:
1. Open-road launch: vehicle starts from rest and accelerates without being blocked by false crash risk.
2. Stop-and-resume: vehicle resumes moving forward after stopping on an open road.
3. Hazardous braking: vehicle safely brakes and avoids collision when facing a close stationary lead.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from controller.trajectory_tracker import TrajectoryTracker
from dataset.recording import vehicle_state
from env.data_v2 import configure_egl, NativeBEVProvider
from configs.stage2 import BEVConfig
from configs.integrated import IntegratedConfig
from env.integrated import make_external_env, checked_step
from env.navigation import NavigationProvider, vehicle_parameters
from env.lane_permissions import maneuver_permissions
from env.traffic_teacher import spawn_slow_actor
from evaluation.history import ObservedHistory
from models.navigation_encoder import load_navigation_encoder
from models.navigation_flow import NavigationFlow
from models.integrated_world import load_integrated_world
from models.risk_head import ConsequenceRisk
from planning.integrated_policy import IntegratedPolicy


def run_gate_check(flow_ckpt, risk_ckpt, output_dir=None):
    if not torch.version.hip or torch.version.cuda:
        raise RuntimeError('Native ROCm required')
    torch.set_num_threads(4)

    flow_saved = torch.load(flow_ckpt, map_location='cpu', weights_only=False)
    risk_saved = torch.load(risk_ckpt, map_location='cpu', weights_only=False)

    encoder, _ = load_navigation_encoder('joint', flow_saved['encoder_checkpoint'], 'cuda')
    flow = NavigationFlow(**flow_saved['arch'], residual_std=flow_saved['residual_std']).to('cuda').eval()
    flow.load_state_dict(flow_saved['model'])

    risk = ConsequenceRisk(**risk_saved['arch']).to('cuda').eval()
    risk.load_state_dict(risk_saved['model'])
    thresholds = risk_saved['thresholds']

    world_bundle = load_integrated_world('cuda')
    world = world_bundle.world

    policy = IntegratedPolicy(encoder, flow, world, risk, thresholds).eval()

    _, restore = configure_egl(1)
    results = {}
    try:
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            # Test 1: Open road launch from rest
            print("--- Running Gate Test 1: Open Road Launch ---", flush=True)
            overrides = dict(
                map=3, map_config=dict(type='block_sequence', config='S', lane_num=3, exit_length=250),
                start_seed=4300000, num_scenarios=1, traffic_density=0., horizon=200, decision_repeat=5,
                physics_world_step_size=.02, force_destroy=True,
                agent_configs={'default_agent': {'spawn_lane_index': ('>>', '>>>', 1), 'spawn_longitude': 5.}}
            )
            sim = make_external_env(overrides)
            try:
                obs, _ = sim.reset(seed=4300000)
                provider = NativeBEVProvider(sim, BEVConfig()); provider.reset(obs)
                nav = NavigationProvider(sim, IntegratedConfig())
                tracker = TrajectoryTracker(sim.agent, .1)
                history = ObservedHistory(.1, .1)
                observed_bev = []; observed_states = []
                fallbacks = 0
                speeds = []
                for step in range(35):
                    bev = provider.capture().raster.copy(); state = vehicle_state(sim.agent)
                    observed_bev.append(bev); observed_states.append(state)
                    encoded = encoder(torch.from_numpy(bev[None]).to('cuda'))
                    h, ego, _ = history.append(step, encoded, float(state[3]), float(state[2]))
                    permission = maneuver_permissions(sim)
                    decision = policy.decide(
                        h, ego, bev, observed_bev[-4:], np.asarray(observed_states[-4:], np.float32),
                        nav.capture(), vehicle_parameters(sim.agent), permission, 20261025 + step
                    )
                    action, _ = tracker.act(decision['plan'], state[3])
                    obs, _, tm, tc, info = checked_step(sim, action); provider.update(obs)
                    speeds.append(float(sim.agent.speed))
                    if decision['fallback']:
                        fallbacks += 1
                    if tm or tc:
                        break
                final_speed = float(sim.agent.speed)
                crashed = bool(info.get('crash', False))
                offroad = bool(info.get('out_of_road', False))
                print(f"Test 1 result: final_speed={final_speed:.2f} m/s, fallbacks={fallbacks}/35, crash={crashed}, offroad={offroad}", flush=True)
                test1_passed = (final_speed > 1.5) and (fallbacks <= 5) and (not crashed) and (not offroad)
                results['test1_open_road_launch'] = dict(
                    passed=test1_passed, final_speed=final_speed, fallbacks=fallbacks, crashed=crashed, offroad=offroad
                )
            finally:
                sim.close()

            # Test 2: Stop and resume
            print("--- Running Gate Test 2: Stop and Resume ---", flush=True)
            sim = make_external_env(overrides)
            try:
                obs, _ = sim.reset(seed=4300000)
                provider = NativeBEVProvider(sim, BEVConfig()); provider.reset(obs)
                nav = NavigationProvider(sim, IntegratedConfig())
                tracker = TrajectoryTracker(sim.agent, .1)
                history = ObservedHistory(.1, .1)
                # First, bring car to a stop by applying brake for 20 steps
                for step in range(20):
                    obs, _, tm, tc, info = checked_step(sim, np.array([-1.0, 0.0], np.float32))
                    provider.update(obs)
                assert sim.agent.speed < 0.1, f"Expected stopped agent, got {sim.agent.speed}"
                # Now resume from stopped state using policy
                observed_bev = []; observed_states = []
                fallbacks = 0
                for step in range(35):
                    bev = provider.capture().raster.copy(); state = vehicle_state(sim.agent)
                    observed_bev.append(bev); observed_states.append(state)
                    encoded = encoder(torch.from_numpy(bev[None]).to('cuda'))
                    h, ego, _ = history.append(step, encoded, float(state[3]), float(state[2]))
                    permission = maneuver_permissions(sim)
                    decision = policy.decide(
                        h, ego, bev, observed_bev[-4:], np.asarray(observed_states[-4:], np.float32),
                        nav.capture(), vehicle_parameters(sim.agent), permission, 20261026 + step
                    )
                    action, _ = tracker.act(decision['plan'], state[3])
                    obs, _, tm, tc, info = checked_step(sim, action); provider.update(obs)
                    if decision['fallback']:
                        fallbacks += 1
                    if tm or tc:
                        break
                final_speed = float(sim.agent.speed)
                crashed = bool(info.get('crash', False))
                offroad = bool(info.get('out_of_road', False))
                print(f"Test 2 result: final_speed={final_speed:.2f} m/s, fallbacks={fallbacks}/35, crash={crashed}, offroad={offroad}", flush=True)
                test2_passed = (final_speed > 1.2) and (fallbacks <= 5) and (not crashed) and (not offroad)
                results['test2_stop_and_resume'] = dict(
                    passed=test2_passed, final_speed=final_speed, fallbacks=fallbacks, crashed=crashed, offroad=offroad
                )
            finally:
                sim.close()

            # Test 3: Hazardous lead braking
            print("--- Running Gate Test 3: Hazardous Lead Braking ---", flush=True)
            hazard_overrides = dict(
                map=3, map_config=dict(type='block_sequence', config='S', lane_num=1, exit_length=250),
                start_seed=4101000, num_scenarios=1, traffic_density=0., horizon=200, decision_repeat=5,
                physics_world_step_size=.02, force_destroy=True,
                agent_configs={'default_agent': {'spawn_lane_index': ('>>', '>>>', 0), 'spawn_longitude': 5.}}
            )
            sim = make_external_env(hazard_overrides)
            try:
                obs, _ = sim.reset(seed=4101000)
                npc = spawn_slow_actor(sim, sim.agent.lane, 5. + 7.5, 4101000, stationary=True)
                provider = NativeBEVProvider(sim, BEVConfig()); provider.reset(None)
                nav = NavigationProvider(sim, IntegratedConfig())
                tracker = TrajectoryTracker(sim.agent, .1)
                history = ObservedHistory(.1, .1)
                observed_bev = []; observed_states = []
                crashed = False
                min_distance = float('inf')
                for step in range(30):
                    bev = provider.capture().raster.copy(); state = vehicle_state(sim.agent)
                    dist = float(np.linalg.norm(sim.agent.position - npc.position))
                    min_distance = min(min_distance, dist)
                    observed_bev.append(bev); observed_states.append(state)
                    encoded = encoder(torch.from_numpy(bev[None]).to('cuda'))
                    h, ego, _ = history.append(step, encoded, float(state[3]), float(state[2]))
                    permission = maneuver_permissions(sim)
                    decision = policy.decide(
                        h, ego, bev, observed_bev[-4:], np.asarray(observed_states[-4:], np.float32),
                        nav.capture(), vehicle_parameters(sim.agent), permission, 20261027 + step
                    )
                    action, _ = tracker.act(decision['plan'], state[3])
                    obs, _, tm, tc, info = checked_step(sim, action); provider.update(obs)
                    if info.get('crash', False):
                        crashed = True
                        break
                print(f"Test 3 result: crashed={crashed}, min_distance={min_distance:.2f} m", flush=True)
                test3_passed = (not crashed) and (min_distance > 1.5)
                results['test3_hazardous_lead_braking'] = dict(
                    passed=test3_passed, crashed=crashed, min_distance=min_distance
                )
            finally:
                sim.close()

    finally:
        restore()

    all_passed = all(r['passed'] for r in results.values())
    summary = dict(
        gate_check_passed=all_passed,
        thresholds=thresholds,
        tests=results
    )
    print("=== Gate Check Summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if output_dir is not None:
        (Path(output_dir) / 'gate_check.json').write_text(json.dumps(summary, indent=2))
    return all_passed


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flow', type=Path, required=True)
    p.add_argument('--risk', type=Path, required=True)
    p.add_argument('--output', type=Path, default=None)
    args = p.parse_args()
    ok = run_gate_check(args.flow, args.risk, args.output)
    sys.exit(0 if ok else 1)
