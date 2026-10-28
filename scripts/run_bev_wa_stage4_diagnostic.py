"""Read-only existing-weight Stage 4 development diagnostics, never training/acceptance.

Use --list-scenes to review the frozen scenario list without importing simulator or torch.
Each invocation requires a fresh output directory. Expert and learned runs use the
same explicit scenarios; scenario_seed and training_seed are separate quantities.
"""
import argparse
from pathlib import Path
import sys
import hashlib
import json
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from evaluation.stage4_diagnostic import EVENTS, causal_ego, classify_outcome, scene_overrides, scene_suite, recovery_pulse
from configs.stage3 import Stage3Config
from mini_wa_jepa.operators import sdpa_context, configure_operator_optimizations, verify_attention_fidelity
from controller.route_governor import route_plan, should_override


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def actor_diagnostics(sim, tracked_id, home_lane):
    """Privileged measurement only. Never fed to learned model or tracker."""
    from dataset.recording import vehicle_state
    from mini_wa_jepa.geometry import world_to_ego
    actors = sim.engine.get_objects(lambda o: getattr(o, 'metadrive_type', None) == 'VEHICLE')
    car = sim.agent
    es, _ = home_lane.local_coordinates(car.position)
    tracked_gap = None
    snapshots = []
    for key, actor in actors.items():
        if actor is car:
            continue
        local = world_to_ego(actor.position, car.position, car.heading_theta)
        rear_gap = float(local[0]-(car.LENGTH+actor.LENGTH)*.5)
        entry = dict(id=str(key), state=vehicle_state(actor).tolist(), relative_ego_xy=local.tolist(),
                     longitudinal_bumper_gap_m=rear_gap, lateral_body_overlap=bool(abs(local[1])<(car.WIDTH+actor.WIDTH)*.5),
                     lane_index=list(actor.lane_index) if actor.lane_index is not None else None)
        snapshots.append(entry)
        if str(key) == tracked_id:
            os, _ = home_lane.local_coordinates(actor.position)
            tracked_gap = float(os-es-(car.LENGTH+actor.LENGTH)*.5)
    return snapshots, tracked_gap


def load_model(arm, checkpoint, device):
    import torch
    from configs.stage3 import Stage3Config
    from models.bev_wa_joint_flow import ArmBActionOnlyFlowModel, ArmDJointFlowModel
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = {'D': ArmDJointFlowModel, 'B': ArmBActionOnlyFlowModel}[arm](cfg=Stage3Config(), use_applied_action=True)
    model.load_state_dict(saved['model_state'], strict=True)
    return model.to(device).eval(), {k:saved[k] for k in ('seed', 'epoch', 'arm')}


def export_plots(folder, rows, bevs, targets, valid, interval, video):
    """Observed BEV and actual replanned rollout; no synthetic camera images."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mini_wa_jepa.visualization.rerun_logger import colorize_bev_4ch
    from scripts.report_stage3 import create_video
    frames = folder/'figures'; frames.mkdir()
    paths = []
    for t in range(0, len(rows), interval):
        row = rows[t]; fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4), dpi=100)
        axes[0].imshow(colorize_bev_4ch(bevs[t]), extent=(32,-32,-32,32))
        plan=np.asarray(row['plan']) if row['plan'] is not None else np.empty((0,4))
        model_plan=np.asarray(row['plan_model']) if row.get('plan_model') is not None else np.empty((0,4))
        actual=targets[t][valid[t]]
        if len(plan): axes[0].plot(plan[:,1], plan[:,0], color='deepskyblue', label='planned (used)')
        if len(model_plan): axes[0].plot(model_plan[:,1], model_plan[:,0], color='0.6', ls='--', label='model plan')
        if len(actual): axes[0].plot(actual[:,1],actual[:,0], color='gold', label='observed after replanning')
        axes[0].set(xlabel='ego left (m)', ylabel='ego forward (m)', title=f"step {t}, t={t*.1:.1f}s")
        if len(plan) or len(actual): axes[0].legend()
        times=np.arange(t+1)*.1
        axes[1].plot(times,[r['state'][3] for r in rows[:t+1]],label='speed m/s')
        axes[1].plot(times,[r['lateral_m'] for r in rows[:t+1]],label='lane lateral m')
        axes[1].plot(times,[r['applied_action'][0] for r in rows[:t+1]],label='steer')
        axes[1].plot(times,[r['applied_action'][1] for r in rows[:t+1]],label='throttle/brake')
        axes[1].legend(); axes[1].set(xlabel='elapsed seconds', title='Actual control and state')
        fig.suptitle('Development diagnostic — not Stage 4 acceptance')
        fig.tight_layout(); p=frames/f'{t:05d}.png'; fig.savefig(p); plt.close(fig); paths.append(p)
    return create_video(paths,folder/'diagnostic.mp4') if video and paths else None


def evaluate_episode(args, scene, model, folder, device):
    import torch
    from dataclasses import replace
    from configs.integrated import IntegratedConfig
    from configs.stage2 import BEVConfig
    from controller.trajectory_tracker import TrajectoryTracker
    from dataset.bev_wa_dataset import extract_causal_actions
    from dataset.recording import vehicle_state, future_labels_v2
    from env.data_v2 import NativeBEVProvider
    from env.integrated import make_external_env, checked_step
    from env.navigation import NavigationProvider, vehicle_parameters
    from env.lane_permissions import maneuver_permissions
    from env.expert import NavigationExpert
    from env.traffic_teacher import spawn_slow_actor
    from mini_wa_jepa.provenance import write_json
    from mini_wa_jepa.visualization import RerunLogger
    from mini_wa_jepa.geometry import wrap_angle

    folder.mkdir(); config=scene_overrides(scene); sim=None; rows=[]; bevs=[]; ego=[]; applied=[]; predicted_latents={}
    write_json(folder/'scene.json', dict(scene=scene, simulator_overrides=config, tracker_dt_s=args.tracker_dt))
    try:
        sim=make_external_env(config); obs,_=sim.reset(seed=scene['seed'])
        provider=NativeBEVProvider(sim,BEVConfig()); provider.reset(obs)
        home=sim.agent.lane; tracked=None
        if scene.get('lead'):
            actor=spawn_slow_actor(sim, home, 5.+scene['lead_distance'], scene['seed'],
                                  dynamic_stop_resume=scene['dynamic'],stop_step=25,resume_step=90)
            tracked=str(actor.id)
        nav=NavigationProvider(sim,replace(IntegratedConfig(),preferred_lane=min(1,scene.get('lanes',3)-1)))
        tracker=TrajectoryTracker(sim.agent,dt_s=args.tracker_dt)
        expert=NavigationExpert(sim,dt=.1) if model is None else None
        params=vehicle_parameters(sim.agent); previous_heading=None
        write_json(folder/'initial.json',dict(calibration=provider.calibration,map_hash=provider.geometry_hash,
                    actual_control_dt_s=float(sim.config['decision_repeat']*sim.config['physics_world_step_size']),
                    tracked_lead_id=tracked,vehicle_parameters=params.tolist(),initial_state=vehicle_state(sim.agent).tolist()))
        if not np.isclose(sim.config['decision_repeat']*sim.config['physics_world_step_size'], .1):
            raise RuntimeError('Expected measured decision interval 0.1s')
        with RerunLogger(app_id='stage4-diagnostic',spawn=False,save_path=folder/'driving.rrd') as log:
            log.log_context(static=True, arm=args.arm,scenario_seed=scene['seed'],control_dt_s=.1,
                            tracker_dt_s=args.tracker_dt,latent_target='observed_after_replanning_not_fixed_command_counterfactual')
            with (folder/'steps.jsonl').open('x') as stream:
                for step in range(scene['horizon']):
                    tick=time.perf_counter(); bev=provider.capture().raster.copy(); state=vehicle_state(sim.agent)
                    frame=nav.capture(); permission=maneuver_permissions(sim)
                    bevs.append(bev); ego.append(causal_ego(state[3],state[2],previous_heading)); previous_heading=float(state[2])
                    h_app,h_valid=extract_causal_actions(applied,step,4)
                    indices=np.maximum(np.arange(step-3,step+1),0)
                    inputs=(np.stack([bevs[i] for i in indices]),np.stack([ego[i] for i in indices]),frame.points.astype(np.float32),
                            frame.valid,params.astype(np.float32),permission)
                    plan=None; inference_ms=None
                    plan_model=None; plan_source='expert' if model is None else 'model'
                    route_override_info=None; route_plan_info=None
                    if model is None:
                        action,detail=expert.act()
                    else:
                        torch.cuda.synchronize(); inference_tick=time.perf_counter()
                        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                            result=model.solve_euler(*[torch.from_numpy(v[None]).to(device) for v in inputs],
                                applied_action=torch.from_numpy(h_app[None]).to(device),action_valid=torch.from_numpy(h_valid[None]).to(device),
                                num_steps=5,noise_seed=args.noise_seed)
                            plan_t=result[0] if isinstance(result,tuple) else result
                            plan_model=plan_t[0].float().cpu().numpy()
                            plan=plan_model
                            plan_source='model'
                            if args.route_governor:
                                route_plan_used, route_plan_info = route_plan(frame.points, frame.valid, speed=args.route_speed, model_plan=plan_model)
                                override, route_override_info = should_override(
                                    plan_model, frame.points, frame.valid,
                                    turn_threshold_deg=args.route_turn_deg,
                                    deviation_threshold_deg=args.route_deviation_deg)
                                if override and route_plan_used is not None:
                                    plan=route_plan_used
                                    plan_source='route_governor'
                            if isinstance(result,tuple) and step%args.visual_interval==0:
                                predicted_latents[step]=result[1][0].float().cpu().numpy()
                        torch.cuda.synchronize(); inference_ms=(time.perf_counter()-inference_tick)*1000
                        action,detail=tracker.act(plan,float(state[3]))
                        detail.update(plan_source=plan_source,
                                      route_turn_deg=(route_override_info or {}).get('route_turn_deg'),
                                      route_deviation_deg=(route_override_info or {}).get('deviation_deg'),
                                      route_speed_m_s=float(args.route_speed))
                        distances=np.linalg.norm(plan[:,:2],axis=-1)
                        eligible=np.flatnonzero((distances>=detail['lookahead_m'])&(plan[:,0]>0))
                        chosen=int(eligible[0]) if len(eligible) else 9
                        detail.update(target_index=chosen,target_waypoint=plan[chosen].tolist())
                    recovery_active, recovery_sign = recovery_pulse(
                        int(scene.get('seed', 0)), step,
                        bool(args.recovery_pulses and scene.get('recovery', False)))
                    if recovery_active:
                        action = np.asarray(action, np.float32).copy()
                        action[0] = float(np.clip(action[0] + recovery_sign * 0.25, -1., 1.))
                    pipeline_ms=(time.perf_counter()-tick)*1000
                    lane=sim.agent.lane; lane_s,lateral=lane.local_coordinates(sim.agent.position)
                    heading_error=float(wrap_angle(state[2]-lane.heading_theta_at(np.clip(lane_s,0,lane.length))))
                    actors,gap=actor_diagnostics(sim,tracked,home)
                    obs,reward,terminated,truncated,info=checked_step(sim,action);provider.update(obs)
                    actual=np.asarray(info['raw_action'],np.float32); applied.append(actual)
                    post=vehicle_state(sim.agent)
                    row=dict(step=step,time_s=step*.1,state=state.tolist(),post_state=post.tolist(),ego_motion=ego[-1].tolist(),
                        causal_actions=h_app.tolist(),action_valid=h_valid.tolist(),route_points=frame.points.tolist(),route_valid=frame.valid.tolist(),
                        permission=permission.tolist(),lane_index=list(lane.index),lane_s_m=float(lane_s),lane_length_m=float(lane.length),
                        lane_width_m=float(lane.width_at(np.clip(lane_s,0,lane.length))),lateral_m=float(lateral),heading_error_rad=heading_error,
                        plan=plan.tolist() if plan is not None else None,plan_model=plan_model.tolist() if plan_model is not None else None,
                        plan_source=plan_source,route_turn_deg=(route_override_info or {}).get('route_turn_deg'),
                        route_deviation_deg=(route_override_info or {}).get('deviation_deg'),
                        route_plan_length_m=(route_plan_info or {}).get('route_length_m'),
                        recovery_active=recovery_active,recovery_sign=recovery_sign,
                        requested_action=np.asarray(action).tolist(),applied_action=actual.tolist(),
                        tracker=detail,npcs=actors,tracked_lead_clearance_m=gap,tracked_lead_id=tracked,
                        events={k:bool(info.get(k,False)) for k in EVENTS},terminated=bool(terminated),truncated=bool(truncated),
                        route_completion=float(info['route_completion']) if 'route_completion' in info else None,
                        inference_ms=inference_ms,pipeline_ms=pipeline_ms,reward=float(reward))
                    rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                    log.set_step(step,step*.1);log.log_bev(bev);log.log_state(state)
                    log.log_navigation(frame.points,frame.valid,source=frame.source)
                    if plan is not None: log.log_trajectories(planned=plan)
                    log.log_telemetry(speed_m_s=float(state[3]),steering=float(actual[0]),throttle_brake=float(actual[1]),
                                      lateral_error_m=float(lateral),route_completion=row['route_completion'])
                    if terminated or truncated or any(row['events'][k] for k in ('arrive_dest','crash','out_of_road')):break
            states=np.asarray([r['state'] for r in rows]+[rows[-1]['post_state']],np.float32)
            failure=np.zeros(len(states),bool);failure[-1]=any(rows[-1]['events'][k] for k in ('crash','out_of_road'))
            targets,valid,_=future_labels_v2(states,failure,10,2)
            latent_rows=[]
            for step in range(len(rows)):
                log.set_step(step,step*.1);log.log_trajectories(ground_truth=targets[step][valid[step]])
                if step in predicted_latents and step+20<len(bevs):
                    future=np.stack([bevs[step+i] for i in (5,10,15,20)])
                    with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                        target=model.target_encoder.encode_future(torch.from_numpy(future[None]).to(device))[0].float().cpu().numpy()
                    log.log_latents(predicted=predicted_latents[step],ground_truth=target)
                    latent_rows.append((step,predicted_latents[step],target))
            if latent_rows:
                np.savez_compressed(folder/'latents_observed_replanning.npz',steps=[x[0] for x in latent_rows],
                    predicted=np.stack([x[1] for x in latent_rows]),observed=np.stack([x[2] for x in latent_rows]))
            plans_used=np.asarray([r['plan'] for r in rows],np.float32) if model is not None else np.empty((0,10,4))
            plans_model=(np.asarray([r['plan_model'] for r in rows],np.float32)
                         if model is not None and all(r['plan_model'] is not None for r in rows) else np.empty((0,10,4)))
            np.savez_compressed(folder/'actual.npz',bev=np.asarray(bevs),states=states,
                plans=plans_used,plans_model=plans_model,
                applied_action=np.asarray(applied),ego=np.asarray(ego),trajectory=targets[:len(rows)],valid=valid[:len(rows)])
        result=classify_outcome(scene['category'],rows,scene['horizon'])
        timings=[r['inference_ms'] for r in rows if r['inference_ms'] is not None]
        plan_sources={name: sum(1 for r in rows if r.get('plan_source')==name) for name in ('model','route_governor','expert')}
        result.update(steps=len(rows),inference_p95_ms=float(np.percentile(timings,95)) if timings else None,
                      pipeline_p95_ms=float(np.percentile([r['pipeline_ms'] for r in rows],95)),
                      raw=str(folder/'steps.jsonl'),latent_pairs=len(latent_rows),
                      route_governor=bool(args.route_governor),plan_sources=plan_sources,
                      recovery_pulses=bool(args.recovery_pulses),recovery_steps=sum(1 for r in rows if r.get('recovery_active')),
                      video=export_plots(folder,rows,bevs,targets,valid,args.visual_interval,args.video))
        write_json(folder/'outcome.json',result)
        return result
    except BaseException:
        from mini_wa_jepa.provenance import write_json
        write_json(folder/'failure_mre.json',dict(scene=scene,rows=len(rows),traceback=traceback.format_exc()))
        raise
    finally:
        if sim is not None:sim.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--arm',choices=('D','B','expert'),default='D')
    p.add_argument('--training-seed',type=int,choices=(20261022,20261023),default=20261022)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--suite',choices=('legacy_four','legacy_six_S','manifest_nav','manifest_nav12'),default='legacy_six_S')
    p.add_argument('--manifest',type=Path,default=ROOT/'artifacts/stage7_integrated/mixed_validation_010/manifest.json')
    p.add_argument('--scenario-seed',type=int,choices=(20261022,20261023),default=20261022)
    p.add_argument('--scene-indices',type=int,nargs='+')
    p.add_argument('--max-steps',type=int,help='Shortens only; labels run truncated diagnostic')
    p.add_argument('--tracker-dt',type=float,choices=(.1,.02),default=.1)
    p.add_argument('--noise-seed',type=int,default=42)
    p.add_argument('--visual-interval',type=int,default=25)
    p.add_argument('--video',action='store_true')
    p.add_argument('--route-governor',action='store_true',help='Use route-aware fallback at sharp turns')
    p.add_argument('--route-speed',type=float,default=6.0,help='Fallback speed cap m/s; actual speed is min(cap, model lookahead speed)')
    p.add_argument('--route-turn-deg',type=float,default=20.0,help='Route net turn threshold deg')
    p.add_argument('--route-deviation-deg',type=float,default=20.0,help='Model-vs-route heading deviation threshold deg')
    p.add_argument('--recovery-pulses',action='store_true',help='Recreate collection-time recovery steering pulses on recovery scenes')
    p.add_argument('--sdpa-backend',choices=('AUTO','MATH','FLASH','EFFICIENT','CUDNN'),default=Stage3Config().sdpa_backend,help='SDPA backend contract; AUTO is the ROCm default')
    p.add_argument('--allow-tf32',action='store_true',help='Allow TF32 matmul/MIOpen (default off for determinism)')
    p.add_argument('--cudnn-benchmark',action='store_true',help='Enable MIOpen benchmark (default off)')
    p.add_argument('--list-scenes',action='store_true')
    args=p.parse_args()
    scenes=scene_suite(args.suite,args.manifest,args.scenario_seed)
    if args.scene_indices is not None: scenes=[scenes[i] for i in args.scene_indices]
    if args.max_steps is not None:
        if args.max_steps<1:p.error('--max-steps must be positive')
        scenes=[dict(s,horizon=min(s['horizon'],args.max_steps),original_horizon=s['horizon']) for s in scenes]
    if args.visual_interval<1:p.error('--visual-interval must be positive')
    if args.list_scenes:print(json.dumps(scenes,indent=2));return
    if args.output is None:p.error('--output is required')
    import torch
    from env.data_v2 import configure_egl
    from mini_wa_jepa.provenance import provenance,write_json
    if args.arm!='expert' and (not torch.version.hip or torch.version.cuda):raise RuntimeError('Native ROCm required')
    torch.set_num_threads(4);args.output.mkdir(parents=True,exist_ok=False)
    model=None; checkpoint=args.checkpoint; saved=None
    if args.arm!='expert':
        stem={'D':'arm_d_repaired_joint_wa-flow','B':'arm_b_action-only_flow'}[args.arm]
        checkpoint=checkpoint or ROOT/f'artifacts/bev_wa_alignment/stage4_remediated/{stem}_seed_{args.training_seed}.pt'
        model,saved=load_model(args.arm,checkpoint,'cuda')
        if saved['seed']!=args.training_seed:raise ValueError('Checkpoint training seed mismatch')
    prov = provenance()
    prov.pop('config', None)
    report = dict(**prov, completed=False, development_only=True, stage4_accepted=False,
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},scenes=scenes,episodes=[],
        checkpoint=str(checkpoint) if checkpoint else None,checkpoint_sha256=digest(checkpoint) if checkpoint else None,
        checkpoint_metadata=saved,protocol_sha256=digest(ROOT/'docs/BEV_WA_ALIGNED_PROTOCOL.md'),
        scenario_manifest_sha256=digest(args.manifest),
        expert_note='Privileged NavigationExpert; success demonstrates reachability; failure does not prove impossibility',
        latent_target='Observed subsequent replanning; not fixed-command counterfactual truth')
    report['sdpa_backend']=args.sdpa_backend
    report['operator_optimizations']=configure_operator_optimizations(vendor='AMD',allow_tf32=args.allow_tf32,cudnn_benchmark=args.cudnn_benchmark,deterministic=True)
    report['sdpa_verification']=verify_attention_fidelity(
        torch.randn(16,4,64,64,device='cuda',dtype=torch.bfloat16),
        torch.randn(16,4,64,64,device='cuda',dtype=torch.bfloat16),
        torch.randn(16,4,64,64,device='cuda',dtype=torch.bfloat16),
        candidate_backend=args.sdpa_backend,vendor='AMD')
    write_json(args.output/'report.json',report)
    _,restore=configure_egl(1)
    try:
        with sdpa_context(args.sdpa_backend,vendor='AMD'):
            for scene in scenes:
                result=evaluate_episode(args,scene,model,args.output/f"{scene['id']}_seed_{scene['seed']}",'cuda')
                report['episodes'].append(dict(scene=scene,outcome=result));write_json(args.output/'report.json',report)
                print(json.dumps(dict(scene=scene,result=result),ensure_ascii=False),flush=True)
        report['completed']=True
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        restore();write_json(args.output/'report.json',report)


if __name__=='__main__':main()
