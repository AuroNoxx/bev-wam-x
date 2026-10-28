"""Matched Stage 7 receding-horizon policies and auditable unified planning traces."""
import argparse
from collections import deque
from dataclasses import asdict
import faulthandler
import hashlib
import json
from pathlib import Path
import tarfile
import time
import traceback
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import sys
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from configs.stage2 import Stage2Config
from configs.stage7 import Stage7Config
from configs.planner7 import Planner7Config
from controller.trajectory_tracker import TrajectoryTracker
from dataset.recording import vehicle_state, future_labels_v2
from env.data_v2 import configure_egl,make_env,NativeBEVProvider
from env.expert import NavigationExpert
from evaluation.history import ObservedHistory
from evaluation.memory import memory_snapshot
from evaluation.planning_view import decision_figure
from mini_wa_jepa.provenance import provenance,write_json
from mini_wa_jepa.visualization import RerunLogger
from planning.policy7 import Stage7Policy
from scripts.preflight_risk_collection import scene_state
from training.flow_matching import file_sha256


def save_episode(folder,records,states,encoded,future_samples,arm,seed,map_spec,representative):
    folder.mkdir();states=np.asarray(states);n=len(records)
    target,valid,_=future_labels_v2(states,np.zeros(len(states),bool),10,2)
    candidates=np.stack([r['candidates'] for r in records]);k=candidates.shape[1]
    arrays=dict(states=states,bev=np.stack([r['bev'] for r in records]),actions=np.stack([r['action'] for r in records]),
        candidates=candidates,plans=np.stack([r['plan'] for r in records]),
        scores=np.stack([r['scores'] for r in records]),terms=np.stack([r['terms'] for r in records]),
        probability=np.stack([r['probability'] for r in records]),
        selected_index=np.array([r['selected_index'] for r in records]),
        selected_ids=np.array([r['selected_id'] for r in records]),
        selection_sources=np.array([r['selection_source'] for r in records]),
        rejections=np.asarray([[';'.join(x) for x in r['rejection_reasons']] for r in records]),
        trajectory_target=target[:n],trajectory_valid=valid[:n],
        decision_ms=np.array([r['decision_ms'] for r in records]),pipeline_ms=np.array([r['pipeline_ms'] for r in records]),
        route=np.array([r['route'] for r in records]),outcome_flags=np.array([r['flags'] for r in records]))
    sample_steps=np.array(sorted(future_samples),int);horizons=[.1,.2,.3,.4] if arm=='mse' else [.5,1.,1.5,2.]
    latent_prediction=np.stack([future_samples[s] for s in sample_steps]) if len(sample_steps) else np.empty((0,k,4,64,256),np.float16)
    observed=np.zeros((len(sample_steps),4,64,256),np.float16);observed_valid=np.zeros((len(sample_steps),4),bool)
    encoded=np.asarray(encoded)
    for i,step in enumerate(sample_steps):
        for j,t in enumerate(horizons):
            slot=step+round(t/.1)
            if slot<len(encoded):observed[i,j]=encoded[slot];observed_valid[i,j]=True
    arrays.update(latent_steps=sample_steps,latent_prediction=latent_prediction,
        observed_future_under_replanning=observed,latent_valid=observed_valid)
    np.savez_compressed(folder/'actual.npz',**arrays)
    (folder/'recording.json').write_text(json.dumps(dict(arm=arm,training_seed=seed,map_spec=map_spec,
        horizons_s=horizons,future_semantics='actual observations under subsequent replanning; not fixed-command counterfactual',
        term_names=['progress','speed','lateral_accel','curvature','static','dynamic','crash','offroad'],
        raw_sha256=file_sha256(folder/'actual.npz')),indent=2)+'\n')
    if representative:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        lookup={s:i for i,s in enumerate(sample_steps)}
        with RerunLogger(spawn=False,save_path=folder/'planning.rrd') as log:
            for index,row in enumerate(records):
                log.set_step(index,index*.1);log.log_bev(row['bev']);log.log_state(states[index])
                p=future_samples.get(index)
                log.log_planning(candidates=row['candidates'],selected_id=row['selected_id'],
                    selected_trajectory=row['plan'],selection_source=row['selection_source'],
                    scores=row['scores'] if arm!='mse' else None,terms=row['terms'] if arm!='mse' else None,
                    rejection_reasons=row['rejection_reasons'],predicted=p,executed_action=row['action'])
                log.log_trajectories(ground_truth=target[index][valid[index]])
                log.log_telemetry(speed_m_s=states[index,3],steering=row['action'][0],
                    throttle_brake=row['action'][1],route_completion=row['route'])
                if index in lookup and row['selected_index']>=0:
                    slot=lookup[index];mask=observed_valid[slot]
                    if mask.any():
                        log.log_context(actual_future_semantics='replanned actual, not fixed-command counterfactual',
                            future_horizons_s=horizons)
                        log.log_latents(predicted=p[row['selected_index']][mask],ground_truth=observed[slot][mask])
                if index in {0,n//2,n-1}:
                    fig=decision_figure(arrays,index);path=folder/f'decision_{index:05d}.png'
                    fig.savefig(path);plt.close(fig)
    distance=np.linalg.norm(arrays['plans'][...,:2]-target[:n,:,:2],axis=-1)
    return dict(raw=str(folder/'actual.npz'),raw_sha256=file_sha256(folder/'actual.npz'),
        rrd=str(folder/'planning.rrd') if representative else None,
        replanned_execution_ade_m=float(distance[valid[:n]].mean()) if valid[:n].any() else None,
        replanned_execution_fde_m=float(distance[:,-1][valid[:n,-1]].mean()) if valid[:n,-1].any() else None,
        fallback_fraction=float((arrays['selected_index']<0).mean()))


def evaluate(output,arm,seed,smoke=False,flow_checkpoint=None,target_map=None,target_seed=None,traffic_density=None):
    faulthandler.enable()
    if not torch.version.hip or torch.version.cuda:raise RuntimeError('Native ROCm required')
    torch.set_num_threads(4);cfg=Stage7Config();env_cfg=Stage2Config();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    if not smoke and seed not in cfg.training_seeds:
        raise ValueError('Formal evaluation uses the two prespecified training seeds')
    policy=Stage7Policy(arm,seed,flow_path=flow_checkpoint);torch.cuda.reset_peak_memory_stats()
    report=dict(**provenance(cfg),stage=7,completed=False,smoke=smoke,arm=arm,training_seed=seed,
        parameter_count=policy.parameter_count,checkpoints=policy.provenance,planner_config=asdict(Planner7Config()),
        semantic_order=True,
        traffic_density=float(traffic_density if traffic_density is not None else getattr(cfg, 'traffic_density', 0.3)),
        control_dt_s=.1,max_steps=80 if smoke else cfg.closed_loop_max_steps,episodes=[],
        latency_definition='current BEV to HIP encoder, observed history, generated candidates, world/risk as needed, geometry, selection and tracker; first 20 steps excluded',
        ground_truth_boundary='subsequent states only joined after episode; evaluator route geometry is not a policy input')
    write_json(out/'report.json',report)
    with tarfile.open(out/'source_snapshot.tar.gz','w:gz') as arc:
        for path in report['source_files']:arc.add(path,arcname=path)
        for path in ('GOAL.md','AGENTS.md','docs/STAGE7_PROTOCOL.md'):arc.add(path,arcname=path)
    _,restore=configure_egl(1);started=time.perf_counter()
    try:
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            maps_to_run = (target_map,) if target_map else (('SCS','CSC') if not smoke else ('CC','SS'))
            density_use = float(traffic_density if traffic_density is not None else getattr(cfg, 'traffic_density', 0.3))
            overrides = {'force_destroy': True, 'traffic_density': density_use}
            for map_i,map_spec in enumerate(maps_to_run):
                base=cfg.closed_loop_seeds[0] if map_spec=='SCS' else cfg.closed_loop_seeds[1]
                if smoke:base=880000+map_i*10000
                sim=make_env(map_spec,base,env_cfg,num_scenarios=1000,control_dt_s=.1,
                    environment_overrides=overrides,semantic_order=True)
                try:
                    for episode_i in range(1 if smoke else cfg.closed_loop_seeds_per_map):
                        episode_seed=base+episode_i
                        if target_seed and episode_seed != target_seed:
                            continue
                        obs,_=sim.reset(seed=episode_seed)
                        provider=NativeBEVProvider(sim,env_cfg.bev);provider.reset(obs)
                        tracker=TrajectoryTracker(sim.agent,.1);scorer=NavigationExpert(sim,env_cfg.expert,.1)
                        history=ObservedHistory(.1,.1);bevs=deque(maxlen=4);past=deque(maxlen=4)
                        initial_state=vehicle_state(sim.agent);initial_scene=scene_state(sim)
                        initial_bev_sha=hashlib.sha256(provider.capture().raster.tobytes()).hexdigest()
                        records=[];states=[];encoded=[];future_samples={};lat=[];pipes=[];laterals=[]
                        collision=offroad=False;episode_tick=time.perf_counter()
                        for step in range(80 if smoke else cfg.closed_loop_max_steps):
                            tick=time.perf_counter();state=vehicle_state(sim.agent);bev=provider.capture().raster
                            z=policy.encoder(torch.from_numpy(bev[None]).to('cuda'))
                            observed,motion,indices=history.append(step,z,float(state[3]),float(state[2]))
                            bevs.append(bev);past.append(state)
                            decision=policy.decide(observed,motion,bev,list(bevs),list(past),
                                sim.agent.LENGTH,sim.agent.WIDTH,42+episode_seed*10000+step)
                            action,_=tracker.act(decision['plan'],sim.agent.speed)
                            torch.cuda.synchronize();decision_ms=(time.perf_counter()-tick)*1000
                            # Evaluation-only metric read AFTER selection; never supplied to the policy.
                            laterals.append(float(scorer.route(0.)[2]))
                            obs,_,tm,tc,info=sim.step(action);provider.update(obs)
                            pipeline_ms=(time.perf_counter()-tick)*1000
                            if step>=20:lat.append(decision_ms);pipes.append(pipeline_ms)
                            states.append(state);encoded.append(z[0].cpu().numpy().astype(np.float16))
                            future=decision.pop('future')
                            if future is not None and step%50==0:future_samples[step]=future.astype(np.float16)
                            k=len(decision['candidates'])
                            if decision.get('terms') is None:decision['terms']=np.full((k,8),np.nan)
                            if decision.get('scores') is None:decision['scores']=np.full(k,np.nan)
                            if decision.get('probability') is None:decision['probability']=np.full((k,2),np.nan)
                            records.append(dict(**decision,bev=bev.copy(),action=action,
                                decision_ms=decision_ms,pipeline_ms=pipeline_ms,
                                route=float(info['route_completion']),flags=[info['crash'],info['out_of_road'],tm,tc]))
                            collision|=bool(info['crash']);offroad|=bool(info['out_of_road'])
                            if step%100==0:
                                write_json(out/'progress.json',dict(map_spec=map_spec,seed=episode_seed,step=step,
                                    selected_id=decision['selected_id'],route_completion=float(info['route_completion'])))
                            if tm or tc:break
                        states.append(vehicle_state(sim.agent))
                        evidence=save_episode(out/f'{map_spec}_{episode_seed}',records,states,encoded,future_samples,
                            arm,seed,map_spec,episode_i==0)
                        row=dict(seed=episode_seed,map_spec=map_spec,steps=step+1,
                            route_completion=float(info['route_completion']),arrive_dest=bool(info['arrive_dest']),
                            collision=collision,out_of_road=offroad,truncated=bool(tc),
                            lateral_error_m=float(np.mean(np.abs(laterals))),
                            inference_ms=float(np.mean(lat)) if lat else None,
                            inference_p95_ms=float(np.percentile(lat,95)) if lat else None,
                            pipeline_p95_ms=float(np.percentile(pipes,95)) if pipes else None,
                            initial_state=initial_state.tolist(),initial_scene=initial_scene.tolist(),
                            initial_bev_sha256=initial_bev_sha,elapsed_s=time.perf_counter()-episode_tick,
                            memory=memory_snapshot(),peak_allocator_bytes=torch.cuda.max_memory_allocated(),**evidence)
                        report['episodes'].append(row);write_json(out/'report.json',report)
                        print(json.dumps({k:row[k] for k in ('seed','steps','route_completion','collision','out_of_road','fallback_fraction')}),flush=True)
                finally:sim.close()
        report.update(completed=True,elapsed_s=time.perf_counter()-started)
    except BaseException:
        report['error']=traceback.format_exc()
        if 'records' in locals():
            torch.save(dict(records=records,states=states),out/'failure_mre.pt')
        raise
    finally:restore();write_json(out/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--arm',choices=['mse','geometry','full'],required=True)
    p.add_argument('--seed',type=int,default=20261021);p.add_argument('--smoke',action='store_true')
    p.add_argument('--flow-checkpoint',default=None)
    p.add_argument('--target-map',default=None)
    p.add_argument('--target-seed',type=int,default=None)
    p.add_argument('--traffic-density',type=float,default=None)
    a=p.parse_args();evaluate(a.output,a.arm,a.seed,a.smoke,a.flow_checkpoint,a.target_map,a.target_seed,a.traffic_density)
