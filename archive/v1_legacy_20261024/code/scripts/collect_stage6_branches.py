"""Repeat exact expert prefixes, execute fixed candidates, retain real outcomes."""
import argparse
from dataclasses import asdict
import faulthandler
import hashlib
import json
from pathlib import Path
import time
import traceback
import tarfile

import numpy as np

from configs.stage2 import Stage2Config
from configs.stage6 import Stage6Config, BRANCH_SPLITS
from controller.trajectory_tracker import TrajectoryTracker
from dataset.action_branches import make_candidates, remaining_command, realized_waypoints
from dataset.recording import vehicle_state
from env.data_v2 import NativeBEVProvider, configure_egl, make_env
from env.expert import NavigationExpert
from mini_wa_jepa.geometry import wrap_angle
from mini_wa_jepa.provenance import provenance, write_json


def scene_state(sim):
    objects=sim.engine.get_objects(lambda obj:getattr(obj,'metadrive_type',None)=='VEHICLE').values()
    values=[vehicle_state(obj) for obj in objects]
    values.sort(key=lambda row:tuple(np.round(row[:2],3)))
    return np.asarray(values,np.float32)


def collect(output, config, splits, smoke=False, map_names=None, deep_smoke=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(**provenance(config),stage=6,completed=False,smoke=smoke,groups=[],failures=[],
                requested_splits=list(splits),requested_maps=map_names)
    write_json(output/'manifest.json',report)
    with tarfile.open(output/'source_snapshot.tar.gz','w:gz') as archive:
        for source in report['source_files']:archive.add(source,arcname=source)
        for source in ('GOAL.md','AGENTS.md','docs/STAGE6_PROTOCOL.md'):archive.add(source,arcname=source)
    _,restore=configure_egl(1);cfg=Stage2Config();started=time.perf_counter()
    try:
        for split in splits:
            spec=BRANCH_SPLITS[split]
            count=1 if smoke else getattr(config,split+'_seeds_per_map')
            maps=list(enumerate(spec['maps']))
            if map_names is not None:maps=[(i,m) for i,m in maps if m in map_names]
            elif smoke:maps=maps[:1]
            for map_i,map_spec in maps:
                base=spec['seed_start']+map_i*10000
                sim=make_env(map_spec,base,cfg,num_scenarios=1000,control_dt_s=config.control_dt_s,
                    environment_overrides={'force_destroy':config.force_destroy_between_resets})
                try:
                    for seed_i in range(count):
                        seed=base+seed_i
                        for anchor_step in ((20,) if smoke and not deep_smoke else config.prefix_steps):
                            group=f'{split}_{map_spec}_{seed}_{anchor_step}'
                            folder=output/split/group;folder.mkdir(parents=True,exist_ok=False)
                            prefix_actions=None;reference_history=reference_states=commands=reference_scene=None
                            row=dict(id=group,split=split,map_spec=map_spec,seed=seed,anchor_step=anchor_step,
                                     candidate_ids=list(config.candidates),branches=[])
                            names=config.candidates+(('keep_repeat',) if anchor_step==config.prefix_steps[0] else ())
                            for ci,name in enumerate(names):
                                command_index=0 if name=='keep_repeat' else ci
                                obs,_=sim.reset(seed=seed)
                                provider=NativeBEVProvider(sim,cfg.bev);provider.reset(obs)
                                expert=NavigationExpert(sim,cfg.expert,config.control_dt_s)
                                states=[vehicle_state(sim.agent)];bevs=[provider.capture().raster.copy()]
                                actions=[];tm=tc=False
                                for step in range(anchor_step):
                                    action=expert.act()[0] if prefix_actions is None else prefix_actions[step]
                                    actions.append(action.copy());obs,_,tm,tc,info=sim.step(action);provider.update(obs)
                                    states.append(vehicle_state(sim.agent));bevs.append(provider.capture().raster.copy())
                                    if tm or tc:raise RuntimeError(f'Prefix terminated: {group} {step} {info}')
                                history=np.stack(bevs[-4:]);past=np.stack(states[-4:])
                                ego=np.zeros((4,2),np.float32);ego[:,0]=past[:,3]/10
                                ego[:,1]=wrap_angle(np.stack(states[-4:])[:,2]-np.stack(states[-5:-1])[:,2])/.1
                                if prefix_actions is None:
                                    prefix_actions=actions;reference_history=history;reference_states=past;reference_scene=scene_state(sim)
                                    rng=np.random.default_rng(config.seed+seed+anchor_step)
                                    commands=make_candidates(expert,float(rng.uniform(1.,2.)))
                                    np.savez_compressed(folder/'context.npz',history=history,ego=ego,
                                        states=past,commands=commands,prefix_actions=np.asarray(actions),scene_state=reference_scene)
                                state_error=float(np.max(np.abs(reference_states-past)))
                                current_scene=scene_state(sim)
                                scene_error=(float(np.max(np.abs(reference_scene-current_scene)))
                                             if reference_scene.shape==current_scene.shape else float('inf'))
                                scene_bounds=np.array([config.replay_scene_position_m]*2+[config.replay_scene_heading_rad,
                                    config.replay_scene_speed_m_s]+[config.replay_scene_velocity_m_s]*2)
                                scene_ok=(reference_scene.shape==current_scene.shape and
                                          bool(np.all(np.abs(reference_scene-current_scene)<=scene_bounds)))
                                different=float(np.mean(reference_history!=history))
                                if state_error>config.replay_state_atol or not scene_ok or different>config.replay_bev_different_fraction:
                                    np.savez_compressed(folder/f'replay_mismatch_{ci}.npz',history=history,states=past,
                                                        reference_history=reference_history,reference_states=reference_states,
                                                        scene=current_scene,reference_scene=reference_scene)
                                    raise RuntimeError(f'Branch prefix outside frozen tolerance {group}: state={state_error}, scene={scene_error}, BEV fraction={different}')
                                anchor_state=states[-1].copy();tracker=TrajectoryTracker(sim.agent,config.control_dt_s)
                                future=[];executed=[];flags=[];branch_states=[anchor_state]
                                for step in range(20):
                                    command=remaining_command(commands[command_index],anchor_state,vehicle_state(sim.agent),step*.1)
                                    action,_=tracker.act(command,sim.agent.speed)
                                    obs,_,tm,tc,info=sim.step(action);provider.update(obs)
                                    executed.append(action);branch_states.append(vehicle_state(sim.agent))
                                    future.append(provider.capture().raster.copy())
                                    flags.append([bool(info['crash']),bool(info['out_of_road']),bool(tm),bool(tc)])
                                    if tm or tc:break
                                length=len(future);future_steps=np.rint(np.asarray(config.future_times_s)/.1).astype(int)
                                valid=future_steps<=length
                                targets=np.zeros((4,4,128,128),np.uint8)
                                targets[valid]=np.asarray(future)[future_steps[valid]-1]
                                realized=np.zeros((10,4),np.float32);way_valid=np.arange(2,21,2)<=length
                                realized[way_valid]=realized_waypoints(branch_states,0,np.arange(2,21,2)[way_valid])
                                np.savez_compressed(folder/f'{name}.npz',future=targets,valid=valid,
                                    command=commands[command_index],states=np.asarray(branch_states),executed_actions=np.asarray(executed),
                                    outcome_flags=np.asarray(flags),realized=realized,waypoint_valid=way_valid,
                                    observed_history=history,observed_states=past,scene_state=current_scene)
                                path=folder/f'{name}.npz'
                                branch_row=dict(id=name,path=str(path.relative_to(output)),
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),steps=length,
                                    world_valid=valid.tolist(),prefix_max_state_error=state_error,
                                    prefix_scene_max_state_error=scene_error,
                                    prefix_bev_different_fraction=different,collision=bool(np.any(np.asarray(flags)[:,0])),
                                    out_of_road=bool(np.any(np.asarray(flags)[:,1])))
                                if name=='keep_repeat':row['noise_replicate']=branch_row
                                else:row['branches'].append(branch_row)
                            row['context']=str((folder/'context.npz').relative_to(output));report['groups'].append(row)
                            report['elapsed_seconds']=time.perf_counter()-started
                            write_json(output/'manifest.json',report)
                            print(json.dumps(dict(group=group,groups=len(report['groups']),elapsed_s=report['elapsed_seconds'])),flush=True)
                finally:sim.close()
        report['completed']=True
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        restore();report['elapsed_seconds']=time.perf_counter()-started;write_json(output/'manifest.json',report)


def main():
    faulthandler.enable();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--splits',nargs='+',choices=list(BRANCH_SPLITS),default=list(BRANCH_SPLITS))
    p.add_argument('--maps',nargs='+',help='Partition independent maps across collector processes')
    p.add_argument('--deep-smoke',action='store_true',help='One seed per map, all formal prefix lengths')
    args=p.parse_args();collect(args.output,Stage6Config(),args.splits,args.smoke or args.deep_smoke,args.maps,args.deep_smoke)


if __name__=='__main__':main()
