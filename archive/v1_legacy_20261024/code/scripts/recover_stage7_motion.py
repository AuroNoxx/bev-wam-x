"""Recover exact observed motion by verified prefix replay; never rewrite raw data."""
import argparse
from collections import deque
import json
from pathlib import Path
import tarfile
import time
import traceback

import numpy as np

from configs.stage2 import Stage2Config
from configs.stage7 import Stage7Config
from dataset.recording import vehicle_state
from env.data_v2 import configure_egl,make_env,NativeBEVProvider
from mini_wa_jepa.geometry import wrap_angle
from mini_wa_jepa.provenance import provenance,write_json
from scripts.preflight_risk_collection import scene_state
from training.flow_matching import file_sha256


def observed_motion(states,dt=.1):
    states=np.asarray(states)
    if states.shape != (5,6) or not np.isfinite(states).all():
        raise ValueError('Five actual observed vehicle states required')
    motion=np.zeros((4,2),np.float32)
    motion[:,0]=states[-4:,3]/10
    motion[:,1]=wrap_angle(np.diff(states[:,2]))/dt
    return motion


def recover(inputs,output,smoke=False,reuse=()):
    cfg=Stage7Config();env_cfg=Stage2Config();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(**provenance(cfg),completed=False,smoke=smoke,groups={},inputs=[],
        reason='Original collector stored placeholder zero ego before computing observed motion; original files retained unchanged',
        derived_fields='speed/10 and wrapped backward heading difference per .1s for four observed frames')
    with tarfile.open(output/'source_snapshot.tar.gz','w:gz') as arc:
        for path in report['source_files']:arc.add(path,arcname=path)
    write_json(output/'manifest.json',report)
    items=[]
    for root in map(Path,inputs):
        manifest=json.loads((root/'manifest.json').read_text())
        if not smoke and not manifest['completed']:raise ValueError('Incomplete partition')
        report['inputs'].append(dict(path=str(root),manifest_sha256=file_sha256(root/'manifest.json')))
        items.extend((root,g) for g in (manifest['groups'][:2] if smoke else manifest['groups']))
    available={}
    report['reused_recoveries']=[]
    for previous in map(Path,reuse):
        prior=json.loads((previous/'manifest.json').read_text())
        if not prior['completed'] or prior['smoke']:raise ValueError('Only completed formal recovery can be reused')
        report['reused_recoveries'].append(dict(path=str(previous),manifest_sha256=file_sha256(previous/'manifest.json')))
        for key,value in prior['groups'].items():
            if key in available:raise ValueError('Duplicate recovered context')
            available[key]=value
    _,restore=configure_egl(1);started=time.perf_counter();sim=None;current=None
    try:
        for root,g in items:
            context_path=root/g['context'];context=np.load(context_path)
            if g['id'] in available:
                old=available[g['id']]
                if old['source_context_sha256']!=file_sha256(context_path) or old['sha256']!=file_sha256(old['path']):
                    raise ValueError('Reused recovery/context changed')
                report['groups'][g['id']]=old;write_json(output/'manifest.json',report)
                continue
            ordered=bool(g.get('semantic_order',False))
            map_key=(g['split'],g['map_spec'],ordered)
            if current!=map_key:
                if sim is not None:sim.close()
                sim=make_env(g['map_spec'],g['seed'],env_cfg,num_scenarios=1000,control_dt_s=.1,
                             environment_overrides={'force_destroy':True},semantic_order=ordered);current=map_key
            obs,_=sim.reset(seed=g['seed']);provider=NativeBEVProvider(sim,env_cfg.bev);provider.reset(obs)
            states=deque([vehicle_state(sim.agent)],maxlen=5);bevs=deque(maxlen=4)
            prefix=context['prefix_actions']
            assert len(prefix)==g['anchor_step']
            for step,action in enumerate(prefix):
                obs,_,tm,tc,info=sim.step(action);provider.update(obs)
                states.append(vehicle_state(sim.agent))
                if step>=len(prefix)-4:bevs.append(provider.capture().raster.copy())
                if tm or tc:raise RuntimeError('Saved prefix now terminates; preserve MRE')
            states=np.asarray(states);bev_error=int(np.count_nonzero(np.asarray(bevs)!=context['history']))
            state_error=float(np.max(np.abs(states[-4:]-context['states'])))
            current_scene=scene_state(sim);reference=context['scene_state']
            if current_scene.shape!=reference.shape:raise ValueError('Scene cardinality changed')
            scene_error=float(np.max(np.abs(current_scene-reference),initial=0))
            if bev_error or max(state_error,scene_error)>1e-5:
                np.savez_compressed(output/f'{g["id"]}_mismatch.npz',states=states,bevs=np.asarray(bevs),scene=current_scene)
                raise RuntimeError(f'Replay mismatch {g["id"]}: {state_error}, {scene_error}, {bev_error}')
            ego=observed_motion(states);path=output/f'{g["id"]}.npz'
            np.savez_compressed(path,ego=ego,states=states,source_context=str(context_path),
                                source_context_sha256=file_sha256(context_path))
            report['groups'][g['id']]=dict(path=str(path),sha256=file_sha256(path),
                source_context=str(context_path),source_context_sha256=file_sha256(context_path),
                state_max_abs_error=state_error,scene_max_abs_error=scene_error,bev_different_values=bev_error,
                original_placeholder_zero=bool(np.all(context['ego']==0)))
            write_json(output/'manifest.json',report)
            if len(report['groups'])%8==0:print(json.dumps(dict(groups=len(report['groups']),total=len(items))),flush=True)
        report.update(completed=True,elapsed_s=time.perf_counter()-started)
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        if sim is not None:sim.close()
        restore();write_json(output/'manifest.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--reuse',nargs='*',default=[])
    a=p.parse_args();recover(a.inputs,a.output,a.smoke,a.reuse)
