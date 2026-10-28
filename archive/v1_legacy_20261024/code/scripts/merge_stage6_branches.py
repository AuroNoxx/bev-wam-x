"""Audit and merge disjoint complete collection partitions using local hard links."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import numpy as np

from configs.stage6 import Stage6Config,BRANCH_SPLITS
from mini_wa_jepa.provenance import provenance,write_json
from dataset.action_branches import realized_waypoints


def merge(inputs,output):
    cfg=Stage6Config();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(**provenance(cfg),stage=6,data_schema_version=2,completed=False,smoke=False,groups=[],sources=[])
    found=set();expected=set()
    for split,spec in BRANCH_SPLITS.items():
        for mi,m in enumerate(spec['maps']):
            for si in range(getattr(cfg,split+'_seeds_per_map')):
                for step in cfg.prefix_steps:expected.add(f'{split}_{m}_{spec["seed_start"]+mi*10000+si}_{step}')
    configs=[];core_sources=[]
    for source in map(Path,inputs):
        path=source/'manifest.json';manifest=json.loads(path.read_text())
        if not manifest['completed'] or manifest['smoke'] or manifest.get('error'):raise ValueError(f'Incomplete collection {source}')
        configs.append(manifest['config'])
        core_sources.append({k:v for k,v in manifest['source_files'].items() if k in (
            'configs/stage6.py','scripts/collect_stage6_branches.py','dataset/action_branches.py',
            'env/data_v2.py','env/semantic_bev.py','env/expert.py','controller/trajectory_tracker.py')})
        for group in manifest['groups']:
            if group['id'] in found:raise ValueError('Overlapping context '+group['id'])
            found.add(group['id']);original=source/group['context'];folder=output/group['split']/group['id'];folder.mkdir(parents=True)
            for f in original.parent.iterdir():
                if f.is_file():os.link(f,folder/f.name)
            context=np.load(output/group['context'])
            if context['history'].shape!=(4,4,128,128) or context['history'].dtype!=np.uint8:
                raise ValueError('BEV schema mismatch '+group['id'])
            for branch in group['branches']+([group['noise_replicate']] if 'noise_replicate' in group else []):
                actual=hashlib.sha256((output/branch['path']).read_bytes()).hexdigest()
                if actual!=branch['sha256']:raise ValueError('Corrupt branch '+branch['path'])
                with np.load(output/branch['path']) as arrays:
                    if not np.array_equal(arrays['observed_history'],context['history']):raise ValueError('Raw history mismatch '+branch['path'])
                    if not np.allclose(arrays['observed_states'],context['states'],atol=cfg.replay_state_atol,rtol=0):raise ValueError('Raw ego mismatch '+branch['path'])
                    if not np.allclose(arrays['scene_state'],context['scene_state'],atol=cfg.replay_state_atol,rtol=0):raise ValueError('Raw scene mismatch '+branch['path'])
                    ci=0 if branch['id']=='keep_repeat' else cfg.candidates.index(branch['id'])
                    if not np.array_equal(arrays['command'],context['commands'][ci]):raise ValueError('Command identity mismatch')
                    length=len(arrays['executed_actions'])
                    if len(arrays['states'])!=length+1:raise ValueError('Transition state count mismatch')
                    if not np.array_equal(arrays['valid'],np.rint(np.asarray(cfg.future_times_s)/cfg.control_dt_s).astype(int)<=length):raise ValueError('World mask/time mismatch')
                    indices=np.arange(2,21,2);mask=indices<=length
                    if not np.array_equal(arrays['waypoint_valid'],mask):raise ValueError('Waypoint mask/time mismatch')
                    np.testing.assert_allclose(arrays['realized'][mask],realized_waypoints(arrays['states'],0,indices[mask]),rtol=0,atol=1e-6)
            group['context_sha256']=hashlib.sha256((output/group['context']).read_bytes()).hexdigest()
            report['groups'].append(group)
        report['sources'].append(dict(path=str(source.resolve()),manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if any(c!=configs[0] for c in configs) or any(c!=core_sources[0] for c in core_sources):raise ValueError('Partition config/core source mismatch')
    if json.dumps(configs[0],sort_keys=True)!=json.dumps(asdict(cfg),sort_keys=True):raise ValueError('Collection config changed')
    if found!=expected:raise ValueError(f'Cohort mismatch: missing={sorted(expected-found)},extra={sorted(found-expected)}')
    report['groups'].sort(key=lambda r:(list(BRANCH_SPLITS).index(r['split']),BRANCH_SPLITS[r['split']]['maps'].index(r['map_spec']),r['seed'],r['anchor_step']))
    report.update(completed=True,context_count=len(found),branch_count=sum(len(g['branches']) for g in report['groups']),
                  raw_identity_and_alignment_audit_passed=True)
    write_json(output/'manifest.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=merge(a.inputs,a.output)
    print(json.dumps(dict(completed=r['completed'],contexts=r['context_count'],branches=r['branch_count'])),flush=True)


if __name__=='__main__':main()
