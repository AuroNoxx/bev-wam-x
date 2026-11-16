"""Aggregate all preregistered runs, matched metrics and logged-box risk proxy."""
import argparse, json
import numpy as np
import torch
from lightning_task import ROOT
from train_joint_occ import digest
from audit_joint_occ import write
from diagnose_occ_frozen import groups
from diagnose_v3_aggregation import annotations_and_collisions, from_xy


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='artifacts/joint-occ-v1/runs');a=p.parse_args()
    root=ROOT/a.runs;cfg=json.loads((root/'experiment-config.json').read_text());selection=json.loads((ROOT/cfg['selection']).read_text())
    names=np.asarray(selection['splits']['validation']);states=[];targets=[]
    for name in names:
        with np.load(ROOT/'data/occ_v3'/name) as f:states.append(f['state']);targets.append(f['trajectory'])
    state,target=np.stack(states),np.stack(targets);masks=groups(state,target)
    methods={'expert':target[None]};rows={};checkpoint_checks=[]
    for seed in cfg['seeds']:
        for variant in cfg['variants']:
            key=f'{variant}-seed{seed}';folder=root/key
            assert json.loads((folder/'completion.json').read_text())['global_step']==cfg['max_steps']
            report=json.loads((folder/'validation.json').read_text());train=json.loads((folder/'train.json').read_text());before=json.loads((folder/'before.json').read_text())
            resource=json.loads((folder/'resources.json').read_text())
            with np.load(folder/'validation-predictions.npz') as f:
                assert np.array_equal(names,f['names']);prediction=f['normal']
            methods[key+'/candidates']=prediction.transpose(0,2,1,3,4).reshape(-1,len(names),25,5)
            methods[key+'/median']=from_xy(np.median(prediction[...,:2]*[50,10],axis=2))
            for filename in ['latest.ckpt','best.ckpt']:
                ck=torch.load(folder/filename,map_location='cpu',weights_only=False)
                assert ck['task_state']['manifest_sha']==selection['manifest_sha']
                assert ck['task_state']['settings']['selection_sha']==digest(ROOT/cfg['selection'])
                assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
                steps={int(v['step']) for v in ck['optimizer_states'][0]['state'].values() if 'step' in v}
                assert steps=={ck['global_step']}
                if filename=='latest.ckpt':assert ck['global_step']==cfg['max_steps']
                checkpoint_checks.append({'run':key,'file':filename,'step':ck['global_step'],'sha256':digest(folder/filename),'finite':True})
            assert sorted(p.name for p in folder.glob('*.ckpt'))==['best.ckpt','latest.ckpt']
            rows[key]={'validation':report,'train':train['normal']['groups'],'before_train':before['normal']['groups'],'resources':resource}
    hits,_,boxcount=annotations_and_collisions(names,methods,target,masks['deceleration'])
    for key,row in rows.items():
        candidate=hits[key+'/candidates'].any(-1);median=hits[key+'/median'].any(-1)
        row['logged_box_risk']={g:{'first_candidate_overlap_fraction':float(candidate[0,mask].mean()),
            'all_candidates_overlap_fraction':float(candidate[:,mask].all(0).mean()),
            'mean_candidate_overlap_fraction':float(candidate[:,mask].mean()),
            'median_tangent_overlap_fraction':float(median[:,mask].mean())} for g,mask in masks.items() if mask.any()}
    gates={}
    for variant in ['auxiliary','fusion']:
        per_seed=[]
        for seed in cfg['seeds']:
            row=rows[f'{variant}-seed{seed}'];control=rows[f'temporal_flow-seed{seed}']
            val=row['validation']['normal']['groups'];base=control['validation']['normal']['groups']
            per_seed.append({'seed':seed,
                'deceleration_ADE_improvement_at_least10pct':val['deceleration']['meanADE_m']<=base['deceleration']['meanADE_m']*.9,
                'overall_ADE_regression_at_most5pct':val['all']['meanADE_m']<=base['all']['meanADE_m']*1.05,
                'stationary_forward_bias_not_increased':max(0,val['stationary']['median_final_x_bias_m'])<=max(0,base['stationary']['median_final_x_bias_m']),
                'noncollapse':row['validation']['representation']['noncollapse_probe_pass']})
        gates[variant]={'per_seed':per_seed,'passed_both_seeds':all(all(v for k,v in p.items() if k!='seed') for p in per_seed)}
    protected=json.loads((ROOT/'artifacts/joint-occ-v1/audit/protected-before.json').read_text())
    changed=[name for name,h in protected.items() if not (ROOT/name).exists() or digest(ROOT/name)!=h]
    assert changed==[],changed
    # Data hash audit files also pinned all 6420 caches; recheck at completion.
    count=0
    m=json.loads((ROOT/cfg['manifest']).read_text())
    for r in m['routes']:
        records=json.loads((ROOT/'artifacts/diagnostics/alignment-v3/routes'/f"{r['route']}.json").read_text())
        for record in records['samples']:
            assert digest(ROOT/'data/occ_v3'/record['sample'])==record['new_sha256'];count+=1
    summary={'runs':rows,'gates':gates,'checkpoint_checks':checkpoint_checks,'original_files_unchanged':len(protected),'cache_hashes_rechecked':count,
        'expert_logged_box_overlap_fraction':float(hits['expert'].any(-1).mean()),'boxes_checked':boxcount,
        'test_read':False,'closed_loop_run':False,'limitations':['64 selected validation windows, not full-split generalization; overlapping windows',
            'Two initialization seeds with one fixed sampling seed sequence; not statistical significance',
            'Bounding boxes follow logged futures, not responses to new ego actions; no static geometry, continuous sweep or legal lane test',
            'Per-step full epoch projections exclude disk I/O, validation and saves; small fit loss curves do not establish convergence',
            'Source-level future fusion and annotation timestamps unresolved; oracle OCC research only']}
    write(root.parent/'summary.json',summary)
    compact=[]
    for key,row in rows.items():
        v=row['validation']['normal']['groups'];t=row['train'];b=row['before_train']
        compact.append({'run':key,'val_ADE':v['all']['meanADE_m'],'val_stop_ADE':v['stationary']['meanADE_m'],
            'val_decel_ADE':v['deceleration']['meanADE_m'],'train_ADE_before':b['all']['meanADE_m'],'train_ADE_after':t['all']['meanADE_m'],
            'step_ms':row['resources']['step_seconds_mean']*1000,'peak_reserved_gib':row['resources']['peak_reserved_gib'],
            'first_box_overlap':row['logged_box_risk']['all']['first_candidate_overlap_fraction']})
    write(root.parent/'compact-results.json',compact)
    print(json.dumps({'results':compact,'gates':gates,'protected':len(protected),'caches':count}),flush=True)


if __name__=='__main__':main()
