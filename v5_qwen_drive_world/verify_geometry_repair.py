"""Replay the frozen validation manifest with the same actor noise and occupancy masks."""
import hashlib
import json

import numpy as np
from .runtime import ROOT,configure,math_attention

configure()
import torch
from .geometry_repair import GeometryRepair,WorldRepairActor
from .geometry_repair_features import BASE,CFG,RUN,load_base
from .kinematics import integrate
from .real_dataset import RealData
from .train_geometry_repair import RepairData


def load_repaired_actor(data,motion=False,world_only=False):
    identity=json.loads((RUN/'feature-identity.json').read_text())
    with (BASE/'latest.ckpt').open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=identity['base_sha256']:
            raise ValueError('Base checkpoint SHA mismatch')
    checkpoint=torch.load(RUN/'latest.ckpt',map_location='cpu',weights_only=False,mmap=True)
    if checkpoint['task_state']['feature_identity']!=identity or data.identity!=identity['manifest']:
        raise ValueError('Validation or repair checkpoint identity mismatch')
    step=checkpoint['global_step']
    assert step==CFG['train_windows']*CFG['epochs']//CFG['effective_batch']
    assert all(int(s['step'])==step and s['exp_avg'].dtype==torch.float32
               for s in checkpoint['optimizer_states'][0]['state'].values())
    base=load_base(data)
    if motion and world_only:
        raise ValueError('Choose a single longitudinal readout')
    net=(WorldRepairActor if world_only else GeometryRepair)(base.world).cuda().eval()
    net.load_state_dict({k.removeprefix('net.'):v for k,v in checkpoint['state_dict'].items()})
    if motion:
        from .motion_profile import ProfileActor
        from .train_motion_profile import DEST,CONFIG
        final=torch.load(DEST/'latest.ckpt',map_location='cpu',weights_only=False,mmap=True)
        best=torch.load(DEST/'best.ckpt',map_location='cpu',weights_only=False,mmap=True)
        fit=best['task_state']['identity']
        assert fit==json.loads((DEST/'identity.json').read_text())==final['task_state']['identity']
        with (RUN/'latest.ckpt').open('rb') as stream:
            assert fit['source_geometry_checkpoint']==hashlib.file_digest(stream,'sha256').hexdigest()
        assert fit['source_features']==identity and final['global_step']==len(fit['indices']['fit'])*CONFIG['epochs']//8
        history=[json.loads(line) for line in (DEST/'development.jsonl').read_text().splitlines()]
        assert best['global_step']==min(history,key=lambda r:r['score'])['step']
        revised=ProfileActor(base.world).cuda().eval()
        revised.load_state_dict({**net.state_dict(),**best['state_dict']})
        net=revised
    return base,net,step


def map_scores(field,target,mask):
    p=field[mask].float().cpu().numpy()
    y=target[mask].bool().cpu().numpy()
    k=int(y.sum())
    # Ranking ties on the Kth score share the remaining selection weight equally.
    cutoff=np.partition(p,len(p)-k)[len(p)-k] if k else np.inf
    above=p>cutoff
    tied=p==cutoff
    hit=float(y[above].sum()+y[tied].sum()*(k-above.sum())/max(1,int(tied.sum())))
    bins=np.minimum((p*512).astype(int),511)
    hist=np.stack([np.bincount(bins[y==flag],minlength=512) for flag in (False,True)])
    return hist,np.array([len(p),k,hit,np.square(p-y).sum(dtype=np.float64)])


def paired_summary(proof,logged,old):
    audit=BASE/'final-window-audit'
    previous=json.loads((audit/'windows.json').read_text())
    changed=json.loads((audit/'world-probe.json').read_text())
    rows=logged['samples']
    def column(records,key):
        return np.asarray([r[key] for r in records])
    def summarize(indices,arrays):
        return {k:float(v[indices].mean()) for k,v in arrays.items()} if indices.any() else {}
    speed=column(rows,'speed_kmh')
    dmax=column(rows,'dmax')
    target=column(proof['windows'],'target_kmh')
    routes=column(rows,'route')
    assert np.array_equal(routes,column(previous,'route'))
    assert np.array_equal(column(rows,'index'),column(previous,'index'))
    assert np.allclose(speed,column(previous,'speed_kmh'))
    assert np.allclose(dmax,column(previous,'max_expert_displacement'),atol=.0001)
    trajectory={'before_ade':column(previous,'model_ade'),'after_ade':column(rows,'ade'),
                'cv_ade':column(previous,'cv_ade'),'before_x5':column(proof['windows'],'old_x5'),
                'after_x5':column(rows,'x5'),'true_x5':column(rows,'true_x')}
    for i,h in enumerate((1,3,5)):
        trajectory[f'before_fde{h}']=column(previous,f'model_fde{h}')
        trajectory[f'after_fde{h}']=column(rows,'fde')[:,i]
        trajectory[f'cv_fde{h}']=column(previous,f'cv_fde{h}')
    holds=dmax<.5
    launch=(speed<=3)&(dmax>=3)
    groups={'all':speed>=0,'hold':holds,'moving_from_low':(speed<=3)&~holds,
            'clear_launch':launch,'zero_target_launch':launch&(target==0),
            '0_40':speed<40,'40_80':(speed>=40)&(speed<80),'80_120':speed>=80}
    cohorts={}
    for name,idx in groups.items():
        cohorts[name]=dict(windows=int(idx.sum()),routes=len(set(routes[idx])),**summarize(idx,trajectory))
        if idx.any():
            cohorts[name]['ade_wins_vs_before']=int((trajectory['after_ade'][idx]<trajectory['before_ade'][idx]).sum())
    balanced={key:float(np.mean([values[routes==r].mean() for r in set(routes)])) for key,values in trajectory.items()}
    behavior={}
    for name,key in [('before','old_displacement'),('after','model_displacement')]:
        distance=column(proof['windows'],key)
        stationary=distance[holds&(speed<=3)&(target==0)]
        moving=distance[launch&(target==0)]
        pairs=moving[:,None]-stationary
        behavior[name]=dict(hold_within_half_meter=int((distance[holds]<.5).sum()),
            launches_ge_3m=int((distance[launch]>=3).sum()),zero_target_launches_ge_3m=int((moving>=3).sum()),
            zero_target_auc=float(((pairs>0)+.5*(pairs==0)).mean()),
            zero_target_pairs=pairs.size)
    dynamics=[]
    for h in range(3):
        old_h=[r for r in changed if r['h']==h+1]
        n=int(column(old_h,'changed').sum())
        assert n==int(column(rows,'changed')[:,h].sum()) and n>0
        entry={'seconds':h+1,'changed_cells':n}
        for name,record,w,s in [('before',old_h,'world_changed_sse','static_changed_sse'),
                                ('after',rows,'world_changed_sse','persistence_changed_sse')]:
            a,b=column(record,w),column(record,s)
            if name=='after':
                a,b=a[:,h],b[:,h]
            entry[name+'_world']=float(a.sum()/n)
            entry[name+'_static']=float(b.sum()/n)
            entry[name+'_relative_gain']=float(1-a.sum()/b.sum())
        dynamics.append(entry)
    return dict(cohorts=cohorts,equal_route=balanced,behavior=behavior,dynamics=dynamics,world=proof['world'],
                max_validation_speed_kmh=float(speed.max()),
                before_jerk=float(column(old,'jerk_abs_max').mean()),after_jerk=float(column(rows,'jerk').mean()),
                before_static=float(column(old,'persistence_brier').mean()),after_static=float(column(rows,'persistence_brier').mean()))


@torch.no_grad()
def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--motion',action='store_true')
    parser.add_argument('--world-only',action='store_true')
    args=parser.parse_args()
    dest=RUN/'motion-profile' if args.motion else RUN/'world-only' if args.world_only else RUN
    dest.mkdir(exist_ok=True)
    torch.set_num_threads(6)
    data=RealData('subset2000')
    base,net,step=load_repaired_actor(data,args.motion,args.world_only)
    split=RepairData(data.validation,'validation')
    logged=json.loads((RUN/'validation.json').read_text())
    assert logged['step']==step and len(split)==len(logged['samples'])==128
    audit=BASE/'final-window-audit'
    methods=('before_observed','after_observed','before_proposed','after_proposed')
    hist={m:np.zeros((2,512),np.int64) for m in methods}
    sums={m:np.zeros(4) for m in methods}
    future_brier={m:[] for m in methods}
    rows=[]
    verified_raw={}
    max_original_trajectory_diff=0.
    max_logged_ade_diff=0.
    max_prefix_diff=0.
    max_logged_brier_diff=0.
    frozen_rows=json.loads((BASE/'validation.jsonl').read_text().splitlines()[-1])['samples']
    reference=np.load(audit/'predictions.npz')
    x=torch.arange(128,device='cuda')+.5-16
    y=(torch.arange(128,device='cuda')+.5)*.3-19.2
    corridor=(x>0)&(x<80)&(y[:,None].abs()<=2)
    with math_attention(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i,record in enumerate(data.validation):
            feature,label,_=split[i]
            feat={k:v[None].cuda() for k,v in feature.items()}
            truth=label['trajectory'][None].cuda()
            target=label['occupancy'][None].cuda()
            mask=(label['support'] & label['current_support'][None])[None].bool().cuda()
            original=integrate(feat['ctrl'],feat['v0'].flatten())
            canonical=torch.as_tensor(reference['trajectories'][i],device='cuda')
            assert str(reference['route'][i])==record['route'] and int(reference['index'][i])==record['index']
            max_original_trajectory_diff=max(max_original_trajectory_diff,float((original-canonical).abs().max()))
            latent=feat['latent']
            v0=feat['v0'].flatten()
            before_obs=base.world(latent,truth,v0)['occupancy']
            before_plan=base.world(latent,original[...,:2],v0)['occupancy']
            encoded=net.appearance(feat['bev'].float())
            if args.motion:
                from .motion_profile import profile_features
                observed,base_v=profile_features(feat)
                _,after,_=net.profile(observed,base_v,feat['ctrl'],v0)
            else:
                _,after=net.plan(encoded,feat['state'],feat['ctrl'],v0)
            after_obs=net.occupancy(encoded,latent,truth,v0).sigmoid()[:,1:]
            after_plan=net.occupancy(encoded,latent,after[...,:2],v0).sigmoid()[:,1:]
            saved=logged['samples'][i]
            assert (saved['route'],saved['index'])==(record['route'],record['index'])
            error=(after[...,:2]-truth).norm(dim=-1).mean()
            if args.world_only:
                assert torch.equal(after[0],canonical)
            if args.motion or args.world_only:
                err=(after[...,:2]-truth).norm(dim=-1)[0]
                saved.update(ade=float(error),fde=[float(err[t]) for t in (9,29,49)],x5=float(after[0,-1,0]),
                             jerk=float((after[...,4:6].diff(dim=1)/.1).norm(dim=-1).max()))
            else:
                max_logged_ade_diff=max(max_logged_ade_diff,abs(float(error)-saved['ade']))
            for method,prob in zip(methods,(before_obs,after_obs,before_plan,after_plan)):
                h,s=map_scores(prob[:,1],target[:,1],mask[:,1]&corridor)
                hist[method]+=h
                sums[method]+=s
                future_brier[method].append(float((prob-target).square()[mask].mean()))
            max_logged_brier_diff=max(max_logged_brier_diff,
                abs(future_brier['before_observed'][-1]-frozen_rows[i]['world_brier']),
                abs(future_brier['after_observed'][-1]-saved['world_brier']))
            low=record['speed_kmh']<=3
            cohort='hold' if saved['dmax']<.5 else 'launch' if low and saved['dmax']>=3 else ''
            if cohort and cohort not in verified_raw:
                _,live_tr,live_logits=net.predict_actor(base,record,1907+i)
                delta=float((after-live_tr).abs().max())
                delta_map=float((after_plan-live_logits[:,1:].sigmoid()).abs().max())
                verified_raw[cohort]={'route':record['route'],'index':record['index'],
                                      'trajectory_max_abs':delta,'map_probability_max_abs':delta_map}
                altered=truth.clone()
                altered[:,10:]+=20
                early=net.occupancy(encoded,latent,altered,v0).sigmoid()[:,1]
                max_prefix_diff=max(max_prefix_diff,float((early-after_obs[:,0]).abs().max()))
            actor=json.loads((ROOT/record['actor_path']).read_text(encoding='utf-8'))
            rows.append({'target_kmh':actor['target_speed_kmh'],'old_x5':float(original[0,-1,0]),
                         'model_displacement':float(after[...,:2].norm(dim=-1).max()),
                         'old_displacement':float(original[...,:2].norm(dim=-1).max())})
    result={}
    for method,(negative,positive) in hist.items():
        tp=np.cumsum(positive[::-1]); predicted=np.cumsum((negative+positive)[::-1])
        precision=tp/np.maximum(1,predicted)
        n,k,hit,sse=sums[method]
        result[method]=dict(ap=float(np.sum(precision*positive[::-1])/k),front_brier=float(sse/n),
                            recall_at_true_count=float(hit/k),positive=int(k),valid=int(n),
                            recall_at_half=float(tp[255]/k),precision_at_half=float(precision[255]) if predicted[255] else None,
                            reported_at_half=int(predicted[255]),future_brier=float(np.mean(future_brier[method])))
    proof=dict(step=step,raw=verified_raw,replay_max_abs=max_original_trajectory_diff,
               ade_replay_max_abs=max_logged_ade_diff,brier_replay_max_abs=max_logged_brier_diff,
               prefix_probability_max_abs=max_prefix_diff,world=result,windows=rows)
    if args.motion:
        history=[json.loads(line) for line in (dest/'development.jsonl').read_text().splitlines()]
        proof['profile_selection']=min(history,key=lambda r:r['score'])
        proof['profile_final_step']=history[-1]['step']
        proof['ade_replay_max_abs']=None
        logged['profile_selection']=proof['profile_selection']
        (dest/'validation.json').write_text(json.dumps(logged))
    if args.world_only:
        proof['unchanged_source_trajectories']=len(rows)
        (dest/'validation.json').write_text(json.dumps(logged))
    (dest/'verification.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps({k:v for k,v in proof.items() if k!='windows'},indent=2),flush=True)
    assert max_original_trajectory_diff<.001 and max_logged_ade_diff<.001 and max_logged_brier_diff<.00005
    assert max_prefix_diff==0 and len(verified_raw)==2
    assert all(v['trajectory_max_abs']<.001 and v['map_probability_max_abs']<.001 for v in verified_raw.values())
    summary=paired_summary(proof,logged,frozen_rows)
    (dest/'paired-summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
