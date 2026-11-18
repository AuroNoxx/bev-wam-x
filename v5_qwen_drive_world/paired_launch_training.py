"""Prepare balanced paired demonstrations for the existing longitudinal Lightning task."""
import json
import numpy as np
from .train_scale_profile import extend_observation, fingerprint
from .motion_profile import profile_features
from .mine_launch_pairs import PAIR, SOURCE, ROOT, load
import torch


def samples():
    records=load(PAIR,'cohort.json')
    observed=load(PAIR,'pair-observations.json')
    grouped={'fit':[], 'paired_holdout':[]}
    for i,(r,o) in enumerate(zip(records,observed['observations'],strict=True)):
        assert r['actor_path']==o['actor_path']
        if r['split']=='fit' and not observed['eligible'][r['route']]:
            continue
        item=torch.load(PAIR/'features'/f'{i:03}.pt',weights_only=True,map_location='cpu')
        assert item['identity']['actor_sha256']==fingerprint(ROOT/r['actor_path'])
        feat=item['features']
        obs,base=profile_features({k:v[None] for k,v in feat.items()})
        with np.load(ROOT/r['label_path']) as y:
            truth,expert=[torch.tensor(y[k]) for k in ('trajectory','controls')]
        hold=r['branch']=='hold'
        assert (float(truth.norm(dim=1).max())<.5)==hold
        sample=(extend_observation(obs[0],o),base[0],feat['ctrl'],feat['v0'],truth,expert,hold,not hold)
        grouped[r['split']].append((r,sample))
    return grouped


def augment(fit,identity):
    groups=samples()
    extra=groups['fit']
    assert extra and len(extra)%2==0
    assert not {r['route'] for r,_ in extra}&(set(identity['development_routes'])|
                                    {r['route'] for r,_ in groups['paired_holdout']})
    # Present complete pairs twice per epoch. Fill the last effective batch with
    # original fitting windows, retaining the original route split and all paired windows.
    expanded=fit+[s for _,s in extra]*2
    padding=(-len(expanded))%8
    expanded+=fit[:padding]
    identity.update(fit_epoch_windows=len(expanded),pair_fit_windows=len(extra),pair_repeats=2,
                    original_padding=padding,source_profile=fingerprint(SOURCE/'best.ckpt'),
                    pair_protocol=fingerprint(ROOT/'configs/v5-qwen-drive-world/paired-launch.json'),
                    pair_manifest=fingerprint(PAIR/'cohort.json'),pair_observations=fingerprint(PAIR/'pair-observations.json'))
    return expanded


@torch.inference_mode()
def evaluate_holdout():
    from .train_scale_profile import ScaleMotion
    records=samples()['paired_holdout']
    results=[]
    for folder in (SOURCE,PAIR):
        ck=torch.load(folder/'best.ckpt',map_location='cpu',weights_only=False)
        net=ScaleMotion(ck['task_state']['identity']).cuda().eval()
        net.load_state_dict(ck['state_dict'])
        rows=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for r,s in records:
                obs,base,raw,v0=(v.cuda()[None] for v in s[:4])
                _,traj,_=net.profile(obs,base,raw,v0)
                truth=s[4].cuda()
                rows.append(dict(route=r['route'],branch=r['branch'],true=float(truth.norm(dim=1).max()),
                            pred=float(traj[0,:,:2].norm(dim=1).max()),
                            ade=float((traj[0,:,:2]-truth).norm(dim=1).mean())))
        results.append(dict(checkpoint=fingerprint(folder/'best.ckpt'),rows=rows))
    (PAIR/'paired-holdout.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':evaluate_holdout()
