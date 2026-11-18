"""Matched fixed-weight replay with per-second drift and launch diagnostics."""
import json
from statistics import median
import numpy as np
from .train_scale_profile import ScaleMotion, extend_observation, fingerprint
from .motion_profile import profile_features
from .independent_launch_roads import OUT, PAIR, load
from .mine_launch_pairs import ROOT
import torch


def measures(trajectory, truth):
    displacement=trajectory[:, :2].norm(dim=1)
    crossings=torch.where(displacement>=.5)[0]
    return dict(ade=float((trajectory[:, :2]-truth).norm(dim=1).mean()),
                max_distance=float(displacement.max()),d_by_sec=displacement[9::10].tolist(),
                speed_by_sec=trajectory[9::10,3].tolist(),
                cross_half_s=(int(crossings[0])+1)/10 if len(crossings) else None)


def summarize(rows, name):
    hold=[r[name] for r in rows if r['branch']=='hold']
    launch=[r[name] for r in rows if r['branch']=='launch']
    crosses=[r['cross_half_s'] for r in hold if r['cross_half_s'] is not None]
    paired=sum(a['max_distance']<.5 and b['max_distance']>=3 for a,b in zip(hold,launch,strict=True))
    return dict(roads=len(hold),strict_holds=sum(r['max_distance']<.5 for r in hold),
                launches=sum(r['max_distance']>=3 for r in launch),both=paired,
                large_drifts=sum(r['max_distance']>=3 for r in hold),
                hold_ade=np.mean([r['ade'] for r in hold]),launch_ade=np.mean([r['ade'] for r in launch]),
                median_hold_distance_by_sec=np.median([r['d_by_sec'] for r in hold],axis=0).tolist(),
                hold_cross_half_median=median(crosses) if crosses else None)


@torch.inference_mode()
def main():
    protocol=load(OUT,'protocol.json')
    nets={}
    for name,folder,key in [('previous',PAIR.parent,'reference_sha'),('candidate',PAIR,'candidate_sha')]:
        path=folder/'best.ckpt'
        assert fingerprint(path)==protocol[key]
        checkpoint=torch.load(path,weights_only=False,map_location='cpu')
        if name=='candidate':assert checkpoint['global_step']==protocol['step']
        net=ScaleMotion(checkpoint['task_state']['identity']).cuda().eval()
        net.load_state_dict(checkpoint['state_dict']); nets[name]=net
    records=load(OUT,'cohort.json')
    obs=load(OUT,'pair-observations.json')
    results=[]
    for i,(r,o,overlap) in enumerate(zip(records,obs['observations'],obs['three_frame_iou'],strict=True)):
        cache=torch.load(OUT/'features'/f'{i:03}.pt',map_location='cpu',weights_only=True)
        identity=cache['identity']
        assert identity['actor_path']==r['actor_path']==o['actor_path']
        assert identity['actor_sha256']==fingerprint(ROOT/r['actor_path']) and identity['seed']==65001+i//2
        actor=load(ROOT,r['actor_path'])
        assert actor['target_speed_kmh']==0
        if i%2==0:
            assert r['branch']=='hold'
            paired_actor=actor
        else:
            assert r['route']==records[i-1]['route'] and r['branch']=='launch'
            for k in ('speed_kmh','target_speed_kmh','navigation_flu','history','ego_acceleration','world_from_ego'):
                np.testing.assert_allclose(actor[k],paired_actor[k],atol=1e-4,rtol=0)
        feat={k:v[None].cuda() for k,v in cache['features'].items()}
        with np.load(ROOT/r['label_path']) as labels:truth=torch.tensor(labels['trajectory']).cuda()
        row=dict(route=r['route'],branch=r['branch'],rgb_dx=o['displacement'],true_dx=r['lead_dx'],
                 min_iou=min(overlap),true_distance=float(truth.norm(dim=1).max()))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            base,v=profile_features(feat)
            inputs=extend_observation(base[0],o)[None]
            for name,net in nets.items():
                _,traj,activity=net.profile(inputs,v,feat['ctrl'],feat['v0'])
                row[name]=measures(traj[0],truth)
                row[name]['activity_by_half_second']=activity[0].sigmoid().tolist()
        results.append(row)
    summary={name:summarize(results,name) for name in nets}
    (OUT/'scores.json').write_text(json.dumps(dict(protocol=protocol,summary=summary,rows=results),indent=2,allow_nan=False))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
