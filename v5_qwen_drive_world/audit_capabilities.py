"""Paired open-loop audit across speed, maneuvers, dynamics and candidate-conditioned occupancy."""
import gzip
import json
from functools import lru_cache

import numpy as np
from .train_scale_profile import ScaleMotion, extend_observation, fingerprint
from .mine_launch_pairs import ROOT, PAIR, SOURCE, load
from .geometry_repair_features import RUN, BASE, split_records
from .train_geometry_repair import GeometryRepair, ActionConditionedWorld2D, load_perception, RepairData
from .real_dataset import RealData
from .motion_profile import profile_features
from .runtime import math_attention
from .verify_geometry_repair import map_scores
import torch

OUT = PAIR / 'capability-audit'


@lru_cache(maxsize=8192)
def bench_frame(route, i):
    p=ROOT/'data/public/bench2drive/clips'/route/'anno'/f'{i:05d}.json.gz'
    with gzip.open(p,'rt') as s: frame=json.load(s)
    ego=next(b for b in frame['bounding_boxes'] if b['class']=='ego_vehicle')
    return frame['speed'],ego['rotation'][1],ego['road_id'],ego['lane_id']


@lru_cache(maxsize=4)
def highway_frames(route):
    with (ROOT/'data/v5-qwen-drive-world/highway'/route/'frames.jsonl').open() as f:
        frames={}
        for line in f:
            r=json.loads(line)
            pose=r['current_world_from_ego']
            frames[r['index']]=(r['speed'],np.degrees(np.arctan2(pose[1][0],pose[0][0])),r['road_id'],r['lane_id'])
        return frames


def scenarios(r, truth, ctrl):
    i=r['index']; route=r['route']
    states=np.array([bench_frame(route,j) if r['source']=='bench2drive' else highway_frames(route)[j]
                     for j in range(i,i+51)])
    speeds,_,roads,lanes=states.T
    # Derive heading change from the trajectory itself, using 1 s chords to reduce pose noise.
    path=np.vstack([np.zeros(2),truth])
    start,end=path[10]-path[0],path[-1]-path[-11]
    heading=np.degrees(np.arctan2(start[0]*end[1]-start[1]*end[0],start@end))
    unique=np.unique(lanes)
    change=(len(set(roads))==1 and len(unique)==2 and unique[1]-unique[0]==1 and np.prod(unique)>0
            and np.count_nonzero(np.diff(lanes))==1 and np.all(lanes[:5]==lanes[0]) and np.all(lanes[-5:]==lanes[-1]) and lanes[0]!=lanes[-1])
    bend=min(np.linalg.norm(start),np.linalg.norm(end))>=2 and abs(heading)>=15
    groups=['all',f'speed_{0 if r["speed_kmh"]<40 else 40 if r["speed_kmh"]<80 else 80}']
    if speeds[0]*3.6>=20 and np.ptp(speeds)<=2 and not bend and not change:groups.append('steady')
    if ctrl[:30,0].mean()<-.5:groups.append('decel')
    if speeds[0]*3.6>=10 and max(speeds[-5:])<=.5:groups.append('brake_to_stop')
    if bend:groups.append('bend')
    if change:groups.append('lane_change')
    return groups


def aggregate(rows):
    keys=('ade','cv','old','fde','cv_fde','old_fde','abs_dx5','abs_dy5','peak_ay','peak_jerk',
          'dx5','dy5','old_abs_dx5','old_abs_dy5','cv_abs_dx5','cv_abs_dy5','expert_peak_ay','jerk_p95','expert_jerk_p95')
    result={}
    for name in ('all','speed_0','speed_40','speed_80','steady','decel','brake_to_stop','bend','lane_change'):
        cohort=[r for r in rows if name in r['groups']]
        result[name]=dict(n=len(cohort),routes=len({r['route'] for r in cohort}))
        if cohort:result[name].update({k:np.mean([r[k] for r in cohort],axis=0).tolist() for k in keys},
                                      wins=sum(r['ade']<r['cv'] for r in cohort))
    return result


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=True)
    data=RealData('subset2000')
    saved=load(PAIR,'validation.json')
    assert fingerprint(PAIR/'best.ckpt')==saved['checkpoint_sha256']
    ck=torch.load(PAIR/'best.ckpt',map_location='cpu',weights_only=False)
    ident=ck['task_state']['identity']
    assert ck['global_step']==162 and data.identity==ident['manifest']
    plan=ScaleMotion(ident).cuda().eval()
    plan.load_state_dict(ck['state_dict'])
    assert fingerprint(RUN/'latest.ckpt')==ident['source_geometry_checkpoint']
    per,_=load_perception(load(BASE,'config.json'),device='cpu')
    world=GeometryRepair(ActionConditionedWorld2D(per)).cuda().eval()
    del per
    state=torch.load(RUN/'latest.ckpt',map_location='cpu',weights_only=False,mmap=True)['state_dict']
    world.load_state_dict({k.removeprefix('net.'):v for k,v in state.items()})
    cues=load(SOURCE/'seeded','observations.json')['splits']
    assert fingerprint(SOURCE/'seeded'/'observations.json')==saved['evaluation_observations_sha256']
    hist=np.zeros((2,512),np.int64); sums=np.zeros(4)
    changed=np.zeros((3,3)); threshold=np.zeros(2)
    wmetrics=[]; all_rows={}; outputs={}
    x=torch.arange(128,device='cuda')+.5-16
    y=(torch.arange(128,device='cuda')+.5)*.3-19.2
    corridor=(x>0)&(x<80)&(y[:,None].abs()<=2)
    with math_attention(),torch.autocast('cuda',dtype=torch.bfloat16):
        for split,records in split_records(data).items():
            samples=RepairData(records,split);rows=[]
            for i,r in enumerate(records):
                if split=='train' and r['speed_kmh']<40:continue
                feat,lab,_=samples[i]
                feat={k:v[None].cuda() for k,v in feat.items()}
                obs,base=profile_features(feat)
                cue=cues[split][i];assert cue['actor_path']==r['actor_path']
                v=feat['v0'].flatten()
                _,tr,_=plan.profile(extend_observation(obs[0],cue)[None],base,feat['ctrl'],v)
                from .kinematics import integrate
                old=integrate(feat['ctrl'],v)[0,:,:2]
                tr=tr[0].float()
                truth=lab['trajectory'].cuda()
                cv=torch.zeros_like(truth)
                cv[:,0]=v*torch.arange(1,51,device='cuda')*.1
                e=(tr[:,:2]-truth).norm(dim=1); ce=(cv-truth).norm(dim=1); oe=(old-truth).norm(dim=1)
                ctrl=lab['controls'].cuda()
                jerk=tr[:,4:6].diff(dim=0).abs()/.1; ej=ctrl.diff(dim=0).abs()/.1
                rows.append(dict(route=r['route'],index=r['index'],groups=scenarios(r,truth.cpu().numpy(),ctrl.cpu().numpy()),
                    ade=float(e.mean()),cv=float(ce.mean()),old=float(oe.mean()),fde=e[[9,29,49]].tolist(),
                    cv_fde=ce[[9,29,49]].tolist(),old_fde=oe[[9,29,49]].tolist(),
                    dx5=float(tr[-1,0]-truth[-1,0]),dy5=float(tr[-1,1]-truth[-1,1]),
                    old_abs_dx5=float(abs(old[-1,0]-truth[-1,0])),old_abs_dy5=float(abs(old[-1,1]-truth[-1,1])),
                    cv_abs_dx5=float(abs(cv[-1,0]-truth[-1,0])),cv_abs_dy5=float(abs(cv[-1,1]-truth[-1,1])),
                    abs_dx5=float(abs(tr[-1,0]-truth[-1,0])),abs_dy5=float(abs(tr[-1,1]-truth[-1,1])),
                    peak_ay=float(tr[:,5].abs().max()),expert_peak_ay=float(ctrl[:,1].abs().max()),
                    peak_jerk=float(jerk.norm(dim=1).max()),jerk_p95=jerk.quantile(.95,dim=0).tolist(),
                    expert_jerk_p95=ej.quantile(.95,dim=0).tolist()))
                if split=='validation':
                    a=saved['rows'][i]
                    assert (r['route'],r['index'])==(a['route'],a['index']) and abs(a['ade']-float(e.mean()))<1e-4
                    logits=world.occupancy(world.appearance(feat['bev'].float()),feat['latent'],tr[None,:,:2],v).sigmoid()
                    future=logits[0,1:].float(); target=lab['occupancy'].cuda().float()
                    mask=(lab['support']&lab['current_support'][None]).cuda().bool()
                    persist=logits[0,:1].float().expand_as(future)
                    err=(future-target).square(); static=(persist-target).square()
                    wmetrics.append([float(err[mask].mean()),float(static[mask].mean())])
                    h,s=map_scores(future[1],target[1],mask[1]&corridor)
                    hist+=h;sums+=s
                    area=mask[1]&corridor
                    positive_pred=(future[1]>=.5)&area
                    threshold += [int(positive_pred.sum()), int((positive_pred & target[1].bool()).sum())]
                    for t in range(3):
                        moving=mask[t] & (target[t]!=lab['current_occupancy'].cuda())
                        changed[t]+=[int(moving.sum()),float(err[t][moving].sum()),float(static[t][moving].sum())]
            outputs[split]=aggregate(rows);all_rows[split]=rows
            print(split,len(rows),flush=True)
    neg,pos=hist[:,::-1];precision=np.cumsum(pos)/np.maximum(1,np.cumsum(pos+neg))
    outputs['world']=dict(brier=np.mean(wmetrics,axis=0).tolist(),ap_2s=float((precision*pos).sum()/sum(pos)),
                          recall_at_true_count=sums[2]/sums[1],cells=int(sums[0]),positive=int(sums[1]),
                          threshold_half_predictions=int(threshold[0]),threshold_half_hits=int(threshold[1]),
                          changes=[dict(cells=int(n),brier=a/n,persist=b/n,gain=1-a/b) for n,a,b in changed])
    (OUT/'scores.json').write_text(json.dumps(dict(checkpoint=saved['checkpoint_sha256'],summary=outputs,rows=all_rows),
                                               allow_nan=False,indent=2))
    print(outputs['world'],flush=True)


if __name__=='__main__':main()
