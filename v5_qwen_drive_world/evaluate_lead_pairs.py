"""Evaluate the locked displacement readout without exposing CARLA truth to the actor."""
import hashlib
import json
import numpy as np
from .lead_motion import ROOT,DEST,RealData,load_base,front_observation,torch
from .train_lead_motion import LeadReadout,metrics
from qwen_drive import CameraFrame


def main():
    datafolder=ROOT/'data/v5-qwen-drive-world/lead-pairs-aligned'
    doc=json.loads((datafolder/'manifest.json').read_text())
    cfg=json.loads((ROOT/'configs/v5-qwen-drive-world/lead-pair-evaluation.json').read_text())
    assert doc['config']==cfg
    source=DEST/'directed/latest.ckpt'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg['checkpoint_sha256']
    base=load_base(RealData('subset2000'))
    x=torch.load(DEST/'directed/observations.pt',weights_only=True)
    head=LeadReadout(x,{},[]).cuda().eval()
    ckpt=torch.load(source,map_location='cpu',weights_only=False)
    assert ckpt['global_step']==700
    head.load_state_dict(ckpt['state_dict'])
    assert ckpt['task_state']['config']['movement_m']==cfg['decision_threshold_m']
    windows=[]
    initial={}
    with torch.no_grad():
        for r in doc['windows']:
            path=ROOT/r['actor']
            actor=json.loads(path.read_text())
            frames=actor['observations']
            assert len(frames)==4 and np.allclose(np.diff([f['time'] for f in frames]),.5,atol=.001)
            timeline=json.loads((path.parent/'physics.json').read_text())
            q=cfg['query_tick']
            start=np.stack([timeline[0][k] for k in ('ego','lead')])
            if r['route'] in initial: np.testing.assert_allclose(initial[r['route']],start,atol=.001,rtol=0)
            initial[r['route']]=start
            T=np.linalg.inv(timeline[q]['ego'])
            true_dx=(T@(np.array(timeline[q]['lead'])-np.array(timeline[q-10]['lead'])))[0,3]
            assert abs(true_dx-r['target'])<1e-5
            for f,idx in zip(frames,(q-15,q-10,q-5,q)):
                snap=timeline[idx]
                mount=np.linalg.solve(snap['ego'],np.linalg.inv(f['camera']['world2cam']))
                assert abs(f['time']-snap['time'])<=.005 and f['physics_frame']==snap['frame']
                assert np.linalg.norm(mount[:3,3]-[.8,0,1.6])<cfg['max_mount_error_m']
                assert np.rad2deg(np.arccos(np.clip((np.trace(mount[:3,:3])-1)/2,-1,1)))<=cfg['max_mount_rotation_deg']
            obs=front_observation(base,[CameraFrame(ROOT/f['path']) for f in frames],actor,
                                  [f['camera'] for f in frames],directed=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=float(head(obs[None].cuda()).item())
            assert np.isfinite(pred)
            windows.append(dict(**r,prediction=pred))
    pairs=[]
    for route in sorted({r['route'] for r in windows}):
        h,d=[next(r for r in windows if r['route']==route and r['branch']==b) for b in ('hold','depart')]
        pairs.append(dict(route=route,correct_order=d['prediction']>h['prediction'],
                          both_correct=h['prediction']<=cfg['decision_threshold_m']<d['prediction'],
                          gap_difference_m=abs(d['gap']-h['gap'])))
    assert len(pairs)==len(cfg['maps'])*cfg['pairs_per_map'] and len(windows)==2*len(pairs)
    summary=dict(overall=metrics(windows),by_town={t:metrics([r for r in windows if r['route'].startswith(t+'-')]) for t in cfg['maps']},
                 correctly_ordered=sum(r['correct_order'] for r in pairs),both_correct=sum(r['both_correct'] for r in pairs))
    out=DEST/'directed/carla-pair-evaluation.json'
    out.write_text(json.dumps(dict(config=cfg,summary=summary,pairs=pairs,windows=windows),indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
