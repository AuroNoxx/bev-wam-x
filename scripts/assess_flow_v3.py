"""Frozen validation acceptance diagnostics. No training and no reserved test reads."""
import json,gzip,time,hashlib
from pathlib import Path
import numpy as np
import torch
from lightning_task import ROOT,configure_runtime,OccData
from models_v1 import FlowPlanner
from audit_logged_collision import overlap

OUT=ROOT/'artifacts/diagnostics/flow-v3-assessment'
OLD=ROOT/'artifacts/diagnostics/20261115-contract'


def collision_proxy(names,target,predictions):
    chosen={k:p[:,:,0] for k,p in predictions.items()}
    chosen['logged_expert']=target[None]
    hits={k:np.zeros(p.shape[:2],bool) for k,p in chosen.items()};cached={};box_count=0
    for n,name in enumerate(names):
        route=Path(name).parts[-2];frame=int(Path(name).stem)
        if route not in cached:
            clip=ROOT/'data/public/bench2drive/clips'/route/'anno'
            cached={route:{int(p.name.split('.')[0]):json.load(gzip.open(p,'rt')) for p in clip.glob('*.gz')}}
        ann=cached[route];ego=next(b for b in ann[frame]['bounding_boxes'] if b['class']=='ego_vehicle')
        w=np.asarray(ego['world2ego']);yaw0=np.deg2rad(ego['rotation'][2])
        for ti in range(25):
            boxes=[]
            for b in ann[frame+2*(ti+1)]['bounding_boxes']:
                if b['class'] not in ['vehicle','walker'] or 'extent' not in b or 'center' not in b:continue
                c=(w@np.r_[b['center'],1])[:3];c[1]*=-1
                if abs(c[2]-ego['bbx_loc'][2])>b['extent'][2]+ego['extent'][2]+.3:continue
                boxes.append((c[:2],yaw0-np.deg2rad(b['rotation'][2]),np.asarray(b['extent'][:2])))
            box_count+=len(boxes)
            for key,p in chosen.items():
                for seed in range(len(p)):
                    if hits[key][seed,n]:continue
                    q=p[seed,n,ti];yaw=np.arctan2(q[2],q[3]);c=q[:2]*[50,10]
                    off=np.asarray(ego['bbx_loc'][:2])*[1,-1];co,si=np.cos(yaw),np.sin(yaw)
                    c=c+np.array([co*off[0]-si*off[1],si*off[0]+co*off[1]])
                    if any(overlap(c,yaw,ego['extent'],oc,oa,oe) for oc,oa,oe in boxes):hits[key][seed,n]=True
    return hits,box_count


def metrics(p,state,target,flags,routes,families):
    xy=p[...,:2]*[50,10];truth=target[...,:2]*[50,10]
    error=np.linalg.norm(xy-truth[None,:,None],axis=-1)
    ade=error.mean(-1);fde=error[...,-1]
    pos=np.concatenate([np.zeros_like(xy[:,:,:,:1]),xy],3);vel=np.diff(pos,axis=3)/.2
    initial=np.zeros_like(vel[:,:,:,:1]);initial[...,0]=state[None,:,None,0,None]*30
    acc=np.diff(np.concatenate([initial,vel],3),axis=3)/.2;jerk=np.diff(acc,axis=3)/.2
    head=p[...,4]*30
    head_acc=np.diff(np.concatenate([np.broadcast_to(state[None,:,None,0,None]*30,head[:,:,:,:1].shape),head],3),axis=3)/.2
    values={'first_ADE_m':ade[:,:,0],'first_FDE_m':fde[:,:,0],
            'meanADE_m':ade.mean(2),'meanFDE_m':fde.mean(2),'minADE_m':ade.min(2),'minFDE_m':fde.min(2),
            'first_abs_speed_acceleration_mps2':np.abs(head_acc[:,:,0]).mean(-1),
            'first_vector_acceleration_mps2':np.linalg.norm(acc[:,:,0],axis=-1).mean(-1),
            'first_vector_jerk_mps3':np.linalg.norm(jerk[:,:,0],axis=-1).mean(-1),
            'first_any_vector_jerk_over10':(np.linalg.norm(jerk[:,:,0],axis=-1)>10).any(-1),
            'first_logged_bbox_overlap_window_rate':flags}
    def group(mask):return {'windows':int(mask.sum()),'routes':len(set(routes[mask])),
                           **{k:float(v[:,mask].mean()) if mask.any() else None for k,v in values.items()}}
    inside=(np.abs(truth[...,0])<=25.6)&(np.abs(truth[...,1])<=25.6)
    horizons={}
    for second in range(1,6):
        i=second*5-1
        row={'windows':len(state),'first_error_m':float(error[:,:,0,i].mean()),'all_candidate_error_m':float(error[:,:,:,i].mean())}
        for tag,mask in [('inside',inside[:,i]),('outside',~inside[:,i])]:
            row[tag]={'windows':int(mask.sum()),'first_error_m':float(error[:,mask,0,i].mean()) if mask.any() else None}
        horizons[str(second)]=row
    speed=state[:,0]*108
    return {'seeds':len(p),'candidates':p.shape[2],'overall':group(np.ones(len(state),bool)),
            'by_family':{f:group(families==f) for f in np.unique(families)},
            'by_route':{r:group(routes==r) for r in np.unique(routes)},
            'by_seed_meanADE':[float(a.mean()) for a in ade],
            'first_ADE_window_seed_p50_p90_p95_m':np.percentile(ade[:,:,0],[50,90,95]).tolist(),
            'horizons':horizons,'by_speed':{f'{lo}-{hi}':group((speed>=lo)&(speed<hi)) for lo,hi in [(-1000,0),(0,10),(10,30),(30,50),(50,80),(80,121)]}}


def main():
    configure_runtime();started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'artifacts/lightning/flow-v3/best.ckpt';before=hashlib.sha256(path.read_bytes()).hexdigest()
    ck=torch.load(path,map_location='cpu',weights_only=False);data=OccData('flow',ck['task_state']['settings'])
    assert data.sha==ck['task_state']['manifest_sha']
    names=np.asarray(data.manifest['splits']['validation']);state=[];target=[]
    for name in names:
        with np.load(ROOT/'data/occ_v3'/name) as a:state.append(a['state']);target.append(a['trajectory'])
    state=np.stack(state);target=np.stack(target)
    with np.load(OLD/'evaluation-labels.npz') as a:
        assert np.array_equal(names,a['names']) and np.array_equal(state,a['state']) and np.array_equal(target,a['target'])
    model=FlowPlanner().cuda().eval();model.load_state_dict({k.removeprefix('model.'):v for k,v in ck['state_dict'].items()},strict=True)
    if ck['task_state']['settings']['channels_last']:
        torch.nn.utils.convert_conv2d_weight_memory_format(model,torch.channels_last);model.encoder.channels_last=True
    outputs=[];seeds=[1908,2911,3911]
    for seed in seeds:
        batches=[]
        for bi,b in enumerate(data.loader('validation')):
            b={k:v.cuda(non_blocking=True) for k,v in b.items()}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),torch.random.fork_rng(devices=[0]):
                torch.manual_seed(seed+bi);p=model.sample(b,k=8,steps=16)
            batches.append(p.cpu().numpy())
        outputs.append(np.concatenate(batches));print('INFER',seed,flush=True)
    predictions={'flow_v3':np.stack(outputs)}
    with np.load(OLD/'ridge_state_nav.npz') as a:predictions['ridge_state_nav']=a['prediction'].copy()
    constant=np.zeros_like(target);constant[:,:,0]=state[:,0,None]*30*np.arange(1,26)[None,:]*.2/50
    constant[:,:,3]=1;constant[:,:,4]=state[:,0,None]
    predictions['constant_velocity']=constant[None,:,None]
    for key,p in predictions.items():np.savez_compressed(OUT/(key+'.npz'),prediction=p)
    del model,ck;torch.cuda.empty_cache()
    hits,box_count=collision_proxy(names,target,predictions)
    np.savez_compressed(OUT/'overlap-flags.npz',**hits)
    routes=np.asarray([Path(n).parts[-2] for n in names]);lookup={r['route']:r['family'] for r in data.manifest['routes']}
    families=np.asarray([lookup[r] for r in routes])
    result={'checkpoint_sha256':before,'manifest_sha256':data.sha,'seeds':seeds,'test_opened':False,
            'state_nav_baseline_reused_after_exact_state_label_and_order_equality_check':True,
            'models':{k:metrics(p,state,target,hits[k],routes,families) for k,p in predictions.items()},
            'expert_proxy':{'hits':int(hits['logged_expert'].sum()),'windows':len(names),'actor_time_boxes':box_count},
            'limitations':['Development validation used for checkpoint selection, not independent test',
                           'Fixed candidate0 is independent of ground truth but is not a risk selector',
                           'minADE/minFDE are independent oracle selections, not deployment performance',
                           '3 sampling seeds are not 3 training seeds; 1124 overlapping windows belong to 9 routes',
                           'State/nav ridge has different model capacity; this is not a matched architecture ablation',
                           'BBox proxy uses logged responses, discrete0.2s and incomplete actors/static obstacles; not closed-loop collision rate',
                           'Position-differenced accel/jerk are unsmoothed diagnostics; threshold10 is not a calibrated vehicle limit',
                           'No legal-lane boundary metric or controller tracking evaluation',
                           'Coverage groups use future truth position in original XY rectangle; time/speed/scene confounded'],
            'seconds':time.perf_counter()-started}
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    (OUT/'assessment.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'overall':{k:v['overall'] for k,v in result['models'].items()},'expert_proxy':result['expert_proxy'],'seconds':result['seconds']}),flush=True)


if __name__=='__main__':main()
