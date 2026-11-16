"""Physical-unit, stratified frozen validation summaries with explicit denominators."""
import json,gzip,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/diagnostics/20261115-contract'
def average(x):return float(np.mean(x))
def main():
    t0=time.perf_counter();m=json.loads((ROOT/'data/occ_v1/manifest.json').read_text());lookup={r['route']:r['family'] for r in m['routes']}
    with np.load(OUT/'evaluation-labels.npz') as f:s=f['state'];target=f['target'];names=f['names']
    routes=np.asarray([Path(n).parts[-2] for n in names]);family=np.asarray([lookup[r] for r in routes]);speed=s[:,0]*108
    maneuver=np.asarray(['near_command_'+str(np.argmax(x[8:14])+1) for x in s]);xygt=target[:,:,:2]*[50,10];inside=(np.abs(xygt[:,:,0])<=25.6)&(np.abs(xygt[:,:,1])<=25.6)
    models={}
    for key in ['ridge_state_nav','ridge_state_nav_occ','flow_v1','flow_v2','flow_v2_unknown_occ']:
        with np.load(OUT/(key+'.npz')) as f:p=f['prediction']
        xy=p[...,:2]*[50,10];err=np.linalg.norm(xy-xygt[None,:,None],axis=-1);ade=err.mean(-1);fde=err[...,-1]
        spread=np.sqrt(np.mean(np.sum((xy-xy.mean(2,keepdims=True))**2,axis=-1),axis=(2,3)))
        # Include t=0 boundary; positions, scalar speed head and vector finite differences are separate.
        pos=np.concatenate([np.zeros_like(xy[:,:,:,:1]),xy],3);vel=np.diff(pos,axis=3)/.2
        initial=np.zeros_like(vel[:,:,:,:1]);initial[...,0]=s[None,:,None,0,None]*30
        acc=np.diff(np.concatenate([initial,vel],3),axis=3)/.2;jerk=np.diff(acc,axis=3)/.2
        head=p[...,4]*30;headacc=np.diff(np.concatenate([np.broadcast_to(s[None,:,None,0,None]*30,head[:,:,:,:1].shape),head],3),axis=3)/.2
        metrics={'first_ADE_m':ade[:,:,0],'first_FDE_m':fde[:,:,0],'all_candidate_ADE_m':ade.mean(2),'minADE_m':ade.min(2),'minFDE_m':fde.min(2),'spread_m':spread,
          'first_speed_head_mps':head[:,:,0].mean(-1),'first_position_speed_mps':np.linalg.norm(vel[:,:,0],axis=-1).mean(-1),
          'first_abs_head_accel_mps2':np.abs(headacc[:,:,0]).mean(-1),'first_vector_accel_mps2':np.linalg.norm(acc[:,:,0],axis=-1).mean(-1),'first_vector_jerk_mps3':np.linalg.norm(jerk[:,:,0],axis=-1).mean(-1),
          'first_speed_error_mps':np.abs(head[:,:,0]-target[None,:,:,4]*30).mean(-1),
          'first_any_abs_head_accel_over6':(np.abs(headacc[:,:,0])>6).any(-1),'first_any_vector_jerk_over10':(np.linalg.norm(jerk[:,:,0],axis=-1)>10).any(-1)}
        flags_path=OUT/'logged-collision-flags.npz'
        if flags_path.exists():
            with np.load(flags_path) as f:metrics['first_logged_bbox_overlap_fraction']=f[key].copy()
        def summary(mask):return {'windows':int(mask.sum()),'routes':int(len(np.unique(routes[mask]))),**{k:average(v[:,mask]) if mask.any() else None for k,v in metrics.items()}}
        def point_summary(mask):
            return {'window_time_points':int(mask.sum()),'windows':int(mask.any(1).sum()),'routes':int(len(np.unique(routes[mask.any(1)]))),
              'first_point_error_m':average(err[:,:,0][:,mask]) if mask.any() else None,'all_candidate_point_error_m':average(err.transpose(0,2,1,3)[:,:,mask]) if mask.any() else None}
        groups={}
        for group,labels in [('family',family),('route',routes),('near_command',maneuver)]:groups[group]={v:summary(labels==v) for v in np.unique(labels)}
        groups['speed_kmh']={f'{lo}-{hi}':summary((speed>=lo)&(speed<hi)) for lo,hi in [(-1000,0),(0,10),(10,30),(30,50),(50,70),(70,90),(90,121),(121,1000)]}
        groups['point_coverage']={'inside_original_xy':point_summary(inside),'outside_original_xy':point_summary(~inside)}
        groups['horizon']={str(round((i+1)*.2,1)):{'windows':len(s),'first_error_m':average(err[:,:,0,i]),'all_candidate_error_m':average(err[:,:,:,i]),'inside':point_summary(inside & (np.arange(25)[None,:]==i)),'outside':point_summary((~inside)&(np.arange(25)[None,:]==i))} for i in range(25)}
        groups['speed_family_cross']={f'{lo}-{hi}/{f}':summary((speed>=lo)&(speed<hi)&(family==f)) for lo,hi in [(-1000,0),(0,10),(10,30),(30,50),(50,70),(70,90),(90,121),(121,1000)] for f in np.unique(family)}
        models[key]={'seeds':p.shape[0],'candidates':p.shape[2],'overall':summary(np.ones(len(s),bool)),'groups':groups}
    result={'models':models,'point_definition':'coverage is ground-truth future XY within current original +/-25.6m rectangle; not sensor visibility or validity','selection':'first index chosen before seeing ground truth, no risk selector; minADE and minFDE independently optimized oracle indices','thresholds':'6m/s2 scalar acceleration,10m/s3 vector jerk are diagnostic thresholds, not calibrated vehicle limits','lane_boundary_violation':None,'lane_reason':'No legal-lane polygons or trajectory-aligned map boundary in current cache; road support cannot certify lane compliance','seconds':time.perf_counter()-t0}
    (OUT/'stratified-evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v['overall'] for k,v in models.items()},indent=2))
if __name__=='__main__':main()
