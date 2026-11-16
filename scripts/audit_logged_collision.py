"""2D footprint overlap against logged actors at 0.2s, not closed-loop collisions."""
import json,gzip,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/diagnostics/20261115-contract'
def overlap(center,angle,extent,other_center,other_angle,other_extent):
    u=np.array([np.cos(angle),np.sin(angle)]);v=np.array([-u[1],u[0]])
    ou=np.array([np.cos(other_angle),np.sin(other_angle)]);ov=np.array([-ou[1],ou[0]]);delta=other_center-center
    for axis in [u,v,ou,ov]:
        if abs(delta@axis)>extent[0]*abs(u@axis)+extent[1]*abs(v@axis)+other_extent[0]*abs(ou@axis)+other_extent[1]*abs(ov@axis):return False
    return True
def main():
    started=time.perf_counter()
    with np.load(OUT/'evaluation-labels.npz') as f:names=f['names'];gt=f['target']
    pred={}
    for key in ['ridge_state_nav','ridge_state_nav_occ','flow_v1','flow_v2','flow_v2_unknown_occ']:
        with np.load(OUT/(key+'.npz')) as f:pred[key]=f['prediction'][:,:,0] # deployable first candidate, each seed
    pred['logged_expert']=gt[None]
    hits={k:np.zeros(p.shape[:2],bool) for k,p in pred.items()};counts=[];cached={};extentrows=[]
    for n,name in enumerate(names):
        route=Path(name).parts[-2];frame=int(Path(name).stem)
        if route not in cached:
            clip=ROOT/'data/public/bench2drive/clips'/route/'anno';cached={route:{int(p.name.split('.')[0]):json.load(gzip.open(p,'rt')) for p in clip.glob('*.gz')}}
        ann=cached[route];ego=next(b for b in ann[frame]['bounding_boxes'] if b['class']=='ego_vehicle');w=np.asarray(ego['world2ego']);yaw0=np.deg2rad(ego['rotation'][2]);extentrows.append(ego['extent'])
        total=0
        for ti in range(25):
            a=ann[frame+2*(ti+1)];boxes=[]
            for b in a['bounding_boxes']:
                if b['class'] not in ['vehicle','walker'] or 'extent' not in b or 'center' not in b:continue
                c=(w@np.r_[b['center'],1])[:3];c[1]*=-1
                if abs(c[2]-ego['bbx_loc'][2])>b['extent'][2]+ego['extent'][2]+.3:continue
                boxes.append((c[:2],yaw0-np.deg2rad(b['rotation'][2]),np.asarray(b['extent'][:2])))
            total+=len(boxes)
            for key,p in pred.items():
                for seed in range(len(p)):
                    if hits[key][seed,n]:continue
                    q=p[seed,n,ti];yaw=np.arctan2(q[2],q[3]);c=q[:2]*[50,10]
                    # Pose origin differs from bbox center; rotate measured local bbox offset.
                    off=np.asarray(ego['bbx_loc'][:2])*[1,-1];co,si=np.cos(yaw),np.sin(yaw);c=c+np.array([co*off[0]-si*off[1],si*off[0]+co*off[1]])
                    for oc,oa,oe in boxes:
                        if overlap(c,yaw,ego['extent'],oc,oa,oe):hits[key][seed,n]=True;break
        counts.append(total)
    result={'definition':'Any 2D oriented bbox overlap against logged vehicle/walker future at 25 discrete times; vertical gate; own actor excluded; first candidate only','limitations':['Not swept continuous collision; may miss inter-step contact','Other actors follow recorded expert response, not candidate counterfactual response','Annotations may omit actors; no static obstacle collision completeness','No lane boundary metric without legal lane map; do not interpret missing metric as zero'],'windows':len(names),'actor_time_boxes':sum(counts),'models':{k:{'overlap_window_fraction':float(v.mean()),'by_seed_counts':v.sum(1).tolist(),'windows_per_seed':len(names)} for k,v in hits.items()},'ego_extent_min':np.min(extentrows,0).tolist(),'ego_extent_max':np.max(extentrows,0).tolist(),'seconds':time.perf_counter()-started}
    (OUT/'logged-collision-proxy.json').write_text(json.dumps(result,indent=2));np.savez_compressed(OUT/'logged-collision-flags.npz',**hits);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
