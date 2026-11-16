"""Read-only train/validation contract probes. Never opens reserved test samples."""
from pathlib import Path
import json,gzip,time,hashlib,tarfile
import numpy as np
from dataset_v1 import compact,warp
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/diagnostics/20261115-contract'
def anno(p):return json.load(gzip.open(p,'rt'))
def ego(a):return next(b for b in a['bounding_boxes'] if b['class']=='ego_vehicle')
def points(o,s,kind):
    mask=(o==1)&(np.isin(s,[14,15,16,17,18,19,21]) if kind=='dynamic' else np.isin(s,[1,2,6,8,9,10,24,25,27,28]))
    y,x,z=np.nonzero(mask);p=np.stack([-.0-25.6+(x+.5)*.4,-25.6+(y+.5)*.4,-5+(z+.5)*.4],1)
    return p[(np.abs(p[:,0])>3)|(np.abs(p[:,1])>1.5)]
def box_score(p,a):
    if not len(p):return None
    w=np.asarray(ego(a)['world2ego']);best=np.full(len(p),100.)
    for b in a['bounding_boxes']:
        if b['class']!='vehicle':continue
        center=(w@np.r_[b['center'],1])[:3]
        yaw=np.deg2rad(b['rotation'][2]-ego(a)['rotation'][2]);c,s=np.cos(yaw),np.sin(yaw)
        q=p-center;local=np.stack([q[:,0]*c+q[:,1]*s,-q[:,0]*s+q[:,1]*c,q[:,2]],1)
        dist=np.linalg.norm(np.maximum(np.abs(local)-np.asarray(b['extent'])-.3,0),axis=1)
        best=np.minimum(best,dist)
    return dict(points=len(p),fraction_within_expanded_bbox=float((best==0).mean()),mean_outside_distance_m=float(best.mean()))
def main():
    started=time.perf_counter();m=json.loads((ROOT/'data/occ_v1/manifest.json').read_text());rows=[];units=[];coverage=[];sourcecounts={};inventory=[]
    for route in [r for r in m['routes'] if r['split']=='validation']:
        clip=ROOT/'data/public/bench2drive/clips'/route['route'];aa={int(p.name.split('.')[0]):anno(p) for p in (clip/'anno').glob('*.gz')}
        ids=sorted(i for i in aa if 3<=i<=max(aa)-3 and (clip/'occ'/f'{i:05}.npz').exists())
        selected=sorted(set([ids[int(x)] for x in np.linspace(0,len(ids)-1,12)]))
        for i in selected:
            a=aa[i];pose=np.asarray(ego(a)['world2ego']);poses=[np.linalg.inv(np.asarray(ego(aa[j])['world2ego'])) for j in [i-1,i,i+1]]
            v=(poses[2][:3,3]-poses[0][:3,3])/.2
            yaw=np.arctan2(poses[2][1,0],poses[2][0,0])-np.arctan2(poses[0][1,0],poses[0][0,0]);yaw=np.arctan2(np.sin(yaw),np.cos(yaw))/.2
            units.append(dict(route=route['route'],frame=i,speed=a['speed'],pose_speed=float(np.linalg.norm(v)),gyro_z=a['angular_velocity'][2],pose_yaw_rate=float(yaw),imu_z=a['acceleration'][2]))
            with np.load(clip/'occ'/f'{i:05}.npz') as f:o,s=f['occ'],f['seg'];sourcecounts[str((route['route'],i))]={str(k):int(v) for k,v in zip(*np.unique(o,return_counts=True))};p=points(o,s,'dynamic');base=compact(o,s)
            for d in [-2,-1,0,1,2]:
                score=box_score(p,aa[i+d]);row=dict(route=route['route'],family=route['family'],frame=i,offset=d,ego_speed=a['speed'],dynamic=score)
                with np.load(clip/'occ'/f'{i+1:05}.npz') as f:other=compact(f['occ'],f['seg'])
                a0=np.asarray(ego(aa[i+d])['world2ego']);a1=np.asarray(ego(aa[i+1+d])['world2ego'])
                g0=warp(base,a0,np.linalg.inv(a0));g1=warp(other,a1,np.linalg.inv(a0));valid=(g0!=255)&(g1!=255)
                static=[1,2,3,6,7];b0=np.isin(g0,static)&valid;b1=np.isin(g1,static)&valid
                row['static_iou']=float((b0&b1).sum()/max(1,(b0|b1).sum()));rows.append(row)
        print('ALIGN',route['route'],len(selected),flush=True)
    for name in m['splits']['validation']:
        with np.load(ROOT/'data/occ_v1'/name) as f:
            h=f['history'];t=f['trajectory'];state=f['state'];known=h!=255
            coverage.append(dict(name=name,speed_kmh=float(state[0]*108),history_not255=[float(x.mean()) for x in known],history_union_not255=float(known.any(0).mean()),current_occupied=float(((h[-1]>0)&(h[-1]<255)).mean()),current_unfilled=float((h[-1]==0).mean()),current_forward_extension_not255=float(known[-1,:,:,64:].mean()),future_points_outside_original=int(((np.abs(t[:,0]*50)>25.6)|(np.abs(t[:,1]*10)>25.6)).sum()),future_points_forward_outside=int((t[:,0]*50>25.6).sum())))
    # Inspect archive member names and a few RGB/semantic frames from two validation routes only.
    for route in [r for r in m['routes'] if r['split']=='validation'][:2]:
        counts={};examples={};dest=OUT/'rgb-probe'/route['route'];dest.mkdir(parents=True,exist_ok=True)
        with tarfile.open(ROOT/'data/public/bench2drive/archives'/route['file'],'r|gz') as tar:
            for member in tar:
                if not member.isfile():continue
                parts=Path(member.name).parts;key='/'.join(parts[-3:-1]);counts[key]=counts.get(key,0)+1;examples.setdefault(key,member.name)
                if len(parts)>2 and parts[-2] in ['rgb_front','semantic_front'] and parts[-1].split('.')[0] in ['00030','00060','00090']:
                    (dest/(parts[-2]+'-'+parts[-1])).write_bytes(tar.extractfile(member).read())
        inventory.append(dict(route=route['route'],counts=counts,examples=examples))
    result=dict(offset_definition='OCC file i compared to annotation i+offset; no index edits',alignment=rows,units=units,coverage=coverage,source_occ_counts=sourcecounts,archives=inventory,seconds=time.perf_counter()-started)
    (OUT/'contract-probes.json').write_text(json.dumps(result,indent=2));print('DONE',result['seconds'])
if __name__=='__main__':main()
