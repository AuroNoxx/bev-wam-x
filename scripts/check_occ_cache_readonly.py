"""Recheck current v3 hashes and raw-to-cache transforms; no cache writes."""
import gzip
import json
import time
import numpy as np
from dataset_v1 import ROOT, compact, warp, digest


def main():
    started=time.perf_counter()
    out=ROOT/'artifacts/diagnostics/occ-frozen-audit-20261115'
    assert out.exists()
    manifest=json.loads((ROOT/'data/occ_v3/manifest.json').read_text())
    assert manifest['splits']['test']==[]
    checked=0; raw=[]
    for row in manifest['routes']:
        records=json.loads((ROOT/'artifacts/diagnostics/alignment-v3/routes'/(row['route']+'.json')).read_text())
        for r in records['samples']:
            assert digest(ROOT/'data/occ_v3'/r['sample'])==r['new_sha256']
            checked+=1
        if row['split']!='validation':continue
        names=row['samples']
        for name in [names[0],names[len(names)//2],names[-1]]:
            with np.load(ROOT/'data/occ_v3'/name) as f:
                i=int(f['frame']); w=f['world2ego']; inverse=np.linalg.inv(w)
                current_ids=np.array([-6,-4,-2,0])+i
                assert np.array_equal(f['history_pose_ids'],current_ids)
                assert np.array_equal(f['history_occ_ids'],current_ids-1)
                assert (f['history'][-1,:,:,64:]==255).all()
                clip=ROOT/'data/public/bench2drive/clips'/row['route']
                for hi,j in enumerate(current_ids):
                    with gzip.open(clip/'anno'/f'{j:05}.json.gz','rt') as stream: a=json.load(stream)
                    pose=np.array(next(b for b in a['bounding_boxes'] if b['class']=='ego_vehicle')['world2ego'])
                    with np.load(clip/'occ'/f'{j-1:05}.npz') as occ: h=warp(compact(occ['occ'],occ['seg']),pose,inverse)
                    assert np.array_equal(h,f['history'][hi])
                # Independent rotation/translation formulation for future XY and yaw.
                positions=[]; headings=[]; speeds=[]
                for j in range(i+2,i+51,2):
                    with gzip.open(clip/'anno'/f'{j:05}.json.gz','rt') as stream: a=json.load(stream)
                    future=np.linalg.inv(np.array(next(b for b in a['bounding_boxes'] if b['class']=='ego_vehicle')['world2ego']))
                    delta=future[:3,3]-inverse[:3,3]
                    local=inverse[:3,:3].T@delta
                    positions.append(local[:2]*[1,-1])
                    forward=inverse[:3,:3].T@future[:3,0]
                    headings.append(-np.arctan2(forward[1],forward[0]));speeds.append(a['speed'])
                xy=np.array(positions); yaw=np.array(headings)
                assert np.allclose(xy,f['trajectory'][:,:2]*[50,10],atol=1e-5)
                assert np.allclose(np.sin(yaw),f['trajectory'][:,2],atol=1e-6)
                assert np.allclose(np.cos(yaw),f['trajectory'][:,3],atol=1e-6)
                assert np.allclose(speeds,f['trajectory'][:,4]*30,atol=1e-5)
                raw.append({'window':name,'raw_history_exact':True,'trajectory_geometry_pass':True,
                            'max_xy_error_m':float(np.abs(xy-f['trajectory'][:,:2]*[50,10]).max())})
    source=np.zeros((128,128,20),np.uint8); seg=np.zeros_like(source)
    # Source storage is [y_right,x_forward,z], verified by raw LiDAR indexing.
    # Compact halves XY; cached FLU flips y: row=63-source_y//2.
    source[80,90,10]=1;seg[80,90,10]=14
    source[20,30,10]=2
    cached=warp(compact(source,seg),np.eye(4),np.eye(4))
    assert cached[10,23,45]==4 and cached[10,53,15]==255
    assert cached[10,32,32]==0 and (cached[:,:,64:]==255).all()
    result={'pass':True,'cache_hashes_verified':checked,'raw_history_and_label_windows':len(raw),
            'validation_routes':9,'raw_checks':raw,'synthetic_axis_unknown_check':True,
            'test_opened':False,'seconds':time.perf_counter()-started}
    (out/'cache-recheck.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='raw_checks'}),flush=True)


if __name__=='__main__':main()
