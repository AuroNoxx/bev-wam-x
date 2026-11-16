"""Source-pairing, leakage bounds, loader rejection and behavior negative controls."""
import gzip, json
from pathlib import Path
import numpy as np
from dataset_v1 import ROOT
from lightning_task import OccSamples
from build_aligned_cache import frame_ids, atomic_json
from screen_logged_behaviors import scan_route


def behavior_controls():
    # Continuous passing of the SAME moving actor, including departure and return.
    # A separate road-boundary fixture must not count as a lane change.
    results={}
    for kind in ['overtake', 'road_boundary', 'queue']:
        route={'route':'fixture_'+kind, 'split':'synthetic', 'family':'synthetic'}
        folder=ROOT/'artifacts/diagnostics/alignment-v3/fixtures'/kind
        folder.mkdir(parents=True,exist_ok=True)
        for i in range(121):
            t=i*.1
            road=2 if kind=='road_boundary' and i>=20 else 1
            lane=-2 if (kind=='overtake' and 20<=i<90) or (kind=='road_boundary' and i>=20) else -1
            v=1 if kind=='queue' else 10
            x=v*t
            transform=np.eye(4);transform[0,3]=-x
            ego={'class':'ego_vehicle','world2ego':transform.tolist(),'road_id':road,'section_id':0,'lane_id':lane,'rotation':[0,0,0]}
            other_v=1 if kind=='queue' else 6
            actors=[{'class':'vehicle','id':'target','center':[20+other_v*t,0,0],
                     'rotation':[0,0,0],'speed':other_v,'road_id':road,'section_id':0,'lane_id':-1}]
            if kind=='queue':
                actors.append({**actors[0],'id':'target2','center':[30+other_v*t,0,0]})
            a={'speed':v,'bounding_boxes':[ego]+actors,'command_near':4,'command_far':4}
            with gzip.open(folder/f'{i:05}.json.gz','wt') as stream:json.dump(a,stream)
        result=scan_route(route,folder)
        kinds=[e['kind'] for e in result['events']]
        if kind=='overtake':assert 'passing_target' in kinds and 'completed_overtake' in kinds
        if kind=='road_boundary':assert 'lane_change' not in kinds
        if kind=='queue':assert 'slow_queue' in kinds and 'following' in kinds and 'completed_overtake' not in kinds
        results[kind]=kinds
    return results


def main():
    manifest=json.loads((ROOT/'data/occ_v3/smoke.json').read_text())
    checked=0
    for r in manifest['routes']:
        name=r['samples'][0]
        with np.load(ROOT/'data/occ_v3'/name) as a:
            i=int(a['frame'])
            assert np.array_equal(a['history_occ_ids'],np.array([-6,-4,-2,0])+i-1)
            assert np.array_equal(a['future_occ_ids'],np.array([10,20,30,40,50])+i-1)
            # The current source is native near-field only: forward padding stays unknown.
            assert (a['history'][-1,:,:,64:]==255).all()
            if r['split']=='validation':
                side=ROOT/'artifacts/diagnostics/frame-alignment/corrected-smoke'/r['route']/Path(name).name
                with np.load(side) as reference:
                    for key in ['history','future','state','trajectory','world2ego']:
                        assert np.array_equal(a[key],reference[key]),(name,key)
                checked+=1
    h,f=frame_ids(6)
    assert h[0]-1==-1 and h[0]==0 # Negative OCC file ID is valid; pose stays anno 0.
    loader=OccSamples(manifest,'train','world')
    batch=loader[0]
    assert batch['history'].shape==(4,20,64,160)
    # A v2 file accidentally listed in a v3 manifest must fail, rather than train silently.
    loader.files[0]=ROOT/'data/occ_v1'/manifest['splits']['train'][0]
    try:loader[0]
    except RuntimeError as e:assert 'contract mismatch' in str(e)
    else:raise AssertionError('Mixed-version sample accepted')
    result={'pass':True,'smoke_routes':len(manifest['routes']),
            'independent_sidecar_array_matches':checked,'unknown_padding_preserved':True,
            'negative_occ_boundary_supported':True,'mixed_version_rejected':True,
            'behavior_controls':behavior_controls()}
    atomic_json(ROOT/'artifacts/diagnostics/alignment-v3/unit-result.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
