"""Read-only v3 audit and frozen small-sample selection; never opens test caches."""
import argparse, collections, gzip, hashlib, json, re, time
from pathlib import Path
import numpy as np
from dataset_v1 import ROOT, compact, warp, digest
from diagnose_occ_frozen import groups
import screen_logged_behaviors as screen


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def merged_intervals(names):
    """Union entire [-.6,+5] windows, NOT independent maneuver ground truth."""
    by = collections.defaultdict(list)
    for name in names:
        p = Path(name); i = int(p.stem)
        by[p.parts[-2]].append((i-6, i+50))
    out = []
    for route, intervals in sorted(by.items()):
        current = None
        for lo, hi in sorted(intervals):
            if current is None or lo > current[1]:
                if current: out.append(dict(route=route, start=current[0], end=current[1]))
                current = [lo, hi]
            else: current[1] = max(current[1], hi)
        if current: out.append(dict(route=route, start=current[0], end=current[1]))
    return out


def select(names, masks, n=64):
    selected = []
    # Round-robin routes; spread picks inside each route. Labels only select a
    # diagnostic TRAIN subset, never become actor inputs or redefine split.
    for group, quota in [('stationary',24), ('deceleration',24), ('all',16)]:
        by = collections.defaultdict(list)
        for i in np.flatnonzero(masks[group]):
            if int(i) not in selected: by[Path(names[i]).parts[-2]].append(int(i))
        buckets = []
        for route in sorted(by):
            ids = by[route]
            order = np.argsort([hashlib.sha256(names[i].encode()).hexdigest() for i in ids])
            buckets.append([ids[j] for j in order])
        count = 0
        while buckets and count < quota:
            for b in buckets:
                if b and count < quota: selected.append(b.pop()); count += 1
            buckets = [b for b in buckets if b]
    assert len(selected)==n and len(set(selected))==n
    return [names[i] for i in selected]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='artifacts/joint-occ-v1/audit');a=p.parse_args()
    out=ROOT/a.out;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    manifest_path=ROOT/'data/occ_v3/manifest.json';m=json.loads(manifest_path.read_text())
    assert m['preprocess_version']==3 and m['splits']['test']==[]
    protected=[manifest_path, ROOT/'data/occ_v1/manifest.json']
    protected += list((ROOT/'scripts').glob('*.py')) + list((ROOT/'configs').glob('*.json'))
    protected += list((ROOT/'artifacts').rglob('*.ckpt'))
    hashes={str(x.relative_to(ROOT)):digest(x) for x in protected}
    write(out/'protected-before.json',hashes)
    summary={};selection={};raw=[];cache_checked=0
    for split in ['train','validation']:
        names=m['splits'][split];states=[];targets=[]
        expected={r['sample']:r['new_sha256'] for route in m['routes'] if route['split']==split
                  for r in json.loads((ROOT/'artifacts/diagnostics/alignment-v3/routes'/f"{route['route']}.json").read_text())['samples']}
        for name in names:
            path=ROOT/'data/occ_v3'/name
            assert digest(path)==expected[name];cache_checked+=1
            with np.load(path) as f:
                i=int(f['frame']);h=f['history_pose_ids'];future=f['future_pose_ids']
                assert np.array_equal(h,i+np.array([-6,-4,-2,0]))
                assert np.array_equal(future,i+np.array([10,20,30,40,50]))
                assert np.array_equal(f['history_occ_ids'],h-1) and np.array_equal(f['future_occ_ids'],future-1)
                states.append(f['state']);targets.append(f['trajectory'])
        state,target=np.stack(states),np.stack(targets); masks=groups(state,target)
        rows={}
        for group,mask in masks.items():
            ns=np.asarray(names)[mask].tolist();intervals=merged_intervals(ns)
            rows[group]={'windows':len(ns),'route_recordings':len({Path(n).parts[-2] for n in ns}),
                         'road_route_ids':len({re.search(r'(Town\d+_Route\d+)',n)[0] for n in ns}),
                         'nonoverlapping_support_components':len(intervals),'support_intervals':intervals}
        rows['current_speed_max_kmh']=float(state[:,0].max()*108)
        rows['current_speed_ge80_windows']=int((state[:,0]*108>=80).sum())
        summary[split]=rows;selection[split]=select(names,masks)
        # Independently rebuild history AND future and trajectory for 3 routes/split.
        for route in [r for r in m['routes'] if r['split']==split][:3]:
            name=route['samples'][len(route['samples'])//2];clip=ROOT/'data/public/bench2drive/clips'/route['route']
            with np.load(ROOT/'data/occ_v3'/name) as f:
                current=np.linalg.inv(f['world2ego']);i=int(f['frame'])
                for prefix in ['history','future']:
                    for k,j in enumerate(f[prefix+'_pose_ids']):
                        with gzip.open(clip/'anno'/f'{j:05d}.json.gz','rt') as stream:ann=json.load(stream)
                        pose=np.asarray(next(b for b in ann['bounding_boxes'] if b['class']=='ego_vehicle')['world2ego'])
                        with np.load(clip/'occ'/f'{j-1:05d}.npz') as src:rebuilt=warp(compact(src['occ'],src['seg']),pose,current)
                        assert np.array_equal(rebuilt,f[prefix][k])
                for k,j in enumerate(range(i+2,i+51,2)):
                    with gzip.open(clip/'anno'/f'{j:05d}.json.gz','rt') as stream:ann=json.load(stream)
                    future_pose=np.linalg.inv(np.asarray(next(b for b in ann['bounding_boxes'] if b['class']=='ego_vehicle')['world2ego']))
                    local=current[:3,:3].T@(future_pose[:3,3]-current[:3,3])
                    forward=current[:3,:3].T@future_pose[:3,0];yaw=-np.arctan2(forward[1],forward[0])
                    assert np.allclose(f['trajectory'][k],[local[0]/50,-local[1]/10,np.sin(yaw),np.cos(yaw),ann['speed']/30],atol=1e-5)
                assert (f['history'][-1,:,:,64:]==255).all()
            raw.append(name)
    screen.OUT=out/'behavior-rescan'
    events=[screen.scan_route(r) for r in m['routes']]
    event_summary={}
    for split in ['train','validation']:
        routes=[r for r in events if r['split']==split]
        kinds=sorted({e['kind'] for r in routes for e in r['events']})
        event_summary[split]={k:{'candidate_segments':sum(e['kind']==k for r in routes for e in r['events']),
             'route_recordings':sum(any(e['kind']==k for e in r['events']) for r in routes)} for k in kinds}
        event_summary[split]['speed_bins']=[{'range_kmh':r,'frames':sum(next(b['frames'] for b in row['speed_bins'] if b['range_kmh']==r) for row in routes),
            'route_recordings':sum(next(b['frames'] for b in row['speed_bins'] if b['range_kmh']==r)>0 for row in routes)} for r in [[0,20],[20,40],[40,60],[60,80],[80,100],[100,120]]]
    write(out/'behavior-rescan.json',events)
    write(out/'selection.json',{'manifest_sha':digest(manifest_path),'splits':selection,'test_read':False,'purpose':'64-window fit diagnostic and enriched validation probe; NOT full-split evaluation'})
    report={'counts':m['counts'],'manifest_sha':digest(manifest_path),'cache_hashes_verified':cache_checked,'raw_rebuild_windows':raw,
        'window_groups':summary,'raw_annotation_events':event_summary,'test_read':False,'seconds':time.perf_counter()-started,
        'event_caveat':'Support unions and heuristic segments are not independent certified events; recordings may repeat road route IDs.',
        'source_causality':'Online source uses historical queues, offline source also exists with centered fusion. Exact published generator invocation unavailable: source-level future fusion NOT certified absent.',
        'timestamp_caveat':'0.1 s inferred from source cadence; no native simulator timestamp per annotation'}
    write(out/'report.json',report)
    print(json.dumps({'audit':str(out),'hashes':cache_checked,'events':event_summary,'seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
