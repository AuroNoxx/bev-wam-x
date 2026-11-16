"""Read-only train/validation timing inventory; no test file access."""
import json,gzip,numpy as np
from pathlib import Path
root=Path(__file__).resolve().parents[1];m=json.loads((root/'data/occ_v3/manifest.json').read_text());rows=[];examples=[]
for row in m['routes']:
    if row['split'] not in ['train','validation']:continue
    clip=root/'data/public/bench2drive/clips'/row['route'];a={int(p.name.split('.')[0]):p for p in (clip/'anno').glob('*.json.gz')};o={int(p.stem):p for p in (clip/'occ').glob('*.npz')};ids=sorted(i for i in a if i-1 in o);runs=[]
    for i in ids:
        if not runs or i!=runs[-1][-1]+1:runs.append([i])
        else:runs[-1].append(i)
    cached={int(Path(x).stem) for x in row.get('samples',[])};eligible={str(s):0 for s in [.6,2,5,10,30]};raweligible=eligible.copy()
    for run in runs:
        for s in eligible:
            start=run[0]+round(float(s)*10);end=run[-1]-50
            raweligible[s]+=max(0,end-start+1);eligible[s]+=sum(start<=i<=end for i in cached)
    ex={}
    if ids:
        with gzip.open(a[ids[len(ids)//2]],'rt') as f:ann=json.load(f)
        boxes=ann.get('bounding_boxes',[]);ego=next((b for b in boxes if b.get('class')=='ego_vehicle'),{})
        ex={'annotation_keys':sorted(ann),'ego_keys':sorted(ego),'object_keys':sorted(next((b for b in boxes if b.get('class')!='ego_vehicle'),{})),'objects_count':len(boxes),'explicit_timestamp_keys':[k for k in ann if 'time' in k.lower()]}
        with np.load(o[ids[len(ids)//2]-1]) as f:ex['occ_fields']={k:{'shape':list(f[k].shape),'dtype':str(f[k].dtype)} for k in f.files}
        examples.append({'route':row['route'],**ex})
    rows.append({'route':row['route'],'split':row['split'],'anno_count':len(a),'occ_count':len(o),'paired_count':len(ids),'continuous_run_frames':[len(r) for r in runs],'longest_span_seconds_inferred':max((len(r)-1)*.1 for r in runs) if runs else 0,'cached_windows':len(cached),'eligible_cached_history_plus_5s_future':eligible,'eligible_raw_history_plus_5s_future':raweligible})
summary={}
for split in ['train','validation']:
    rr=[r for r in rows if r['split']==split];spans=[r['longest_span_seconds_inferred'] for r in rr]
    summary[split]={'recordings':len(rr),'longest_span_min_median_max_seconds':[min(spans),float(np.median(spans)),max(spans)],'recordings_span_ge30s':sum(s>=30 for s in spans),'recordings_span_ge35s':sum(s>=35 for s in spans),'cached_windows':sum(r['cached_windows'] for r in rr),'eligible_cached_history_plus_5s_future':{str(s):sum(r['eligible_cached_history_plus_5s_future'][str(s)] for r in rr) for s in [.6,2,5,10,30]},'eligible_raw_history_plus_5s_future':{str(s):sum(r['eligible_raw_history_plus_5s_future'][str(s)] for r in rr) for s in [.6,2,5,10,30]}}
r={'cadence_assumption_seconds':.1,'timestamp_provenance':'manifest nominal cadence, no assertion measured true timestamp','test_files_read':False,'summary':summary,'routes':rows,'sampled_annotation_schema':examples}
p=root/'artifacts/temporal-history-inventory-20261115.json';assert not p.exists();p.write_text(json.dumps(r,indent=2));print(json.dumps(summary));print(json.dumps(examples[:1]))
