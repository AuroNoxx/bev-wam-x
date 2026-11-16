import json,re,collections,hashlib,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[1];out=R/'artifacts/route-regression-flow1024';out.mkdir(exist_ok=True)
items=[]
for p in sorted((R/'data/occ_v3/cache/train').rglob('*.npz')):
 with np.load(p) as d:
  items.append(dict(file=str(p.relative_to(R/'data/occ_v3')),route=re.search(r'Town\d+_Route\d+',p.parent.name).group(),scene=p.parent.name.split('_Town')[0],start=int(min(d['history_pose_ids'])),end=max(int(max(d['future_pose_ids'])),int(d['frame'])+50),v0=float(d['state'][0])*30,v1=float(d['trajectory'][-1,4])*30))
assert len(items)==5296
rng=np.random.default_rng(2050);split={};routes={}
for scene in sorted({x['scene'] for x in items}):
 gs=sorted({x['route'] for x in items if x['scene']==scene});rng.shuffle(gs);routes[scene]={'held':[gs[0]],'tuning':[gs[1]],'train':gs[2:]}
 for part,groups in routes[scene].items():
  for g in groups:assert g not in split;split[g]=part
pools={k:[x for x in items if split[x['route']]==k] for k in ['train','tuning','held']}
def subset(pool):
 selected=[]
 for g in sorted({x['route'] for x in pool}):
  last=-1
  for x in sorted([x for x in pool if x['route']==g],key=lambda x:(x['start'],x['file'])):
   if x['start']>last:selected.append(x);last=x['end']
 return selected
held=subset(pools['held']);tuning=subset(pools['tuning']);fitdiag=subset(pools['train'])[::4];streams={};reports={}
for seed in [2050,2051]:
 r=np.random.default_rng(seed);per={s:[x['file'] for x in pools['train'] if x['scene']==s] for s in routes};seq=[]
 # Equal scene allocation, shuffled cycles; all windows per scene covered before repeat.
 for s,names in per.items():
  a=[]
  while len(a)<2048:a.extend(r.permutation(names).tolist())
  per[s]=a[:2048]
 for j in range(2048):
  z=[per[s][j] for s in sorted(per)];r.shuffle(z);seq.extend(z)
 streams[str(seed)]=seq;reports[str(seed)]={'samples':len(seq),'unique':len(set(seq))}
d={'routes':routes,'counts':{k:len(v) for k,v in pools.items()},'pool_scenarios':{k:dict(collections.Counter(x['scene'] for x in v)) for k,v in pools.items()},'held':held,'tuning':tuning,'fit_diagnostic':fitdiag,'streams':streams,'stream_reports':reports,'limits':'All source5296 train examined; new fresh models and route-ID disjoint train/tuning/held incl repeated weather. Previously inspected development data, not untouched final holdout. Physical roads may overlap across route IDs; source OCC causality unverified, offline only. Scenario names not event verification. No officialval/test. Equal-scene sampling has repeats; not independent samples.'}
(out/'design.json').write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k not in ['streams','held','tuning','fit_diagnostic']},indent=2));print('eval',len(held),len(tuning),len(fitdiag))
