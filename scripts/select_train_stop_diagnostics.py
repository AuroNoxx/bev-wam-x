import json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'artifacts/implementation-plan-v2';sel=json.loads((ROOT/'artifacts/structural-prototypes-light256/selection.json').read_text());excluded=set(sel['train']+sel['validation']);buckets={'initial_stopped_and_end_stopped':[],'moving_to_stopped':[],'decelerating_not_stopped':[]}
for p in sorted((ROOT/'data/occ_v3/cache/train').rglob('*.npz')):
 name=str(p.relative_to(ROOT/'data/occ_v3'))
 if name in excluded:continue
 with np.load(p,allow_pickle=False) as f:
  v0=float(f['state'][0])*30.;v1=float(f['trajectory'][-1,4])*30.
 kind='initial_stopped_and_end_stopped' if abs(v0)<.5 and abs(v1)<.5 else 'moving_to_stopped' if v0>2 and abs(v1)<.5 else 'decelerating_not_stopped' if v0-v1>3 and v1>.5 else None
 if kind:buckets[kind].append({'file':name,'initial_speed_mps':v0,'label_final_speed_mps':v1})
chosen=[]
for kind,items in buckets.items():
 routes={}
 for r in items:routes.setdefault(Path(r['file']).parent.name,[]).append(r)
 candidates=[r[len(r)//2] for _,r in sorted(routes.items())]
 chosen.extend([{**r,'stratum':kind} for r in candidates[:8]])
report={'counts':{k:len(v) for k,v in buckets.items()},'selection':chosen,'excludes_84_train_and_42_dev':True,'limits':'Unseen windows from training split, not independent route-heldout or test. Label stop is a historical outcome, not proof stopping is right. Selection frozen before new evaluation; no model predictions used.'}
(folder/'stop_deceleration_train_only_selection.json').write_text(json.dumps(report,indent=2));print(json.dumps({'counts':report['counts'],'selected':len(chosen)}))
