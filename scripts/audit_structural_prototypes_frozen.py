"""CPU-only frozen label-fit/progress stratification; no training/test/weight mutation."""
import json,sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'artifacts/structural-prototypes-light256'
sel=json.loads((base/'selection.json').read_text())
# Use the same already selected internal train-route diagnostic windows only.
labels=[]
for name in sel['validation']:
 p=ROOT/'data/occ_v3'/name
 with np.load(p,allow_pickle=False) as d:
  if not labels:print('KEYS',d.files)
  labels.append({k:d[k] for k in d.files if k in ['trajectory','state']})
print('FIRST',[(k,v.shape) for k,v in labels[0].items()])
rows=[]
for folder in sorted(base.glob('*-*')):
 if not (folder/'report.json').exists():continue
 r=json.loads((folder/'report.json').read_text());rr=[]
 for x in r['eval']['records']:
  item=labels[x['window']];truth=item['trajectory']*np.array([50,10,1,1,30]);p=np.array(x['points']);v0=float(item['state'][0])*30
  label_stop=abs(truth[-1,4])<.5;pred_stop=p[-1,4]<.5
  rr.append({'v0':v0,'label_stop':label_stop,'pred_stop':pred_stop,'position_fit_m':float(np.linalg.norm(p[:,:2]-truth[:,:2],axis=1).mean()),'end_gap_m':float(np.linalg.norm(p[-1,:2]-truth[-1,:2])),'forward_m':float(p[-1,0]),'label_forward_m':float(truth[-1,0]),'speed_mae_mps':float(abs(p[:,4]-abs(truth[:,4])).mean()),'max_accel_mps2':x['max_accel_mps2'],'max_jerk_mps3':x['max_jerk_mps3']})
 groups={}
 for name,subset in [('all',rr),('initial_low_speed',[v for v in rr if v['v0']<.5]),('label_end_stopped',[v for v in rr if v['label_stop']]),('label_end_moving',[v for v in rr if not v['label_stop']])]:
  groups[name]={'candidate_count':len(subset),'window_count':len(subset)//4,**{k:float(np.mean([v[k] for v in subset])) if subset else None for k in ['position_fit_m','end_gap_m','forward_m','label_forward_m','speed_mae_mps','max_accel_mps2','max_jerk_mps3','pred_stop']}}
 rows.append({'seed':r['seed'],'arm':r['arm'],'groups':groups})
output={'rows':rows,'limits':'Labels describe recorded outcome, not unique correct policy. Parking/advancing labels are diagnostic only. Candidate counts correlated within four samples per window. No free/safety/collision/route progress truth; speed vector uses body heading approximation, t0 lateral speed assumed zero.'}
(base/'frozen_stratified_diagnostics.json').write_text(json.dumps(output,indent=2))
for r in rows:print(r['seed'],r['arm'],json.dumps(r['groups']['all']))
