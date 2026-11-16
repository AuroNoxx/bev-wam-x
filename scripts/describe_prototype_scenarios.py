import json,numpy as np,collections
from pathlib import Path
r=Path('.');s=json.loads((r/'artifacts/structural-prototypes-light512/selection.json').read_text());d=json.loads((r/'artifacts/implementation-plan-v2/stop_deceleration_train_only_selection.json').read_text())['selection'];groups={**s,'stop_diagnostic':[x['file'] for x in d]}
for k,ls in groups.items():
 v=[];kinds=collections.Counter();routes=set()
 for n in ls:
  routes.add(Path(n).parent.name);kinds[Path(n).parent.name.split('_Town')[0]]+=1
  with np.load(r/'data/occ_v3'/n) as f:v.append(float(f['state'][0])*30*3.6)
 print(k,json.dumps({'windows':len(ls),'routes':len(routes),'directory_scenario_labels':dict(kinds),'initial_speed_kmh_min_median_max':np.percentile(v,[0,50,100]).tolist(),'counts_0_20_40_60_80_100_120':np.histogram(v,bins=[-.5,20,40,60,80,100,120,200])[0].tolist()}))
