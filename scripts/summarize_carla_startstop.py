import json,numpy as np
from pathlib import Path
R=Path.cwd();b=R/'artifacts/carla-startstop-mixture512';names={'newdev':json.load(open('data/carla_startstop_cache_v1/manifest.json'))['dev'],'olddev':json.load(open('data/carla_miniloop_cache_v1/manifest.json'))['dev']};results=[]
for label in ['baseline','mixture']:
 report=json.load(open(b/f'{label}-eval.json'));result={'model':label}
 for split,files in names.items():
  truth=[];kinds=[]
  for n in files:
   with np.load(n) as f:
    tr=f['trajectory']*np.array([50,10,1,1,30]);v=float(f['state'][0])*30;end=tr[-1,4];kind='start' if v<.2 and end>2 else 'stationary_stop' if v<.5 and end<.5 else 'deceleration' if v-end>2 else 'moving';truth.append(tr);kinds.append(kind)
  rows=[]
  for r in report[split]['records']:
   q=np.array(r['points']);t=truth[r['window']];rows.append({'kind':kinds[r['window']],'pos':float(np.linalg.norm(q[:,:2]-t[:,:2],axis=-1).mean()),'end_speed':float(q[-1,4]),'label_end_speed':float(t[-1,4]),'forward':float(q[-1,0]),'label_forward':float(t[-1,0]),'reverse_endpoint':float(q[-1,0]<-.5)})
  result[split]={}
  for kind in ['ALL']+sorted(set(kinds)):
   sub=rows if kind=='ALL' else [r for r in rows if r['kind']==kind];result[split][kind]={'windows':len(sub)//4,**{k:float(np.mean([r[k] for r in sub])) for k in sub[0] if k!='kind'}}
 results.append(result)
(b/'stratified.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2));print((b/'train_complete.json').read_text())
