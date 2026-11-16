import json,numpy as np
from pathlib import Path
b=Path('artifacts/route-flow-frozen-full');d=json.load(open(b/'selection.json'));names=d['held'];targets=[]
for n in names:
 with np.load(Path('data/occ_v3')/n) as z:targets.append(z['trajectory']*np.array([50,10,1,1,30]))
for p in sorted(b.glob('*-audit.json')):
 a=json.load(open(p));ev=json.load(open(b/p.name.replace('-audit','-full')));sc={};rows=[]
 for r in ev['records']:
  pred=np.array(r['points']);t=targets[r['window']];row={'scene':Path(names[r['window']]).parent.name.split('_Town')[0],'position':np.linalg.norm(pred[:,:2]-t[:,:2],axis=-1).mean(),'velocity':np.sqrt(((pred[:,4:5]*pred[:,[3,2]]-t[:,4:5]*t[:,[3,2]])**2).mean())};rows.append(row)
 for s in sorted({x['scene'] for x in rows}):sc[s]={k:float(np.mean([x[k] for x in rows if x['scene']==s])) for k in ['position','velocity']}
 macro={k:float(np.mean([x[k] for x in sc.values()])) for k in ['position','velocity']};micro={k:float(np.mean([x[k] for x in rows])) for k in ['position','velocity']}
 path={}
 for k in sorted({x['kind'] for x in a['path_rows']}):path[k]={f:float(np.mean([x[f] for x in a['path_rows'] if x['kind']==k])) for f in ['bias_rmse_mps','noise_spread_mps','position_fit_m','forward_m','vector_peak_jerk_mps3']}
 a.update(full_scenarios=sc,full_macro=macro,full_micro=micro,path_summary=path);p.write_text(json.dumps(a,indent=2));print(p.stem,json.dumps({'macro':macro,'micro':micro,'path':path,'full_summary':a['full_summary']}))
print((b/'complete.json').read_text())
