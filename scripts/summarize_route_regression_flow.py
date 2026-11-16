import json,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[1];base=R/'artifacts/route-regression-flow1024';d=json.loads((base/'design.json').read_text());names={'fit':[r['file'] for r in d['fit_diagnostic']],'held':[r['file'] for r in d['held']]};labels={}
for split,items in names.items():
 labels[split]=[]
 for n in items:
  with np.load(R/'data/occ_v3'/n) as f:labels[split].append((f['trajectory']*np.array([50,10,1,1,30]),float(f['state'][0])*30))
allrows=[]
for folder in sorted(base.glob('20*-*')):
 if not (folder/'report.json').exists():continue
 r=json.loads((folder/'report.json').read_text());grouped={}
 for split,items in names.items():
  rows=[]
  for p in r[split]['records']:
   i=p['window'];q=np.array(p['points']);truth,v0=labels[split][i];vel=q[:,4:5]*q[:,[3,2]];tv=truth[:,4:5]*truth[:,[3,2]];acc=np.diff(np.vstack(([v0,0],vel)),axis=0)/.2;jerk=np.diff(acc,axis=0)/.2
   rows.append({'scene':Path(items[i]).parent.name.split('_Town')[0],'position_fit_m':float(np.linalg.norm(q[:,:2]-truth[:,:2],axis=-1).mean()),'velocity_rmse_mps':float(np.sqrt(((vel-tv)**2).mean())),'vector_peak_accel_mps2':float(np.linalg.norm(acc,axis=-1).max()),'vector_peak_jerk_mps3':float(np.linalg.norm(jerk,axis=-1).max()),'final_speed_mps':float(q[-1,4]),'label_final_speed_mps':float(truth[-1,4]),'forward_proxy_m':float(q[-1,0]),'label_forward_m':float(truth[-1,0]),'end_stopped':float(q[-1,4]<.5)})
  means={}
  for scene in ['ALL']+sorted(set(v['scene'] for v in rows)):
   sub=rows if scene=='ALL' else [v for v in rows if v['scene']==scene];means[scene]={'windows':len(sub)//4,'mean':{k:float(np.mean([x[k] for x in sub])) for k in sub[0] if k!='scene'}}
  grouped[split]=means
 allrows.append({'seed':r['seed'],'method':r['method'],'groups':grouped,'fit_diversity':r['fit']['summary']['spread_rms_m'],'held_diversity':r['held']['summary']['spread_rms_m']})
(base/'stratified.json').write_text(json.dumps({'results':allrows,'limits':d['limits']+' Raw candidate means,not best-of-K; regression repeat4 is deterministic,not independent. Actual event success/collision/TTC unavailable.'},indent=2))
for r in allrows:print(r['seed'],r['method'],json.dumps({s:r['groups'][s]['ALL'] for s in ['fit','held']}))
