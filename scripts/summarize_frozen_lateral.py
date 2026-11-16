import json,numpy as np
from pathlib import Path
b=Path('artifacts/carla-frozen-lateral512');d=json.load(open(b/'design.json'));labels={}
for split,names in d['devs'].items():
 labels[split]=[]
 for n in names:
  with np.load(n) as z:
   t=z['trajectory']*np.array([50,10,1,1,30]);v=float(z['state'][0])*30;e=float(z['lane_error_right_m']) if 'lane_error_right_m' in z else 0.;kind='start' if v<.2 and t[-1,4]>2 else 'stopped' if v<.5 and t[-1,4]<.5 else 'deceleration' if v-t[-1,4]>2 else 'moving';labels[split].append((t,v,e,kind))
results=[]
for arm in d['cells']:
 arm=arm.replace('(reused)','');ev=json.load(open(b/f'{arm}-eval.json'));o={'arm':arm}
 for split in labels:
  rows=[]
  for r in ev[split]['records']:
   q=np.array(r['points']);t,v,e,kind=labels[split][r['window']];vel=q[:,4:5]*q[:,[3,2]];acc=np.diff(np.vstack(([v,0],vel)),axis=0)/.2;jerk=np.diff(acc,axis=0)/.2;rows.append({'kind':kind,'offcenter':abs(e)>.1,'pos':float(np.linalg.norm(q[:,:2]-t[:,:2],axis=-1).mean()),'velocity_rmse':float(np.sqrt(((vel-t[:,4:5]*t[:,[3,2]])**2).mean())),'abs_y06_error':float(abs(q[2,1]-t[2,1])),'peak_vector_jerk':float(np.linalg.norm(jerk,axis=-1).max()),'forward':float(q[-1,0]),'label_forward':float(t[-1,0]),'end_speed':float(q[-1,4]),'label_end_speed':float(t[-1,4]),'reverse_endpoint':float(q[-1,0]<-.5)})
  o[split]={}
  for group in ['ALL','offcenter','start','stopped','deceleration','moving']:
   sub=[r for r in rows if group=='ALL' or (r['offcenter'] if group=='offcenter' else r['kind']==group)]
   if sub:o[split][group]={'windows':len(sub)//4,**{k:float(np.mean([r[k] for r in sub])) for k in sub[0] if k not in ['kind','offcenter']}}
 results.append(o)
(b/'stratified.json').write_text(json.dumps(results,indent=2))
for o in results:print(o['arm'],json.dumps({s:{g:o[s][g] for g in ['ALL','offcenter','deceleration'] if g in o[s]} for s in o if s!='arm'}))
print((b/'complete.json').read_text())

