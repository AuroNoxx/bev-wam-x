import json,numpy as np
from pathlib import Path
R=Path.cwd();b=R/'artifacts/carla-miniloop-flow512';d=json.load(open(b/'design.json'));out=[]
for p in sorted(b.glob('*/report.json')):
 r=json.load(open(p));ans={'group':p.parent.name,'wall_s':r['wall_s'],'unique_train':r['unique_train']}
 for split in ['new','old']:
  names=d[split+'_dev'];truth=[]
  for n in names:
   with np.load(Path(n) if Path(n).is_absolute() else R/'data/occ_v3'/n) as z:truth.append(z['trajectory']*np.array([50,10,1,1,30]))
  ev=r[split+'_eval'];pos=[];vel=[]
  for x in ev['records']:
   q=np.array(x['points']);t=truth[x['window']];pos.append(np.linalg.norm(q[:,:2]-t[:,:2],axis=-1).mean());vel.append(np.sqrt(((q[:,4:5]*q[:,[3,2]]-t[:,4:5]*t[:,[3,2]])**2).mean()))
  ans[split]={'position_fit_m':float(np.mean(pos)),'vector_velocity_rmse_mps':float(np.mean(vel)),**ev['summary'],'label_forward_m':float(np.mean([t[-1,0] for t in truth]))}
 out.append(ans)
(b/'comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
