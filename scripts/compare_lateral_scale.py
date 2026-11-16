import json,numpy as np
from pathlib import Path
b=Path('artifacts/carla-lateral-scale512');base=Path('artifacts/carla-nav-recovery-factorial512');a=json.load(open(base/'stratified.json'));old=next(x for x in a if x['arm']=='on_recovery');new=json.load(open(b/'stratified.json'))[0];r={'old':old,'new':new,'rng_first3_equal':json.load(open(b/'train_complete.json'))['rng']==json.load(open(base/'on_recovery/train_complete.json'))['rng'],'noise':{}}
for arm,path in [('old',base/'on_recovery-eval.json'),('scaled',b/'scaled-eval.json')]:
 ev=json.load(open(path));r['noise'][arm]={}
 for split,v in ev.items():
  rows=v['records'];groups={}
  for row in rows:groups.setdefault(row['window'],[]).append(np.array(row['points']))
  ys=[];xs=[];acc=[]
  for ps in groups.values():
   p=np.stack(ps);ys.append(float(p[:,-1,1].std()));xs.append(float(p[:,-1,0].std()));vy=p[:,:,4]*p[:,:,2];acc.extend(np.max(np.abs(np.diff(np.concatenate([np.zeros((len(p),1)),vy],axis=1),axis=1)/.2),axis=1).tolist())
  r['noise'][arm][split]={'candidate_endpoint_y_std_mean_m':float(np.mean(ys)),'candidate_endpoint_x_std_mean_m':float(np.mean(xs)),'peak_lateral_acc_mean_mps2':float(np.mean(acc))}
(b/'comparison.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
