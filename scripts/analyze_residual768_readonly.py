import json,numpy as np
from pathlib import Path
root=Path('E:/ai/projects/highway-occ-flow');src=root/'artifacts/residual-clip-paired768';base=root/'artifacts/residual-representation-paired128';out=root/'artifacts/residual768-readonly-dynamics-world';out.mkdir(exist_ok=False);sel=json.load(open(base/'selection.json'));v0=[];truth=[]
for f in sel['validation']:
 with np.load(root/'data/occ_v3'/f,allow_pickle=False) as z:v0.append(float(z['state'][0])*30);truth.append(z['trajectory']*np.array([50,10,1,1,30]))
v0=np.array(v0);truth=np.array(truth);results=[]
for seed in [1980,1981]:
 for clip in [1,2]:
  r=json.load(open(src/f'{seed}-clip{clip}/report.json'));rec=r['eval']['records'];segments=[];windowerr={}
  for z in rec:
   i=z['window'];p=np.array(z['points']);xy=p[:,:2];chord=np.diff(np.vstack(([0,0],xy)),axis=0)/.2;vel=p[:,4,None]*p[:,[3,2]];prev=np.vstack(([v0[i],0],vel[:-1]));mid=(prev+vel)/2;res=chord-mid;direction=mid/np.maximum(np.linalg.norm(mid,axis=1,keepdims=True),1e-8);along=(res*direction).sum(-1);cross=res[:,0]*direction[:,1]-res[:,1]*direction[:,0];windowerr.setdefault(i,[]).append(float(np.linalg.norm(res,axis=-1).mean()));segments.append({'res':np.linalg.norm(res,axis=-1),'along':abs(along),'cross':abs(cross),'chord_speed':np.linalg.norm(chord,axis=-1),'head_speed':p[:,4],'negative_x':chord[:,0]<0,'heading_backward':p[:,3]<0})
  horizons={}
  for name,lo,hi in [('first_second',0,5),('one_to_three',5,15),('three_to_five',15,25)]:horizons[name]={k:float(np.mean([a[k][lo:hi] for a in segments])) for k in segments[0]}
  world=r['eval']['world_rows'];wins=sum(w['own_ema_mse']<w['own_persistence_mse'] for w in world);ratios=[w['own_ema_mse']/w['own_persistence_mse'] for w in world if w['own_persistence_mse']>1e-8];delta=[w['own_ema_mse']-w['own_persistence_mse'] for w in world];err=[np.mean(windowerr[i]) for i in range(42)];groups={}
  for name,lo,hi in [('under5',-1,5),('5to10',5,10),('10plus',10,100)]:
   ids=np.where((v0>=lo)&(v0<hi))[0];groups[name]={'windows':len(ids),'prediction':float(np.mean([world[i]['own_ema_mse'] for i in ids])) if len(ids) else None,'persistence':float(np.mean([world[i]['own_persistence_mse'] for i in ids])) if len(ids) else None}
  results.append({'seed':seed,'clip':clip,'horizon_metrics':horizons,'world':{'prediction_better_windows':wins,'windows':42,'median_ratio':float(np.median(ratios)),'speed_groups':groups,'pearson_gap_vs_trajectory_residual_descriptive_only':float(np.corrcoef(delta,err)[0,1])}})
report={'results':results,'limits':'42same-routeinternaltrain diagnostics x4candidates; worldtargetsownEMAspaces,not comparableacrossmodels; correlationnotcausation; nofree/collision/TTCtruth; entirelyreadonlysavedpredictionsandlabels,noGPUorupdates'};(out/'report.json').write_text(json.dumps(report,indent=2))
for x in results:print(x['seed'],x['clip'],json.dumps(x['horizon_metrics']),json.dumps(x['world']))
