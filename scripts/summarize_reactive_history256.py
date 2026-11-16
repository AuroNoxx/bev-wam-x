import json,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[1];OUT=R/'artifacts/carla-reactive-history-paired256'
def main():
 d=json.loads((R/'data/carla_reactive_hold_cache_v1/manifest.json').read_text());names=d['dev'];labels=[];states=[]
 for name in names:
  with np.load(name) as z:labels.append(z['trajectory']*np.array([50,10,1,1,30]));states.append(z['state'])
 labels=np.array(labels);states=np.array(states);v0=states[:,0]*30;groups=np.array(['command_hold' if s[2]<.5 else 'static_permitted' if v<.2 else 'decelerating' if y[-1,4]<v-1 else 'accelerating' if y[-1,4]>v+1 else 'steady' for s,v,y in zip(states,v0,labels)]);result={}
 for seed in [2070,2071]:
  for arm in ['base','history']:
   folder=OUT/f'{seed}-{arm}'
   if not (folder/'eval.json').exists():continue
   e=json.loads((folder/'eval.json').read_text());p=np.array([r['points'] for r in e['records']]).reshape(len(names),4,25,5);rows=e['records'];res={'training':json.loads((folder/'train_complete.json').read_text()),'world':e['world_summary'],'groups':{},'actor_future_poison_exact':e['actor_future_poison_exact']};res['training'].pop('rng',None)
   for group in ['all','command_hold','static_permitted','decelerating','accelerating','steady']:
    mask=np.ones(len(names),bool) if group=='all' else groups==group;q=p[mask];y=labels[mask];ix=np.flatnonzero(mask);rs=[r for r in rows if r['window'] in set(ix.tolist())]
    if not len(q):res['groups'][group]={'n':0};continue
    res['groups'][group]={'n':len(q),'candidate_count':len(q)*4,'speed_MAE_0.2_0.6_1_5s':np.abs(q[:,:,[0,2,4,24],4]-y[:,None,[0,2,4,24],4]).mean((0,1)).tolist(),'position_fit_diagnostic_m':float(np.linalg.norm(q[...,:2]-y[:,None,:,:2],axis=-1).mean()),'pred_end_x_m':float(q[:,:,-1,0].mean()),'label_end_x_m':float(y[:,-1,0].mean()),'pred_end_speed':float(q[:,:,-1,4].mean()),'label_end_speed':float(y[:,-1,4].mean()),'first_accel_mean':float(np.mean([r['first_accel_mps2'] for r in rs])),'jerk_peak_mean':float(np.mean([r['max_jerk_mps3'] for r in rs])),'negative_forward_segments_mean':float(np.mean([r['negative_forward_segments'] for r in rs])),'collision':None,'TTC':None,'closed_loop':False}
   result[f'{seed}-{arm}']=res
 (OUT/'summary.partial.json').write_text(json.dumps(result,indent=2))
 for key,r in result.items():print(key,{g:{k:v for k,v in m.items() if k in ['n','speed_MAE_0.2_0.6_1_5s','jerk_peak_mean','pred_end_speed']} for g,m in r['groups'].items()})
if __name__=='__main__':main()
