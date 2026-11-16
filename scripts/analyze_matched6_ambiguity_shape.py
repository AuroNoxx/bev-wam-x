"""Existing-data-only ambiguity and temporal-shape decomposition. No learning."""
from pathlib import Path
import numpy as np,json,hashlib,time
R=Path(__file__).resolve().parents[1];O=R/'artifacts/matched6-ambiguity-shape-v2';D=json.loads((R/'data/carla_matched6_cache_v1/manifest.json').read_text())
def cosine(a,b):return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)*np.linalg.norm(b)>1e-12 else None
def main():
 O.mkdir(exist_ok=False);start=time.perf_counter();static=[];exact={};truth=[];v0=[]
 for name in D['train']:
  with np.load(name) as z:s=z['state'].copy();h=z['history'].copy();y=z['trajectory'].copy();age=z['history_age_seconds'].copy();valid=z['history_valid'].copy()
  if s[0]*30>=.05 or abs(s[3]*5)>=.05:continue
  digest=hashlib.sha256(h.tobytes()+s.tobytes()+age.tobytes()+valid.tobytes()).hexdigest();exact.setdefault(digest,[]).append({'file':name,'future_speed5':float(y[-1,4]*30)})
  flat=h.flatten();idx=np.flatnonzero(flat!=255);tokens=idx.astype(np.int64)*8+flat[idx];static.append({'file':name,'state':s,'tokens':tokens,'future_speed5':float(y[-1,4]*30),'future_speed1':float(y[4,4]*30)})
 exactdup=[v for v in exact.values() if len(v)>1];pairs=[]
 for i,a in enumerate(static):
  for b in static[i+1:]:
   if Path(a['file']).parent!=Path(b['file']).parent:continue
   sa,sb=a['state'],b['state']
   if abs(sa[0]-sb[0])*30>.02 or abs(sa[3]-sb[3])*5>.05 or np.linalg.norm(sa[4:6]-sb[4:6])*50>.05:continue
   n=np.intersect1d(a['tokens'],b['tokens'],assume_unique=True).size;union=len(a['tokens'])+len(b['tokens'])-n;jac=n/union if union else 1.;pairs.append({'a':a['file'],'b':b['file'],'all4history_observed_semantic_jaccard':jac,'future5speed_difference':abs(a['future_speed5']-b['future_speed5']),'future1speed_difference':abs(a['future_speed1']-b['future_speed1']),'speeds5':[a['future_speed5'],b['future_speed5']]})
 high=sorted([p for p in pairs if p['future5speed_difference']>2],key=lambda p:-p['all4history_observed_semantic_jaccard'])
 for name in D['dev']:
  with np.load(name) as z:truth.append(z['trajectory']);v0.append(z['state'][0]*30)
 y=np.stack(truth);v=np.array(v0);ys=y[...,4]*30;yv=ys*y[...,3];groups={'start':(v<.2)&(ys[:,-1]>2),'decel':(v>=.2)&(ys[:,-1]<v-1),'hold1s':(v<.2)&(ys[:,:5].max(1)<.2),'steady':(v>=.2)&(abs(ys[:,-1]-v)<=1)}
 result={'limits':'Nearidentical != identical: Jaccard measured onobservedsemantic union, unknownnotfree; currentstate thresholds preset, trainonly. Event future is genuinelyunobserved, butnearneighborsnotproofsameexactinput. Shapeanalysis isdescriptive, notgradientcausality.','static_train_count':len(static),'exact_duplicate_actor_groups':exactdup,'similar_state_pairs':len(pairs),'near_pairs_future5_difference_gt2_top10':high[:10],'models':{}}
 for seed in [7201,7202]:
  for step,base in [(512,'matched6-regression512'),(1024,'matched6-regression-resume1024-v3')]:
   p=np.load(R/f'artifacts/{base}/{seed}/dev_predictions.npy');pv=p[...,4]*30*p[...,3];center=pv-pv.mean(1,keepdims=True);u,s,vt=np.linalg.svd(center,full_matrices=False);entry={'temporal_energy_first_mode_fraction':float(s[0]**2/(s*s).sum()),'groups':{}};shapes={};labelshapes={}
   for key,mask in groups.items():
    pred=pv[mask];label=yv[mask];shape=(pred-pred.mean(1,keepdims=True)).mean(0);labelshape=(label-label.mean(1,keepdims=True)).mean(0);shapes[key]=shape;labelshapes[key]=labelshape
    entry['groups'][key]={'n':int(mask.sum()),'pred_mean_velocity_mps':float(pred.mean()),'label_mean_velocity_mps':float(label.mean()),'pred_shape25':shape.tolist(),'label_shape25':labelshape.tolist(),'shape_cosine_to_label':cosine(shape,labelshape),'pred_5_minus02_mean':float((pred[:,-1]-pred[:,0]).mean()),'label_5_minus02_mean':float((label[:,-1]-label[:,0]).mean())}
   entry['start_decel_pred_shape_cosine']=cosine(shapes['start'],shapes['decel']);entry['start_decel_label_shape_cosine']=cosine(labelshapes['start'],labelshapes['decel']);result['models'][f'{seed}_{step}']=entry
 result['wall_s']=time.perf_counter()-start;(O/'report.json').write_text(json.dumps(result,indent=2));print('COMPLETE',len(static),len(exactdup),len(high),result['wall_s'])
if __name__=='__main__':main()

