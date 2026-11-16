"""Frozen report, no parameter updates. Truth used only for evaluation stratification."""
from pathlib import Path
import numpy as np,json
R=Path(__file__).resolve().parents[1];O=R/'artifacts/matched6-regression512';D=json.loads((R/'data/carla_matched6_cache_v1/manifest.json').read_text());names=D['dev'];rows={}
truth=[];v0=[];lead=[]
for name in names:
 p=Path(name);ep=p.parent.name
 if ep not in rows:rows[ep]=json.loads((R/'data/carla_causal_matched6_20261116_v1'/ep/'frames.json').read_text())
 row=rows[ep][int(p.stem)];lead.append(np.linalg.norm(row['teacher_target_post_audit']['velocity_world'][:2]))
 with np.load(name) as z:truth.append(z['trajectory']);v0.append(z['state'][0]*30)
y=np.stack(truth);v0=np.array(v0);lead=np.array(lead);ys=y[...,4]*30
out={'limits':'extra strata use current lead GT ONLY for posthoc diagnosis, not actor; future speed stratification is descriptive not a promise that early restart is observable; no closedloop/test; only2heldepisodes','seeds':{}}
for seed in [7201,7202]:
 f=O/str(seed);p=np.load(f/'dev_predictions.npy');ps=p[...,4]*30;extra={}
 for n in [1,3,5,25]:
  s=(v0<.2)&(ys[:,:n].max(1)<.2);extra[f'hold_prefix_{n*.2:.1f}s']={'n':int(s.sum()),'speed_mae':float(abs(ps[s,:n]-ys[s,:n]).mean()) if s.any() else None,'endpoint_drift_m':float(np.linalg.norm(p[s,n-1,:2]*[50,10],axis=1).mean()) if s.any() else None}
 for moving in [False,True]:
  s=(v0<.2)&(ys[:,-1]>2)&((lead>.2)==moving);extra[f'future_start_current_lead_moving_{moving}']={'n':int(s.sum()),'pred_5s_speed':float(ps[s,-1].mean()) if s.any() else None,'label_5s_speed':float(ys[s,-1].mean()) if s.any() else None}
 ev=json.loads((f/'eval.json').read_text());fit=json.loads((f/'fit_sample_eval.json').read_text());out['seeds'][str(seed)]={'training':json.loads((f/'train_complete.json').read_text()),'dev':ev,'train_stride8':fit,'extra_strata':extra,'resume':json.loads((f/'resume_verified.json').read_text())}
(O/'summary.json').write_text(json.dumps(out,indent=2))
for seed,r in out['seeds'].items():
 print(seed, json.dumps({'all':r['dev']['model']['all'],'extra':r['extra_strata'],'trainall':r['train_stride8']['model']['all']},indent=2))
