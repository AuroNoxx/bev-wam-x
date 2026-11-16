"""Post-run frozen comparison, no training."""
import json,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'artifacts/matched6-regression-resume1024-v3';old=R/'artifacts/matched6-regression512';d=json.loads((R/'data/carla_matched6_cache_v1/manifest.json').read_text());ys=[];v=[]
for n in d['dev']:
 with np.load(n) as z:ys.append(z['trajectory'][:,4]*30);v.append(z['state'][0]*30)
ys=np.stack(ys);v=np.array(v);s=(v<.2)&(ys[:,:5].max(1)<.2);out={}
for seed in [7201,7202]:
 out[str(seed)]={}
 for step,folder in [(512,old/str(seed)),(1024,O/str(seed))]:
  e=json.loads((folder/'eval.json').read_text());p=np.load(folder/'dev_predictions.npy');extra={'hold1s_count':int(s.sum()),'hold1s_speedMAE':float(abs(p[s,:5,4]*30-ys[s,:5]).mean()),'hold1s_drift_m':float(np.linalg.norm(p[s,4,:2]*[50,10],axis=1).mean())};g=e['model'];out[str(seed)][str(step)]={'all':g['all'],'start':g['start'],'decel':g['decelerating'],'steady':g['steady'],'episodes':{k:x for k,x in g.items() if k.startswith('episode')},'hold1s':extra,'train_fit':json.loads((folder/'fit_sample_eval.json').read_text())['model']['all']}
 (O/str(seed)/'comparison_512_1024.json').write_text(json.dumps(out[str(seed)],indent=2))
(O/'comparison_summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
