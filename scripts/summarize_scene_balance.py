"""CPU-only stratified report for completed behavior-sampling experiment."""
import json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'artifacts/scene-balance-light512';design=json.loads((base/'design.json').read_text());infos=design['validation'];truth=[]
for x in infos:
 with np.load(ROOT/'data/occ_v3'/x['file'],allow_pickle=False) as d:truth.append((d['trajectory']*np.array([50,10,1,1,30]),float(d['state'][0])*30))
results=[]
for folder in sorted(base.glob('20*-*')):
 if not (folder/'report.json').exists():continue
 report=json.loads((folder/'report.json').read_text());rows=[]
 for r in report['eval']['records']:
  i=r['window'];p=np.array(r['points']);t,v0=truth[i];v=p[:,4:5]*p[:,[3,2]];a=np.diff(np.vstack(([v0,0.],v)),axis=0)/.2;j=np.diff(a,axis=0)/.2
  rows.append({'kind':infos[i]['stratum'],'stop':float(p[-1,4]<.5),'final_speed_mps':float(p[-1,4]),'label_final_speed_mps':float(t[-1,4]),'forward_proxy_m':float(p[-1,0]),'label_forward_m':float(t[-1,0]),'path_m':float(np.linalg.norm(np.diff(np.vstack(([0,0],p[:,:2])),axis=0),axis=1).sum()),'vector_peak_accel_mps2':float(np.linalg.norm(a,axis=1).max()),'vector_peak_jerk_mps3':float(np.linalg.norm(j,axis=1).max()),'position_label_fit_m':float(np.linalg.norm(p[:,:2]-t[:,:2],axis=1).mean()),'backward_segments':int((np.diff(np.r_[0,p[:,0]])<0).sum())})
 groups={}
 for kind in sorted(set(r['kind'] for r in rows)):
  subset=[r for r in rows if r['kind']==kind];groups[kind]={'windows':len(subset)//4,'candidates':len(subset),'mean':{k:float(np.mean([r[k] for r in subset])) for k in subset[0] if k!='kind'}}
 results.append({'seed':report['seed'],'arm':report['arm'],'groups':groups,'diversity_rms_m':report['eval']['summary']['spread_rms_m'],'world':report['eval']['world_summary']})
(base/'stratified.json').write_text(json.dumps({'results':results,'limits':design['limits'],'collision':None,'TTC':None,'actual_progress':None,'closed_loop':False},indent=2));print(json.dumps(results,indent=2))
