"""Frozen CPU evidence: actual history, no-ID OCC, label horizons, exposure."""
from diagnose_occ_noid_motion import readout,serial
from pathlib import Path
import numpy as np,json,time
R=Path(__file__).resolve().parents[1];S=R/'data/carla_causal_matched6_20261116_v1';C=R/'data/carla_matched6_cache_v1';O=R/'artifacts/matched6-start-frozen-audit';D=json.loads((C/'manifest.json').read_text());threshold=json.loads((R/'artifacts/occ-noid-motion-diagnostic/calibration.json').read_text())['threshold_displacement_m']
def main():
 O.mkdir(exist_ok=False);start=time.perf_counter();records=[];rrs={}
 for split in ['train','dev']:
  for name in D[split]:
   p=Path(name);ep=p.parent.name;i=int(p.stem)
   if ep not in rrs:rrs[ep]=json.loads((S/ep/'frames.json').read_text())
   rr=rrs[ep];row=rr[i]
   with np.load(p) as z:h=z['history'].copy();state=z['state'].copy();ys=z['trajectory'][:,4]*30;truthxy=z['trajectory'][:,:2]*[50,10];ages=z['history_age_seconds'].copy()
   v=float(state[0]*30);lead=np.array([np.linalg.norm(r['teacher_target_post_audit']['velocity_world'][:2]) for r in rr[i-6:i+1]])
   if not (v<.2 or (ys[-1]<v-1)):continue
   read=readout(h,state[4:6]*50);cur=np.linalg.inv(np.array(row['ego_world']));pose=np.array(row['teacher_target_post_audit']['world']);oldpose=np.array(rr[i-6]['teacher_target_post_audit']['world']);delta=(cur@pose)[:2,3]-(cur@oldpose)[:2,3];delta[1]*=-1;axis=state[4:6]/np.linalg.norm(state[4:6]);disp=float(delta@axis)
   speeds=np.array([np.linalg.norm(r['teacher_target_post_audit']['velocity_world'][:2]) for r in rr[:i+1]]);onsets=np.flatnonzero((speeds[1:]>=.2)&(speeds[:-1]<.2))+1;onset=int(onsets[-1]) if len(onsets) and speeds[-1]>=.2 else None
   entry={'file':name,'split':split,'episode':ep,'index':i,'v0':v,'ego_history_accel_mps2':float(state[3]*5),'permission':float(state[2]),'cruise':float(state[1]*30),'history_age_s':ages.tolist(),'lead_current_speed_eval_only':float(lead[-1]),'lead_past06_displacement_eval_only_m':disp,'lead_onset_age_s_eval_only':None if onset is None else (i-onset)*.1,'noid':serial(read),'noid_motion_above_OLD_train_threshold':bool(read['valid'] and abs(read['displacement_s'])>threshold),'future_start_label':bool(v<.2 and ys[-1]>2),'hold_1s_label':bool(v<.2 and ys[:5].max()<.2),'decel_label':bool(v>=.2 and ys[-1]<v-1),'label_speed_02_06_1_5':ys[[0,2,4,24]].tolist(),'label_end_x':float(truthxy[-1,0])};records.append(entry)
  print('DONE_SPLIT',split,flush=True)
 report={'old_train_threshold_m':threshold,'limits':'Threshold frozen from older train, not re-estimated ondev. GT used only for after-the-fact grouping; no-ID readout usesactual4OCC andnavigation only. Readout valid is not proof of correct object association. Observed0.2m/s instantaneouslead onset neednot exceed .8m voxel/historynoise. No training.','splits':{}}
 for split in ['train','dev']:
  rs=[r for r in records if r['split']==split];groups={'future_start':[r for r in rs if r['future_start_label']],'start_current_lead_moving':[r for r in rs if r['future_start_label'] and r['lead_current_speed_eval_only']>.2],'start_motion_detected':[r for r in rs if r['future_start_label'] and r['noid_motion_above_OLD_train_threshold']],'hold1s':[r for r in rs if r['hold_1s_label']],'decel':[r for r in rs if r['decel_label']]};report['splits'][split]={}
  for key,g in groups.items():
   a={'n':len(g),'episodes':sorted(set(r['episode'] for r in g)),'noid_valid':sum(r['noid']['valid'] for r in g),'noid_motion':sum(r['noid_motion_above_OLD_train_threshold'] for r in g),'label_speeds_mean':np.mean([r['label_speed_02_06_1_5'] for r in g],axis=0).tolist() if g else None}
   if split=='train':
    for seed in [7201,7202]:
     seq=json.loads((R/f'artifacts/matched6-regression512/{seed}/selection.json').read_text());chosen=set(r['file'] for r in g);a[f'exposure_{seed}']=sum(x in chosen for x in seq)
   else:
    ids=[D['dev'].index(r['file']) for r in g]
    for seed in [7201,7202]:
     pred=np.load(R/f'artifacts/matched6-regression512/{seed}/dev_predictions.npy')[ids];a[f'pred_speed_mean_{seed}']=(pred[...,4]*30).mean(0)[[0,2,4,24]].tolist() if g else None
   report['splits'][split][key]=a
 (O/'rows.json').write_text(json.dumps(records));report['wall_s']=time.perf_counter()-start;(O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
