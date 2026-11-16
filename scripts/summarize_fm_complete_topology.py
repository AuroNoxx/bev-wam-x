"""Post-run complete-window and command-prefix stratification, no updates."""
import json,numpy as np,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'artifacts/fm-complete-topology-paired256'
def main():
 manifest=json.loads((R/'data/carla_reactive_hold_cache_v1/manifest.json').read_text());meta=json.loads((R/'artifacts/permission-prefix-audit-v1/dev_label_only_manifest.json').read_text());truth=[];v0=[];permission=[]
 for path in manifest['dev']:
  with np.load(path) as z:truth.append(z['trajectory']*[50,10,1,1,30]);v0.append(float(z['state'][0]*30));permission.append(bool(z['state'][2]))
 truth=np.array(truth);v0=np.array(v0);complete=np.array([x['change_post_frame_index'] is None for x in meta]);masks=np.array([np.arange(25)<x['valid_points'] for x in meta]);summary={}
 for folder in sorted(O.iterdir()):
  if not folder.is_dir() or not (folder/'eval.json').exists():continue
  ev=json.loads((folder/'eval.json').read_text());rows=ev['records'];idx=np.array([r['window'] for r in rows]);p=np.array([r['points'] for r in rows]);tr=truth[idx];vv=p[:,:,4,None]*p[:,:,[3,2]];initial=np.stack([v0[idx],np.zeros(len(idx))],1);a=np.diff(np.concatenate([initial[:,None],vv],1),axis=1)/.2;j=np.diff(a,axis=1)/.2;groups={}
  definitions={'primary_complete':complete[idx],'complete_static_permitted':complete[idx]&(v0[idx]<.2),'complete_decelerating':complete[idx]&(v0[idx]>=.2)&(tr[:,-1,4]<v0[idx]-1),'complete_accelerating':complete[idx]&(v0[idx]>=.2)&(tr[:,-1,4]>v0[idx]+1),'complete_steady':complete[idx]&(v0[idx]>=.2)&(abs(tr[:,-1,4]-v0[idx])<=1),'secondary_command_changes':~complete[idx]}
  for name,sel in definitions.items():
   if not sel.any():continue
   valid=masks[idx[sel]] if name=='secondary_command_changes' else np.ones((int(sel.sum()),25),bool);pe=np.linalg.norm(p[sel,:,:2]-tr[sel,:,:2],axis=-1);se=abs(p[sel,:,4]-tr[sel,:,4]);groups[name]={'windows':int(len(set(idx[sel]))),'candidates':int(sel.sum()),'position_error_m':float(pe[valid].mean()) if valid.any() else None,'speed_MAE_0.6_5s':[float(se[:,t][valid[:,t]].mean()) if valid[:,t].any() else None for t in [2,24]],'valid_speed_MAE_all_times':float(se[valid].mean()) if valid.any() else None,'mean_peak_vector_accel_with_t0':float(np.linalg.norm(a[sel],axis=-1).max(1).mean()),'mean_peak_vector_jerk_after_t0':float(np.linalg.norm(j[sel],axis=-1).max(1).mean()),'endpoint_forward_m':float(p[sel,-1,0].mean()),'collision':None,'TTC':None,'dynamics_note':'full factual horizon, command-change subgroup dynamics not prefix-only'}
  world=[w for w in ev['world_rows'] if complete[w['window']]];worldmetrics={k:float(np.mean([w[k] for w in world if w[k] is not None])) for k in ['fixed_reference_mse','fixed_persistence_mse','own_ema_mse','own_persistence_mse']};spread=[]
  for wi in np.where(complete)[0]:
   q=p[idx==wi,:,:2];spread.append(float(np.sqrt(((q-q.mean(0))**2).sum(-1).mean())))
  train=json.loads((folder/'train_complete.json').read_text());ck=torch.load(folder/'latest.ckpt',map_location='cpu',weights_only=False);summary[folder.name]={'groups':groups,'world_primary':worldmetrics,'candidate_spread_rms_m':float(np.mean(spread)),'train_wall_s':train['train_wall_s'],'peak_gib':train['peak_gib'],'checkpoint':{'step':ck['global_step'],'cursor':ck['CompleteDM']['loader']['_sampler_iter_state']['samples_yielded'],'optimizer':bool(ck['optimizer_states']),'scheduler':bool(ck['lr_schedulers']),'task_state':bool(ck.get('task_state'))}}
 (O/'stratified_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
