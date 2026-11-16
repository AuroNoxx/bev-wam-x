"""Read-only label-prefix audit. Never returns switch information to actor."""
import json,sys,numpy as np,torch
from pathlib import Path
from audit_signed_curvature_expressivity import rows
R=Path(__file__).resolve().parents[1];O=R/'artifacts/permission-prefix-audit-v1';O.mkdir(exist_ok=False);m=json.loads((R/'data/carla_reactive_hold_cache_v1/manifest.json').read_text());report={};masks={}
for split in ['train','dev']:
 masklist=[];info=[];eventkeys=set()
 for path in m[split]:
  p=Path(path);rr=rows('carla_causal_reactive_20261116_hold_v3',p.parent.name);i=int(p.stem);now=rr[i]['teacher_inputs'];change=None;why=[]
  for j in range(i+1,i+51):
   c=rr[j]['teacher_inputs'];reasons=[]
   if c['movement_permitted']!=now['movement_permitted']:reasons.append('permission')
   if c['cruise_mps']!=now['cruise_mps']:reasons.append('cruise')
   if reasons:change=j;why=reasons;eventkeys.add((p.parent.name,j,tuple(reasons)));break
  valid=np.ones(25,bool) if change is None else i+np.arange(2,51,2)<change
  masklist.append(valid);info.append({'episode':p.parent.name,'anchor':i,'change_post_frame_index':change,'reason':why,'valid_points':int(valid.sum()),'current_permission':bool(now['movement_permitted'])})
 masks[split]=np.array(masklist);masksplit=masks[split];changed=np.array([x['change_post_frame_index'] is not None for x in info]);permitted=np.array([x['current_permission'] for x in info]);report[split]={'windows':len(info),'unique_exogenous_change_events':len(eventkeys),'changed_windows':int(changed.sum()),'unchanged_windows':int((~changed).sum()),'valid_points':int(masksplit.sum()),'original_points':masksplit.size,'removed_fraction':float(1-masksplit.mean()),'all_masked_windows':int((masksplit.sum(1)==0).sum()),'valid_count_distribution':{str(k):int((masksplit.sum(1)==k).sum()) for k in np.unique(masksplit.sum(1))},'valid_window_count_at_0.2_0.6_1_5s':masksplit.sum(0)[[0,2,4,24]].tolist(),'permission_false_share_original_points':float((~permitted).mean()),'permission_false_share_remaining_points':float(masksplit[~permitted].sum()/masksplit.sum()),'nonpermission_future_changes':sum(bool(set(x['reason'])-{'permission'}) for x in info)}
 (O/(split+'_label_only_manifest.json')).write_text(json.dumps(info,indent=2))
# Apply same frozen predictions to masked vs unmasked scoring, never retrain.
metrics={}
for seed in [2070,2071]:
 folder=R/f'artifacts/rear-frozen-diagnostic512/{seed}';d=torch.load(folder/'frozen_features_and_labels.pt',map_location='cpu',weights_only=False);truth=d['signed_speed'][514:].numpy();txy=d['actor_xy'][514:].numpy();p=torch.load(folder/'predictions.pt',map_location='cpu',weights_only=False);v0=d['v0'][514:].numpy();baselinev=np.repeat(v0[:,None],25,1);baselinexy=np.zeros_like(txy);baselinexy[:,:,0]=baselinev*np.arange(1,26)[None]*.2;metrics[str(seed)]={}
 for method,v,xy in [('trained',p['signed_speed'].numpy(),p['actor_xy'].numpy()),('constant_velocity',baselinev,baselinexy)]:
  ve=abs(v-truth);pe=np.linalg.norm(xy-txy,axis=-1);valid=masks['dev'];changed=valid.sum(1)<25;groups={}
  for g,sel in [('all',np.ones(len(v),bool)),('future_command_changes',changed),('no_future_command_change',~changed)]:
   vv=valid[sel];ev=ve[sel];ep=pe[sel];nonempty=vv.sum(1)>0
   groups[g]={'n_windows':int(sel.sum()),'valid_points':int(vv.sum()),'speed_MAE_original':float(ev.mean()),'speed_MAE_prefix_pointweighted':float(ev[vv].mean()) if vv.any() else None,'speed_MAE_prefix_windowweighted':float(((ev*vv).sum(1)[nonempty]/vv.sum(1)[nonempty]).mean()) if nonempty.any() else None,'position_error_original':float(ep.mean()),'position_error_prefix_pointweighted':float(ep[vv].mean()) if vv.any() else None}
  metrics[str(seed)][method]=groups
report['frozen_metrics']=metrics;report['contract']={'change_boundary':'first post-control recorded frame whose issued permission/cruise differs from anchor last-issued command; endpoint strictly before boundary valid','temporal_alignment':'row.teacher_observation_frame = row.frame-1; changed control affects row.frame, exclude that post-frame itself','mask_scope':'label loss/evaluation only, not actor, feature tensor, online branch, sampling or future horizon input','missing_metadata':'unknown, not no-change','no_training':True,'not_new_independent_events':'34overlappingwindows represent1switch per split','not_all_ambiguity_filtered':'lead behavior, sparse observation, multiple safe actions remain, no filter on target error/surprise','world_future':'do not automatically erase generic future-world targets; separately define conditional-vs-marginal prediction and stratify'}
(O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
