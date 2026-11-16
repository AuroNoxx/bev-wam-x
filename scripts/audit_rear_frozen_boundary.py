import sys,json,torch,numpy as np
from pathlib import Path
sys.path.insert(0,'scripts');from train_rear_frozen_diagnostic512 import HeadTask,metrics
O=Path('artifacts/rear-frozen-diagnostic512');manifest=json.loads(Path('data/carla_reactive_hold_cache_v1/manifest.json').read_text());permission=[]
for path in manifest['dev']:
 with np.load(path) as z:permission.append(bool(z['state'][2]))
extra={}
for seed in [2070,2071]:
 folder=O/str(seed);d=torch.load(folder/'frozen_features_and_labels.pt',map_location='cpu',weights_only=False);dev={k:v[514:] for k,v in d.items()};c=torch.load(folder/'latest.ckpt',map_location='cpu',weights_only=False);h=c['hyper_parameters'];task=HeadTask(**h);task.load_state_dict(c['state_dict']);task.eval();ps=[]
 with torch.no_grad():
  for start in range(0,514,32):ps.append(task({k:v[start:start+32] for k,v in dev.items()}))
 pred={k:torch.cat([p[k] for p in ps]) for k in ps[0]};saved=torch.load(folder/'predictions.pt',map_location='cpu',weights_only=False);reloadmax=float(abs(pred['actor_xy']-saved['actor_xy']).max());assert reloadmax<.001
 # Signed rear velocities and curvature define actor-point velocities; include true measured t0 actor velocity.
 from audit_signed_curvature_expressivity import rows
 v0xy=[]
 for path in manifest['dev']:
  p=Path(path);row=rows('carla_causal_reactive_20261116_hold_v3',p.parent.name)[int(p.stem)];v=np.array(row['ego_world'])[:3,:3].T@np.array(row['velocity_world']);v0xy.append([v[0],-v[1]])
 yaw=pred['body_yaw'];v=pred['signed_speed'];om=v*pred['curvature'];dire=torch.stack([yaw.cos(),yaw.sin()],-1);normal=torch.stack([-yaw.sin(),yaw.cos()],-1);actorv=v[...,None]*dire+task.decoder.geometry.rear_distance_m*om[...,None]*normal
 true0=torch.tensor(v0xy,dtype=actorv.dtype);acc=torch.diff(torch.cat([true0[:,None],actorv],1),dim=1)/.2;jerk=torch.diff(acc,dim=1)/.2
 initlat=task.decoder.geometry.rear_distance_m*dev['v0']*pred['coefficients'][:,0,1];boundary=(initlat-true0[:,1]).abs()
 masks={'all':torch.ones(514,dtype=torch.bool),'current_static':dev['v0'].abs()<.2,'permission_false':~torch.tensor(permission),'static_permitted':(dev['v0'].abs()<.2)&torch.tensor(permission)};out={}
 for group,sel in masks.items():
  if not sel.any():continue
  out[group]={'n':int(sel.sum()),'speed_MAE_0.6_5s':(pred['signed_speed'][sel]-dev['signed_speed'][sel]).abs().mean(0)[[2,24]].tolist(),'mean_predicted_speed_0.6_5s':pred['signed_speed'][sel].mean(0)[[2,24]].tolist(),'mean_label_speed_0.6_5s':dev['signed_speed'][sel].mean(0)[[2,24]].tolist(),'actor_peak_accel_including_t0_mean':float(acc[sel].norm(dim=-1).max(1).values.mean()),'actor_peak_jerk_after_t0_mean':float(jerk[sel].norm(dim=-1).max(1).values.mean()),'initial_lateral_velocity_jump_mean':float(boundary[sel].mean()),'initial_lateral_velocity_jump_max':float(boundary[sel].max()),'negative_speed_fraction':float((v[sel]<-.05).float().mean())}
 extra[str(seed)]={'checkpoint_cpu_reload_position_maxdiff_m':reloadmax,'groups':out};print(seed,json.dumps(out,indent=2))
(O/'boundary_and_static_audit.json').write_text(json.dumps(extra,indent=2))
