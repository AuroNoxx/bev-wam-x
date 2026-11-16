"""Isolated no-learning public Bench2Drive contract/baseline audit. Never opens test."""
from dataset_v1 import compact,warp
from pathlib import Path
import json,gzip,hashlib,time,collections
import numpy as np
R=Path(__file__).resolve().parents[1];C=R/'data/occ_v3';P=R/'data/public/bench2drive/clips';O=R/'artifacts/bench2drive-official-contract-baselines-v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ego(a):return next(x for x in a['bounding_boxes'] if x['class']=='ego_vehicle')
def W(a):return np.array(ego(a)['world2ego'],dtype=np.float64)
def load(route):return {int(p.name.split('.')[0]):json.load(gzip.open(p,'rt')) for p in (P/route/'anno').glob('*.json.gz')}
def wrap(x):return np.arctan2(np.sin(x),np.cos(x))
def predict(v,w,ts):
 if abs(w)<1e-7:return np.stack([v*ts,np.zeros_like(ts)],-1)
 return np.stack([v*np.sin(w*ts)/w,v*(1-np.cos(w*ts))/w],-1)
def agg(rows):
 out={'n':len(rows),'routes':len(set(r['route'] for r in rows)),'metrics':{}}
 for method in ['stationary','constant_velocity','CTRV_current_gyro']:
  rr=[r for r in rows if r['predictions'][method] is not None];out['metrics'][method]={'n':len(rr)}
  for key in ['endpoint_position_m','prefix_ADE_m','endpoint_speed_MAE_mps','endpoint_signed_speed_MAE_mps']:
   a=np.array([r['predictions'][method][key] for r in rr]);out['metrics'][method][key]={'mean':a.mean(0).tolist(),'p95':np.quantile(a,.95,axis=0).tolist()} if len(a) else None
 return out
def main():
 O.mkdir(exist_ok=False);start=time.perf_counter();manifest=C/'manifest.json';m=json.loads(manifest.read_text());routes=[r for r in m['routes'] if r['split'] in ['train','validation']];assert collections.Counter(r['split'] for r in routes)=={'train':42,'validation':9};assert m['splits']['test']==[];before=sha(manifest)
 chosen=[next(r for r in routes if r['split']=='train' and r['family']=='HighwayCutIn'),next(r for r in routes if r['split']=='train' and r['family']=='HardBreakRoute'),next(r for r in routes if r['split']=='validation' and r['family']=='SequentialLaneChange')]
 plan={'source_revision':m['revision'],'split_rule':m['split_rule'],'train_routes':[r['route'] for r in routes if r['split']=='train'],'validation_routes':[r['route'] for r in routes if r['split']=='validation'],'validation_windows':len(m['splits']['validation']),'anchors':[{'route':r['route'],'split':r['split'],'sample':r['samples'][len(r['samples'])//2]} for r in chosen],'horizons_seconds':[1,2,3,5],'time':'NOMINAL10Hz, no source timestamp certificate','CTRV':'current annotation gyro_z converted toFLU negative; no future derived rate; speed aslogged; reverseflag reported','groups':'currentstatic<.2; holdall5s<.2; deceleration current>=.2 and5s speed<current-1; labelsONLYevaluate; allwindows frozen','scope':'not official model reproduction; no learning,install,CUDA,defaultchanges or testreading'};(O/'plan.json').write_text(json.dumps(plan,indent=2));aa={};anchors=[]
 for r in chosen:
  route=r['route'];aa[route]=load(route);name=r['samples'][len(r['samples'])//2];i=int(Path(name).stem);a=aa[route][i];w=W(a);pose=np.linalg.inv(w)
  with np.load(C/name) as z:h=z['history'].copy();state=z['state'].copy();trajectory=z['trajectory'].copy();cw=z['world2ego'].copy();fields=z.files
  future=[]
  for j in range(2,51,2):
   rel=w@np.linalg.inv(W(aa[route][i+j]));yaw=-np.arctan2(rel[1,0],rel[0,0]);future.append([rel[0,3]/50,-rel[1,3]/10,np.sin(yaw),np.cos(yaw),aa[route][i+j]['speed']/30])
  nav=[]
  for kind in ['near','far']:
   loc=w@np.array([a['x_command_'+kind],a['y_command_'+kind],pose[2,3],1]);nav.extend([loc[0]/50,-loc[1]/50])
  occpath=P/route/'occ'/f'{i-1:05d}.npz'
  with np.load(occpath) as z:oc=z['occ'].copy();seg=z['seg'].copy();grid=warp(compact(oc,seg),w,pose)
  back=np.linalg.inv(W(aa[route][i-1]));yawpast=wrap(np.arctan2(pose[1,0],pose[0,0])-np.arctan2(back[1,0],back[0,0]))/.1
  row={'route':route,'split':r['split'],'frame':i,'cache_fields':fields,'annotation_sha':sha(P/route/'anno'/f'{i:05d}.json.gz'),'occ_sha':sha(occpath),'pose_cache_max_abs':float(abs(cw-w).max()),'pose_inverse_residual':float(abs(w@pose-np.eye(4)).max()),'ego_actor_location_error_m':float(np.linalg.norm(pose[:3,3]-ego(a)['location'])),'top_xy_minus_actor_xy_m':(np.array([a['x'],a['y']])-pose[:2,3]).tolist(),'compass_vs_yaw_plus90_rad':float(wrap(a['theta']-(np.arctan2(pose[1,0],pose[0,0])+np.pi/2))),'trajectory_cache_max_abs_normalized':float(abs(np.array(future)-trajectory).max()),'navigation_cache_max_abs_normalized':float(abs(np.array(nav)-state[4:8]).max()),'nav_commands':[a['command_near'],a['command_far']],'gyro_rad_s_CARLA':a['angular_velocity'][2],'past_pose_yawrate_rad_s_CARLA':float(yawpast),'cached_FLU_yawrate':float(state[3]),'speed_reported_mps':a['speed'],'past_pose_speed_mps':float(np.linalg.norm(pose[:3,3]-back[:3,3])/.1),'control':{k:a[k] for k in ['throttle','steer','brake','reverse']},'control_application_timing':'unknown: sameannotationnot proof pre/post actuation','occ_source_values':{str(k):int(v) for k,v in zip(*np.unique(oc,return_counts=True))},'occ_current_rebuild_exact':bool(np.array_equal(grid,h[-1])),'occ_unfilled_fraction':float((h[-1]==0).mean()),'occ_unknown_fraction':float((h[-1]==255).mean()),'source_frame_pairing':'OCC filename annotationindex-1, corroborated previousrawLiDAR audit; not timestampcertificate','occ_past_only_generation':'unknown; exactcache matchdoesnot prove sourcegenerationcausality'}
  assert row['pose_cache_max_abs']<1e-9 and row['trajectory_cache_max_abs_normalized']<1e-5 and row['navigation_cache_max_abs_normalized']<1e-5 and row['occ_current_rebuild_exact'];anchors.append(row)
 (O/'anchor_contract.json').write_text(json.dumps(anchors,indent=2));rows=[];times=np.arange(1,26)*.2;idx=np.array([4,9,14,24]);units=[]
 for route in [r for r in routes if r['split']=='validation']:
  rn=route['route'];raw=aa.get(rn) or load(rn)
  for name in route['samples']:
   i=int(Path(name).stem);a=raw[i];w=W(a);pose=np.linalg.inv(w);v=float(a['speed']);omega=-float(a['angular_velocity'][2]);ys=np.array([raw[i+j]['speed'] for j in range(2,51,2)]);xy=[]
   for j in range(2,51,2):rel=w@np.linalg.inv(W(raw[i+j]));xy.append([rel[0,3],-rel[1,3]])
   xy=np.array(xy)
   with np.load(C/name) as z:cacheerr=float(abs(z['trajectory'][:,:2]*[50,10]-xy).max());speederr=float(abs(z['trajectory'][:,4]*30-ys).max())
   assert cacheerr<1e-4 and speederr<1e-4
   back=np.linalg.inv(W(raw[i-1]));pastw=-wrap(np.arctan2(pose[1,0],pose[0,0])-np.arctan2(back[1,0],back[0,0]))/.1;units.append({'route':rn,'frame':i,'speed_mps':v,'pose_past_speed_mps':float(np.linalg.norm(pose[:3,3]-back[:3,3])/.1),'current_gyro_FLU':omega,'past_pose_FLU':float(pastw),'reverse':bool(a['reverse'])})
   preds={'stationary':np.zeros_like(xy),'constant_velocity':predict(v,0,times),'CTRV_current_gyro':predict(v,omega,times) if np.isfinite(omega) else None};metrics={}
   for method,p in preds.items():
    if p is None:metrics[method]=None;continue
    err=np.linalg.norm(p-xy,axis=1);pspeed=0 if method=='stationary' else abs(v);metrics[method]={'endpoint_position_m':err[idx].tolist(),'prefix_ADE_m':[float(err[:j+1].mean()) for j in idx],'endpoint_speed_MAE_mps':abs(pspeed-abs(ys[idx])).tolist(),'endpoint_signed_speed_MAE_mps':abs((0 if method=='stationary' else v)-ys[idx]).tolist()}
   rows.append({'file':name,'route':rn,'family':route['family'],'frame':i,'speed':v,'yawrate':omega,'static_current':abs(v)<.2,'hold_full5s':abs(v)<.2 and abs(ys).max()<.2,'deceleration':abs(v)>=.2 and abs(ys[-1])<abs(v)-1,'acceleration':abs(v)>=.2 and abs(ys[-1])>abs(v)+1,'cache_position_error_max_m':cacheerr,'predictions':metrics})
  print('ROUTE_DONE',rn,len(route['samples']),flush=True)
 groups={'all':rows,'static_current':[r for r in rows if r['static_current']],'hold_full5s':[r for r in rows if r['hold_full5s']],'deceleration':[r for r in rows if r['deceleration']],'acceleration':[r for r in rows if r['acceleration']]}
 groups.update({'route/'+r: [x for x in rows if x['route']==r] for r in sorted(set(x['route'] for x in rows))});groups.update({'family/'+r:[x for x in rows if x['family']==r] for r in sorted(set(x['family'] for x in rows))})
 result={'horizons_seconds':[1,2,3,5],'groups':{k:agg(v) for k,v in groups.items()},'cache_max_position_error_m':max(r['cache_position_error_max_m'] for r in rows),'units':{'reverse_frames':sum(r['reverse'] for r in units),'gyro_minus_past_pose_abs_p50_p95':np.quantile([abs(r['current_gyro_FLU']-r['past_pose_FLU']) for r in units],[.5,.95]).tolist(),'speed_minus_past_pose_abs_p50_p95':np.quantile([abs(r['speed_mps']-r['pose_past_speed_mps']) for r in units],[.5,.95]).tolist()},'limits':['No learning or officialmodel reproduction','Nominal10Hz, gyrobodyz vsplanaryaw crosscheck only, no exact timestampcertificate','OCC past-only unproven; zero unfilled NOTverifiedfree','42train9val existinglocalsubset not officialfullsplit','Validationwindows overlap;9routes not1124independentdrivingevents','Position/speed labelfit only;no collisions,TTC,closedloop or safetyclaims'],'wall_s':time.perf_counter()-start,'manifest_unchanged':sha(manifest)==before,'test_opened':False}
 assert len(rows)==1124 and result['manifest_unchanged'];(O/'rows.json').write_text(json.dumps(rows,default=lambda x:x.item()));(O/'units.json').write_text(json.dumps(units));(O/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps({'all':result['groups']['all'],'counts':{k:len(v) for k,v in groups.items() if '/' not in k},'units':result['units'],'wall_s':result['wall_s']},indent=2))
if __name__=='__main__':main()


