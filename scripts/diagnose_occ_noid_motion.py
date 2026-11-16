"""No-ID final OCC geometric motion diagnostic. GT is evaluated separately."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from carla_miniloop_adapter import voxelize
from carla_nav_contract import near_goal
R=Path(__file__).resolve().parents[1];OUT=R/'artifacts/occ-noid-motion-diagnostic';RES=np.array([.8,.8,.4]);LO=np.array([-25.6,-25.6,-5.])
def components(grid,nav):
 xy=np.argwhere((grid==4).any(0));axis=np.asarray(nav,float);axis/=np.linalg.norm(axis);lateral=np.array([-axis[1],axis[0]]);pos=np.c_[LO[0]+(xy[:,1]+.5)*.8,LO[1]+(xy[:,0]+.5)*.8];good=(pos@axis>3)&(pos@axis<60)&(abs(pos@lateral)<1.75);cells=set(map(tuple,xy[good].tolist()));cs=[]
 while cells:
  start=cells.pop();stack=[start];group=[]
  while stack:
   p=stack.pop();group.append(p)
   for dy in [-1,0,1]:
    for dx in [-1,0,1]:
     q=(p[0]+dy,p[1]+dx)
     if q in cells:cells.remove(q);stack.append(q)
  if len(group)<2:continue
  group=np.array(group);pp=np.c_[LO[0]+(group[:,1]+.5)*.8,LO[1]+(group[:,0]+.5)*.8];cs.append({'center':pp.mean(0),'min_s':float((pp@axis).min()),'cells':group})
 return cs,axis
def readout(history,nav):
 sets=[]
 for g in history:
  cs,axis=components(g,nav);sets.append(cs)
 if any(not c for c in sets):return {'valid':False,'reason':'no_component_in_one_or_more_frames'}
 current=sorted(sets[-1],key=lambda c:c['min_s'])
 if len(current)>1 and current[1]['min_s']-current[0]['min_s']<.8:return {'valid':False,'reason':'ambiguous_current_nearest'}
 selected=[current[0]]
 for cs in reversed(sets[:-1]):
  cand=sorted([(float(np.linalg.norm(c['center']-selected[-1]['center'])),k) for k,c in enumerate(cs)])
  if cand[0][0]>4.6:return {'valid':False,'reason':'association_distance_exceeded'}
  if len(cand)>1 and cand[1][0]-cand[0][0]<.8:return {'valid':False,'reason':'ambiguous_history_association'}
  selected.append(cs[cand[0][1]])
 selected=selected[::-1];centers=np.array([c['center'] for c in selected]);times=np.array([-.6,-.4,-.2,0]);velocity=((times-times.mean())[:,None]*(centers-centers.mean(0))).sum(0)/((times-times.mean())**2).sum();return {'valid':True,'velocity_xy':velocity,'displacement_s':float(velocity@axis*.6),'selected':selected,'axis':axis}
def serial(result):return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items() if k not in ['selected','axis']}
def loadraw(root,ep,i,extras=False):
 with np.load(root/f'episode{ep}/{i:05d}.npz') as z:return {k:z[k].copy() for k in (['lidar_xyz','lidar_tag','lidar_world','ego_world']+(['lidar_object_idx'] if extras else []))}
def process(split):
 audit=split=='audit';src=R/'data'/('carla_causal_instance_audit_20261116_v1' if audit else 'carla_causal_reactive_20261116_hold_v3');ep=0 if split!='dev' else 1;rr=json.loads((src/f'episode{ep}/frames.json').read_text());rows=[]
 for i in range(36,450 if audit else 550):
  ids=[i-6,i-4,i-2,i];nav=near_goal(rr[i])*50;cur=np.array(rr[i]['ego_world']);raws=[loadraw(src,ep,j) for j in ids]
  if audit:h=np.stack([voxelize(z,cur) for z in raws])
  else:
   with np.load(R/f'data/carla_reactive_hold_cache_v1/episode{ep}/{i:05d}.npz') as z:h=z['history'].copy()
  estimate=readout(h,nav);entry={'index':i,**serial(estimate)};axis=nav/np.linalg.norm(nav)
  # Evaluation-only GT: no fields below are arguments to readout.
  if audit:
   target=rr[i]['teacher_target_post_audit'];old=rr[i-6]['teacher_target_post_audit'];T=np.linalg.inv(cur);gt=(T@np.r_[np.array(target['world'])[:3,3],1])[:2]-(T@np.r_[np.array(old['world'])[:3,3],1])[:2];gt[1]*=-1;truth=float(gt@axis/.6);entry['reference']='true same-current-frame past .6 displacement from audit lead pose';entry['current_gt_speed']=float(np.linalg.norm(target['velocity_world'][:2]))
   if estimate['valid']:
    fractions=[]
    for k,j in enumerate(ids):
     z=loadraw(src,ep,j,True);target_id=rr[j]['teacher_target_post_audit']['actor_id'];keep=z['lidar_object_idx']==target_id;leadraw={**z,'lidar_xyz':z['lidar_xyz'][keep],'lidar_tag':z['lidar_tag'][keep]};leadgrid=voxelize(leadraw,cur);cells=estimate['selected'][k]['cells'];truthbev=(leadgrid==4).any(0);fractions.append(float(np.mean(truthbev[cells[:,0],cells[:,1]])))
    entry['audit_selected_cells_target_fraction']=fractions
  else:
   vals=np.array([r['lead_actual_post_eval_only']['lead_forward_speed_mps'] for r in rr[i-6:i+1]]);truth=float(np.trapezoid(vals,dx=.1)/.6);entry['reference']='trapezoid past .6 post lead-forward speed, approximate ego-heading alignment';entry['current_gt_speed']=float(vals[-1])
  entry['gt_past_mean_speed']=truth;entry['gt_static_history']=all(abs(r['lead_actual_post_eval_only']['lead_forward_speed_mps'])<.05 for r in rr[i-6:i+1]);rows.append(entry)
 print('READ_DONE',split,flush=True);return rows

def main():
 OUT.mkdir(exist_ok=False);start=time.perf_counter();design={'geometry_preset_before_dev':'BEV8connected class4 >=2 cells; nav corridor3..60m abs lateral<1.75m; nearest candidate ambiguous if .8m; backwards centroid association <=4.6m per.2s, second-best ambiguity margin.8m','motion':'least squares 4 centers at[-.6,-.4,-.2,0], no GT ID or lead pose/gap','decision_threshold':'train stationary p95 absolute apparent .6s displacement; frozen before dev/audit','limits':'single lead corridor, no occlusion validation; repeated dev and audit share road/seed, independent recording not independent scenario; no control'};(OUT/'design.json').write_text(json.dumps(design,indent=2));train=process('train');noise=[abs(r['displacement_s']) for r in train if r['valid'] and r['gt_static_history']];threshold=float(np.percentile(noise,95));(OUT/'calibration.json').write_text(json.dumps({'threshold_displacement_m':threshold,'train_static_n':len(noise),'frozen':True},indent=2));allrows={'train':train,'dev':process('dev'),'audit':process('audit')};summary={}
 for split,rr in allrows.items():
  summary[split]={}
  for group,lo,hi in [('all',-1e9,1e9),('static',-.05,.05),('slow',.05,.5),('medium',.5,2),('moving',2,1e9)]:
   rs=[r for r in rr if lo<=r['gt_past_mean_speed']<hi];ok=[r for r in rs if r['valid']];errors=[r['displacement_s']/.6-r['gt_past_mean_speed'] for r in ok];summary[split][group]={'n':len(rs),'valid_n':len(ok),'unknown_n':len(rs)-len(ok),'speed_MAE_mps':float(np.mean(np.abs(errors))) if ok else None,'speed_abs_error_p95':float(np.percentile(np.abs(errors),95)) if ok else None,'zero_motion_MAE':float(np.mean([abs(r['gt_past_mean_speed']) for r in ok])) if ok else None,'motion_detect_fraction':float(np.mean([abs(r['displacement_s'])>threshold for r in ok])) if ok else None}
  if split=='audit':summary[split]['target_association']={'audited_valid_frames':sum(r['valid'] for r in rr),'fourframes_all_selected_cells_target':sum(r['valid'] and min(r['audit_selected_cells_target_fraction'])==1 for r in rr),'min_selected_cell_fraction':min(min(r['audit_selected_cells_target_fraction']) for r in rr if r['valid'])}
  # Current speed onset defines posthoc event; decision only reads past/current OCC.
  events=[]
  for i in range(1,len(rr)):
   if rr[i-1]['current_gt_speed']<.2<=rr[i]['current_gt_speed']:
    event=rr[i]['index'];found=next((r['index'] for r in rr[i:i+31] if r['valid'] and abs(r['displacement_s'])>threshold),None);events.append({'gt_onset_index':event,'first_detection_index':found,'delay_s':None if found is None else (found-event)*.1})
  summary[split]['onsets']=events
 (OUT/'rows.json').write_text(json.dumps(allrows));(OUT/'summary.json').write_text(json.dumps({'threshold':threshold,'groups':summary,'wall_s':time.perf_counter()-start},indent=2));print('DONE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
