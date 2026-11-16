"""Read-only source audit: vehicle-ray proxies are NOT instance lead visibility."""
import json,numpy as np,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];SRC=R/'data/carla_causal_reactive_20261116_hold_v3';OUT=R/'artifacts/reactive-observer-visibility-audit';OUT.mkdir(exist_ok=False)
rowsout=[];start=time.perf_counter()
for ep in [0,1]:
 rr=json.loads((SRC/f'episode{ep}/frames.json').read_text())
 for i,r in enumerate(rr):
  with np.load(SRC/f'episode{ep}/{i:05d}.npz') as z:
   assert set(z.files)=={'rgb_bgra','lidar_xyz','lidar_tag','lidar_world','ego_world'}
   xyz=z['lidar_xyz'];tag=z['lidar_tag'];p=np.c_[xyz,np.ones(len(xyz))]@(np.linalg.inv(z['ego_world'])@z['lidar_world']).T
  vehicle=np.isin(tag,[14,15,16,17,18,19,21]);own=(abs(p[:,0])<3)&(abs(p[:,1])<1.4)&(p[:,2]>-1)&(p[:,2]<2.8);vehicle&=~own
  forward=vehicle&(p[:,0]>3)&(p[:,0]<60)&(abs(p[:,1])<1.75)
  q=p[forward];ix=np.floor((q[:,:3]-[-25.6,-25.6,-5])/[.8,.8,.4]).astype(int);valid=(ix[:,0]>=0)&(ix[:,0]<160)&(ix[:,1]>=0)&(ix[:,1]<64)&(ix[:,2]>=0)&(ix[:,2]<20)
  occupied=len(np.unique(ix[valid],axis=0));v=float(np.linalg.norm(r['velocity_world'][:2]));gap=r['lead_actual_post_eval_only']['gap_m']
  rowsout.append({'ep':ep,'index':i,'speed':v,'gap_eval_only':gap,'teacher_hold_eval_only':r['teacher_hold'],'vehicle_rays_after_ego_filter':int(vehicle.sum()),'forward_corridor_vehicle_rays':int(forward.sum()),'forward_corridor_vehicle_unique_voxels':occupied,'proxy_nearest_x':float(q[:,0].min()) if len(q) else None})
def stats(rs):
 return {'n':len(rs),'vehicle_ray_nonempty_fraction':float(np.mean([x['vehicle_rays_after_ego_filter']>0 for x in rs])),'forward_corridor_nonempty_fraction':float(np.mean([x['forward_corridor_vehicle_rays']>0 for x in rs])),'forward_corridor_rays_min_median_max':np.percentile([x['forward_corridor_vehicle_rays'] for x in rs],[0,50,100]).tolist(),'forward_unique_voxels_min_median_max':np.percentile([x['forward_corridor_vehicle_unique_voxels'] for x in rs],[0,50,100]).tolist()} if rs else {'n':0}
summary={}
for ep in [0,1]:
 a=[x for x in rowsout if x['ep']==ep and 36<=x['index']<550];summary[str(ep)]={'all_actor_anchors':stats(a),'teacher_hold':stats([x for x in a if x['teacher_hold_eval_only']]),'moving':stats([x for x in a if x['speed']>=.2]),'static':stats([x for x in a if x['speed']<.2])}
r={'summary':summary,'actual_lead_instance_visibility':None,'reason':'NPZ discards LiDAR object_idx; no recorded lead instance ID/pose/bbox, cannot assign vehicle rays to actual teacher lead. Corridor proxy only; no certification.','proxy_definition':'same-frame semantic vehicle tags, existing ego-removal box, x in(3,60)m abs(CARLA ego y)<1.75m; voxel size .8/.8/.4m','source_oracle':'semantic class and ego pose truth; not learned detection','rows':rowsout,'wall_s':time.perf_counter()-start};(OUT/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
