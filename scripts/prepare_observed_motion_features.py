"""Derive actor features only from cache history/state/navigation."""
import json,numpy as np,hashlib,time
from pathlib import Path
from diagnose_occ_noid_motion import readout
R=Path(__file__).resolve().parents[1];OUT=R/'artifacts/carla-observed-motion-paired512'
def encode(history,state,threshold):
 nav=state[4:6]*50;r=readout(history,nav)
 if not r['valid']:return np.array([0,0,0,0,1],np.float32),r['reason']
 lead=float(r['velocity_xy']@r['axis']);ego_along=float(state[0]*30*r['axis'][0]);surface=r['selected'][-1]['min_s'];return np.array([surface/50,lead/15,(ego_along-lead)/15,1,float(abs(r['displacement_s'])<=threshold)],np.float32),None

def main():
 OUT.mkdir(exist_ok=False);d=json.loads((R/'data/carla_reactive_hold_cache_v1/manifest.json').read_text());threshold=json.loads((R/'artifacts/occ-noid-motion-diagnostic/calibration.json').read_text())['threshold_displacement_m'];features={};fail={};start=time.perf_counter()
 for split in ['train','dev']:
  counts={}
  for name in d[split]:
   with np.load(name) as z:h=z['history'].copy();s=z['state'].copy()
   f,reason=encode(h,s,threshold);features[name]=f.tolist()
   if reason:counts[reason]=counts.get(reason,0)+1
   assert np.isfinite(f).all() and f[3] in [0,1] and f[4] in [0,1]
  fail[split]=counts
 # Sensor/geometry-only API has no path for teacher/future labels.
 empty=np.full_like(h,255);e,_=encode(empty,s,threshold);assert np.array_equal(e,[0,0,0,0,1])
 (OUT/'features.json').write_text(json.dumps(features));(OUT/'feature_contract.json').write_text(json.dumps({'fields':['nearest_component_front_surface_from_ego_origin_m/50','past0.6_mean_lead_velocity_along_navigation_mps/15','ego_minus_lead_closing_mps/15','association_valid','low_motion_uncertain'],'uncertainty_threshold_train_only_m':threshold,'missing':[0,0,0,0,1],'origin_distance_not_bumper_gap':True,'ego_velocity_projection_assumption':'current scalar horizontal speed aligned with ego x, projected on near-navigation axis; no lead pose or vector truth','history':'four actor grids already registered current ego, no future','unknown_counts':fail,'future_GT_excluded_by_API':True,'empty_unknown_test_pass':True,'wall_s':time.perf_counter()-start},indent=2));print('DONE',fail)
if __name__=='__main__':main()
