"""Pure offline contract for a FUTURE learned-policy CARLA bridge. Never actuates."""
import numpy as np
ACTOR_KEYS={'history','state','history_valid','history_age_seconds','geometry_valid'}
SHAPE=(4,20,64,160)
SCALE=np.array([50.,10.,1.,1.,30.])
def validate_actor_payload(payload):
 if set(payload)!=ACTOR_KEYS:raise ValueError('Actor payload must contain only the five causal input fields')
 h=np.asarray(payload['history']);s=np.asarray(payload['state']);valid=np.asarray(payload['history_valid']);age=np.asarray(payload['history_age_seconds']);geometry=np.asarray(payload['geometry_valid'])
 if h.shape!=SHAPE or geometry.shape!=SHAPE:raise ValueError('Expected TZYX=(4,20,64,160); no implicit XYZ regridding')
 if s.shape!=(20,) or not np.isfinite(s).all():raise ValueError('Expected finite normalized20-state vector')
 if valid.shape!=(4,) or valid.dtype!=np.bool_ or geometry.dtype!=np.bool_:raise ValueError('Validity masks must be explicit booleans')
 if not valid[-1]:raise ValueError('Current observation unavailable')
 if age.shape!=(4,) or not np.isfinite(age).all() or (age<0).any() or (age[1:]>age[:-1]).any() or abs(age[-1])>1e-6:raise ValueError('Noncausal/current-time age sequence')
 if not np.isin(h,[0,1,2,3,4,5,6,7,255]).all():raise ValueError('Unknown semantic encoding')
 canonical=np.where(geometry&valid[:,None,None,None]&(h>0)&(h<8),h,255).astype(np.uint8)
 return {'canonical_history':canonical,'observed_occupied_fraction':float((canonical!=255).mean()),'free_space_certified':False,'source_causality_certified':None}
def validate_timing(frame_ids,timestamps_s,current_time_s):
 ids=np.asarray(frame_ids);t=np.asarray(timestamps_s,dtype=float)
 if ids.shape!=(4,) or t.shape!=(4,) or not np.isfinite(t).all() or not np.isfinite(current_time_s):raise ValueError('Four finite timestamped observations required')
 if (np.diff(ids)<=0).any() or (np.diff(t)<=0).any():raise ValueError('Duplicate/reordered observation')
 if (t>current_time_s+1e-6).any() or abs(t[-1]-current_time_s)>1e-6:raise ValueError('Future or stale current observation')
 return current_time_s-t

def candidate_to_physical(candidate):
 p=np.asarray(candidate,dtype=float)
 if p.shape!=(25,5) or not np.isfinite(p).all():raise ValueError('Expected finite25x5 normalized candidate')
 q=p*SCALE;n=np.linalg.norm(q[:,2:4],axis=1)
 if (n<1e-6).any() or np.max(abs(n-1))>1e-3:raise ValueError('Heading sin/cos must already be normalized; no silent repair')
 if (q[:,4]<0).any():raise ValueError('Negative predicted speed; do not silently clamp at bridge')
 return {'trajectory_flu_m_sin_cos_mps':q,'time_s':np.arange(1,26)*.2,'frame':'current_ego_FLU','collision':None,'TTC':None,'road_compliance':None,'driving_pass':None,'execution_authorized':False}
