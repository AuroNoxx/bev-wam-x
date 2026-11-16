"""Explicit current map-near-goal only. Does not fabricate far navigation or commands."""
import json,numpy as np
from pathlib import Path
from functools import lru_cache
R=Path(__file__).resolve().parents[1]
def near_goal(row):
 T=np.asarray(row['ego_world'],float);goal=np.asarray(row['navigation_world'],float);assert T.shape==(4,4) and goal.shape==(3,) and np.isfinite(T).all() and np.isfinite(goal).all();q=np.linalg.inv(T)@np.r_[goal,1.];assert np.linalg.norm(q[:2])<100,'invalid map goal distance';return np.array([q[0],-q[1]],np.float32)/50
@lru_cache(maxsize=32)
def raw_rows(root,episode):return json.loads((Path(root)/episode/'frames.json').read_text())
def cache_near(path):
 p=Path(path);folder=p.parent.parent
 with np.load(p) as z:
  if 'near_goal_flu_normalized' in z:return z['near_goal_flu_normalized'].copy()
 mapping={'carla_miniloop_cache_v1':'carla_causal_miniloop_20261116_town12_r2','carla_startstop_cache_v1':'carla_causal_miniloop_20261116_startstop_v1'}
 if folder.name not in mapping:raise ValueError('No verified navigation source '+str(folder))
 return near_goal(raw_rows(str(R/'data'/mapping[folder.name]),p.parent.name)[int(p.stem)])
if __name__=='__main__':
 assert np.allclose(near_goal({'ego_world':np.eye(4),'navigation_world':[10,-5,0]}),[.2,.1]);print('near nav axes/normalization pass')
