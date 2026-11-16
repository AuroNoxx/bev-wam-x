"""Matched episode cache and independent actor information audit. No training."""
from carla_reactive_hold_adapter import actor as old_actor, voxelize
from pathlib import Path
import numpy as np,json,copy,time,hashlib
R=Path(__file__).resolve().parents[1];SRC=R/'data/carla_causal_matched6_20261116_v1';OUT=R/'data/carla_matched6_cache_v1'
def actor(raws,history):
 h=[]
 for r in history:
  q={k:copy.deepcopy(r[k]) for k in ['timestamp','frame','teacher_observation_timestamp','teacher_observation_frame','velocity_world','ego_world','navigation_world']}
  q['teacher_inputs']=copy.deepcopy(r['command_current']);h.append(q)
 return old_actor([{k:z[k] for k in ['lidar_xyz','lidar_tag','lidar_world','ego_world']} for z in raws],h)
def main():
 OUT.mkdir(exist_ok=False);start=time.perf_counter();m=json.loads((SRC/'manifest.json').read_text());assert m['status']=='completed';result={'train':[],'dev':[],'episodes':[],'tests':{},'source':str(SRC),'source_manifest_sha256':hashlib.sha256((SRC/'manifest.json').read_bytes()).hexdigest(),'filtered':[],'limits':'4train2dev; train cruise8.33..8.76/dev7.887..7.894 confounded; semantic pose map oracle; no test/closedloop'}
 for ep in m['episodes']:
  e=ep['episode'];root=SRC/f'episode{e}';rr=json.loads((root/'frames.json').read_text());raw=[];naverr=[];command=rr[0]['command_current'];assert command['movement_permitted'] is True
  for i,r in enumerate(rr):
   with np.load(root/f'{i:05d}.npz') as z:raw.append({k:z[k].copy() for k in ['lidar_xyz','lidar_tag','lidar_world','ego_world']})
   assert r['frame']==r['rgb_frame']==r['lidar_frame'] and np.allclose(raw[-1]['ego_world'],r['ego_world'])
   assert r['teacher_observation_frame']==r['frame']-1 and abs(r['timestamp']-r['teacher_observation_timestamp']-.1)<1e-4
   assert r['command_current']==command and all(r['teacher_inputs'][k]==v for k,v in command.items())
   assert i==0 or (r['frame']==rr[i-1]['frame']+1 and abs(r['timestamp']-rr[i-1]['timestamp']-.1)<1e-4)
   local=np.linalg.inv(np.array(r['ego_world']))@np.r_[r['navigation_world'],1];naverr.append(float(np.max(abs(local[:3]-r['near_goal_local_carla']))))
  assert max(naverr)<.002
  folder=OUT/f'episode{e}';folder.mkdir();unknown=[];names=[]
  for i in range(6,len(rr)-50):
   span=rr[i-6:i+51]
   if any(not x['valid_for_training'] or x['guard'] or x['safety_takeover'] for x in span):result['filtered'].append({'ep':e,'i':i});continue
   ids=[i-6,i-4,i-2,i];b=actor([raw[j] for j in ids],rr[i-6:i+1]);T=np.linalg.inv(np.array(rr[i]['ego_world']));traj=[]
   for j in range(i+2,i+51,2):
    rel=T@np.array(rr[j]['ego_world']);yaw=-np.arctan2(rel[1,0],rel[0,0]);v=np.linalg.norm(rr[j]['velocity_world'][:2]);traj.append([rel[0,3]/50,-rel[1,3]/10,np.sin(yaw),np.cos(yaw),v/30])
   if i in [6,100,340,549]:
    poisoned=copy.deepcopy(rr[i-6:i+1])
    for row in poisoned:
     for key in list(row):
      if key not in ['timestamp','frame','teacher_observation_timestamp','teacher_observation_frame','velocity_world','ego_world','navigation_world','command_current']:row[key]='FORBIDDEN'
    p=actor([raw[j] for j in ids],poisoned);assert all(np.array_equal(b[k],p[k]) for k in b)
    result['tests'][f'ep{e}_i{i}_whitelist_poison_exact']=True
   assert set(np.unique(b['history'])).issubset(set(range(1,8))|{255});unknown.append(float(np.mean(b['history']==255)))
   b.update(trajectory=np.array(traj,np.float32),future=np.stack([voxelize(raw[j],np.array(rr[i]['ego_world'])) for j in [i+10,i+20]]));p=folder/f'{i:05d}.npz';np.savez_compressed(p,**b);names.append(str(p));result[ep['split']].append(str(p))
  result['episodes'].append({'episode':e,'split':ep['split'],'windows':len(names),'cruise_mps':command['cruise_mps'],'nav_error_max_m':max(naverr),'unknown_fraction_mean':float(np.mean(unknown))});print('EP_DONE',result['episodes'][-1],flush=True)
 assert not set(result['train'])&set(result['dev']);result['counts']={k:len(result[k]) for k in ['train','dev']};assert result['counts']=={'train':2176,'dev':1088};assert not result['filtered'];result['tests'].update(sync=True,commands_constant=True,nav_current_pose=True,episode_isolation=True);result['wall_s']=time.perf_counter()-start;(OUT/'manifest.json').write_text(json.dumps(result,indent=2));print('DONE',result['counts'],result['wall_s'],flush=True)
if __name__=='__main__':main()
