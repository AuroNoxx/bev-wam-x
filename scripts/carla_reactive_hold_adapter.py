"""Causal hold-v3 adapter. Teacher privileged fields are never actor input."""
from carla_miniloop_adapter import voxelize
from carla_nav_contract import near_goal
from pathlib import Path
import numpy as np,json,copy,time
R=Path(__file__).resolve().parents[1];SRC=R/'data/carla_causal_reactive_20261116_hold_v3';OUT=R/'data/carla_reactive_hold_cache_v1'
def actor(raws,history):
 assert len(history)==7 and len(raws)==4
 times=np.array([r['timestamp'] for r in history]);assert np.allclose(np.diff(times),.1,atol=1e-4)
 current=history[-1];age=current['timestamp']-current['teacher_observation_timestamp'];assert abs(age-.1)<1e-4 and current['teacher_observation_frame']==current['frame']-1
 v=np.array([np.linalg.norm(r['velocity_world'][:2]) for r in history]);s=np.zeros(20,np.float32);s[0]=v[-1]/30;s[1]=current['teacher_inputs']['cruise_mps']/30;s[2]=float(current['teacher_inputs']['movement_permitted']);s[3]=(v[-1]-v[0])/(times[-1]-times[0])/5;s[4:6]=near_goal(current);s[6]=1
 return dict(history=np.stack([voxelize(z,np.array(current['ego_world'])) for z in raws]),state=s,history_valid=np.ones(4,bool),history_age_seconds=(times[-1]-times[[0,2,4,6]]).astype(np.float32))
def main():
 OUT.mkdir(exist_ok=False);start=time.perf_counter();manifest=json.loads((SRC/'manifest.json').read_text());assert manifest['status']=='completed';result={'train':[],'dev':[],'source':str(SRC),'state_contract':{'0':'current horizontal speed/30','1':'last issued cruise/30','2':'last issued movement permission','3':'past .6sec acceleration/5','4:6':'current near nav FLU/50','6':'history acceleration valid'},'command_age_s':.1,'unknown':255,'limits':'oracle semantic rays/ego/HDmap; teacher lead GT excluded; only1train1dev; cruise8 vs8.5 and road confounded','tests':{},'filtered':[]}
 unknown=[]
 for ep in manifest['episodes']:
  e=ep['episode'];root=SRC/f'episode{e}';rr=json.loads((root/'frames.json').read_text());raw=[]
  for i,r in enumerate(rr):
   with np.load(root/f'{i:05d}.npz') as z: raw.append({k:z[k].copy() for k in ['lidar_xyz','lidar_tag','lidar_world','ego_world']})
   assert r['frame']==r['rgb_frame']==r['lidar_frame'];assert np.allclose(raw[-1]['ego_world'],r['ego_world']);assert i==0 or r['frame']==rr[i-1]['frame']+1
  folder=OUT/f'episode{e}';folder.mkdir()
  for i in range(36,len(rr)-50):
   span=rr[i-6:i+51]
   if any(not r['valid_for_training'] or r['guard'] or r['safety_takeover'] for r in span):result['filtered'].append({'ep':e,'index':i,'reason':'invalid/guard/takeover in history or supervision interval'});continue
   ids=[i-6,i-4,i-2,i];b=actor([raw[j] for j in ids],rr[i-6:i+1]);T=np.linalg.inv(np.array(rr[i]['ego_world']));traj=[]
   for j in range(i+2,i+51,2):
    rel=T@np.array(rr[j]['ego_world']);yaw=-np.arctan2(rel[1,0],rel[0,0]);v=np.linalg.norm(rr[j]['velocity_world'][:2]);traj.append([rel[0,3]/50,-rel[1,3]/10,np.sin(yaw),np.cos(yaw),v/30])
   if i==100:
    poisoned=copy.deepcopy(rr[i-6:i+1])
    for row in poisoned:
     for key in ['gap_m','lead_forward_speed_mps','closing_mps']:row['teacher_inputs'][key]=1e9
     for key in ['target_speed','phase','teacher_hold','lead_target_eval_only','lead_actual_post_eval_only']:row[key]='FORBIDDEN'
    p=actor([raw[j] for j in ids],poisoned);assert all(np.array_equal(b[k],p[k]) for k in b)
    result['tests'][f'ep{e}_teacher_field_poison_exact']=True
    assert set(np.unique(b['history'])).issubset(set(range(1,8))|{255})
   unknown.append(float(np.mean(b['history']==255)));b.update(trajectory=np.array(traj,np.float32),future=np.stack([voxelize(raw[j],np.array(rr[i]['ego_world'])) for j in [i+10,i+20]]));p=folder/f'{i:05d}.npz';np.savez_compressed(p,**b);result[ep['split']].append(str(p))
  print('EP_DONE',e,flush=True)
 result['counts']={k:len(result[k]) for k in ['train','dev']};assert result['counts']=={'train':514,'dev':514};result['tests'].update(sync_pose_checked=True,causal_history_only=True,teacher_privileged_excluded=True,unknown_not_free=True);result['unknown_fraction']={'min':min(unknown),'mean':float(np.mean(unknown)),'max':max(unknown)};result['wall_s']=time.perf_counter()-start;(OUT/'manifest.json').write_text(json.dumps(result,indent=2));print('DONE',result['counts'],result['unknown_fraction'],result['wall_s'])
if __name__=='__main__':main()
