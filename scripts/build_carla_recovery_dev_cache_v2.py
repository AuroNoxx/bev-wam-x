from carla_miniloop_adapter import *
from carla_nav_contract import near_goal

def main():
 src=R/'data/carla_causal_miniloop_20261116_recovery_dev_v2';out=R/'data/carla_recovery_dev_cache_v2';manifest=json.loads((src/'manifest.json').read_text());assert manifest['status']=='completed';out.mkdir(exist_ok=False);report={'train':[],'dev':[],'excluded':[],'source':str(src),'limits':'Real recovery labels, semanticGT/poseoracle; unsafe/incomplete windows excluded with explicit counts, dev episode failure not erased.'}
 for epinfo in manifest['episodes']:
  ep=epinfo['episode'];split=epinfo['split'];root=src/f'episode{ep}';rows=json.loads((root/'frames.json').read_text());raw=[]
  for i in range(len(rows)):
   with np.load(root/f'{i:05d}.npz') as z:raw.append({k:z[k] for k in ['lidar_xyz','lidar_tag','lidar_world','ego_world']})
  folder=out/f'episode{ep}';folder.mkdir()
  for i in range(36,len(rows)-50):
   bad=[j for j in range(i-6,i+51) if rows[j].get('guard') or not rows[j].get('valid_for_training',False)]
   if bad:report['excluded'].append({'episode':ep,'anchor':i,'split':split,'reason':'source invalid/guard support','bad':bad});continue
   ids=[i-6,i-4,i-2,i];b=actor_inputs([raw[j] for j in ids],[rows[j] for j in ids]);T=np.linalg.inv(np.array(rows[i]['ego_world']));traj=[]
   for j in range(i+2,i+51,2):
    rel=T@np.array(rows[j]['ego_world']);yaw=-np.arctan2(rel[1,0],rel[0,0]);v=np.linalg.norm(rows[j]['velocity_world'][:2]);traj.append([rel[0,3]/50,-rel[1,3]/10,np.sin(yaw),np.cos(yaw),v/30])
   row={**rows[i],'navigation_world':rows[i]['controller_goal_previous_frame']};goal=near_goal(row)
   b.update(trajectory=np.array(traj,np.float32),future=np.stack([voxelize(raw[j],np.array(rows[i]['ego_world'])) for j in [i+10,i+20]]),near_goal_flu_normalized=goal,lane_error_right_m=np.float32(rows[i]['lane_error_right_m']),heading_error_deg=np.float32(rows[i]['heading_error_deg']));path=folder/f'{i:05d}.npz';np.savez_compressed(path,**b);report[split].append(str(path))
 report['source_episodes']=manifest['episodes'];report['counts']={k:len(report[k]) for k in ['train','dev','excluded']};(out/'manifest.json').write_text(json.dumps(report,indent=2));print(report['counts'])
if __name__=='__main__':main()
