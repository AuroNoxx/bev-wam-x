"""Same-frame semantic ray-hit adapter; simulator semantics and ego pose are oracle."""
import numpy as np,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def voxelize(raw,current_world):
 xyz=np.asarray(raw['lidar_xyz']);T=np.asarray(raw['lidar_world']);source_ego=np.asarray(raw['ego_world']);pts=np.column_stack([xyz,np.ones(len(xyz))]);own=pts@(np.linalg.inv(source_ego)@T).T
 keep=~((abs(own[:,0])<3)&(abs(own[:,1])<1.4)&(own[:,2]>-1)&(own[:,2]<2.8))
 p=pts@(np.linalg.inv(current_world)@T).T;p[:,1]*=-1;idx=np.floor((p[:,:3]-[-25.6,-25.6,-5])/[.8,.8,.4]).astype(int);ok=keep&(idx[:,0]>=0)&(idx[:,0]<160)&(idx[:,1]>=0)&(idx[:,1]<64)&(idx[:,2]>=0)&(idx[:,2]<20);tags=np.asarray(raw['lidar_tag'])[ok];idx=idx[ok];cls=np.full(len(tags),3,np.uint8)
 for c,ids in [(1,[1,25,27]),(2,[9,10]),(4,[14,15,16,17,18,19,21]),(5,[12,13]),(6,[24]),(7,[2,28])]:cls[np.isin(tags,ids)]=c
 g=np.full((20,64,160),255,np.uint8)
 for c in [1,2,3,6,7,4,5]:
  q=idx[cls==c];g[q[:,2],q[:,1],q[:,0]]=c
 return g
def actor_inputs(raws,rows):
 assert len(raws)==len(rows)==4;times=np.array([r['timestamp'] for r in rows]);assert np.allclose(times[-1]-times,[.6,.4,.2,0],atol=1e-3)
 current=np.array(rows[-1]['ego_world']);state=np.zeros(20,np.float32);state[0]=np.linalg.norm(rows[-1]['velocity_world'][:2])/30
 return dict(history=np.stack([voxelize(z,current) for z in raws]),state=state,history_valid=np.ones(4,bool),history_age_seconds=(times[-1]-times).astype(np.float32))
def main():
 src=R/'data/carla_causal_miniloop_20261116_startstop_v1';out=R/'data/carla_startstop_cache_v1';out.mkdir(exist_ok=False);report={'train':[],'dev':[],'limits':'Semantic ray-hit GT and ego pose GT; no free. State speed-only for both data sources, navigation/acceleration withheld. First3s excluded to avoid spawning transient. New vs old occupancy generation differs, not pure-data-quality causal experiment.'}
 for ep in range(3):
  root=src/f'episode{ep}';rows=json.loads((root/'frames.json').read_text());raw=[]
  for i in range(len(rows)):
   with np.load(root/f'{i:05d}.npz') as z:raw.append({k:z[k] for k in ['lidar_xyz','lidar_tag','lidar_world','ego_world']})
  folder=out/f'episode{ep}';folder.mkdir();split='train' if ep<2 else 'dev'
  for i in range(36,len(rows)-50):
   ids=[i-6,i-4,i-2,i];b=actor_inputs([raw[j] for j in ids],[rows[j] for j in ids]);T=np.linalg.inv(np.array(rows[i]['ego_world']));traj=[]
   for j in range(i+2,i+51,2):
    rel=T@np.array(rows[j]['ego_world']);yaw=-np.arctan2(rel[1,0],rel[0,0]);v=np.linalg.norm(rows[j]['velocity_world'][:2]);traj.append([rel[0,3]/50,-rel[1,3]/10,np.sin(yaw),np.cos(yaw),v/30])
   b.update(trajectory=np.array(traj,np.float32),future=np.stack([voxelize(raw[j],np.array(rows[i]['ego_world'])) for j in [i+10,i+20]]));path=folder/f'{i:05d}.npz';np.savez_compressed(path,**b);report[split].append(str(path))
 report['counts']={k:len(report[k]) for k in ['train','dev']};(out/'manifest.json').write_text(json.dumps(report,indent=2));print(report['counts'])
if __name__=='__main__':main()
