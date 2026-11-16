"""Original semantic-LiDAR to OCC frame matching, validation only; no cache mutation."""
from pathlib import Path
import json,gzip,tarfile,time,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/diagnostics/frame-alignment'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--train-probe',action='store_true');args=parser.parse_args()
 started=time.perf_counter();manifest=json.loads((ROOT/'data/occ_v1/manifest.json').read_text());old=json.loads((ROOT/'artifacts/diagnostics/20261115-contract/contract-probes.json').read_text());results=[];bounds=[]
 selected=[r for r in manifest['routes'] if r['split']=='validation']
 if args.train_probe:
  selected=[]
  for family in sorted(set(r['family'] for r in manifest['routes'])):
   selected.append(next(r for r in manifest['routes'] if r['split']=='train' and r['family']==family))
 for route in selected:
  name=route['route'];frames=sorted(set(x['frame'] for x in old['alignment'] if x['route']==name));dest=OUT/'raw'/name;dest.mkdir(parents=True,exist_ok=True)
  if args.train_probe:
   limit=max(int(p.stem) for p in (ROOT/'data/public/bench2drive/clips'/name/'occ').glob('*.npz'))
   frames=sorted(set(np.linspace(3,limit-3,12).astype(int).tolist()))
  if not all((dest/f'{i:05}.npz').exists() for i in frames):
   with tarfile.open(ROOT/'data/public/bench2drive/archives'/route['file'],'r|gz') as tar:
    for member in tar:
     parts=Path(member.name).parts
     if member.isfile() and len(parts)>1 and parts[-2]=='lidar_seg' and int(Path(parts[-1]).stem) in frames:
      (dest/parts[-1]).write_bytes(tar.extractfile(member).read())
  clip=ROOT/'data/public/bench2drive/clips'/name
  for i in frames:
   a=json.load(gzip.open(clip/'anno'/f'{i:05}.json.gz','rt'));eg=next(b for b in a['bounding_boxes'] if b['class']=='ego_vehicle')
   with np.load(dest/f'{i:05}.npz',allow_pickle=False) as f:p=f['xyz'];tags=f['obj_tag'];ids=f['obj_idx']
   # Native saved point coordinates are tested as ego coordinates; independent actor-ID bbox containment below.
   valid=(p[:,0]>=-25.6)&(p[:,0]<25.6)&(p[:,1]>=-25.6)&(p[:,1]<25.6)&(p[:,2]>=-5)&(p[:,2]<3)&(ids!=int(eg['id']))
   p,tags,ids=p[valid],tags[valid],ids[valid];co=np.floor((p-[-25.6,-25.6,-5]-1e-5)/.4).astype(int);co=np.clip(co,[0,0,0],[127,127,19])
   for d in [-2,-1,0,1,2]:
    path=clip/'occ'/f'{i+d:05}.npz'
    if not path.exists():continue
    with np.load(path,allow_pickle=False) as f:o=f['occ'];seg=f['seg']
    match=(o[co[:,1],co[:,0],co[:,2]]==1)&(seg[co[:,1],co[:,0],co[:,2]]==tags)
    occupied=o[co[:,1],co[:,0],co[:,2]]==1
    # Unique voxel+semantic rows remove dense LiDAR angular sampling weights.
    _,idx=np.unique(np.c_[co,tags],axis=0,return_index=True)
    for kind,mask in [('vehicles',np.isin(tags,[14,15,16,17,18,19])),('static_pole_sign',np.isin(tags,[6,8])),('all',np.ones(len(p),bool))]:
     choose=idx[mask[idx]]
     if len(choose):results.append(dict(route=name,family=route['family'],frame=i,occ_offset=d,kind=kind,unique_voxels=len(choose),semantic_hits=int(match[choose].sum()),occupied_hits=int(occupied[choose].sum()),semantic_recall=float(match[choose].mean()),ego_speed=a['speed']))
   # Match raw sensor point actor IDs to annotations i+d, exact oriented 3D actor transform + bbox center.
   for d in [-2,-1,0,1,2]:
    path=clip/'anno'/f'{i+d:05}.json.gz'
    if not path.exists():continue
    ann=json.load(gzip.open(path,'rt'));ego=next(b for b in ann['bounding_boxes'] if b['class']=='ego_vehicle');e2w=np.linalg.inv(ego['world2ego']);world=np.c_[p,np.ones(len(p))]@e2w.T;hits=0;count=0
    for b in ann['bounding_boxes']:
     if b['class']!='vehicle' or 'world2vehicle' not in b:continue
     mask=(ids==int(b['id']))&np.isin(tags,[14,15,16,17,18,19]);q=world[mask]@np.asarray(b['world2vehicle']).T
     if not len(q):continue
     good=(np.abs(q[:,:3]-np.asarray(b['bbx_loc']))<=np.asarray(b['extent'])+.1).all(1);hits+=int(good.sum());count+=len(q)
    bounds.append(dict(route=name,frame=i,anno_offset=d,points=count,hits=hits,recall=hits/count if count else None))
  print('DONE ROUTE',name,flush=True)
 report=dict(offset_definition='sensor lidar_seg i versus OCC i+offset; bbox uses annotation i+offset',voxel_rows=results,bbox_rows=bounds,seconds=time.perf_counter()-started,limitations=['Raw lidar may share preprocessing with OCC; not timestamp proof','Point recall not dense IoU; static fusion may favor older points','Validation 108 frames or train 72 frames; no test read'])
 (OUT/('raw-lidar-train-alignment.json' if args.train_probe else 'raw-lidar-alignment.json')).write_text(json.dumps(report,indent=2));print('SECONDS',report['seconds'])
if __name__=='__main__':main()
