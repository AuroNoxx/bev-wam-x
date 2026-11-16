"""Sidecar correction probe only. Never changes formal caches/manifests/checkpoints."""
from pathlib import Path
import gzip,json,hashlib
import numpy as np
from dataset_v1 import compact,warp,CFG
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/diagnostics/frame-alignment'
def main():
 m=json.loads((ROOT/'data/occ_v1/manifest.json').read_text());rows=[]
 for route in [r for r in m['routes'] if r['split']=='validation']:
  name=next(n for n in m['splits']['validation'] if route['route'] in n);path=ROOT/'data/occ_v1'/name;i=int(Path(name).stem);clip=ROOT/'data/public/bench2drive/clips'/route['route'];original_sha=hashlib.sha256(path.read_bytes()).hexdigest()
  with np.load(path) as f:original={k:f[k].copy() for k in f.files}
  def pose(j):
   a=json.load(gzip.open(clip/'anno'/f'{j:05}.json.gz','rt'));return np.asarray(next(b for b in a['bounding_boxes'] if b['class']=='ego_vehicle')['world2ego'])
  current=np.linalg.inv(pose(i))
  def aligned(offsets):
   grids=[]
   for h in offsets:
    with np.load(clip/'occ'/f'{i+h-1:05}.npz') as f:g=compact(f['occ'],f['seg'])
    grids.append(warp(g,pose(i+h),current))
   return np.stack(grids)
  changed={**original,'history':aligned(CFG['history_offsets']),'future':aligned(CFG['future_occ_offsets']),'diagnostic_only':np.bool_(True),'occ_file_offset':np.int32(-1),'preprocess_version':np.int32(3),'history_occ_ids':np.asarray([i+h-1 for h in CFG['history_offsets']]),'future_occ_ids':np.asarray([i+h-1 for h in CFG['future_occ_offsets']])}
  dest=OUT/'corrected-smoke'/route['route']/Path(name).name;dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**changed)
  with np.load(dest) as f:
   for k in ['state','trajectory','frame','time_s','world2ego']:assert np.array_equal(f[k],original[k])
   assert f['history'].shape==(4,20,64,160) and f['future'].shape==(5,20,64,160)
  assert hashlib.sha256(path.read_bytes()).hexdigest()==original_sha
  rows.append(dict(route=route['route'],frame=i,history_occ_ids=changed['history_occ_ids'].tolist(),future_occ_ids=changed['future_occ_ids'].tolist(),changed_history_fraction=float((changed['history']!=original['history']).mean()),changed_future_fraction=float((changed['future']!=original['future']).mean()),formal_cache_sha=original_sha))
 (OUT/'correction-smoke.json').write_text(json.dumps({'diagnostic_only':True,'rows':rows,'unchanged':['state','navigation','trajectory','time','world2ego','formal_cache','formal_manifest'],'note':'Only OCC file indices shifted -1; pose index remains annotation index. No formal training or inference run.'},indent=2));print('PASS',len(rows),'sidecar windows')
if __name__=='__main__':main()
