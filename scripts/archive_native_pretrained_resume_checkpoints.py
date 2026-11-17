"""Lossless logical checkpoint archive/rebuild for three named engineering tests.
Does not delete or modify originals. CPU-only; no model execution.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['HIP_VISIBLE_DEVICES']='-1';os.environ['ROCR_VISIBLE_DEVICES']='-1'
import argparse,copy,gc,hashlib,json,math,uuid,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/occ-qwen-native-pretrained-resume-test'
ARCHIVE=SOURCE/'lossless-archive-v1'
NAMES=('full.ckpt','split.ckpt','resumed.ckpt')
PREFIX='model.backbone.'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def equal(a,b,path='root',counts=None):
 if counts is None:counts={'tensors':0,'numpy_arrays':0,'non_tensor_leaves':0}
 assert type(a) is type(b),(path,type(a),type(b))
 if isinstance(a,torch.Tensor):
  assert a.dtype==b.dtype and a.shape==b.shape and a.layout==b.layout,(path,'tensor metadata')
  assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8)),(path,'tensor bits')
  counts['tensors']+=1
 elif isinstance(a,np.ndarray):
  assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),(path,'numpy bytes');counts['numpy_arrays']+=1
 elif isinstance(a,dict):
  assert list(a.keys())==list(b.keys()),(path,'ordered dict keys')
  for k in a:equal(a[k],b[k],path+'/'+str(k),counts)
  if hasattr(a,'_metadata'):equal(a._metadata,b._metadata,path+'/_metadata',counts)
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b),(path,'length')
  for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i),counts)
 elif isinstance(a,float):
  import struct
  assert struct.pack('d',a)==struct.pack('d',b),(path,'float bits');counts['non_tensor_leaves']+=1
 else:
  assert a==b,(path,'value');counts['non_tensor_leaves']+=1
 return counts

def save_atomic(obj,path):
 path=Path(path);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
 try:
  with tmp.open('xb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
  loaded=torch.load(tmp,map_location='cpu',weights_only=False);equal(obj,loaded);del loaded
  os.replace(tmp,path)
 finally:
  if tmp.exists():tmp.unlink()

def reconstruct(compact,shared):
 cp=compact['checkpoint'];state=cp['state_dict'];ordered=type(state)()
 for key in compact['original_state_key_order']:ordered[key]=shared[key] if key.startswith(PREFIX) else state[key]
 if hasattr(state,'_metadata'):ordered._metadata=state._metadata
 cp['state_dict']=ordered
 return cp

def build():
 if ARCHIVE.exists():raise RuntimeError('archive already exists; do not overwrite verified archive')
 ARCHIVE.mkdir();start=time.perf_counter();report={'schema':'lossless-logical-checkpoint-v1','originals_deleted':False,'prefix':PREFIX,'files':[]}
 shared=None
 for name in NAMES:
  source=SOURCE/name;before=source.stat();original=torch.load(source,map_location='cpu',weights_only=False)
  backbone={k:v for k,v in original['state_dict'].items() if k.startswith(PREFIX)}
  assert backbone,'missing backbone'
  if shared is None:
   shared=backbone;save_atomic(shared,ARCHIVE/'shared-backbone.pt');report['shared']={'file':'shared-backbone.pt','sha256':sha(ARCHIVE/'shared-backbone.pt'),'bytes':(ARCHIVE/'shared-backbone.pt').stat().st_size,'tensor_count':len(shared)}
  else:equal(shared,backbone)
  # Shallow copy containers; tensor bytes unchanged. All non-backbone state kept.
  cp=original.copy();state=original['state_dict'].copy()
  if hasattr(original['state_dict'],'_metadata'):state._metadata=copy.deepcopy(original['state_dict']._metadata)
  for k in backbone:del state[k]
  cp['state_dict']=state
  compact={'schema':report['schema'],'shared_sha256':report['shared']['sha256'],'original_state_key_order':list(original['state_dict']),'checkpoint':cp}
  target=ARCHIVE/(name+'.compact.pt');save_atomic(compact,target)
  disk_compact=torch.load(target,map_location='cpu',weights_only=False);disk_shared=torch.load(ARCHIVE/'shared-backbone.pt',map_location='cpu',weights_only=False)
  rebuilt=reconstruct(disk_compact,disk_shared);counts=equal(original,rebuilt)
  row={'original_name':name,'original_bytes':before.st_size,'original_sha256':sha(source),'compact_file':target.name,'compact_bytes':target.stat().st_size,'compact_sha256':sha(target),'logical_content_exact':True,'comparison_counts':counts}
  assert source.stat().st_mtime_ns==before.st_mtime_ns and source.stat().st_size==before.st_size
  report['files'].append(row);print(json.dumps(row),flush=True)
  del original,backbone,cp,state,compact,disk_compact,disk_shared,rebuilt;gc.collect()
 report['original_total_bytes']=sum(x['original_bytes'] for x in report['files']);report['archive_tensor_bytes']=report['shared']['bytes']+sum(x['compact_bytes'] for x in report['files']);report['potential_net_release_bytes']=report['original_total_bytes']-report['archive_tensor_bytes'];report['seconds']=time.perf_counter()-start
 p=ARCHIVE/'manifest.json';tmp=p.with_suffix('.pending.json')
 with tmp.open('x') as f:json.dump(report,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);print(json.dumps(report,indent=2))

def rebuild(name,destination):
 if name not in NAMES:raise ValueError('unknown checkpoint')
 m=json.loads((ARCHIVE/'manifest.json').read_text());r=next(x for x in m['files'] if x['original_name']==name)
 shared=ARCHIVE/m['shared']['file'];compact=ARCHIVE/r['compact_file'];out=Path(destination)
 if out.exists():raise ValueError('destination exists; never overwrite')
 assert sha(shared)==m['shared']['sha256'] and sha(compact)==r['compact_sha256'],'archive checksum mismatch'
 c=torch.load(compact,map_location='cpu',weights_only=False);assert c['shared_sha256']==m['shared']['sha256']
 original=reconstruct(c,torch.load(shared,map_location='cpu',weights_only=False));save_atomic(original,out)
 print(json.dumps({'restored':str(out),'logical_equivalent_to':r['original_sha256'],'note':'serialized file hash may differ; all checkpoint logical content retained'}))

def main():
 torch.set_num_threads(1)
 p=argparse.ArgumentParser();p.add_argument('--build',action='store_true');p.add_argument('--restore',choices=NAMES);p.add_argument('--output');a=p.parse_args()
 if a.build:build()
 elif a.restore and a.output:rebuild(a.restore,a.output)
 else:p.error('choose --build or --restore NAME --output NEW_PATH')
if __name__=='__main__':main()
