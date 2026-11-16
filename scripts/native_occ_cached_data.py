"""Independent persistent native OCC cache. Does not modify the active trainer.

Content-addressed by manifest, loader/contract code and EVERY input file hash.
Dataset construction revalidates source hashes; reads then reject stat changes.
No GPU, resampling, semantic conversion or dtype conversion in cached reads.
"""
import hashlib, io, json, os, time, uuid
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from b2d_native_occ import ROOT, load_sample

SCHEMA='native-aligned-occ-cache-v1'
KEYS=('history','state','trajectory','future','history_source_support','future_source_support')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def contract():
    return {'schema':SCHEMA,'manifest_sha256':sha(ROOT/'data/occ_v3/manifest.json'),
            'loader_sha256':sha(ROOT/'scripts/b2d_native_occ.py'),
            'cache_code_sha256':sha(Path(__file__)),
            'history_shape':[4,20,128,128],'future_shape':[2,20,128,128],
            'spacing_m':.4,'xy_bounds_m':[-25.6,25.6],'future_labels_only':True}

def atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    try:
        with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        if hashlib.sha256(tmp.read_bytes()).digest()!=hashlib.sha256(data).digest():raise IOError('temporary write checksum')
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()

def source_paths(name):
    p=Path(name)
    if len(p.parts)!=4 or p.parts[0]!='cache' or p.parts[1] not in ('train','validation') or '..' in p.parts:raise ValueError('invalid sample')
    base=ROOT/'data/occ_v3'/p
    with np.load(base,allow_pickle=False) as z:
        ids=list(z['history_pose_ids'])+list(z['future_pose_ids'][:2])
    clip=ROOT/'data/public/bench2drive/clips'/p.parent.name
    return [base]+[clip/k/f'{int(i)-(k=="occ"):05d}{ext}' for i in ids for k,ext in [('anno','.json.gz'),('occ','.npz')]]

def fingerprints(name):
    return {str(p.relative_to(ROOT)):sha(p) for p in source_paths(name)}

def validate(row):
    shapes={'history':(4,20,128,128),'future':(2,20,128,128),'state':(20,),'trajectory':(25,5),'history_source_support':(4,20,128,128),'future_source_support':(2,20,128,128)}
    if set(row)!=set(KEYS):raise ValueError('cache fields')
    for k,s in shapes.items():
        if tuple(row[k].shape)!=s:raise ValueError('shape '+k)
    if row['history'].dtype!=torch.uint8 or row['future'].dtype!=torch.uint8:raise ValueError('OCC dtype')
    if any(row[k].dtype!=torch.bool for k in ('history_source_support','future_source_support')):raise ValueError('support dtype')
    if not all(torch.isfinite(row[k]).all() for k in ('state','trajectory')):raise ValueError('nonfinite scalar data')

def build_cache(folder,names):
    folder=Path(folder); c=contract(); rows=[]; started=time.perf_counter()
    for name in names:
        before=fingerprints(name); key=digest({'contract':c,'sample':name,'sources':before})
        target=folder/'objects'/(key+'.npz')
        if not target.exists():
            tensors,meta=load_sample(name,include_labels=True);validate(tensors)
            if fingerprints(name)!=before:raise RuntimeError('source changed while building')
            buf=io.BytesIO();np.savez_compressed(buf,**{k:tensors[k].numpy() for k in KEYS})
            atomic_bytes(target,buf.getvalue())
        with np.load(target,allow_pickle=False) as z:validate({k:torch.from_numpy(z[k]) for k in z.files})
        rows.append({'name':name,'object':'objects/'+key+'.npz','sha256':sha(target),'sources':before})
        print(json.dumps({'built':len(rows),'total':len(names),'sample':name}),flush=True)
    index={'contract':c,'rows':rows,'build_seconds':time.perf_counter()-started}
    index_path=folder/('index-'+digest(index)+'.json')
    atomic_bytes(index_path,json.dumps(index,indent=2).encode())
    return index_path

class NativeCachedDataset(Dataset):
    def __init__(self,index_path,split,actor_only=False):
        if split not in ('train','validation'):raise ValueError('test excluded')
        self.index_path=str(Path(index_path).resolve());index=json.loads(Path(index_path).read_text())
        if index['contract']!=contract():raise ValueError('stale cache contract/code/manifest')
        self.rows=[r for r in index['rows'] if Path(r['name']).parts[1]==split]
        self.names=tuple(r['name'] for r in self.rows)
        if not self.rows or len(set(self.names))!=len(self.names):raise ValueError('empty/duplicate split')
        manifest=json.loads((ROOT/'data/occ_v3/manifest.json').read_text())
        if not set(self.names)<=set(manifest['splits'][split]):raise ValueError('sample outside split')
        self.actor_only=actor_only;self.folder=Path(index_path).parent; self.source_stats={}
        self.identity=digest({'contract':index['contract'],'rows':self.rows,'actor_only':actor_only})
        expected={}
        for row in self.rows:
            for p,h in row['sources'].items():
                if p in expected and expected[p]!=h:raise ValueError('conflicting source hash')
                expected[p]=h
        for p,h in expected.items():
            path=ROOT/p
            if sha(path)!=h:raise ValueError('source changed: '+p)
            s=path.stat();self.source_stats[p]=(s.st_size,s.st_mtime_ns)
    def state_dict(self):return {'cache_identity':self.identity}
    def load_state_dict(self,state):
        if state!={'cache_identity':self.identity}:raise ValueError('resume cache dataset/order/contract mismatch')
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        for p in r['sources']:
            s=(ROOT/p).stat()
            if (s.st_size,s.st_mtime_ns)!=self.source_stats[p]:raise RuntimeError('source changed after validation: '+p)
        obj=(self.folder/r['object']).resolve()
        if obj.parent!=(self.folder/'objects').resolve():raise ValueError('invalid object path')
        data=obj.read_bytes()
        if hashlib.sha256(data).hexdigest()!=r['sha256']:raise ValueError('corrupt cached object')
        with np.load(io.BytesIO(data),allow_pickle=False) as z:
            keys=('history','state') if self.actor_only else ('history','state','trajectory','future')
            return {k:torch.from_numpy(z[k].copy()) for k in keys}

