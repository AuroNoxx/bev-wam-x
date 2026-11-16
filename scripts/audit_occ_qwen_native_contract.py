"""One deterministic window per train/validation route; read-only data audit."""
import json,time
from pathlib import Path
import numpy as np
import torch
from b2d_native_occ import ROOT,load_sample,align_native,semantic_grid
from occ_qwen_native_data import NativeDataset,require_training_authorization,loader

start=time.perf_counter();m=json.loads((ROOT/'data/occ_v3/manifest.json').read_text());rows=[];errors=[]
for r in m['routes']:
    if r['split'] not in ['train','validation'] or not r['samples']:continue
    name=r['samples'][len(r['samples'])//2]
    try:
        b,p=load_sample(name)
        assert b['history'].shape==(4,20,128,128) and b['future'].shape==(2,20,128,128)
        actor,am=load_sample(name,include_labels=False)
        assert set(actor)=={'history','state','history_source_support'}
        assert all(x['role']=='history' for x in am['sources'])
        assert torch.equal(b['history'],actor['history']) and torch.equal(b['state'],actor['state'])
        valid=b['future_source_support']
        assert (b['future'][~valid]==255).all()
        rows.append({'split':r['split'],'route':r['route'],'sample':name,'history_unknown':float((b['history']==255).float().mean()),'future_unknown':float((b['future']==255).float().mean()),'future_source_support':float(valid.float().mean())})
    except Exception as e:errors.append({'sample':name,'error':repr(e)})
    print('AUDIT',len(rows),len(errors),r['route'],flush=True)
# Strict source-coordinate synthetic regression tests.
g=np.zeros((20,128,128),np.uint8);g[4,55,70]=4
z,v=align_native(g,np.eye(4),np.eye(4));assert z[4,72,70]==4 and v.all()
t=np.eye(4);t[0,3]=.4;z,v=align_native(g,t,np.eye(4));assert z[4,72,69]==4 and (z[:,:,-1]==255).all()
try:load_sample('../test/route/0.npz')
except ValueError:pass
else:raise AssertionError('unsafe path accepted')
try:require_training_authorization()
except RuntimeError as e:training_guard=str(e)
else:raise AssertionError('training guard failed')
# Actual stateful loader restore, no model/optimizer, compare exact next batch.
dl=loader();it=iter(dl);first=next(it);state=dl.state_dict();expected=next(it)
restored=loader();restored.load_state_dict(state);actual=next(iter(restored))
assert all(torch.equal(actual[k],expected[k]) for k in actual)
train=NativeDataset('train');val=NativeDataset('validation')
assert not set(train.names)&set(val.names)
assert not {Path(n).parent.name for n in train.names}&{Path(n).parent.name for n in val.names}
report={'passed':not errors,'seconds':time.perf_counter()-start,'route_samples_read':len(rows),'errors':errors,'rows':rows,'dataset_windows':{'train':len(train),'validation':len(val)},'manifest_sha256':train.manifest_sha256,'loader_next_batch_exact_after_restore':True,'training_guard':training_guard,'formal_training':False,'source_causality_certified':False,'future_occ_decoder':False,'scene_to_ego':False}
path=ROOT/'artifacts/occ-qwen-native-contract-audit.json';path.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
