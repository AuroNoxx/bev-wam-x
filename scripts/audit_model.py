"""Frozen checkpoints, fixed ridge controls, validation only. No Trainer.fit."""
from pathlib import Path
import json,time,hashlib
import numpy as np
import torch
from lightning_task import configure_runtime,OccData
from models_v1 import FlowPlanner
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/diagnostics/20261115-contract'
def features(state):
    return np.concatenate([state,state[:,:8]**2,state[:,0:1]*state[:,4:8]],1).astype(np.float64)
def occ_features(h):
    # Current-frame coarse statistics; separate all source classes including unknown.
    h=h[-1];return np.stack([(h==k).reshape(20,4,16,8,20).mean((0,2,4)) for k in [0,1,2,3,4,5,6,7,255]]).flatten()
def load_split(m,split):
    ss=[];tt=[];oo=[]
    for name in m['splits'][split]:
        with np.load(ROOT/'data/occ_v1'/name) as f:ss.append(f['state']);tt.append(f['trajectory']);oo.append(occ_features(f['history']))
    return np.stack(ss),np.stack(tt),np.stack(oo)
def ridge(x,y,v):
    mean=x.mean(0);std=x.std(0);std[std<1e-6]=1
    a=np.c_[np.ones(len(x)),(x-mean)/std];b=np.c_[np.ones(len(v)),(v-mean)/std];reg=np.eye(a.shape[1])*10;reg[0,0]=0
    w=np.linalg.solve(a.T@a+reg,a.T@y.reshape(len(y),-1));p=(b@w).reshape(len(v),25,5)
    p[:,:,2:4]/=np.maximum(np.linalg.norm(p[:,:,2:4],axis=-1,keepdims=True),1e-6);p[:,:,4]=np.maximum(p[:,:,4],0)
    return p[:,None].astype(np.float32),dict(mean=mean,std=std,weights=w)
def main():
    configure_runtime();started=time.perf_counter();m=json.loads((ROOT/'data/occ_v1/manifest.json').read_text());timing={}
    ts,ty,to=load_split(m,'train');vs,vy,vo=load_split(m,'validation');print('FEATURES',len(ts),len(vs),flush=True)
    np.savez_compressed(OUT/'evaluation-labels.npz',state=vs,target=vy,names=np.asarray(m['splits']['validation']))
    for key,x,v in [('ridge_state_nav',features(ts),features(vs)),('ridge_state_nav_occ',np.c_[features(ts),to],np.c_[features(vs),vo])]:
        t=time.perf_counter();p,w=ridge(x,ty,v);np.savez_compressed(OUT/(key+'.npz'),prediction=p[None]);np.savez_compressed(OUT/(key+'-fit.npz'),**w);timing[key]=time.perf_counter()-t
    del ts,ty,to,vo
    for label,path in [('flow_v1','artifacts/lightning/flow/best.ckpt'),('flow_v2','artifacts/lightning/flow-v2/best.ckpt')]:
        t=time.perf_counter();ck=torch.load(ROOT/path,map_location='cpu',weights_only=False);settings=ck['task_state']['settings'];data=OccData('flow',settings)
        assert data.sha==ck['task_state']['manifest_sha'];model=FlowPlanner().cuda().eval();model.load_state_dict({k.removeprefix('model.'):v for k,v in ck['state_dict'].items()},strict=True)
        for mode in (['normal'] if label=='flow_v1' else ['normal','unknown_occ']):
            seeds=[1908,2911,3911] if mode=='normal' else [1908];outputs=[]
            for seed in seeds:
                batches=[]
                for bi,batch in enumerate(data.loader('validation')):
                    batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
                    if mode=='unknown_occ':batch['history']=torch.full_like(batch['history'],255)
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),torch.random.fork_rng(devices=[0]):
                        torch.manual_seed(seed+bi);p=model.sample(batch,k=8,steps=16)
                    batches.append(p.cpu().numpy())
                outputs.append(np.concatenate(batches));print('INFER',label,mode,seed,flush=True)
            name=label if mode=='normal' else label+'_'+mode;np.savez_compressed(OUT/(name+'.npz'),prediction=np.stack(outputs),seeds=np.asarray(seeds))
        timing[label]=time.perf_counter()-t;del model,ck;torch.cuda.empty_cache()
    result={'timing_s':timing,'total_s':time.perf_counter()-started,'manifest_sha':data.sha,'train_windows':len(m['splits']['train']),'validation_windows':len(vs),'test_opened':False,'ridge_alpha':10,'ridge_selection':'Fixed in advance, train-only normalization; no validation tuning','unknown_occ':'Frozen-model OOD sensitivity probe, NOT a retrained state-only model','model_settings':'8 candidates,16 Euler steps,bf16,original batch size8,paired original seeds','ridge_comparison':'Same solver/regularization/state features; added coarse current OCC only; not capacity matched to Flow'}
    (OUT/'evaluation-run.json').write_text(json.dumps(result,indent=2));print('DONE',result)
if __name__=='__main__':main()
