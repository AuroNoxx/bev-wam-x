"""Contract tests for frame alignment, split separation and crash-safe resume."""
from pathlib import Path
from unittest.mock import patch
import hashlib,json,tempfile
import numpy as np
import torch
from dataset_v1 import warp,compact,plan,ROOT
from train_v1 import atomic_checkpoint

def main():
    grid=np.zeros((20,64,64),np.uint8);grid[7,5,10]=4
    identity=warp(grid,np.eye(4),np.eye(4))
    assert identity[7,58,10]==4 and (identity[:,:,64:]==255).all()
    transform=np.eye(4);transform[0,3]=-.8
    shifted=warp(grid,transform,np.eye(4));assert shifted[7,58,11]==4
    occ=np.zeros((128,128,20),np.uint8);seg=np.zeros_like(occ)
    occ[0,0,0]=2;occ[1,1,1]=1;seg[1,1,1]=14
    pooled=compact(occ,seg);assert pooled[0,0,0]==255 and pooled[1,0,0]==4
    rows=plan();assert len(rows)==60
    towns={s:{r['file'].split('_')[1] for r in rows if r['split']==s} for s in ['train','validation','test']}
    assert not towns['train']&towns['validation'] and not towns['train']&towns['test'] and not towns['validation']&towns['test']
    torch.manual_seed(123);model=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters(),lr=.001);scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,.9)
    def step():
        optimizer.zero_grad();x=torch.randn(4,3);y=model(x).square().mean();y.backward();optimizer.step();scheduler.step()
    step()
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':torch.get_rng_state(),'epoch':0,'next_batch':1,'global_step':1,'config':{},'manifest_sha':'test','order':[0,1,2],'best_loss':1.,'bad_epochs':0}
    directory=ROOT/'artifacts/v1/contract-test';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'latest.pt';atomic_checkpoint(path,state);checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    step();expected={k:v.clone() for k,v in model.state_dict().items()};expected_lr=scheduler.get_last_lr()
    saved=torch.load(path,weights_only=False);model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);torch.set_rng_state(saved['rng']);step()
    assert all(torch.equal(v,expected[k]) for k,v in model.state_dict().items()) and scheduler.get_last_lr()==expected_lr
    with patch('train_v1.os.replace',side_effect=OSError('simulated replace failure')):
        try:atomic_checkpoint(path,state)
        except OSError:pass
        else:raise AssertionError('Expected save failure')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==checksum
    result={'coordinate_identity_flip':True,'translation_alignment':True,'unknown_not_free':True,'map_disjoint_split':True,'optimizer_scheduler_rng_resume_exact':True,'failed_save_preserves_previous':True}
    (directory/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
