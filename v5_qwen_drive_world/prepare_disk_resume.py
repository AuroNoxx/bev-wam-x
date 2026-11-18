"""Read-only verification of the latest completed update before switching."""
from .runtime import ROOT,configure
configure()
import hashlib,json
import torch
from .real_dataset import RealData


def main():
    folder=ROOT/'artifacts/v5-qwen-drive-world/subset2000';path=folder/'latest.ckpt'
    ckpt=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    step=ckpt['global_step'];states=ckpt['optimizer_states'][0]['state']
    assert ckpt['task_state']['ema_step']==step
    assert states and all(int(s['step'])==step and s['exp_avg'].dtype==s['exp_avg_sq'].dtype==torch.float32 for s in states.values())
    saved=ckpt['loops']['fit_loop']['state_dict']['combined_loader'][0]
    expected_in_epoch=step*8%2000 or 2000
    assert saved['_num_yielded']==expected_in_epoch,'Checkpoint is inside an uncommitted accumulation window'
    loader=RealData('subset2000').train_dataloader();loader.load_state_dict(saved);iterator=iter(loader)
    ids=[]
    for _ in range(8):
        r=next(iterator)['record'];ids.append(f"{r['source']}:{r['route']}:{r['index']}")
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    result={'step':step,'epoch':ckpt['epoch'],'next_sample_ids':ids,'optimizer_states':len(states),
            'ema_step':ckpt['task_state']['ema_step'],'loader_yielded':saved['_num_yielded'],'sha256':digest}
    (folder/'disk-resume-expected.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
