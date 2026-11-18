"""Inspect the actual SFT checkpoint and verify atomic failure + next samples."""
from .runtime import ROOT,configure
configure()
import argparse,hashlib,json
import torch
from .real_dataset import RealData
from .checkpoint_io import VerifiedCheckpointIO


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,required=True);args=parser.parse_args()
    folder=ROOT/'artifacts/v5-qwen-drive-world/overfit64';path=folder/'latest.ckpt'
    checkpoint=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    assert checkpoint['global_step']==args.step
    assert checkpoint['task_state']['ema_step']==args.step
    states=checkpoint['optimizer_states'][0]['state']
    assert states and all(int(s['step'])==args.step and s['exp_avg'].dtype==s['exp_avg_sq'].dtype==torch.float32 for s in states.values())
    loader_state=checkpoint['loops']['fit_loop']['state_dict']['combined_loader'][0]
    assert loader_state['_num_yielded']==args.step*8,loader_state['_num_yielded']
    loader=RealData('overfit64').train_dataloader();loader.load_state_dict(loader_state)
    iterator=iter(loader);next_ids=[]
    for _ in range(8):
        r=next(iterator)['record'];next_ids.append(f"{r['source']}:{r['route']}:{r['index']}")
    if args.step==2:
        first=json.loads((folder/'resume-step1.json').read_text())
        assert checkpoint['task_state']['seen_tail'][-8:]==first['next_sample_ids'],'Resume skipped or repeated samples'
    before=digest(path)
    try:VerifiedCheckpointIO().save_checkpoint({'state_dict':{}},path)
    except ValueError as e:failure=str(e)
    else:raise AssertionError('Invalid checkpoint was accepted')
    assert digest(path)==before,'Previous valid checkpoint changed after failed save'
    row={'step':args.step,'ema_step':checkpoint['task_state']['ema_step'],'optimizer_states':len(states),
         'optimizer_fp32':True,'loader_samples_yielded':loader_state['_num_yielded'],
         'epoch':checkpoint['epoch'],'failed_save_preserved_previous':True,'injected_failure':failure,
         'next_sample_ids':next_ids,'sha256':before,'checkpoint_bytes':path.stat().st_size,
         'resume_sample_order_verified':args.step==2}
    (folder/f'resume-step{args.step}.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)


if __name__=='__main__':main()
