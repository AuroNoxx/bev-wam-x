"""Frozen inference on all training stop/deceleration examples; no fitting."""
import json
import time
import numpy as np
import torch
from diagnose_occ_frozen import ROOT, sha, groups, summarize, batch_at
from lightning_task import configure_runtime, OccSamples
from models_v1 import FlowPlanner


def main():
    configure_runtime();starttime=time.perf_counter()
    out=ROOT/'artifacts/diagnostics/occ-frozen-audit-20261115'
    manifest=json.loads((ROOT/'data/occ_v3/manifest.json').read_text())
    names=np.asarray(manifest['splits']['train']);states=[];targets=[]
    for name in names:
        with np.load(ROOT/'data/occ_v3'/name) as f:states.append(f['state']);targets.append(f['trajectory'])
    state,target=np.stack(states),np.stack(targets);masks=groups(state,target)
    ids=np.flatnonzero(masks['stationary']|masks['deceleration'])
    ckpath=ROOT/'artifacts/lightning/flow-v3/best.ckpt';before=sha(ckpath)
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    model=FlowPlanner().cuda().eval().requires_grad_(False)
    model.load_state_dict({k.removeprefix('model.'):v for k,v in ck['state_dict'].items()},strict=True)
    torch.nn.utils.convert_conv2d_weight_memory_format(model,torch.channels_last);model.encoder.channels_last=True
    ds=OccSamples(manifest,'train','flow');parts=[]
    for bi,start in enumerate(range(0,len(ids),8)):
        b=batch_at(ds,ids[start:start+8])
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),torch.random.fork_rng(devices=[0]):
            torch.manual_seed(1908+bi);p=model.sample(b,k=8,steps=16)
        parts.append(p.cpu().numpy())
    pred=np.concatenate(parts)[None]
    result={'metrics':summarize(pred,state[ids],target[ids],names[ids]),'seed':1908,
            'selection':'All training windows satisfying pre-existing stationary or deceleration definitions',
            'seconds':time.perf_counter()-starttime,'test_opened':False,'training_run':False,
            'limitation':'Train and validation have different roads/behavior difficulty; descriptive fit diagnostic, not causal gap.'}
    assert sha(ckpath)==before
    np.savez_compressed(out/'train_behaviors.npz',prediction=pred,names=names[ids],state=state[ids],target=target[ids])
    (out/'train-behaviors.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'seconds':result['seconds'],'groups':result['metrics']['groups']}),flush=True)


if __name__=='__main__':main()
