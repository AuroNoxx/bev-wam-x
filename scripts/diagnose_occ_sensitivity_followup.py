"""Frozen follow-up: distinguish semantic content from coverage-only inputs."""
import json
import time
import numpy as np
import torch
from diagnose_occ_frozen import ROOT, sha, summarize, batch_at, groups
from lightning_task import OccSamples, configure_runtime
from models_v1 import FlowPlanner


def main():
    configure_runtime();starttime=time.perf_counter()
    out=ROOT/'artifacts/diagnostics/occ-frozen-audit-20261115'
    report=json.loads((out/'report.json').read_text())
    ckpath=ROOT/'artifacts/lightning/flow-v3/best.ckpt'
    assert sha(ckpath)==report['checkpoint_sha256']
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    manifest=json.loads((ROOT/'data/occ_v3/manifest.json').read_text())
    dataset=OccSamples(manifest,'validation','flow')
    model=FlowPlanner().cuda().eval().requires_grad_(False)
    model.load_state_dict({k.removeprefix('model.'):v for k,v in ck['state_dict'].items()},strict=True)
    torch.nn.utils.convert_conv2d_weight_memory_format(model,torch.channels_last);model.encoder.channels_last=True
    with np.load(out/'labels.npz') as f:state,target,names=f['state'],f['target'],f['names']
    with np.load(out/'normal.npz') as f:baseline=f['prediction'][:1]
    modes=['vehicles_ground','only_coverage','reverse_old_common']
    parts={m:[] for m in modes}; feature=[]
    for bi,start in enumerate(range(0,len(dataset),8)):
        b=batch_at(dataset,range(start,min(start+8,len(dataset))));h=b['history']
        for mode in modes:
            hh=h.clone()
            if mode=='vehicles_ground':hh[hh==4]=1
            elif mode=='only_coverage':hh[hh!=255]=0
            elif mode=='reverse_old_common':
                reverse=h[:,:3].flip(1);common=(h[:,:3]!=255)&(reverse!=255)
                hh[:,:3]=torch.where(common,reverse,h[:,:3])
            bb={**b,'history':hh}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),torch.random.fork_rng(devices=[0]):
                torch.manual_seed(1908+bi);p=model.sample(bb,k=8,steps=16)
            parts[mode].append(p.cpu().numpy())
        # Measure perturbation at encoder output and pooled tokens, before state addition.
        hh=h.clone();hh[hh==4]=0
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            real=model.encoder(h).float();erased=model.encoder(hh).float()
            pooled=torch.nn.functional.adaptive_avg_pool2d(real,(8,20))
            epool=torch.nn.functional.adaptive_avg_pool2d(erased,(8,20))
            rel=(real-erased).flatten(1).norm(dim=1)/real.flatten(1).norm(dim=1).clamp_min(1e-8)
            prel=(pooled-epool).flatten(1).norm(dim=1)/pooled.flatten(1).norm(dim=1).clamp_min(1e-8)
            feature.append(torch.stack([rel,prel],1).cpu().numpy())
        if (bi+1)%50==0:print('FOLLOWUP',start+len(h),flush=True)
    predictions={m:np.concatenate(v)[None] for m,v in parts.items()}
    for m,p in predictions.items():np.savez_compressed(out/(m+'.npz'),prediction=p)
    f=np.concatenate(feature)
    result={'seed':1908,'validation':{m:summarize(p,state,target,names,baseline) for m,p in predictions.items()},
            'feature_relative_l2':{g:{'windows':int(mask.sum()),'conv':float(f[mask,0].mean()),'pooled':float(f[mask,1].mean())} for g,mask in groups(state,target).items()},
            'test_opened':False,'training_run':False,'seconds':time.perf_counter()-starttime,
            'note':'OOD sensitivity only; coverage-only retains ego-motion-dependent observation boundaries and is not a no-OCC baseline. Reverse-old-common keeps current frame and all unknown masks fixed.'}
    assert sha(ckpath)==report['checkpoint_sha256']
    (out/'followup.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('FOLLOWUP_DONE',result['seconds'],flush=True)


if __name__=='__main__':main()
