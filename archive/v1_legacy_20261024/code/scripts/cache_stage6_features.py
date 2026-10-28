"""Encode branch inputs/actual targets with the immutable selected Stage 4 encoder."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from configs.stage6 import Stage6Config
from mini_wa_jepa.provenance import provenance,write_json
from scripts.evaluate_joint import load_joint


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    cfg=Stage6Config();torch.set_num_threads(cfg.cpu_threads)
    if not torch.version.hip or torch.version.cuda:raise RuntimeError('Native ROCm required')
    args.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((args.data/'manifest.json').read_text())
    if not source['completed']:raise ValueError('Complete branch collection first')
    model,_,_=load_joint(cfg.checkpoint,'cuda');model.eval()
    report=dict(**provenance(cfg),stage=6,data_schema_version=2,completed=False,data=str(args.data.resolve()),
        data_manifest_sha256=hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(Path(cfg.checkpoint).read_bytes()).hexdigest(),splits={})
    started=time.perf_counter()
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        for split in ('train','val','test'):
            groups=[g for g in source['groups'] if g['split']==split]
            if not groups:continue
            records=[];rows=[];noise_errors=[];noise_masks=[];noise_seeds=[]
            for gi,g in enumerate(groups):
                ctx=np.load(args.data/g['context'])
                hist=model.encoder(torch.from_numpy(ctx['history']).to('cuda')).float().cpu().numpy()
                for ci,branch in enumerate(g['branches']):
                    raw=np.load(args.data/branch['path'])
                    target=model.encoder(torch.from_numpy(raw['future']).to('cuda')).float().cpu().numpy()
                    # Invalid zeros are encoded for shape only and ALWAYS masked in metrics/loss.
                    records.append(dict(history=hist,ego=ctx['ego'],command=raw['command'],target=target,
                        valid=raw['valid'],group=gi,candidate=ci,seed=g['seed'],
                        realized=raw['realized'],waypoint_valid=raw['waypoint_valid']))
                    rows.append(dict(group=g['id'],candidate=branch['id'],context=g['context'],raw=branch['path'],
                                     map_spec=g['map_spec'],seed=g['seed'],anchor_step=g['anchor_step']))
                if 'noise_replicate' in g:
                    repeat=np.load(args.data/g['noise_replicate']['path'])
                    repeated_target=model.encoder(torch.from_numpy(repeat['future']).to('cuda')).float().cpu().numpy()
                    reference=records[-4]
                    noise_errors.append(np.square(repeated_target-reference['target']).mean((-1,-2)))
                    noise_masks.append(repeat['valid']&reference['valid']);noise_seeds.append(g['seed'])
                if gi%16==0:print(json.dumps(dict(split=split,group=gi,total=len(groups))),flush=True)
            folder=args.output/split;folder.mkdir()
            for key in records[0]:np.save(folder/f'{key}.npy',np.asarray([r[key] for r in records]))
            np.save(folder/'noise_error.npy',np.asarray(noise_errors));np.save(folder/'noise_valid.npy',np.asarray(noise_masks))
            np.save(folder/'noise_seeds.npy',np.asarray(noise_seeds))
            write_json(folder/'rows.json',rows)
            noise_errors=np.asarray(noise_errors);noise_masks=np.asarray(noise_masks)
            report['splits'][split]=dict(groups=len(groups),samples=len(records),seeds=sorted({g['seed'] for g in groups}),
                noise_replicate_contexts=len(noise_errors),noise_latent_mse=float(noise_errors[noise_masks].mean()) if len(noise_errors) else None)
            write_json(args.output/'manifest.json',report)
    report.update(completed=True,elapsed_seconds=time.perf_counter()-started)
    write_json(args.output/'manifest.json',report)


if __name__=='__main__':main()
