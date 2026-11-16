"""Evaluate exact-budget latest, preserving current v3 and held-out test."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import numpy as np
import torch
from lightning_task import ROOT, configure_runtime, OccData
from models_v1 import FlowPlanner


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--arm', choices=['absolute', 'cv_residual'], required=True)
    args = parser.parse_args(); configure_runtime(); start = time.perf_counter()
    run = ROOT / ('artifacts/lightning/flow-v3-' + args.arm + '-pilot10')
    ckpath = run / 'latest.ckpt'; before = hashlib.sha256(ckpath.read_bytes()).hexdigest()
    ck = torch.load(ckpath, map_location='cpu', weights_only=False)
    assert ck['global_step'] == 6620
    settings = ck['task_state']['settings']; assert settings['trajectory_parameterization'] == args.arm
    data = OccData('flow', settings); assert data.sha == ck['task_state']['manifest_sha']
    model = FlowPlanner(parameterization=args.arm).cuda().eval()
    model.load_state_dict({k.removeprefix('model.'): v for k, v in ck['state_dict'].items()}, strict=True)
    if settings['channels_last']:
        torch.nn.utils.convert_conv2d_weight_memory_format(model, torch.channels_last); model.encoder.channels_last = True
    out = ROOT / ('artifacts/diagnostics/v3-residual-pilot/' + args.arm)
    out.mkdir(parents=True, exist_ok=True); seeds = [1908, 2911, 3911]; results = []
    for seed in seeds:
        batches = []
        for bi, batch in enumerate(data.loader('validation')):
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), torch.random.fork_rng(devices=[0]):
                torch.manual_seed(seed + bi); pred = model.sample(batch, k=8, steps=16)
            batches.append(pred.cpu().numpy())
        results.append(np.concatenate(batches)); print('INFER', args.arm, seed, flush=True)
    np.savez_compressed(out / 'flow_v3.npz', prediction=np.stack(results))
    metadata = {'checkpoint_sha256': before, 'manifest_sha256': data.sha, 'seeds': seeds,
                'parameterization': args.arm, 'step': ck['global_step'], 'test_opened': False,
                'checkpoint': str(ckpath.relative_to(ROOT)), 'seconds': time.perf_counter() - start}
    assert hashlib.sha256(ckpath.read_bytes()).hexdigest() == before
    (out / 'assessment.json').write_text(json.dumps(metadata, indent=2))
    del model, ck; torch.cuda.empty_cache()
    subprocess.run([sys.executable, str(ROOT / 'scripts/diagnose_v3_aggregation.py'),
                    '--prediction-directory', str(out.relative_to(ROOT)),
                    '--checkpoint', str(ckpath.relative_to(ROOT)), '--out', str((out / 'diagnostics').relative_to(ROOT))], check=True, cwd=ROOT)


if __name__ == '__main__': main()
