"""Read-only v2 inference worker. IPC accepts history/state only, never labels."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from occ_qwen_native_v2 import NativeOccQwenV2,load_v2_backbone
from occ_qwen_performance import enable_channels_last

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    path=ROOT/'artifacts/occ-qwen-native-v2-training/best.ckpt'
    checkpoint_sha=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    cp=torch.load(path,map_location='cpu',weights_only=False)
    task=cp['task_state'];cfg=task['settings']
    live=json.loads((ROOT/'configs/occ-qwen-native-v2.json').read_text())
    if cfg!=live or task['schema']!='native-occ-qwen-v2-lightning-v1' or task['contract_test']:
        raise RuntimeError('Checkpoint/config/schema mismatch')
    manifest_path=ROOT/cfg['manifest'];manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if task['manifest_sha']!=manifest_sha:raise RuntimeError('Training manifest has drifted')
    manifest=json.loads(manifest_path.read_text())
    model=NativeOccQwenV2(load_v2_backbone(ROOT/'models/Qwen3-VL-2B-Instruct',cfg['qwen_layers']),
                          accel_weight=cfg['accel_weight'],jerk_weight=cfg['jerk_weight'])
    model.load_state_dict({k.removeprefix('model.'):v for k,v in cp['state_dict'].items()},strict=True)
    meta={'ready':True,'checkpoint':str(path),'checkpoint_sha256':checkpoint_sha,
          'epoch_one_based':cp['epoch']+1,'global_step':cp['global_step'],'config':cfg,
          'manifest_sha256':manifest_sha,'train_routes':[r['route'] for r in manifest['routes'] if r['split']=='train'],
          'validation_routes':[r['route'] for r in manifest['routes'] if r['split']=='validation'],
          'version':'NativeOccQwenV2; strict all 288 tensor load; no optimizer or training',
          'trajectory':'25x2 physical meters, current ego FLU; t=.2...5s',
          'sampling':'single candidate; 8 Euler steps; CUDA generator1907+planning_index per run',
          'actor_fields':['history','state'],'device':torch.cuda.get_device_name(0)}
    enable_channels_last(model);model.cuda().eval();del cp
    (output/'model-identity.json').write_text(json.dumps(meta,indent=2))
    # Warm-up is synthetic and never actuates; its latency is reported separately.
    h=torch.zeros(1,4,20,128,128,dtype=torch.uint8,device='cuda');s=torch.zeros(1,20,device='cuda')
    t=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        model.sample(h,s,steps=8,generator=torch.Generator(device='cuda').manual_seed(0))
    torch.cuda.synchronize();meta['warmup_seconds']=time.perf_counter()-t
    print(json.dumps(meta),flush=True)
    for line in sys.stdin:
        try:
            req=json.loads(line)
            if req.get('close'):break
            if req.get('operation')=='upstream_occ':
                from probe_native_v2_upstream_occ import load_upstream_generator
                from b2d_native_occ import semantic_grid
                if req.get('reset') or 'occ_generator' not in locals():occ_generator,occ_identity=load_upstream_generator()
                t=time.perf_counter()
                with np.load(req['raw_file'],allow_pickle=False) as z:raw=z['raw'].copy()
                occ,seg=occ_generator.occ_from_carla_tick(raw,req['boxes'],{'LIDAR_0':{'world2lidar':req['world2lidar']}})
                grid=semantic_grid(occ,seg)[:,::-1,:].copy()
                destination=Path(req['grid_file']);np.save(destination,grid)
                print(json.dumps({'grid_file':str(destination),'generation_seconds':time.perf_counter()-t,
                                  'generator':occ_identity,'native_counts':{str(k):int(v) for k,v in zip(*np.unique(grid,return_counts=True))}}),flush=True)
                continue
            t=time.perf_counter()
            with np.load(req['input'],allow_pickle=False) as f:
                if set(f.files)!={'history','state'}:raise ValueError('Only actor input fields permitted')
                history=f['history'];state=f['state']
            if history.shape!=(4,20,128,128) or history.dtype!=np.uint8 or state.shape!=(20,) or not np.isfinite(state).all():
                raise ValueError('Native actor shape/numeric mismatch')
            if not np.isin(history,[0,1,2,3,4,5,6,7,255]).all():raise ValueError('Semantic mismatch')
            h=torch.from_numpy(history.copy())[None].cuda();s=torch.from_numpy(state.copy())[None].float().cuda()
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                out=model.sample(h,s,steps=8,generator=torch.Generator(device='cuda').manual_seed(req['seed']))
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            xy=out['trajectory_xy'][0].float().cpu().numpy()
            residual=out['residual_velocity'][0].float().cpu().numpy()
            velocity=residual*10+np.array([state[0]*30,0.],np.float32)
            jerk=np.linalg.norm(np.diff(velocity,n=2,axis=0)/.04,axis=1)
            print(json.dumps({'xy_flu_m':xy.tolist(),'velocity_flu_mps':velocity.tolist(),
                              'residual_velocity':residual.tolist(),'inference_seconds':seconds,
                              'worker_seconds':time.perf_counter()-t,'seed':req['seed'],
                              'vector_jerk_mean_mps3':float(jerk.mean()),'vector_jerk_max_mps3':float(jerk.max()),
                              'jerk_used_for_control_or_veto':False}),flush=True)
        except Exception as e:
            print(json.dumps({'error':repr(e)}),flush=True)


if __name__=='__main__':main()
