"""Measured batch-one inference and disk-loader costs, after training exits."""
import json, math, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from lightning_task import ROOT, OccSamples, configure_runtime
from train_joint_occ import JointTask, JointData, batch_to, digest, resource_guard
from audit_joint_occ import write


def main():
    resource_guard();configure_runtime()
    cfg=json.loads((ROOT/'configs/joint-occ-v1.json').read_text());selection=json.loads((ROOT/cfg['selection']).read_text())
    data=JointData(cfg,selection,1701);device='cuda' if torch.cuda.is_available() else 'cpu'
    b=batch_to(data.datasets['validation'].rows[:1],device);actor={k:b[k] for k in ['history','state']}
    result={}
    for variant in cfg['variants']:
        path=ROOT/f'artifacts/joint-occ-v1/runs/{variant}-seed1701/latest.ckpt'
        model=JointTask(variant,cfg,selection['manifest_sha'],digest(ROOT/cfg['selection']),1701).to(device).eval()
        model.load_state_dict(torch.load(path,map_location=device,weights_only=False)['state_dict'])
        times=[]
        with torch.inference_mode(),torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):
            for i in range(25):
                if device=='cuda':torch.cuda.synchronize()
                start=time.perf_counter();model.model.sample(actor,k=8,steps=16)
                if device=='cuda':torch.cuda.synchronize()
                if i>=5:times.append(time.perf_counter()-start)
        result[variant]={'batch1_k8_steps16_ms_mean':float(np.mean(times)*1000),'ms_p50':float(np.median(times)*1000),
                         'ms_p95':float(np.quantile(times,.95)*1000),'ms_max':float(max(times)*1000),'timed_repetitions':20}
        del model
        if device=='cuda':torch.cuda.empty_cache()
    # Single sequential pass reads only train split. No optimizer and no cache writes.
    manifest=json.loads((ROOT/cfg['manifest']).read_text());ds=OccSamples(manifest,'train','world')
    started=time.perf_counter();count=0
    for batch in DataLoader(ds,batch_size=8,shuffle=False,num_workers=0):count+=len(batch['state'])
    io_seconds=time.perf_counter()-started
    report={'inference':result,'train_disk_loading':{'windows':count,'wall_seconds':io_seconds,'workers':0,
        'note':'reads history plus all five future frames (conservative I/O proxy); OS cache warm status uncontrolled'},
        'scope':'GPU model only; excludes OCC construction, transfers, candidate safety selection, controller, CARLA; not closed-loop deadline acceptance',
        'simulation_seconds':None,'test_read':False,'epoch_projections':{}}
    for variant in cfg['variants']:
        costs=[json.loads((ROOT/f'artifacts/joint-occ-v1/runs/{variant}-seed{s}/resources.json').read_text())['step_seconds_mean'] for s in cfg['seeds']]
        report['epoch_projections'][variant]={'optimizer_only_seconds_range':[math.ceil(len(ds)/8)*min(costs),math.ceil(len(ds)/8)*max(costs)],
            'serial_optimizer_plus_measured_io_seconds_range':[math.ceil(len(ds)/8)*min(costs)+io_seconds,math.ceil(len(ds)/8)*max(costs)+io_seconds],
            'validation_and_checkpoint_seconds':'not included; must measure at full-split run','full_training_commitment':False}
    write(ROOT/'artifacts/joint-occ-v1/benchmark.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
