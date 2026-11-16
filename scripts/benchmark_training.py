"""Timed real optimizer updates, held-out route loss, checkpoints, and TensorBoard."""
from pathlib import Path
import argparse,json,time,os,statistics,math,faulthandler
import numpy as np
import psutil
import torch
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
from benchmark_models import FlowPlanner,ActionWorld,VisualOcc,compute_loss

ROOT=Path(__file__).resolve().parents[1]
os.environ['TORCH_HOME']=str(ROOT/'.cache/torch')
KEYS={'flow':['history','state','trajectory'],
      'world':['history','state','trajectory','controls','future'],
      'vision':['rgb','projection','visible','current']}


class Samples(Dataset):
    def __init__(self,split,mode):
        self.files=sorted((ROOT/'data/benchmark_cache'/split).glob('*.npz'))
        self.keys=KEYS[mode]
    def __len__(self): return len(self.files)
    def __getitem__(self,index):
        with np.load(self.files[index],allow_pickle=False) as data:
            return {k:torch.from_numpy(data[k].copy()) for k in self.keys}


def main():
    faulthandler.dump_traceback_later(45,repeat=True)
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['flow','world','vision'],required=True)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--steps',type=int,default=30)
    parser.add_argument('--warmup',type=int,default=5)
    args=parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(20261114)
    out=ROOT/'artifacts/benchmark'/f'{args.mode}-b{args.batch}'
    out.mkdir(parents=True,exist_ok=True)
    log=SummaryWriter(str(ROOT/'logs/tensorboard'/out.name))
    train=Samples('train',args.mode); val=Samples('validation',args.mode)
    loader=DataLoader(train,batch_size=args.batch,shuffle=True,drop_last=True,num_workers=0,pin_memory=True)
    iterator=iter(loader)
    constructors={'flow':FlowPlanner,'world':ActionWorld,'vision':VisualOcc}
    model=constructors[args.mode]().cuda()
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    before=params[0].detach().clone()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    timings=[]
    initial_loss=None
    total_started=time.monotonic()
    startup_free,total_gpu=torch.cuda.mem_get_info()
    with (out/'steps.jsonl').open('w',encoding='utf-8') as stream:
        for step in range(args.warmup+args.steps):
            wall=time.perf_counter()
            try: batch=next(iterator)
            except StopIteration:
                iterator=iter(loader); batch=next(iterator)
            data_seconds=time.perf_counter()-wall
            batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
            torch.cuda.synchronize()
            compute_start=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): loss=compute_loss(args.mode,model,batch)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward()
            if step==0:
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params): raise RuntimeError('Nonfinite gradient')
                initial_loss=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(params,1.)
            optimizer.step()
            torch.cuda.synchronize()
            compute_seconds=time.perf_counter()-compute_start
            elapsed=time.perf_counter()-wall
            row={'step':step,'warmup':step<args.warmup,'loss':float(loss.detach()),'data_seconds':data_seconds,
                 'compute_seconds':compute_seconds,'step_seconds':elapsed,'samples_per_second':args.batch/elapsed,
                 'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                 'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
                 'system_available_gib':psutil.virtual_memory().available/1024**3}
            stream.write(json.dumps(row)+'\n'); stream.flush()
            if step>=args.warmup: timings.append(row)
            log.add_scalar('train/loss',row['loss'],step)
            log.add_scalar('performance/samples_per_second',row['samples_per_second'],step)
            log.add_scalar('memory/peak_allocated_gib',row['peak_allocated_gib'],step)
            log.add_scalar('train/learning_rate',1e-4,step)
            if step%5==0: print(json.dumps(row),flush=True)
    update=float((params[0].detach()-before).abs().max())
    if update<=0: raise RuntimeError('Optimizer did not update parameter')
    model.eval()
    val_started=time.perf_counter(); validation=[]
    with torch.no_grad():
        for batch in DataLoader(val,batch_size=args.batch,num_workers=0):
            batch={k:v.cuda() for k,v in batch.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16): loss=compute_loss(args.mode,model,batch)
            validation.append((float(loss),len(next(iter(batch.values())))))
    torch.cuda.synchronize()
    validation_seconds=time.perf_counter()-val_started
    val_loss=sum(v*n for v,n in validation)/sum(n for _,n in validation)
    log.add_scalar('validation/loss',val_loss,args.warmup+args.steps)
    log.close()
    save_started=time.perf_counter()
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':vars(args),'validation_loss':val_loss},out/'checkpoint.pt')
    save_seconds=time.perf_counter()-save_started
    mean=statistics.mean(r['step_seconds'] for r in timings)
    report={'mode':args.mode,'batch':args.batch,'train_windows':len(train),'validation_windows':len(val),
            'torch':torch.__version__,'hip':torch.version.hip,'gpu':torch.cuda.get_device_name(0),
            'precision':'BF16 autocast, FP32 master weights, AdamW',
            'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in params),
            'timed_steps':args.steps,'warmup_steps':args.warmup,'mean_step_seconds':mean,
            'median_step_seconds':statistics.median(r['step_seconds'] for r in timings),
            'p95_step_seconds':float(np.percentile([r['step_seconds'] for r in timings],95)),
            'mean_data_seconds':statistics.mean(r['data_seconds'] for r in timings),
            'mean_compute_seconds':statistics.mean(r['compute_seconds'] for r in timings),
            'samples_per_second':args.batch/mean,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,'startup_free_gpu_gib':startup_free/1024**3,
            'initial_train_loss':initial_loss,'last_train_loss':timings[-1]['loss'],'held_out_loss':val_loss,
            'parameter_max_delta':update,'validation_seconds':validation_seconds,'checkpoint_seconds':save_seconds,
            'pilot_wall_seconds':time.monotonic()-total_started,
            'hours_per_100000_windows_epoch':math.ceil(100000/args.batch)*mean/3600,
            'caveat':'Throughput of this specific small model/grid. Does not establish convergence or safe driving.'}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('RESULT '+json.dumps(report),flush=True)


if __name__=='__main__': main()
