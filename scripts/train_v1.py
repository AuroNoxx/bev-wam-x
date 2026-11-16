"""Fresh OCC-only training, measured cost probes, atomic rolling resume checkpoints."""
from pathlib import Path
import argparse, copy, hashlib, json, math, os, random, statistics, time
import numpy as np
import psutil
import torch
from torch.utils.data import Dataset,DataLoader,Subset
from torch.utils.tensorboard import SummaryWriter
from models_v1 import MODELS,loss

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/train_v1.json'
KEYS={'flow':['history','state','trajectory'],'world':['history','state','trajectory','future']}

class Samples(Dataset):
    def __init__(self,manifest,split,mode):
        self.files=[ROOT/'data/occ_v1'/p for p in manifest['splits'][split]];self.keys=KEYS[mode]
    def __len__(self):return len(self.files)
    def __getitem__(self,i):
        with np.load(self.files[i],allow_pickle=False) as a:return {k:torch.from_numpy(a[k].copy()) for k in self.keys}

def seed(value):random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])

def atomic_checkpoint(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    check=torch.load(temporary,map_location='cpu',weights_only=False)
    required={'model','optimizer','scheduler','rng','epoch','next_batch','global_step','config','manifest_sha','order','best_loss','bad_epochs'}
    if not required<=check.keys():raise RuntimeError('Checkpoint missing required resume state')
    if not all(torch.isfinite(t).all() for t in check['model'].values()):raise RuntimeError('Nonfinite checkpoint model')
    del check
    os.replace(temporary,path)

def move(batch):return {k:v.cuda(non_blocking=True) for k,v in batch.items()}

@torch.no_grad()
def validate(mode,model,data,batch_size):
    saved=rng();seed(1907);model.eval();total=0.;count=0;started=time.perf_counter()
    try:
        for batch in DataLoader(data,batch_size=batch_size,num_workers=0,generator=torch.Generator().manual_seed(1907)):
            batch=move(batch)
            with torch.autocast('cuda',dtype=torch.bfloat16):value=loss(mode,model,batch)
            n=len(batch['state']);total+=float(value)*n;count+=n
        torch.cuda.synchronize()
        return total/count,time.perf_counter()-started
    finally:restore_rng(saved);model.train()

@torch.no_grad()
def trajectory_metrics(model,data,batch_size):
    saved=rng();seed(1908);model.eval();totals={};count=0
    try:
        for batch in DataLoader(data,batch_size=batch_size,num_workers=0,generator=torch.Generator().manual_seed(1908)):
            batch=move(batch)
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model.sample(batch,k=8,steps=16)
            error=((pred[...,:2]-batch['trajectory'][:,None,:,:2])*pred.new_tensor([50,10])).norm(dim=-1)
            ade=error.mean(-1);chosen=ade.argmin(1);idx=torch.arange(len(ade),device=ade.device)
            values={'minADE8_m':ade.min(1).values,'meanADE8_m':ade.mean(1),'minFDE8_m':error[:,:,-1].min(1).values,
                    'speed_RMSE_bestADE_m_s':((pred[idx,chosen,:,4]-batch['trajectory'][:,:,4]).square().mean(-1)).sqrt()*30,
                    'endpoint_diversity_m':((pred[:,:,-1,:2]*pred.new_tensor([50,10])).std(dim=1)).norm(dim=-1)}
            for key,value in values.items():totals[key]=totals.get(key,0.)+float(value.sum())
            count+=len(ade)
        return {**{key:value/count for key,value in totals.items()},'metric_windows':count}
    finally:restore_rng(saved);model.train()

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['flow','world'],required=True)
    p.add_argument('--benchmark',action='store_true');p.add_argument('--manifest',default='data/occ_v1/manifest.json')
    p.add_argument('--steps',type=int,default=100);p.add_argument('--warmup',type=int,default=8)
    p.add_argument('--resume',action='store_true');p.add_argument('--run',default='v1');args=p.parse_args()
    cfg=json.loads(CONFIG.read_text());settings=cfg[args.mode];batch_size=settings['batch_size']
    path=ROOT/args.manifest;manifest=json.loads(path.read_text());manifest_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest.get('preprocess_version')!=2:raise RuntimeError('Rebuild data with the current units/coordinate contract')
    if manifest['config_sha256']!=hashlib.sha256(CONFIG.read_bytes()).hexdigest():raise RuntimeError('Cache/config mismatch; rebuild manifest')
    if not args.benchmark and manifest['pilot']:raise RuntimeError('Pilot is for timing only, not formal training')
    train=Samples(manifest,'train',args.mode);validation=Samples(manifest,'validation',args.mode)
    if not len(train) or not len(validation):raise ValueError('Empty split')
    torch.set_num_threads(6);seed(cfg['seed'])
    if not torch.cuda.is_available():raise RuntimeError('ROCm GPU unavailable')
    if psutil.virtual_memory().available<4*1024**3:raise RuntimeError('Insufficient available host RAM before training')
    # Simulation and training alternate; do not compete with the simulator's renderer.
    if any(p.info['name']=='CarlaUnreal-Win64-Shipping.exe' for p in psutil.process_iter(['name'])):raise RuntimeError('Stop CARLA before training')
    model=MODELS[args.mode]().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
    per_epoch=math.ceil(len(train)/batch_size);total_steps=per_epoch*settings['max_epochs']
    def rate(step):
        warm=max(1,per_epoch*2)
        return (step+1)/warm if step<warm else .1+.9*.5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,total_steps-warm))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
    name=f'{args.run}-{args.mode}'+('-speed' if args.benchmark else '')
    out=ROOT/'artifacts/v1'/name;out.mkdir(parents=True,exist_ok=True)
    writer=SummaryWriter(str(ROOT/'logs/tensorboard/v1'/name))
    start_epoch=0;next_batch=0;global_step=0;best=float('inf');bad_epochs=0;order=[];epoch_times=[]
    if args.resume:
        saved=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        if saved['config']!=cfg or saved['manifest_sha']!=manifest_sha:raise RuntimeError('Resume configuration or split mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        # Optimizer.load_state_dict casts parameter states while preserving host step counters.
        start_epoch=saved['epoch'];next_batch=saved['next_batch'];global_step=saved['global_step']
        best=saved['best_loss'];bad_epochs=saved['bad_epochs'];order=saved['order'];restore_rng(saved['rng'])
    elif (out/'latest.pt').exists():raise RuntimeError('Run already exists; use --resume or a new --run')
    torch.cuda.reset_peak_memory_stats();started=time.monotonic();last_save=started;timings=[];losses=[];first_parameter=next(model.parameters());before=first_parameter.detach().clone()
    stats={'mode':args.mode,'parameters':sum(p.numel() for p in model.parameters()),'batch':batch_size,'train_windows':len(train),'validation_windows':len(validation),
           'torch':torch.__version__,'hip':torch.version.hip,'gpu':torch.cuda.get_device_name(0),'precision':cfg['precision'],'benchmark':args.benchmark,
           'manifest_sha':manifest_sha,'gpu_utilization':'not measured; allocator memory is not whole-system VRAM'}
    (out/'config.json').write_text(json.dumps({'config':cfg,'runtime':stats},indent=2),encoding='utf-8')
    print('START '+json.dumps(stats),flush=True)
    def payload(epoch,next_index,permutation):
        return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng(),'epoch':epoch,'next_batch':next_index,'global_step':global_step,
                'config':cfg,'manifest_sha':manifest_sha,'order':permutation,'best_loss':best,'bad_epochs':bad_epochs,'amp_scaler':None,'precision':cfg['precision']}
    if args.benchmark:
        loader=DataLoader(train,batch_size=batch_size,shuffle=False,drop_last=True,pin_memory=True);iterator=iter(loader)
        model.train()
        for step in range(args.warmup+args.steps):
            begin=time.perf_counter()
            try:batch=next(iterator)
            except StopIteration:iterator=iter(loader);batch=next(iterator)
            batch=move(batch);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):value=loss(args.mode,model,batch)
            if not torch.isfinite(value):raise RuntimeError('Nonfinite loss')
            value.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
            seconds=time.perf_counter()-begin
            if step>=args.warmup:timings.append(seconds)
            if step%10==0:print('SPEED '+json.dumps({'step':step,'loss':float(value),'seconds':seconds,'allocated_gib':torch.cuda.max_memory_allocated()/1024**3}),flush=True)
        delta=float((first_parameter.detach()-before).abs().max())
        if not delta>0:raise RuntimeError('No parameter update')
        val,val_seconds=validate(args.mode,model,validation,batch_size)
        stats.update(samples_per_second=batch_size/statistics.mean(timings),mean_step_seconds=statistics.mean(timings),median_step_seconds=statistics.median(timings),p95_step_seconds=float(np.percentile(timings,95)),
                     peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,validation_loss=val,validation_seconds=val_seconds,parameter_delta=delta,timed_steps=args.steps)
        if args.mode=='flow':
            batch=move(next(iter(DataLoader(validation,batch_size=batch_size))));torch.cuda.synchronize();tick=time.perf_counter()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model.sample(batch,k=8,steps=16)
            torch.cuda.synchronize();stats['candidate_inference_batch_seconds']=time.perf_counter()-tick
            metric=((pred[...,:2]-batch['trajectory'][:,None,:,:2])*pred.new_tensor([50,10])).norm(dim=-1).mean(-1)
            stats['pilot_minade8_m']=float(metric.min(1).values.mean())
        (out/'speed.json').write_text(json.dumps(stats,indent=2),encoding='utf-8');writer.close();print('RESULT '+json.dumps(stats),flush=True);return
    with (out/'steps.jsonl').open('a',encoding='utf-8') as stream:
        for epoch in range(start_epoch,settings['max_epochs']):
            if epoch!=start_epoch or not order:order=np.random.default_rng(cfg['seed']+epoch).permutation(len(train)).tolist();next_batch=0
            loader=DataLoader(Subset(train,order[next_batch*batch_size:]),batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(cfg['seed']+epoch))
            epoch_start=time.monotonic();epoch_losses=[];initial_batch=next_batch;batch_clock=time.perf_counter()
            for local,batch in enumerate(loader):
                batch_index=initial_batch+local;begin=time.perf_counter();data_seconds=begin-batch_clock;batch=move(batch);optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):value=loss(args.mode,model,batch)
                if not torch.isfinite(value):raise RuntimeError('Nonfinite train loss')
                value.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step();torch.cuda.synchronize()
                global_step+=1;elapsed=time.perf_counter()-begin+data_seconds;timings.append(elapsed);epoch_losses.append(float(value.detach()))
                row={'step':global_step,'epoch':epoch+(batch_index+1)/per_epoch,'train_loss':float(value.detach()),'lr':optimizer.param_groups[0]['lr'],'grad_norm':float(grad),
                     'samples_s':len(batch['state'])/elapsed,'data_seconds':data_seconds,'allocated_gib':torch.cuda.memory_allocated()/1024**3,'reserved_gib':torch.cuda.memory_reserved()/1024**3,
                     'available_ram_gib':psutil.virtual_memory().available/1024**3,'elapsed_s':time.monotonic()-started,'remaining_max_train_s':max(0,total_steps-global_step)*statistics.mean(timings[-100:])}
                stream.write(json.dumps(row)+'\n');stream.flush()
                for key in ['train_loss','lr','grad_norm','samples_s','allocated_gib','reserved_gib','available_ram_gib','elapsed_s','remaining_max_train_s','epoch']:writer.add_scalar(key,row[key],global_step)
                if global_step%10==0:print('TRAIN '+json.dumps(row),flush=True)
                if global_step%cfg['checkpoint_steps']==0 or time.monotonic()-last_save>=cfg['checkpoint_seconds']:
                    atomic_checkpoint(out/'latest.pt',payload(epoch,batch_index+1,order));last_save=time.monotonic()
                batch_clock=time.perf_counter()
            validation_loss,validation_seconds=validate(args.mode,model,validation,batch_size)
            metrics=trajectory_metrics(model,validation,batch_size) if args.mode=='flow' and (epoch==0 or (epoch+1)%5==0 or epoch+1==settings['max_epochs']) else {}
            improved=validation_loss<best-settings['min_delta']
            if improved:best=validation_loss;bad_epochs=0
            else:bad_epochs+=1
            state=payload(epoch+1,0,[]);atomic_checkpoint(out/'latest.pt',state)
            if improved:atomic_checkpoint(out/'best.pt',state)
            row={'epoch':epoch+1,'step':global_step,'train_loss':statistics.mean(epoch_losses) if epoch_losses else None,'validation_loss':validation_loss,'best_loss':best,'bad_epochs':bad_epochs,
                 'epoch_wall_seconds':time.monotonic()-epoch_start,'validation_seconds':validation_seconds,'status':'running'}
            epoch_times.append(row['epoch_wall_seconds']);row['remaining_max_total_seconds']=statistics.mean(epoch_times[-5:])*(settings['max_epochs']-epoch-1)
            writer.add_scalar('validation_loss',validation_loss,global_step);writer.flush()
            for key,value in metrics.items():writer.add_scalar('trajectory/'+key,value,global_step)
            if metrics:
                row.update(metrics)
                (out/'trajectory-metrics.json').write_text(json.dumps({'epoch':epoch+1,**metrics},indent=2),encoding='utf-8')
            (out/'status.json').write_text(json.dumps(row,indent=2),encoding='utf-8');print('EPOCH '+json.dumps(row),flush=True)
            next_batch=0;order=[]
            if epoch+1>=settings['min_epochs'] and bad_epochs>=settings['patience']:break
    row.update(status='completed',wall_seconds=time.monotonic()-started,parameter_delta=float((first_parameter.detach()-before).abs().max()),stop_reason='validation_plateau' if bad_epochs>=settings['patience'] else 'max_epochs')
    if row['parameter_delta']<=0:raise RuntimeError('No model parameter update')
    (out/'status.json').write_text(json.dumps(row,indent=2),encoding='utf-8');writer.close();print('COMPLETE '+json.dumps(row),flush=True)

if __name__=='__main__':main()
