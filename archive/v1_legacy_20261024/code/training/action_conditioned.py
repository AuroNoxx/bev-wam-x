"""Matched frozen-target action-conditioning experiments with durable evidence."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.attention import SDPBackend,sdpa_kernel

from dataset.branch_features import BranchFeatures
from evaluation.memory import memory_snapshot
from mini_wa_jepa.hardware import detect_hardware
from mini_wa_jepa.provenance import provenance,write_json
from models.action_conditioned import ActionConditionedWorld,masked_world_error
from scripts.evaluate_joint import load_joint


def make_model(config,conditioned):
    base,_,saved=load_joint(config.checkpoint,'cpu')
    return ActionConditionedWorld(base,conditioned),saved


def loader(data,config,shuffle=False):
    return DataLoader(data,batch_size=config.batch_size,shuffle=shuffle,num_workers=0,
                      generator=torch.Generator().manual_seed(config.seed),drop_last=False)


def batch_gpu(raw):return {k:v.to('cuda',non_blocking=False) for k,v in raw.items()}


@torch.no_grad()
def evaluate(model,data,config,return_predictions=False):
    model.eval();errors=[];copies=[];masks=[];predictions=[]
    for raw in loader(data,config):
        batch=batch_gpu(raw);prediction=model(batch['history'],batch['ego'],batch['command'])
        errors.append((prediction-batch['target']).float().square().mean((-1,-2)).cpu().numpy())
        copies.append((batch['history'][:,-1,None]-batch['target']).float().square().mean((-1,-2)).cpu().numpy())
        masks.append(raw['valid'].numpy())
        if return_predictions:predictions.append(prediction.float().cpu().numpy())
    errors=np.concatenate(errors);copies=np.concatenate(copies);valid=np.concatenate(masks)
    if not np.isfinite(errors).all():raise FloatingPointError('Nonfinite validation errors')
    metrics=dict(latent_mse=float(errors[valid].mean()),copy_last_mse=float(copies[valid].mean()),
                 valid_future_count=int(valid.sum()),samples=len(data),
                 per_horizon_mse=[float(errors[:,i][valid[:,i]].mean()) if valid[:,i].any() else None for i in range(4)])
    arrays=dict(error=errors,copy_error=copies,valid=valid)
    if return_predictions:arrays['prediction']=np.concatenate(predictions)
    return metrics,arrays


@torch.no_grad()
def visualize(model,data,config,output,epoch,step,writer=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from mini_wa_jepa.geometry import ego_to_pixels
    from mini_wa_jepa.visualization import RerunLogger
    from mini_wa_jepa.visualization.rerun_logger import colorize_bev_4ch
    model.eval();output=Path(output)/'samples'/f'epoch_{epoch:03d}';output.mkdir(parents=True,exist_ok=False)
    raw_root=Path(data.manifest['data']);selected=[0,min(16,len(data)//4-1)*4]
    paths=[]
    for case,index in enumerate(selected):
        rows=[data[i] for i in range(index,index+4)]
        gpu={key:torch.from_numpy(np.stack([r[key] for r in rows])).to('cuda') for key in ('history','ego','command')}
        prediction=model(gpu['history'],gpu['ego'],gpu['command']).float().cpu().numpy()
        target=np.stack([r['target'] for r in rows]);valid=np.stack([r['valid'] for r in rows])
        meta=data.rows[index];context=np.load(raw_root/meta['context']);bev=context['history'][-1]
        names=[data.rows[index+i]['candidate'] for i in range(4)]
        fig,axes=plt.subplots(1,2,figsize=(12,5));axes[0].imshow(colorize_bev_4ch(bev))
        for ci,(r,name,color) in enumerate(zip(rows,names,['#16a6ef','#e7ad23','#cf5dd1','#ee6559'])):
            command=ego_to_pixels(r['command'][:,:2],128,64)
            actual=ego_to_pixels(r['realized'][r['waypoint_valid'],:2],128,64)
            axes[0].plot(command[:,0],command[:,1],'--',color=color,alpha=.7)
            axes[0].plot(actual[:,0],actual[:,1],'-o',color=color,markersize=2,label=name)
        axes[0].set_title('Same context: dashed command, solid actual\nExperiment branches; no learned selection')
        axes[0].legend();axes[0].set_xlim(10,117);axes[0].set_ylim(100,0)
        errors=np.square(prediction-target).mean((-1,-2))
        for ci,name in enumerate(names):axes[1].plot(config.future_times_s,np.where(valid[ci],errors[ci],np.nan),'-o',
            color=['#16a6ef','#e7ad23','#cf5dd1','#ee6559'][ci],label=name)
        axes[1].set_title('Actual latent prediction error (not decoded images)')
        axes[1].set_xlabel('Future seconds');axes[1].set_ylabel('Latent MSE');axes[1].legend()
        fig.suptitle(f'Stage 6 | epoch {epoch} | conditioned={model.conditioned} | {meta["group"]}')
        fig.tight_layout();image_path=output/f'case_{case}.png';fig.savefig(image_path);plt.close(fig)
        np.savez_compressed(output/f'case_{case}.npz',prediction=prediction,target=target,valid=valid,
                            command=np.stack([r['command'] for r in rows]),realized=np.stack([r['realized'] for r in rows]))
        with RerunLogger(app_id='mini-wa-jepa-stage6',spawn=False,save_path=output/f'case_{case}.rrd') as log:
            anchor_step=int(meta.get('anchor_step',meta['group'].rsplit('_',1)[-1]))
            log.set_step(step,anchor_step*config.control_dt_s)
            log.log_camera(np.asarray(Image.open(image_path).convert('RGB')),name='camera/branch_comparison')
            log.log_bev(bev)
            log.log_context(epoch=epoch,training_step=step,group=meta['group'],candidate_ids=names,
                selection_source='experiment-specified branches; no model selector',future_times_s=list(config.future_times_s),
                anchor_step=anchor_step,anchor_sim_time_s=anchor_step*config.control_dt_s,
                candidate_future_valid=valid.tolist(),waypoint_valid=np.stack([r['waypoint_valid'] for r in rows]).tolist(),
                latent_semantics='features, not reconstructed road or vehicles')
            for ci,r in enumerate(rows):
                log.log_candidate(names[ci],commanded=r['command'],actual=r['realized'][r['waypoint_valid']],
                    predicted_latent=prediction[ci][valid[ci]],target_latent=target[ci][valid[ci]],
                    color=[(22,166,239),(231,173,35),(207,93,209),(238,101,89)][ci])
        if writer:writer.add_image(f'cases/{case}',np.asarray(Image.open(image_path).convert('RGB')).transpose(2,0,1),step)
        paths.append(str(image_path))
    write_json(output/'manifest.json',dict(epoch=epoch,global_step=step,paths=paths,selected_indices=selected))
    return paths


def train(data_root,output,config,conditioned=True,smoke=False):
    if not torch.version.hip or torch.version.cuda:raise RuntimeError('Native ROCm required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(config.cpu_threads);random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    hardware=detect_hardware();print(hardware.summary(),flush=True)
    torch.backends.cudnn.benchmark=False
    data=BranchFeatures(data_root,'train');val=BranchFeatures(data_root,'val')
    checkpoint_hash=hashlib.sha256(Path(config.checkpoint).read_bytes()).hexdigest()
    if data.manifest['checkpoint_sha256']!=checkpoint_hash:raise ValueError('Frozen target encoder checkpoint mismatch')
    if set(map(int,data.arrays['seed']))&set(map(int,val.arrays['seed'])):raise ValueError('Training/validation seed leakage')
    model,initial=make_model(config,conditioned);model=model.to('cuda')
    count=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not 15e6<=count<=100e6:raise ValueError(f'Parameter budget: {count}')
    digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):digest.update(key.encode());digest.update(value.cpu().contiguous().numpy().tobytes())
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config.learning_rate,
                               weight_decay=config.weight_decay,fused=False,foreach=False)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.epochs)
    writer=SummaryWriter(str(output/'tensorboard'));started=time.perf_counter();step=0;latest=None
    report=dict(**provenance(config),stage=6,completed=False,smoke=smoke,conditioned=conditioned,epochs=[],visualizations=[],
        parameter_count=count,trainable_parameters=trainable,hardware=hardware.to_dict(),
        initial_model_sha256=digest.hexdigest(),data=str(Path(data_root).resolve()),
        feature_cache_manifest_sha256=hashlib.sha256((Path(data_root)/'manifest.json').read_bytes()).hexdigest(),
        initialization_checkpoint_sha256=checkpoint_hash,
        checkpoint_selection='minimum full validation masked latent MSE; test untouched',
        sdpa_backend='MATH',encoder_frozen=True,train_samples=len(data),val_samples=len(val))
    write_json(output/'report.json',report);best=float('inf')
    try:
        with sdpa_kernel(SDPBackend.MATH):
            # Probe a smaller batch and the fixed paired-design batch before any update.
            raw=next(iter(loader(data,config)));latest=raw;report['calibration_trials']=[]
            for size in sorted({min(32,len(raw['history'])),len(raw['history'])}):
                optimizer.zero_grad(set_to_none=True)
                batch=batch_gpu({key:value[:size] for key,value in raw.items()})
                torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config.mixed_precision=='bf16'):
                    prediction=model(batch['history'],batch['ego'],batch['command'])
                    loss=masked_world_error(prediction,batch['target'],batch['valid'])
                loss.backward();torch.cuda.synchronize()
                if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Calibration failed')
                elapsed=time.perf_counter()-tick
                trial=dict(batch_size=size,seconds=elapsed,samples_per_second=size/elapsed,
                    loss=float(loss.detach()),peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,memory=memory_snapshot())
                if trial['peak_allocated_gib']>.85*hardware.gpu.total_memory_gib:
                    raise RuntimeError('Calibration exceeds dynamic hardware headroom; preserve and stop before changing fixed experiment')
                report['calibration_trials'].append(trial)
            optimizer.zero_grad(set_to_none=True)
            report['calibration']=report['calibration_trials'][-1]
            report['calibration_selection']='Fixed preregistered batch validated against smaller batch; no parameter updates during trials. First trial includes cold-start overhead.'
            report['initial_validation']=evaluate(model,val,config)[0]
            report['visualizations'].append(dict(epoch=0,paths=visualize(model,val,config,output,0,step,writer)))
            train_loader=loader(data,config,True)
            for epoch in range(1,config.epochs+1):
                model.train();total=0.;seen=0;tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                for raw in train_loader:
                    latest=raw;batch=batch_gpu(raw);optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config.mixed_precision=='bf16'):
                        prediction=model(batch['history'],batch['ego'],batch['command'])
                        loss=masked_world_error(prediction,batch['target'],batch['valid'])
                    if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss step {step}')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip)
                    if not torch.isfinite(norm):raise FloatingPointError(f'Nonfinite gradient step {step}')
                    optimizer.step();step+=1;total+=float(loss.detach())*len(raw['history']);seen+=len(raw['history'])
                    writer.add_scalar('train/latent_mse',float(loss.detach()),step);writer.add_scalar('train/gradient_norm',float(norm),step)
                metrics,arrays=evaluate(model,val,config)
                row=dict(epoch=epoch,global_step=step,train_latent_mse=total/seen,validation=metrics,
                    seconds=time.perf_counter()-tick,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,memory=memory_snapshot())
                report['epochs'].append(row)
                for key in ('latent_mse','copy_last_mse'):writer.add_scalar('val/'+key,metrics[key],step)
                writer.add_scalar('memory/peak_allocated_gib',row['peak_allocated_gib'],step)
                np.savez_compressed(output/f'validation_epoch_{epoch:03d}.npz',**arrays)
                scheduler.step()
                saved=dict(model=model.state_dict(),config=asdict(config),conditioned=conditioned,
                    epoch=epoch,global_step=step,validation=metrics,normalization=initial['normalization'],
                    base_model_config=initial['model_config'],data_schema_version=2,stage=6,parameter_count=count,
                    provenance={k:report[k] for k in ('git_commit','source_tree_sha256','goal_sha256')})
                if metrics['latent_mse']<best:
                    best=metrics['latent_mse'];report['best_epoch']=epoch;torch.save(saved,output/'best.pt')
                torch.save(dict(**saved,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    torch_rng=torch.get_rng_state(),hip_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
                    python_rng=random.getstate(),loader_rng=train_loader.generator.get_state()),output/'last.pt')
                if epoch%config.visualization_interval_epochs==0 or epoch==config.epochs:
                    report['visualizations'].append(dict(epoch=epoch,paths=visualize(model,val,config,output,epoch,step,writer)))
                writer.flush();write_json(output/'report.json',report)
                print(json.dumps(dict(conditioned=conditioned,seed=config.seed,epoch=epoch,train=row['train_latent_mse'],
                    val=metrics['latent_mse'],seconds=row['seconds'])),flush=True)
            report['completed']=True
    except BaseException:
        report['error']=traceback.format_exc()
        torch.save(dict(batch=latest,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                        config=asdict(config),epoch=locals().get('epoch',0),global_step=step,
                        torch_rng=torch.get_rng_state(),hip_rng=torch.cuda.get_rng_state_all(),
                        numpy_rng=np.random.get_state(),python_rng=random.getstate()),output/'failure_mre.pt')
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,last_global_step=step)
        write_json(output/'report.json',report);writer.close()
    return report
