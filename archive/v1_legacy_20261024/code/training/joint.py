"""End-to-end deterministic world/action learning with validation-only selection."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.baseline import validate_manifest
from dataset.joint import JointSequences
from evaluation.latent import LatentMoments
from evaluation.memory import memory_snapshot
from mini_wa_jepa.calibration import recommend_training_parameters
from mini_wa_jepa.hardware import detect_hardware
from mini_wa_jepa.oom_recovery import OOMRecoveryManager
from mini_wa_jepa.operators import configure_operator_optimizations, sdpa_context
from mini_wa_jepa.provenance import provenance, write_json
from models.joint import JointModel
from training.joint_loss import joint_loss


def make_loader(dataset, config, shuffle=False):
    kwargs = dict(batch_size=config.batch_size, shuffle=shuffle, drop_last=shuffle,
        num_workers=config.workers, pin_memory=True, generator=torch.Generator().manual_seed(config.seed))
    if config.workers:
        kwargs.update(multiprocessing_context='spawn', persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)


def to_device(batch, device):
    return {key:value.to(device, non_blocking=True) for key,value in batch.items()
            if key in ('bev','ego','future','world_valid','trajectory','valid','action_mask')}


@torch.no_grad()
def evaluate(model, loader, config, device, max_batches=None):
    model.eval(); moments = LatentMoments(model.encoder.token_count, config.encoder_dim, device)
    anchors=[];episodes=[];full_masks=[];ade=[];fde=[];speed=[];heading=[];prediction=[];truth=[]
    noisy=[];recovery=[];latent_sum=copy_sum=world_count=0.;action_sum=action_count=0.
    sums=np.zeros(4);horizon_counts=np.zeros(4,np.int64);frame_count=0
    for index, raw in enumerate(loader):
        if max_batches and index>=max_batches:break
        batch=to_device(raw,device);output=model(batch['bev'],batch['ego'],batch['future'])
        predicted=output['trajectory'].float(); target=batch['trajectory']
        physical_error=(predicted-target)*model.trajectory_std
        physical_error[...,:3]/=model.trajectory_scale
        distances=physical_error[...,:2].norm(dim=-1)
        full=batch['valid'].all(1)&batch['action_mask']
        adequate=batch['valid'].any(1)&batch['action_mask']
        action_count+=int(adequate.sum())
        from models.baseline import masked_trajectory_loss
        action_sum+=float(masked_trajectory_loss(predicted,target,batch['valid'],batch['action_mask']))*int(adequate.sum())
        p=(predicted*model.trajectory_std+model.trajectory_mean)
        g=(target*model.trajectory_std+model.trajectory_mean)
        p[...,:3]/=model.trajectory_scale;g[...,:3]/=model.trajectory_scale
        anchors.extend(raw['anchor'].numpy());episodes.extend(raw['episode'].numpy())
        full_masks.extend(full.cpu().numpy());noisy.extend(raw['noisy'].numpy());recovery.extend(raw['recovery'].numpy())
        ade.extend(distances.mean(1).cpu().numpy());fde.extend(distances[:,-1].cpu().numpy())
        speed.extend(physical_error[...,2].abs().mean(1).cpu().numpy())
        heading.extend(torch.atan2(physical_error[...,3].sin(),physical_error[...,3].cos()).abs().mean(1).cpu().numpy())
        prediction.append(p.cpu().numpy());truth.append(g.cpu().numpy())
        mask=batch['world_valid'];error=(output['latent'].float()-output['target']).square().mean((-1,-2))
        copy=(output['current'][:,None]-output['target']).square().mean((-1,-2))
        latent_sum+=float((error*mask).sum());copy_sum+=float((copy*mask).sum());world_count+=int(mask.sum())
        sums+=(error*mask).sum(0).cpu().numpy();horizon_counts+=mask.sum(0).cpu().numpy()
        if int(mask[:,0].sum()):moments.update(output['target'][:,0][mask[:,0]])
        frame_count+=len(raw['bev'])
    health,health_arrays=moments.finish()
    arrays=dict(anchors=np.asarray(anchors),episodes=np.asarray(episodes),full_mask=np.asarray(full_masks),
        ade_m=np.asarray(ade),fde_m=np.asarray(fde),speed_mae_m_s=np.asarray(speed),heading_mae_rad=np.asarray(heading),
        prediction=np.concatenate(prediction),target=np.concatenate(truth),noisy=np.asarray(noisy),recovery=np.asarray(recovery),
        **health_arrays)
    full=arrays['full_mask']
    metrics=dict(ade_m=float(arrays['ade_m'][full].mean()),fde_m=float(arrays['fde_m'][full].mean()),
        speed_mae_m_s=float(arrays['speed_mae_m_s'][full].mean()),heading_mae_rad=float(arrays['heading_mae_rad'][full].mean()),
        action_loss=action_sum/max(action_count,1),action_frames=action_count,
        full_horizon_frames=int(full.sum()),latent_mse=latent_sum/max(world_count,1),
        copy_last_mse=copy_sum/max(world_count,1),prediction_copy_ratio=latent_sum/max(copy_sum,1e-15),
        world_valid_targets=int(world_count),horizon_mse=(sums/np.maximum(horizon_counts,1)).tolist(),**health)
    # Latent health's `frames` is valid +0.1s targets; retain both unambiguous counts.
    metrics['latent_health_frames']=health['frames'];metrics['frames']=frame_count
    return metrics,arrays


@torch.no_grad()
def visualize(model,dataset,config,device,output,step,epoch,writer=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from mini_wa_jepa.geometry import ego_to_pixels
    from mini_wa_jepa.visualization import RerunLogger,colorize_bev_4ch
    model.eval();folder=output/'samples'/f'step_{step:06d}';folder.mkdir(parents=True,exist_ok=False)
    selected=[i for i in config.visualization_sample_ids if i<len(dataset)]
    samples=[dataset[i] for i in selected]
    x=torch.from_numpy(np.stack([s['bev'] for s in samples])).to(device)
    e=torch.from_numpy(np.stack([s['ego'] for s in samples])).to(device)
    y=torch.from_numpy(np.stack([s['future'] for s in samples])).to(device)
    result={k:v.float().cpu().numpy() for k,v in model(x,e,y).items()}
    trajectories=dataset.normalizer.inverse_trajectory(result['trajectory'])
    raw_targets=dataset.normalizer.inverse_trajectory(np.stack([s['trajectory'] for s in samples]))
    np.savez_compressed(folder/'predictions.npz',**result,trajectory_physical=trajectories,target_physical=raw_targets,
        valid=np.stack([s['valid'] for s in samples]),world_valid=np.stack([s['world_valid'] for s in samples]),
        sample_ids=np.asarray(selected),anchors=np.asarray([s['anchor'] for s in samples]),global_step=step,epoch=epoch)
    manifest=json.loads((dataset.root/'manifest.json').read_text());by_episode={r['episode_id']:r for r in manifest['episodes']}
    paths=[]
    for slot,(index,sample) in enumerate(zip(selected,samples)):
        plan=trajectories[slot];gt=raw_targets[slot][sample['valid']];mask=sample['world_valid'];metadata=by_episode[sample['episode']]
        stem=f'sample_{index:05d}'
        with RerunLogger(app_id='mini-wa-jepa-stage4-joint',spawn=False,save_path=folder/f'{stem}.rrd') as logger:
            logger.set_step(step,sample['step']*dataset.dt_s)
            logger.log_context(global_step=step,epoch=epoch,sample_id=index,anchor=sample['anchor'],
                split=dataset.split,episode_id=sample['episode'],episode_step=sample['step'],seed=metadata['seed'],
                map_spec=metadata['map_spec'],noise_active=sample['noisy'],recovery_active=sample['recovery'],
                history_padding=sample['history_padding'],latent_horizons_s=[.1,.2,.3,.4],waypoint_dt_s=.2,
                latent_weight=config.latent_weight,visualization_seed=config.visualization_seed)
            logger.log_bev(sample['bev'][-1]);logger.log_state(dataset.states[sample['anchor']]);logger.log_camera(None)
            logger.log_trajectories(planned=plan,ground_truth=gt)
            if mask.any():logger.log_latents(predicted=result['latent'][slot][mask],ground_truth=result['target'][slot][mask])
        fig,axes=plt.subplots(2,3,figsize=(12.8,6.4),dpi=100)
        axes[0,0].imshow(colorize_bev_4ch(sample['bev'][-1]))
        for values,color,label in [(plan,'cyan','prediction'),(gt,'orange','realized target')]:
            pixels=ego_to_pixels(values[:,:2],128,64)
            axes[0,0].plot(pixels[:,0],pixels[:,1],color=color,label=label)
        axes[0,0].set_title('Current BEV and 2s trajectories');axes[0,0].legend(fontsize=6);axes[0,0].axis('off')
        axes[0,1].plot(np.arange(1,11)*.2,plan[:,2],label='prediction')
        axes[0,1].plot((np.arange(1,11)*.2)[sample['valid']],raw_targets[slot,sample['valid'],2],label='target')
        axes[0,1].set_xlabel('future seconds');axes[0,1].set_ylabel('speed m/s');axes[0,1].legend(fontsize=7)
        axes[0,2].plot(plan[:,0],plan[:,1],label='prediction');axes[0,2].plot(gt[:,0],gt[:,1],label='target')
        axes[0,2].set_xlabel('ego forward m');axes[0,2].set_ylabel('ego left m');axes[0,2].legend(fontsize=7)
        pred=result['latent'][slot][mask].reshape(-1,config.encoder_dim).T
        target=result['target'][slot][mask].reshape(-1,config.encoder_dim).T
        if pred.size:
            bound=max(float(np.quantile(np.abs(np.concatenate([pred,target],1)),.99)),.01)
            for ax,value,label in [(axes[1,0],pred,'Predicted latents'),(axes[1,1],target,'True future latents')]:
                ax.imshow(value,aspect='auto',cmap='coolwarm',vmin=-bound,vmax=bound);ax.set_title(label)
                ax.set_xlabel('future horizon x spatial token');ax.set_ylabel('channel')
            axes[1,2].imshow(np.abs(pred-target),aspect='auto',cmap='magma');axes[1,2].set_title('Absolute latent error')
        else:
            for ax in axes[1]:ax.text(.1,.5,'No valid world future');ax.axis('off')
        fig.suptitle(f'Stage 4 | epoch {epoch} step {step} | {dataset.split} {metadata["map_spec"]} seed {metadata["seed"]} | '
                     f'latent weight {config.latent_weight}\nLatent heatmaps are features, not reconstructed pixels',fontsize=10)
        fig.tight_layout(rect=(0,0,1,.92));path=folder/f'{stem}.png';fig.savefig(path);plt.close(fig);paths.append(str(path))
        if writer:writer.add_image(f'joint/sample_{index}',np.asarray(Image.open(path).convert('RGB')).transpose(2,0,1),step)
    write_json(folder/'manifest.json',dict(epoch=epoch,global_step=step,sample_ids=selected,paths=paths,
        trajectory_horizon_s=2.,latent_horizon_s=.4,latent_weight=config.latent_weight))
    return paths


def train(root,output,config,max_batches=None):
    hardware = detect_hardware()
    target_device = getattr(config, 'device', 'cuda')
    if target_device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA/ROCm GPU device required')
        device = torch.device('cuda')
    else:
        device = torch.device(target_device)
    manifest=validate_manifest(root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(config.cpu_threads);torch.manual_seed(config.seed);np.random.seed(config.seed)
    configure_operator_optimizations(
        vendor=hardware.gpu.vendor,
        allow_tf32=hardware.runtime.tf32_supported,
        cudnn_benchmark=False,
        deterministic=True,
    )
    if getattr(config, 'auto_calibrate', False):
        rec = recommend_training_parameters(hardware, config)
        print(hardware.summary(), flush=True)
        print(rec.summary(), flush=True)
    vram_cap = config.max_vram_gb if getattr(config, 'max_vram_gb', None) is not None else (
        hardware.gpu_total_vram_gib * 0.95 if hardware.gpu_total_vram_gib > 0 else 14.0
    )
    data=JointSequences(root,'train')
    val_full=JointSequences(root,'val');validation=JointSequences(root,'val',config.validation_pairs,config.seed)
    loader=make_loader(data,config,True);val_loader=make_loader(validation,config)
    model=JointModel(config,data.normalizer).to(device)
    state=torch.load(config.stage3_checkpoint,map_location='cpu',weights_only=False)
    if state['data_schema_version']!=2:raise ValueError('Stage 3 checkpoint schema mismatch')
    model.initialize_stage3(state);del state
    total=sum(p.numel() for p in model.parameters())
    if not 15e6<=total<=30e6:raise ValueError(total)
    groups=[dict(params=model.encoder.parameters(),lr=config.encoder_learning_rate),
        dict(params=list(model.predictor.parameters())+list(model.history_fusion.parameters()),lr=config.predictor_learning_rate),
        dict(params=list(model.action_projection.parameters())+list(model.action_head.parameters()),lr=config.learning_rate)]
    optimizer=torch.optim.AdamW(groups,weight_decay=config.weight_decay,fused=False,foreach=False)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.epochs)
    writer=SummaryWriter(output/'tensorboard')
    initial_digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        initial_digest.update(key.encode());initial_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    report=dict(**provenance(config),stage=4,spec_version=2,completed=False,epochs=[],visualizations=[],
        max_batches=max_batches,parameter_count=total,encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
        dataset=str(Path(root).resolve()),dataset_source=manifest['source_tree_sha256'],
        train_frames=len(data),validation_frames=len(validation),torch_version=torch.__version__,hip=torch.version.hip,
        sdpa_backend=getattr(config, 'sdpa_backend', 'MATH'),vram_limit_gib=vram_cap,
        hardware_profile=hardware.to_dict(),normalization=data.normalizer.to_dict(),
        initialization_sha256=hashlib.sha256(Path(config.stage3_checkpoint).read_bytes()).hexdigest(),
        initial_model_sha256=initial_digest.hexdigest(),
        checkpoint_selection='minimum fixed validation ADE; test untouched during training',
        action_prior='constant observed speed/yaw-rate arc plus learned normalized residual; no route/future state')
    write_json(output/'report.json',report);step=0;best=float('inf');started=time.perf_counter();latest=None
    vendor = hardware.gpu.vendor if hardware else 'AMD'
    active_sdpa = getattr(config, 'sdpa_backend', 'MATH')
    oom_mgr = OOMRecoveryManager(
        max_retries=getattr(config, 'oom_max_retries', 3),
        log_path=output/'oom_recovery_log.json'
    ) if getattr(config, 'enable_oom_recovery', False) else None
    current_batch_size = config.batch_size
    current_accum_steps = getattr(config, 'gradient_accumulation_steps', 1)
    try:
        with sdpa_context(active_sdpa, vendor=vendor):
            initial,_=evaluate(model,val_loader,config,device,max_batches)
            report['initial_validation']=initial
            report['visualizations'].append(dict(step=0,epoch=0,paths=visualize(model,val_full,config,device,output,0,0,writer)))
            write_json(output/'report.json',report)
            for epoch in range(1,config.epochs+1):
                model.train();tick=time.perf_counter();sums=np.zeros(6);seen=0;torch.cuda.reset_peak_memory_stats()
                for batch_index,raw in enumerate(loader):
                    if max_batches and batch_index>=max_batches:break
                    latest=raw;batch=to_device(raw,device)
                    if (batch_index % current_accum_steps) == 0:
                        optimizer.zero_grad(set_to_none=True)
                    try:
                        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config.mixed_precision=='bf16'):
                            output_model=model(batch['bev'],batch['ego'],batch['future'] if config.latent_weight else None)
                        loss,terms=joint_loss(output_model,batch,config)
                        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at step {step}')
                        loss_scaled = loss / current_accum_steps
                        loss_scaled.backward()
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        if oom_mgr and oom_mgr.is_oom_error(e):
                            can_recover, current_batch_size, current_accum_steps = oom_mgr.handle_oom(
                                e, model, optimizer, scheduler, epoch, step, output, current_batch_size, current_accum_steps
                            )
                            if can_recover:
                                optimizer.zero_grad(set_to_none=True)
                                continue
                        raise
                    if (batch_index + 1) % current_accum_steps == 0 or (batch_index + 1 == len(loader)):
                        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip)
                        if not torch.isfinite(norm):raise FloatingPointError(f'Nonfinite gradient at step {step}')
                        optimizer.step();step+=1
                    values=[float(loss.detach())]+[float(terms[k].detach()) for k in ['action','prediction','variance','covariance','latent']]
                    sums+=np.asarray(values)*len(raw['bev']);seen+=len(raw['bev'])
                    if step%20==0:
                        for key,value in zip(['total','action','prediction','variance','covariance','latent'],values):
                            writer.add_scalar(f'train/{key}',value,step)
                        writer.add_scalar('train/gradient_norm',float(norm),step)
                    if step%50==0:
                        progress=dict(epoch=epoch,step=step,seen=seen,loss=values,elapsed_s=time.perf_counter()-started)
                        write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
                    if step%config.visualization_interval_steps==0:
                        paths=visualize(model,val_full,config,device,output,step,epoch,writer)
                        report['visualizations'].append(dict(step=step,epoch=epoch,paths=paths));model.train()
                metrics,arrays=evaluate(model,val_loader,config,device,max_batches)
                row=dict(epoch=epoch,global_step=step,train_frames_seen=seen,train_loss=(sums/max(seen,1)).tolist(),
                    validation=metrics,seconds=time.perf_counter()-tick,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,memory=memory_snapshot())
                report['epochs'].append(row)
                np.savez_compressed(output/f'validation_epoch_{epoch:03d}.npz',**arrays)
                for key in ['ade_m','fde_m','action_loss','latent_mse','copy_last_mse','effective_rank_ratio','pooled_effective_rank_ratio','std_mean']:
                    writer.add_scalar(f'val/{key}',metrics[key],step)
                writer.add_histogram('val/channel_std',np.asarray(metrics['channel_std']),step)
                writer.add_histogram('val/covariance_eigenvalues',np.asarray(metrics['eigenvalues']),step)
                writer.add_scalar('memory/peak_allocated_gib',row['peak_allocated_gib'],step)
                saved=dict(model=model.state_dict(),model_config=asdict(config),epoch=epoch,global_step=step,
                    normalization=data.normalizer.to_dict(),validation=metrics,data_schema_version=2,parameter_count=total,
                    provenance={k:report[k] for k in ['git_commit','source_tree_sha256','goal_sha256']})
                if metrics['ade_m']<best:
                    best=metrics['ade_m'];torch.save(saved,output/'best.pt');report['best_epoch']=epoch
                torch.save(dict(**saved,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),output/'last.pt')
                if report['visualizations'][-1]['step']!=step:
                    paths=visualize(model,val_full,config,device,output,step,epoch,writer)
                    report['visualizations'].append(dict(step=step,epoch=epoch,paths=paths))
                write_json(output/'report.json',report);writer.flush()
                print(json.dumps(dict(epoch=epoch,step=step,ade=metrics['ade_m'],fde=metrics['fde_m'],
                    rank=metrics['effective_rank_ratio'],std=metrics['std_mean'],seconds=row['seconds'],peak_gib=row['peak_allocated_gib'])),flush=True)
                if row['peak_allocated_gib'] > vram_cap:
                    raise RuntimeError(f'Training VRAM ({row["peak_allocated_gib"]:.2f} GiB) exceeded limit ({vram_cap:.2f} GiB)')
                if metrics['effective_rank']<2 or metrics['std_mean']<.01:
                    raise RuntimeError('Severe latent degeneration; preserve MRE and stop')
                scheduler.step()
            report['completed']=True
    except BaseException:
        report['error']=traceback.format_exc()
        if latest is not None:
            torch.save(dict(batch=latest,model=model.state_dict(),config=asdict(config),normalization=data.normalizer.to_dict(),
                global_step=step),output/'failure_mre.pt')
        raise
    finally:
        report['last_global_step']=step;report['elapsed_seconds']=time.perf_counter()-started
        write_json(output/'report.json',report);writer.close()
    return report


