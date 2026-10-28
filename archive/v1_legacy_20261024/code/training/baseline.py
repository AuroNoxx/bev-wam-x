"""Masked v2 behavior cloning with validation-only checkpoint selection."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.baseline import BaselineDataset,validate_manifest
from models.baseline import ResNet18Trajectory,masked_trajectory_loss
from mini_wa_jepa.provenance import provenance,write_json
from evaluation.memory import memory_snapshot


def make_model(config):
    return ResNet18Trajectory(pooling_size=config.pooling_size,mlp_hidden=config.mlp_hidden,latent_dim=config.latent_dim)


def make_loader(dataset,config,shuffle=False):
    options=dict(batch_size=config.batch_size,shuffle=shuffle,num_workers=config.workers,
        pin_memory=True,generator=torch.Generator().manual_seed(config.seed))
    if config.workers:options.update(multiprocessing_context='spawn',persistent_workers=True,prefetch_factor=2)
    return DataLoader(dataset,**options)


def batch_to_device(batch,device):
    return {key:value.to(device,non_blocking=True) for key,value in batch.items() if key!='anchor'}


@torch.no_grad()
def evaluate(model,loader,normalizer,device,max_batches=None):
    model.eval();loss_sum=count=full_count=0;ade=fde=speed_error=0.
    means=torch.tensor(normalizer.trajectory_mean,device=device)
    stds=torch.tensor(normalizer.trajectory_std,device=device)
    for index,batch in enumerate(loader):
        if max_batches and index>=max_batches:break
        batch=batch_to_device(batch,device)
        prediction=model(batch['bev'],batch['ego'])
        loss=masked_trajectory_loss(prediction,batch['trajectory'],batch['valid'],batch['action_mask'])
        loss_sum+=float(loss)*len(prediction);count+=len(prediction)
        full=batch['valid'].all(1)&batch['action_mask']
        if full.any():
            error=(prediction.float()-batch['trajectory'])*stds
            distances=error[full,:,:2].norm(dim=-1)/normalizer.scale
            ade+=float(distances.mean(1).sum());fde+=float(distances[:,-1].sum())
            speed_error+=float(error[full,:,2].abs().mean(1).sum())/normalizer.scale
            full_count+=int(full.sum())
    return dict(loss=loss_sum/max(count,1),ade_m=ade/max(full_count,1),fde_m=fde/max(full_count,1),
                speed_mae_m_s=speed_error/max(full_count,1),frames=count,full_horizon_frames=full_count)


def log_samples(model,dataset,device,output,epoch,samples):
    from mini_wa_jepa.visualization import RerunLogger,colorize_bev_4ch
    from dataset.storage import ShardReader
    from mini_wa_jepa.geometry import ego_to_pixels
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    indices=np.linspace(0,len(dataset)-1,min(samples,len(dataset)),dtype=int)
    reader=ShardReader(dataset.root/dataset.split,cache_shards=1)
    model.eval()
    fig,axes=plt.subplots(2,4,figsize=(13,7));axes=axes.ravel()
    with RerunLogger(app_id='mini-wa-jepa-stage2-baseline',spawn=False,
                     save_path=output/'rerun'/f'epoch_{epoch:03d}.rrd') as logger,torch.no_grad():
        for slot,index in enumerate(indices):
            sample=dataset[int(index)];row=reader[int(sample['anchor'])]
            prediction=model(torch.from_numpy(sample['bev'][None]).to(device),torch.from_numpy(sample['ego'][None]).to(device))
            predicted=dataset.normalizer.inverse_trajectory(prediction[0].float().cpu().numpy())
            gt=row['future_trajectory'][row['future_valid']]
            logger.set_step(epoch*100+slot,sim_time_s=float(row['step_id'])*dataset.dt_s)
            logger.log_context(epoch=epoch,episode_id=int(row['episode_id']),episode_step=int(row['step_id']),
                seed=int(row['seed']),noise_active=bool(row['noise_active']),recovery_active=bool(row['recovery_active']))
            logger.log_bev(row['bev']);logger.log_state(row['state'])
            logger.log_trajectories(planned=predicted,ground_truth=gt)
            axes[slot].imshow(colorize_bev_4ch(row['bev']))
            for trajectory,color,label in [(predicted,'cyan','prediction'),(gt,'orange','ground truth')]:
                pts=ego_to_pixels(trajectory[:,:2],128,64)
                axes[slot].plot(pts[:,0],pts[:,1],color=color,label=label)
            axes[slot].set_title(f"val ep {row['episode_id']} / step {row['step_id']}");axes[slot].axis('off')
        if len(indices):axes[0].legend(fontsize=6)
    fig.tight_layout();fig.savefig(output/'figures'/f'epoch_{epoch:03d}.png',dpi=120);plt.close(fig)


def train(root,output,config,max_batches=None):
    if not torch.version.hip or torch.version.cuda is not None or not torch.cuda.is_available():
        raise RuntimeError('Native PyTorch ROCm device is required')
    manifest=validate_manifest(root)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for name in ['figures','rerun','tensorboard']: (output/name).mkdir()
    torch.set_num_threads(config.cpu_threads);torch.manual_seed(config.seed);np.random.seed(config.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda')
    dataset=BaselineDataset(root,'train');validation=BaselineDataset(root,'val')
    loader=make_loader(dataset,config,True);val_loader=make_loader(validation,config)
    model=make_model(config).to(device)
    parameter_count=sum(p.numel() for p in model.parameters())
    if not 15_000_000<=parameter_count<=30_000_000:raise RuntimeError(parameter_count)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay,
                                fused=False,foreach=False)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.epochs)
    writer=SummaryWriter(output/'tensorboard')
    report=dict(**provenance(config),stage=2,spec_version=2,parameter_count=parameter_count,
        dataset_path=str(Path(root).resolve()),dataset_source_tree_sha256=manifest['source_tree_sha256'],
        train_frames=len(dataset),validation_frames=len(validation),completed=False,epochs=[],
        torch_version=torch.__version__,hip=torch.version.hip,input='single BEV + current speed and yaw rate',
        checkpoint_selection='minimum validation ADE; test split untouched',max_batches=max_batches)
    write_json(output/'report.json',report)
    best=float('inf');global_step=0;started=time.perf_counter()
    try:
        for epoch in range(1,config.epochs+1):
            model.train();loss_total=seen=0;tick=time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            for batch_index,batch in enumerate(loader):
                if max_batches and batch_index>=max_batches:break
                batch=batch_to_device(batch,device);optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config.mixed_precision=='bf16'):
                    predicted=model(batch['bev'],batch['ego'])
                    loss=masked_trajectory_loss(predicted,batch['trajectory'],batch['valid'],batch['action_mask'])
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {global_step}')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not torch.isfinite(norm):raise FloatingPointError(f'Nonfinite gradient at {global_step}')
                optimizer.step();global_step+=1
                loss_value=float(loss.detach())
                loss_total+=loss_value*len(predicted);seen+=len(predicted)
                if global_step%20==0:
                    writer.add_scalar('train/action_loss',loss_value,global_step)
                    writer.add_scalar('train/gradient_norm',float(norm),global_step)
                if global_step%100==0:print(json.dumps(dict(epoch=epoch,step=global_step,loss=loss_value)),flush=True)
            metrics=evaluate(model,val_loader,validation.normalizer,device,max_batches)
            peak=torch.cuda.max_memory_allocated()/1024**3
            if peak>14:raise RuntimeError(f'Training memory gate exceeded: {peak} GiB')
            row=dict(epoch=epoch,train_loss=loss_total/max(seen,1),validation=metrics,
                     seconds=time.perf_counter()-tick,peak_allocated_gib=peak,
                     peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,memory=memory_snapshot())
            for key,value in metrics.items():writer.add_scalar(f'val/{key}',value,global_step)
            writer.add_scalar('memory/peak_allocated_gib',peak,global_step)
            checkpoint=dict(model=model.state_dict(),model_config=asdict(config),normalization=dataset.normalizer.to_dict(),
                epoch=epoch,validation=metrics,parameter_count=parameter_count,
                data_schema_version=2,provenance={key:report[key] for key in ['git_commit','goal_sha256','source_tree_sha256']})
            if metrics['ade_m']<best:
                best=metrics['ade_m'];torch.save(checkpoint,output/'best.pt');report['best_epoch']=epoch
            torch.save(dict(**checkpoint,optimizer=optimizer.state_dict()),output/'last.pt')
            if epoch%config.visualization_interval_epochs==0:
                log_samples(model,validation,device,output,epoch,config.visualization_samples)
            report['epochs'].append(row);write_json(output/'report.json',report)
            print(json.dumps(row),flush=True);scheduler.step();writer.flush()
        report['completed']=True
    except BaseException:
        import traceback
        report['error']=traceback.format_exc();raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        write_json(output/'report.json',report);writer.close()
    return report
