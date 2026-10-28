"""Matched world/zero-world risk learning with validation-only selection."""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import tarfile
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from configs.stage7 import Stage7Config
from dataset.risk_features import RiskFeatures
from evaluation.risk import validation_threshold
from mini_wa_jepa.hardware import detect_hardware
from mini_wa_jepa.provenance import provenance, write_json
from models.risk_head import ConsequenceRisk, masked_risk_loss
from models.flow_matcher import ConditionalFlowMatcher
from training.flow_matching import compute_model_hash, file_sha256, get_all_rng_states


def inputs(raw):
    return [raw[k].float().to('cuda') for k in ('history','ego','command','prediction')]


@torch.no_grad()
def predict(model,data,batch_size=64):
    model.eval();logits=[];numerator=denominator=0.
    for raw in DataLoader(data,batch_size=batch_size,shuffle=False):
        output=model(*inputs(raw));y=raw['risk'].to('cuda');mask=raw['risk_mask'].to('cuda')
        if not torch.isfinite(output).all():raise FloatingPointError('Nonfinite risk output')
        if mask.any():
            numerator+=float(masked_risk_loss(output,y,mask))*int(mask.sum());denominator+=int(mask.sum())
        logits.append(output.cpu().numpy())
    if not denominator:raise ValueError('No valid validation outcomes')
    logits=np.concatenate(logits)
    return numerator/denominator,1/(1+np.exp(-np.clip(logits,-80,80)))


@torch.no_grad()
def visualize(model,data,out,epoch,step,writer):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mini_wa_jepa.geometry import ego_to_pixels
    from mini_wa_jepa.visualization import RerunLogger
    from mini_wa_jepa.visualization.rerun_logger import colorize_bev_4ch
    folder=out/'samples'/f'epoch_{epoch:03d}';folder.mkdir(parents=True)
    model.eval()
    # Fixed metadata-defined validation cases, selected before test outcomes exist.
    roadside=next((i for i,r in enumerate(data.rows) if r['context_type']=='right_turn_roadside'
                  and r['candidate']=='right_12m_fast'),min(13,len(data)-1))
    for index in (0,roadside):
        item=data[index];raw={k:torch.from_numpy(v[None]) if isinstance(v,np.ndarray) else v for k,v in item.items()}
        probability=model(*inputs(raw)).sigmoid()[0].cpu().numpy();meta=data.rows[index]
        ctx=np.load(meta['context']);bev=ctx['history'][-1]
        fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].imshow(colorize_bev_4ch(bev))
        for path,style,label in ((item['command'],'--','command'),
                (item['realized'][item['waypoint_valid']],'-','actual')):
            xy=ego_to_pixels(path[:,:2],128,64);axes[0].plot(xy[:,0],xy[:,1],style,label=label)
        axes[0].legend();axes[0].set_title(meta['candidate'])
        errors=np.square(item['prediction'].astype(float)-item['target'].astype(float)).mean((-1,-2))
        axes[1].plot([.5,1,1.5,2],np.where(item['valid'],errors,np.nan),'-o')
        axes[1].set_title('Actual fixed-command latent error (diagnostic)')
        fig.suptitle(f'Risk P(crash/offroad)={probability.round(3)} actual={item["risk"]} valid={item["risk_mask"]}')
        fig.tight_layout();fig.savefig(folder/f'{index}.png');plt.close(fig)
        np.savez_compressed(folder/f'{index}.npz',**item,probability=probability,
            raw_source=meta['raw'],context_source=meta['context'],candidate_id=meta['candidate'])
        with RerunLogger(spawn=False,save_path=folder/f'{index}.rrd') as log:
            log.set_step(step,meta['anchor_step']*.1);log.log_bev(bev)
            log.log_context(epoch=epoch,global_step=step,group=meta['group'],candidate=meta['candidate'],
                risk=probability.tolist(),risk_target=item['risk'].tolist(),risk_valid=item['risk_mask'].tolist())
            log.log_candidate(meta['candidate'],commanded=item['command'],
                actual=item['realized'][item['waypoint_valid']],
                predicted_latent=item['prediction'][item['valid']],target_latent=item['target'][item['valid']])
        writer.add_scalar(f'fixed_case/{index}_crash_probability',float(probability[0]),step)


def train(data_root,output,seed,use_world=True,smoke=False):
    if not torch.version.hip or torch.version.cuda:raise RuntimeError('Native ROCm required')
    torch.set_num_threads(4);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    cfg=replace(Stage7Config(),seed=seed);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    data=RiskFeatures(data_root,'train');val=RiskFeatures(data_root,'val')
    if data.manifest.get('smoke') and not smoke:
        raise ValueError('Formal training cannot use a smoke cache')
    if set(data.arrays['seed']) & set(val.arrays['seed']):raise ValueError('Seed leakage')
    model=ConsequenceRisk(use_world,hidden=cfg.risk_hidden_dim,dropout=cfg.risk_dropout).to('cuda')
    initial_hash=compute_model_hash(model);head_count=sum(p.numel() for p in model.parameters())
    with torch.random.fork_rng(devices=[]):
        from models.flow_matcher_transformer import TransformerKinematicFlowMatcher
        flow_count=sum(p.numel() for p in TransformerKinematicFlowMatcher(token_count=64,token_dim=256,d_model=384,n_heads=6,n_layers=5,dim_feedforward=1536).parameters())
    world_p = data.manifest.get('world_active_parameter_count') or (80_926_276 if data.manifest['world_parameter_count'] > 50e6 else data.manifest['world_parameter_count'])
    total=world_p+flow_count+head_count
    if not 50e6<=total<=100e6:raise ValueError(f'Planner parameter count {total}')
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(data,batch_size=cfg.batch_size,shuffle=True,num_workers=0,generator=generator)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.risk_learning_rate,weight_decay=cfg.weight_decay)
    epochs=2 if smoke else cfg.risk_epochs
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    report=dict(**provenance(cfg),stage=7,completed=False,smoke=smoke,use_world=use_world,
        initial_hash=initial_hash,head_parameters=head_count,planner_parameters=total,
        cache_manifest_sha256=file_sha256(Path(data_root)/'manifest.json'),
        hardware=detect_hardware().to_dict(),epochs=[],visualizations='fixed validation cases every 5 epochs')
    write_json(out/'report.json',report)
    with tarfile.open(out/'source_snapshot.tar.gz','w:gz') as arc:
        for path in report['source_files']:arc.add(path,arcname=path)
        for path in ('GOAL.md','AGENTS.md','docs/STAGE7_PROTOCOL.md'):arc.add(path,arcname=path)
    writer=SummaryWriter(str(out/'tensorboard'));step=0;best=float('inf');started=time.perf_counter()
    def checkpoint(epoch):
        return dict(model=model.state_dict(),arch=model.arch,optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
            rng=get_all_rng_states(loader),epoch=epoch,step=step,provenance=report,initial_hash=initial_hash)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            scratch=copy.deepcopy(model);scratch_opt=torch.optim.AdamW(scratch.parameters(),lr=cfg.risk_learning_rate)
            batch=next(iter(DataLoader(data,batch_size=cfg.batch_size)));torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize();tick=time.perf_counter()
            loss=masked_risk_loss(scratch(*inputs(batch)),batch['risk'].to('cuda'),batch['risk_mask'].to('cuda'))
            loss.backward();scratch_opt.step();torch.cuda.synchronize()
            report['calibration']=dict(seconds=time.perf_counter()-tick,peak_vram_bytes=torch.cuda.max_memory_allocated(),
                initialization_unchanged=compute_model_hash(model)==initial_hash)
            del scratch,scratch_opt
        write_json(out/'calibration.json',report['calibration'])
        visualize(model,val,out,0,0,writer)
        for epoch in range(1,epochs+1):
            model.train();losses=[];tick=time.perf_counter()
            for raw in loader:
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(*inputs(raw));loss=masked_risk_loss(logits,raw['risk'].to('cuda'),raw['risk_mask'].to('cuda'))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite risk loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.gradient_clip)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                opt.step();losses.append(float(loss.detach()));step+=1
                writer.add_scalar('train/bce',float(loss.detach()),step);writer.add_scalar('train/gradient_norm',float(norm),step)
                writer.add_scalar('resources/peak_vram_bytes',torch.cuda.max_memory_allocated(),step)
            value,_=predict(model,val);scheduler.step()
            writer.add_scalar('val/bce',value,step)
            report['epochs'].append(dict(epoch=epoch,train_bce=float(np.mean(losses)),val_bce=value,
                seconds=time.perf_counter()-tick))
            state=checkpoint(epoch);torch.save(state,out/'last.pt')
            if value<best:best=value;torch.save(state,out/'best.pt')
            if epoch%5==0 or epoch==epochs:visualize(model,val,out,epoch,step,writer)
            write_json(out/'report.json',report);print(json.dumps(report['epochs'][-1]),flush=True)
        saved=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model'])
        _,probability=predict(model,val)
        thresholds=[validation_threshold(val.arrays['risk'][:,k][val.arrays['risk_mask'][:,k]],
            probability[:,k][val.arrays['risk_mask'][:,k]]) for k in range(2)]
        prevalence=[float(data.arrays['risk'][:,k][data.arrays['risk_mask'][:,k]].mean()) for k in range(2)]
        write_json(out/'validation.json',dict(thresholds=thresholds,train_prevalence=prevalence,
            selected_epoch=saved['epoch'],best_val_bce=best,rule='max Youden J; lower threshold breaks ties'))
        np.savez_compressed(out/'validation_raw.npz',probability=probability,risk=val.arrays['risk'],mask=val.arrays['risk_mask'])
        report.update(completed=True,elapsed_s=time.perf_counter()-started,best_val_bce=best)
    except BaseException:
        report['error']=traceback.format_exc();torch.save(checkpoint(locals().get('epoch',0)),out/'emergency.pt')
        if 'raw' in locals():torch.save(raw,out/'failure_batch.pt')
        raise
    finally:
        writer.close();write_json(out/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=20261021);p.add_argument('--zero-world',action='store_true')
    p.add_argument('--smoke',action='store_true');a=p.parse_args();train(a.data,a.output,a.seed,not a.zero_world,a.smoke)
