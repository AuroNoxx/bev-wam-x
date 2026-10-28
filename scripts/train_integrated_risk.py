"""Matched world/zero heads on new native branches; validation selection only."""
import argparse
import json
from pathlib import Path
import tarfile
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.risk_features import RiskFeatures
from evaluation.memory import memory_snapshot
from evaluation.risk import binary_metrics, validation_threshold, paired_seed_interval
from mini_wa_jepa.provenance import provenance, write_json
from models.risk_head import ConsequenceRisk, masked_risk_loss
from scripts.train_stage7_risk import inputs, predict, visualize
from scripts.collect_integrated_navigation import digest
from training.flow_matching import compute_model_hash


def run(data_root, output):
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    if not torch.version.hip or torch.version.cuda:
        raise RuntimeError('Native ROCm required')
    train=RiskFeatures(data_root,'train');val=RiskFeatures(data_root,'val')
    if set(train.arrays['seed'])&set(val.arrays['seed']):
        raise ValueError('Seed overlap')
    cfg=dict(epochs=50,batch=64,learning_rate=1e-3,weight_decay=1e-4,seeds=[20261023,20261024],
        selection='minimum_validation_BCE',threshold='max_validation_Youden_J_lower_threshold_tie',
        architecture='unchanged_ConsequenceRisk_hidden576_proj16',calibration='threshold_only_no_test')
    report=dict(**provenance(cfg),completed=False,stage7_acceptance=False,development_only=True,
        cache_manifest_sha256=digest(data_root/'manifest.json'),models=[],world_zero=[],
        source_protocol_sha256=digest('docs/STAGE7_INTEGRATED_PROTOCOL.md'))
    write_json(output/'report.json',report)
    with tarfile.open(output/'source_snapshot.tar.gz','w:gz') as arc:
        for f in report['source_files']:
            arc.add(f,arcname=f)
        arc.add('docs/STAGE7_INTEGRATED_PROTOCOL.md')
    probabilities={}
    try:
        for seed in cfg['seeds']:
            initial=[];probabilities[seed]={}
            for world in (True,False):
                torch.manual_seed(seed);np.random.seed(seed)
                model=ConsequenceRisk(use_world=world).to('cuda');initial.append(compute_model_hash(model))
                arm='world' if world else 'zero';folder=output/f'{arm}_{seed}';folder.mkdir()
                optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
                scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50)
                loader=DataLoader(train,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
                writer=SummaryWriter(str(folder/'tensorboard'));best=float('inf');epochs=[];step=0
                try:
                    for epoch in range(1,51):
                        model.train();losses=[];torch.cuda.reset_peak_memory_stats()
                        for raw in loader:
                            logits=model(*inputs(raw));loss=masked_risk_loss(logits,raw['risk'].to('cuda'),raw['risk_mask'].to('cuda'))
                            if not torch.isfinite(loss):
                                raise FloatingPointError('Nonfinite risk loss')
                            optimizer.zero_grad(set_to_none=True);loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();step+=1
                            losses.append(float(loss.detach()));writer.add_scalar('train/BCE',losses[-1],step)
                        value,_=predict(model,val);scheduler.step();writer.add_scalar('validation/BCE',value,epoch)
                        row=dict(epoch=epoch,train_bce=float(np.mean(losses)),validation_bce=value,
                            memory=memory_snapshot(),peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
                        epochs.append(row);write_json(folder/'epochs.json',epochs)
                        if value<best:
                            best=value
                            torch.save(dict(model=model.state_dict(),arch=model.arch,epoch=epoch,seed=seed,
                                schema_version=3,initial_hash=initial[-1],cache_manifest_sha256=report['cache_manifest_sha256']),folder/'best.pt')
                        if epoch%5==0:
                            visualize(model,val,folder,epoch,step,writer)
                        print(json.dumps(dict(seed=seed,arm=arm,**row)),flush=True)
                    selected=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(selected['model'])
                    _,prob=predict(model,val);probabilities[seed][arm]=prob
                    thresholds=[validation_threshold(val.arrays['risk'][:,k][val.arrays['risk_mask'][:,k]],
                        prob[:,k][val.arrays['risk_mask'][:,k]]) for k in range(2)]
                    selected['thresholds']=thresholds;torch.save(selected,folder/'calibrated.pt')
                    metrics=[]
                    flow_flag = val.arrays.get('is_flow', np.ones(len(val.arrays['seed']), bool))
                    normal_flag = val.arrays.get('is_normal', np.zeros(len(val.arrays['seed']), bool))
                    for k in range(2):
                        mask=val.arrays['risk_mask'][:,k];tm=train.arrays['risk_mask'][:,k]
                        prevalence=float(train.arrays['risk'][:,k][tm].mean())
                        metric=binary_metrics(val.arrays['risk'][:,k][mask],prob[:,k][mask],thresholds[k],prevalence)
                        metric['positive_seed_clusters']=int(len(np.unique(val.arrays['seed'][mask&(val.arrays['risk'][:,k]>0)])))
                        # Flow-only candidate metrics
                        mask_flow = mask & flow_flag
                        if mask_flow.any():
                            m_flow = binary_metrics(val.arrays['risk'][:,k][mask_flow], prob[:,k][mask_flow], thresholds[k], prevalence)
                            metric['flow_candidates'] = dict(
                                count=int(mask_flow.sum()), positives=int((val.arrays['risk'][:,k][mask_flow]>0).sum()),
                                auroc=m_flow['auroc'], average_precision=m_flow['average_precision'], brier=m_flow['brier']
                            )
                        # Diagnostic-only candidate metrics
                        mask_diag = mask & (~flow_flag)
                        if mask_diag.any():
                            m_diag = binary_metrics(val.arrays['risk'][:,k][mask_diag], prob[:,k][mask_diag], thresholds[k], prevalence)
                            metric['diagnostic_candidates'] = dict(
                                count=int(mask_diag.sum()), positives=int((val.arrays['risk'][:,k][mask_diag]>0).sum()),
                                auroc=m_diag['auroc'], average_precision=m_diag['average_precision'], brier=m_diag['brier']
                            )
                        # Normal open-road Flow false rejection rate (误拒率)
                        norm_flow_k = mask & flow_flag & normal_flag
                        if norm_flow_k.any():
                            metric['normal_flow_false_rejection_rate'] = float((prob[:,k][norm_flow_k] > thresholds[k]).mean())
                        metrics.append(metric)
                    np.savez_compressed(folder/'validation.npz',probability=prob,target=val.arrays['risk'],
                        mask=val.arrays['risk_mask'],seed=val.arrays['seed'])
                    result=dict(seed=seed,arm=arm,selected_epoch=selected['epoch'],thresholds=thresholds,validation=metrics,
                        parameters=sum(p.numel() for p in model.parameters()),calibrated_sha256=digest(folder/'calibrated.pt'),initial_hash=initial[-1])
                    report['models'].append(result);write_json(folder/'report.json',result);write_json(output/'report.json',report)
                finally:
                    writer.close()
            if initial[0]!=initial[1]:
                raise AssertionError('Matched world/zero initial parameters differ')
            for k,name in enumerate(('crash','offroad')):
                mask=val.arrays['risk_mask'][:,k];truth=val.arrays['risk'][:,k][mask]
                a=(probabilities[seed]['world'][mask,k]-truth)**2;b=(probabilities[seed]['zero'][mask,k]-truth)**2
                ci=paired_seed_interval(a,b,val.arrays['seed'][mask],2000,20261023)
                report['world_zero'].append(dict(seed=seed,event=name,relative_Brier_gain=float((b.mean()-a.mean())/b.mean()) if b.mean()>0 else None,
                    paired_seed_interval=ci,validation_only=True))
        report['completed']=True
    except BaseException:
        report['error']=traceback.format_exc();raise
    finally:
        write_json(output/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.data,a.output)
