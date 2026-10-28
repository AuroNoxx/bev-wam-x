"""Train a whole-trajectory coefficient flow on existing expert train/val caches."""
import argparse
import json
from pathlib import Path
import random
import tarfile
import time
import traceback

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from dataset.flow_features import FlowFeaturesDataset
from mini_wa_jepa.hardware import detect_hardware
from mini_wa_jepa.provenance import provenance, write_json
from models.flow_matcher_structured import StructuredFlowMatcher, fit_trajectory_basis, sample_structured_anchors
from scripts.report_stage7 import kinematics
from scripts.run_stage7_metric_ablation import (append_json,resource_record,isolated_rng,
    serial_latency,save_predictions,fixed_validation_loss,checkpoint_inventory)
from training.flow_matching import compute_model_hash,file_sha256


@torch.no_grad()
def evaluate(model,dataset,indices,temperatures,out):
    out.mkdir(parents=True,exist_ok=False);model.eval()
    loader=DataLoader(Subset(dataset,indices.tolist()),batch_size=32,shuffle=False)
    predictions={str(t):[] for t in temperatures};targets=[];mse=[];anchors=[];episodes=[]
    for b in loader:
        h=b['history'].to('cuda');e=b['ego'].to('cuda')
        for t in temperatures:
            norm=sample_structured_anchors(model,h,e,b['anchor'].tolist(),t)
            predictions[str(t)].append(dataset.normalizer.inverse_trajectory(norm.cpu().numpy()))
        targets.append(dataset.normalizer.inverse_trajectory(b['target'].numpy()))
        mse.append(dataset.normalizer.inverse_trajectory(b['mse_prediction'].numpy()))
        anchors.append(b['anchor'].numpy());episodes.append(b['episode'].numpy())
    target=np.concatenate(targets);baseline=np.concatenate(mse)
    anchors=np.concatenate(anchors);episodes=np.concatenate(episodes)
    reference_error=np.linalg.norm(baseline[...,:2]-target[...,:2],axis=-1)
    reference=dict(ade_m=float(reference_error.mean()),fde_m=float(reference_error[:,-1].mean()))
    rows={}
    for t in temperatures:
        candidates=np.concatenate(predictions[str(t)])
        error=np.linalg.norm(candidates[...,:2]-target[:,None,:,:2],axis=-1)
        i,j=np.triu_indices(8,1)
        diversity=np.linalg.norm(candidates[:,i,:,:2]-candidates[:,j,:,:2],axis=-1).mean((1,2))
        row=dict(temperature=t,ade_m=float(error[:,0].mean()),fde_m=float(error[:,0,-1].mean()),
                 diversity_m=float(diversity.mean()),all_candidate_ade_m=float(error.mean()),
                 oracle_min_ade_m=float(error.mean(-1).min(-1).mean()),
                 second_difference_position_m=float(np.linalg.norm(np.diff(candidates[...,:2],n=2,axis=2),axis=-1).mean()),
                 **kinematics(candidates))
        row['gates']=dict(ade=row['ade_m']<=1.25*reference['ade_m'],fde=row['fde_m']<=1.25*reference['fde_m'],
                          diversity=row['diversity_m']>=.1,kinematics=row['pointwise_valid_fraction']>=.95)
        row['trajectory_gates_passed']=all(row['gates'].values());rows[str(t)]=row
        np.savez_compressed(out/f'tau_{t:g}.npz',candidates=candidates,target=target,mse=baseline,
            anchors=anchors,episodes=episodes,sample1_ade_m=error[:,0].mean(-1),sample1_fde_m=error[:,0,-1],diversity_m=diversity)
    result=dict(split='val',anchors=len(indices),mse_reference=reference,temperatures=rows,
                selection='Literal first random candidate; all 8 candidates have the same temperature',
                label_contract='Original full expert trajectory, never its PCA projection',
                kinematic_convention='Original strict moving displacement >0.1m/segment')
    write_json(out/'summary.json',result);return result


@torch.no_grad()
def projection_diagnostic(model,dataset,indices):
    """Representation floor using true labels; cannot be reported as model accuracy."""
    from models.joint import constant_turn_trajectory
    prior=dataset.normalizer.trajectory(constant_turn_trajectory(torch.from_numpy(dataset.ego[indices])).numpy())
    residual=(dataset.targets[indices]-prior)/model.residual_std.cpu().numpy()
    norm=prior+model.decode_residual(model.encode_residual(torch.from_numpy(residual).to('cuda'))).cpu().numpy()*model.residual_std.cpu().numpy()
    error=np.linalg.norm(dataset.normalizer.inverse_trajectory(norm)[...,:2]-dataset.normalizer.inverse_trajectory(dataset.targets[indices])[...,:2],axis=-1)
    return dict(split=dataset.split,anchors=len(indices),projection_ade_m=float(error.mean()),
                projection_fde_m=float(error[:,-1].mean()),
                note='Uses true residual coefficients: representation diagnostic, not generated predictions')


def train_one(root,train,val,fit,cfg,seed,indices):
    out=root/f'structured_{seed}';out.mkdir(exist_ok=False)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=StructuredFlowMatcher(dimensions=cfg['dimensions'],normalizer=train.normalizer,
        **{k:fit[k] for k in ('basis','residual_mean','coefficient_std','residual_std')}).to('cuda')
    params=sum(p.numel() for p in model.parameters());full_params=params+22535784+473094
    if not 15e6<=full_params<=30e6:raise ValueError('Policy parameter budget exceeded')
    report=dict(completed=False,seed=seed,arm='structured',config=cfg,head_parameters=params,
                full_policy_parameters=full_params,initial_model_sha256=compute_model_hash(model))
    write_json(out/'report.json',report)
    loader=DataLoader(train,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    val_loader=DataLoader(val,batch_size=64,shuffle=False)
    g=torch.Generator().manual_seed(42)
    noise=torch.randn(len(val),cfg['dimensions'],generator=g);times=torch.rand(len(val),generator=g)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs']*len(loader))
    writer=SummaryWriter(str(out/'tensorboard'));best=float('inf');step=0;started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,cfg['epochs']+1):
        model.train();totals={k:0. for k in ('flow','reconstructed_ade_m','reconstructed_fde_m','total')};count=0
        for b in loader:
            optimizer.zero_grad(set_to_none=True)
            loss,_,parts=model.compute_loss(b['history'].to('cuda'),b['ego'].to('cuda'),b['target'].to('cuda'),
                b['action_mask'].to('cuda'),b['waypoint_valid'].to('cuda'),ade_weight=.25,fde_weight=.5,return_parts=True)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at step {step}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();scheduler.step();step+=1;count+=len(b['target'])
            for k,v in parts.items():totals[k]+=v.item()*len(b['target'])
        with isolated_rng():
            validation=fixed_validation_loss(model,val,val_loader,noise,times,[.25,.5])
            if validation['flow']<best:
                best=validation['flow'];report['selected_epoch']=epoch
                torch.save(dict(model=model.state_dict(),model_arch=model.model_arch,epoch=epoch,val_flow=best,
                     config=cfg,seed=seed,basis_sha256=file_sha256(root/'trajectory_basis.npz')),out/'best.pt')
            row=dict(epoch=epoch,global_step=step,train={k:v/count for k,v in totals.items()},val=validation,
                     elapsed_s=time.perf_counter()-started)
            for k,v in row['train'].items():writer.add_scalar('train/'+k,v,step)
            for k,v in validation.items():writer.add_scalar('val/'+k,v,step)
            if epoch==1 or epoch%5==0 or epoch==cfg['epochs']:
                save_predictions(model,val,out/'samples'/f'epoch_{epoch:03}',epoch,step)
                probe=evaluate(model,val,indices[:64],[1.],out/'probes'/f'epoch_{epoch:03}')
                row['probe_tau1']=probe['temperatures']['1.0']
                for k in ('ade_m','fde_m','diversity_m'):writer.add_scalar('sampled_probe_tau1/'+k,row['probe_tau1'][k],step)
        append_json(out/'history.jsonl',row);append_json(out/'resources.jsonl',dict(epoch=epoch,**resource_record()))
        writer.flush()
        print(json.dumps(dict(run=out.name,epoch=epoch,val_flow=validation['flow'],seconds=row['elapsed_s'],
                              probe=row.get('probe_tau1'))),flush=True)
    model.load_state_dict(torch.load(out/'best.pt',map_location='cuda',weights_only=False)['model'])
    result=evaluate(model,val,indices,cfg['temperatures'],out/'validation')
    timing=serial_latency(model,val,indices,cfg['temperatures']);write_json(out/'serial_latency.json',timing)
    for t,row in result['temperatures'].items():
        row['head_p95_ms']=timing[t]['p95_ms'];row['gates']['latency']=timing[t]['passes_30ms']
        row['all_gates_passed']=all(row['gates'].values())
    write_json(out/'validation'/'summary.json',result)
    report.update(completed=True,validation=result,projection_diagnostic=projection_diagnostic(model,val,indices),
                  training_seconds=time.perf_counter()-started,best_checkpoint_sha256=file_sha256(out/'best.pt'),resource=resource_record())
    write_json(out/'report.json',report);writer.close()
    print(json.dumps(dict(run=out.name,completed=True,validation=result)),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    root=args.output;root.mkdir(parents=True,exist_ok=False);state=dict(completed=False,phase='preflight')
    try:
        if not torch.version.hip or torch.version.cuda:raise RuntimeError('Native ROCm PyTorch required')
        torch.set_num_threads(4)
        cfg=dict(epochs=1 if args.smoke else 25,batch_size=64,dimensions=12,
            training_seeds=[20261021] if args.smoke else [20261021,20261022],
            temperatures=[1.] if args.smoke else [.4,.6,.8,1.],weights=[.25,.5],
            source='N(0,I_12), same train/inference coefficient coordinates; temperature sensitivity is validation-only',
            basis_fit='Train-only PCA of standardized residuals around analytical kinematic prior',
            checkpoint_selection='Minimum fixed-target validation coefficient-flow MSE; no sampled metric epoch selection',
            baseline_root='artifacts/restart_stage7/metric_loss_v2_001',
            comparison='Prior metric-loss runs, same data/seeds/25 epochs/optimizer/body; coefficient state and output head change',
            precision='float32',backend='ROCm / functional SDPA MATH',candidates=8,euler_steps=10,
            test_access=False,smoke=args.smoke)
        prov=provenance(cfg);cfg['source_tree_sha256']=prov['source_tree_sha256']
        write_json(root/'provenance.json',prov);write_json(root/'config.json',cfg)
        with tarfile.open(root/'source.tar.gz','w:gz') as archive:
            for f in prov['source_files']:archive.add(f,arcname=f)
            for f in ('GOAL.md','AGENTS.md','docs/STAGE7_STRUCTURED_ABLATION.md'):archive.add(f,arcname=f)
        write_json(root/'hardware.json',detect_hardware().to_dict())
        before=checkpoint_inventory();write_json(root/'protected_checkpoints.json',before)
        train=FlowFeaturesDataset('artifacts/restart_stage7/flow_features_001','train')
        val=FlowFeaturesDataset('artifacts/restart_stage7/flow_features_001','val')
        if args.smoke:
            for d,n in ((train,128),(val,64)):
                for name in ('history_indices','ego','targets','waypoint_valid','action_mask','anchors','episodes','steps','mse_predictions'):
                    setattr(d,name,getattr(d,name)[:n])
        fit=fit_trajectory_basis(train,12);np.savez_compressed(root/'trajectory_basis.npz',**fit)
        write_json(root/'basis_summary.json',dict(split='train',anchors=len(train),dimensions=12,
                  explained_variance=fit['explained_variance'],basis_sha256=file_sha256(root/'trajectory_basis.npz')))
        indices=np.random.default_rng(20261023).permutation(len(val))
        cohort=dict(train_anchors=train.anchors.tolist(),val_anchors=val.anchors.tolist(),validation_order=indices.tolist(),
                    feature_manifest_sha256=file_sha256(train.root/'manifest.json'),normalization_sha256=file_sha256(train.root/'normalization.json'))
        write_json(root/'cohort.json',cohort)
        if not args.smoke:
            previous=json.loads((Path(cfg['baseline_root'])/'cohort.json').read_text())
            if cohort!=previous:raise AssertionError('Previously measured control cohort differs')
        runs={}
        with sdpa_kernel(SDPBackend.MATH):
            for seed in cfg['training_seeds']:
                state.update(phase='training',seed=seed);write_json(root/'status.json',state)
                runs[f'structured_{seed}']=train_one(root,train,val,fit,cfg,seed,indices)
        for path,digest in before.items():
            if file_sha256(path)!=digest:raise AssertionError(f'Prior checkpoint changed: {path}')
        state.update(completed=True,phase='experiments_complete',protected_checkpoints_unchanged=True)
        write_json(root/'summary.json',dict(config=cfg,runs=runs,**state));write_json(root/'status.json',state)
    except BaseException:
        state['error']=traceback.format_exc();write_json(root/'status.json',state);raise


if __name__=='__main__':main()
