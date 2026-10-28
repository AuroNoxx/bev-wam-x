"""Train-only detail-retention control for the 12D whole-trajectory experiment."""
import argparse
import json
from pathlib import Path
import tarfile
import traceback

import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from dataset.flow_features import FlowFeaturesDataset
from mini_wa_jepa.hardware import detect_hardware
from mini_wa_jepa.provenance import provenance,write_json
from models.flow_matcher_structured import fit_trajectory_basis
from scripts.run_stage7_structured_ablation import train_one
from scripts.run_stage7_metric_ablation import checkpoint_inventory
from training.flow_matching import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=False)
    state=dict(completed=False,phase='preflight')
    try:
        if not torch.version.hip or torch.version.cuda:raise RuntimeError('Native ROCm required')
        torch.set_num_threads(4)
        previous=Path('artifacts/restart_stage7/structured_v2_001')
        if not json.loads((previous/'checks/delivery.json').read_text()).get('completed'):
            raise ValueError('Complete and preserve the12D experiment before starting20D')
        cfg=json.loads((previous/'config.json').read_text())
        cfg.update(dimensions=20,previous_structured_root=str(previous),
                   source='N(0,I_20), same train/inference coefficient coordinates; temperature sensitivity is validation-only',
                   design_amendment='Train-only detail-retention control: PCA20 retains99.9744% vs PCA12 99.8023%; train projection FDE2.19cm vs8.21cm. Fixed before20D validation.',
                   smoke=False)
        cfg.pop('source_tree_sha256',None)
        prov=provenance(cfg);cfg['source_tree_sha256']=prov['source_tree_sha256']
        write_json(root/'provenance.json',prov);write_json(root/'config.json',cfg)
        with tarfile.open(root/'source.tar.gz','w:gz') as archive:
            for file in prov['source_files']:archive.add(file,arcname=file)
            for file in ('GOAL.md','AGENTS.md','docs/STAGE7_STRUCTURED_ABLATION.md'):archive.add(file,arcname=file)
        write_json(root/'hardware.json',detect_hardware().to_dict())
        before=checkpoint_inventory();write_json(root/'protected_checkpoints.json',before)
        protected={str(p):file_sha256(p) for directory in (previous,Path(cfg['baseline_root']))
                   for p in directory.rglob('*') if p.is_file()}
        write_json(root/'protected_reference_files.json',protected)
        train=FlowFeaturesDataset('artifacts/restart_stage7/flow_features_001','train')
        val=FlowFeaturesDataset('artifacts/restart_stage7/flow_features_001','val')
        fit=fit_trajectory_basis(train,20);np.savez_compressed(root/'trajectory_basis.npz',**fit)
        write_json(root/'basis_summary.json',dict(split='train',anchors=len(train),dimensions=20,
                   explained_variance=fit['explained_variance'],basis_sha256=file_sha256(root/'trajectory_basis.npz')))
        cohort=json.loads((previous/'cohort.json').read_text());write_json(root/'cohort.json',cohort)
        if cohort['train_anchors']!=train.anchors.tolist() or cohort['val_anchors']!=val.anchors.tolist():
            raise AssertionError('Expert cohort changed')
        indices=np.asarray(cohort['validation_order']);runs={}
        with sdpa_kernel(SDPBackend.MATH):
            for seed in cfg['training_seeds']:
                state.update(phase='training',seed=seed);write_json(root/'status.json',state)
                runs[f'structured_{seed}']=train_one(root,train,val,fit,cfg,seed,indices)
        for path,digest in before.items():
            if file_sha256(path)!=digest:raise AssertionError(f'Prior checkpoint changed:{path}')
        state.update(completed=True,phase='experiments_complete',protected_checkpoints_unchanged=True)
        write_json(root/'summary.json',dict(config=cfg,runs=runs,**state));write_json(root/'status.json',state)
    except BaseException:
        state['error']=traceback.format_exc();write_json(root/'status.json',state);raise


if __name__=='__main__':main()
