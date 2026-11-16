"""Acceptance of the completed v3 artifact and both model data adapters, CPU only."""
import json
from pathlib import Path
import numpy as np
from dataset_v1 import ROOT, digest
from build_aligned_cache import check_sample, atomic_json
from lightning_task import OccData, OccSamples


def main():
    old_path=ROOT/'data/occ_v1/manifest.json'
    path=ROOT/'data/occ_v3/manifest.json'
    old=json.loads(old_path.read_text());manifest=json.loads(path.read_text())
    assert manifest['parent_manifest_sha256']==digest(old_path)
    assert manifest['preprocess_version']==3 and not manifest['pilot']
    assert manifest['splits']['test']==[] and manifest['reserved_test']['status']=='unread_unmodified_not_prepared'
    for split in ['train','validation']:
        assert manifest['splits'][split]==old['splits'][split]
    checked=0;byte_count=0;boundary_checks=0
    for row in manifest['routes']:
        audit=json.loads((ROOT/'artifacts/diagnostics/alignment-v3/routes'/(row['route']+'.json')).read_text())
        assert [a['sample'] for a in audit['samples']]==row['samples']
        for a in audit['samples']:
            source=ROOT/'data/occ_v1'/a['sample'];target=ROOT/'data/occ_v3'/a['sample']
            assert digest(source)==a['old_sha256']
            assert digest(target)==a['new_sha256']
            byte_count+=target.stat().st_size;checked+=1
        for name in [row['samples'][0],row['samples'][-1]]:
            with np.load(ROOT/'data/occ_v1'/name) as source, np.load(ROOT/'data/occ_v3'/name) as target:
                check_sample(target,source,digest(ROOT/'data/occ_v1'/name))
                boundary_checks+=1
    settings=json.loads((ROOT/'configs/lightning-stage1-v3.json').read_text())
    data=OccData('flow',settings)
    loader=data.train_dataloader()
    batch=next(iter(loader))
    assert batch['history'].shape==(8,4,20,64,160)
    assert batch['state'].shape==(8,20) and batch['trajectory'].shape==(8,25,5)
    world=OccSamples(manifest,'validation','world')[0]
    assert world['future'].shape==(5,20,64,160)
    result={'pass':True,'manifest_sha256':digest(path),'counts':manifest['counts'],
            'all_cache_hash_pairs_verified':checked,'boundary_array_checks':boundary_checks,
            'train_loader_workers':settings['workers'],'train_loader_batch_size':len(batch['state']),
            'world_loader_pass':True,'test_data_opened':False,'old_cache_unchanged':True,
            'new_cache_bytes':byte_count,'preprocess_seconds':manifest['preprocess_seconds']}
    atomic_json(ROOT/'artifacts/diagnostics/alignment-v3/final-verification.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
