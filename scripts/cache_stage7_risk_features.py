"""Audit native branches and cache frozen predicted consequences for risk learning."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from configs.stage6 import Stage6Config
from configs.stage7 import Stage7Config, STAGE7_SPLITS, STAGE7_RISK_CANDIDATE_SPECS
from mini_wa_jepa.provenance import provenance, write_json
from training.action_conditioned import make_model
from training.flow_matching import file_sha256
from scripts.preflight_risk_collection import compute_risk_targets


def load_world(device='cuda'):
    cfg = Stage7Config()
    model, _ = make_model(Stage6Config(), True)
    saved = torch.load(cfg.stage6_checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model']); model.requires_grad_(False)
    return model.to(device).eval(), saved


def build(inputs, output, smoke=False, motion_root=None, collection_audit=None):
    if not torch.version.hip or torch.version.cuda:
        raise RuntimeError('Native ROCm required')
    torch.set_num_threads(4); cfg = Stage7Config()
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    groups = {s: [] for s in STAGE7_SPLITS}; partitions = []
    relevant = ('scripts/collect_stage7_risk.py', 'scripts/preflight_risk_collection.py',
                'configs/stage7.py', 'env/data_v2.py', 'env/semantic_bev.py', 'controller/trajectory_tracker.py')
    audit=json.loads(Path(collection_audit).read_text()) if collection_audit is not None else None
    reference = None
    for root in map(Path, inputs):
        m = json.loads((root/'manifest.json').read_text())
        if not smoke and (not m['completed'] or m['preflight']):
            raise ValueError(f'Incomplete/nonformal partition {root}')
        source = {k: m['source_files'][k] for k in relevant}
        if reference is None: reference = source
        if audit is not None:
            if any(digest not in audit['allowed_sources'][key] for key,digest in source.items()):
                raise ValueError('Collector differs from the explicitly audited repair versions')
        elif source != reference:raise ValueError('Collection implementations differ without a source-repair audit')
        partitions.append(dict(path=str(root), sha256=file_sha256(root/'manifest.json')))
        for g in (m['groups'][:2] if smoke else m['groups']): groups[g['split']].append((root, g))
    expected = {'train': 128, 'val': 32, 'test': 32}
    for split, items in groups.items():
        if smoke:continue
        if len(items) != expected[split] or len({g['id'] for _, g in items}) != len(items):
            raise ValueError(f'Wrong/duplicate context cohort {split}: {len(items)}')
        spec = STAGE7_SPLITS[split]
        expected_ids = {(m, spec['seed_start']+i*10000+j, step)
            for i,m in enumerate(spec['maps']) for j in range(spec['seeds_per_map'])
            for step in cfg.prefix_steps}
        if {(g['map_spec'],g['seed'],g['anchor_step']) for _,g in items} != expected_ids:
            raise ValueError('Frozen map/seed/anchor cohort mismatch')
    motion_manifest=None
    if motion_root is not None:
        motion_manifest=json.loads((Path(motion_root)/'manifest.json').read_text())
        if not motion_manifest['completed'] or (motion_manifest['smoke'] and not smoke):
            raise ValueError('Completed formal motion recovery required')
    elif not smoke:
        sample_root, sample_g = items[0]
        sample_context = np.load(sample_root / sample_g['context'])
        if np.all(sample_context['ego'] == 0):
            raise ValueError('Original placeholder ego must be recovered by exact prefix replay before formal training')
    model, saved = load_world()
    report = dict(**provenance(cfg), stage=7, completed=False, smoke=smoke, inputs=partitions,
        world_checkpoint=cfg.stage6_checkpoint, world_sha256=file_sha256(cfg.stage6_checkpoint),
        world_parameter_count=saved['parameter_count'],
        world_active_parameter_count=sum(p.numel() for p in model.base.encoder.parameters()) + sum(p.numel() for p in model.world.parameters()),
        splits={}, array_hashes={})
    if motion_root is not None:
        report['motion_recovery']=dict(path=str(motion_root),manifest_sha256=file_sha256(Path(motion_root)/'manifest.json'))
    if collection_audit is not None:
        report['collection_source_repair']=dict(path=str(collection_audit),sha256=file_sha256(collection_audit))
    write_json(output/'manifest.json', report)
    with tarfile.open(output/'source_snapshot.tar.gz', 'w:gz') as arc:
        for path in report['source_files']: arc.add(path, arcname=path)
        for path in ('GOAL.md','AGENTS.md','docs/STAGE7_PROTOCOL.md'): arc.add(path,arcname=path)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for split, items in groups.items():
            if not items:continue
            folder = output/split; folder.mkdir()
            rows=[]; records=[]; histories=[]; egos=[]; raw_hashes={}; tick=time.perf_counter()
            for gi,(root,g) in enumerate(sorted(items,key=lambda x:x[1]['id'])):
                context_path=root/g['context']; context=np.load(context_path)
                raw_hashes[str(context_path)] = file_sha256(context_path)
                ego=context['ego']
                if motion_manifest is not None:
                    recovery=motion_manifest['groups'][g['id']]
                    if recovery['source_context_sha256']!=file_sha256(context_path):raise ValueError('Recovery context changed')
                    if file_sha256(recovery['path'])!=recovery['sha256']:raise ValueError('Recovery data changed')
                    with np.load(recovery['path']) as recovered:ego=recovered['ego'].copy()
                    np.testing.assert_allclose(ego[:,0],context['states'][:,3]/10,atol=1e-6,rtol=0)
                history=model.base.encoder(torch.from_numpy(context['history']).to('cuda'))
                histories.append(history.cpu().numpy().astype(np.float16)); egos.append(ego)
                branches=g['branches']
                if [b['id'] for b in branches] != [r[0] for r in STAGE7_RISK_CANDIDATE_SPECS]:
                    raise ValueError('Candidate set/order changed')
                commands=[]; raws=[]
                for branch in branches:
                    path=root/branch['path']; digest=file_sha256(path)
                    if digest != branch['sha256']: raise ValueError(f'Raw branch hash changed: {path}')
                    raw_hashes[str(path)] = digest
                    with np.load(path) as z: raw={k:z[k] for k in z.files}
                    np.testing.assert_array_equal(raw['observed_history'],context['history'])
                    np.testing.assert_allclose(raw['observed_states'],context['states'],atol=1e-5,rtol=0)
                    flags=raw['outcome_flags']; n=len(flags)
                    if len(raw['states']) != n+1 or len(raw['executed_actions']) != n:
                        raise ValueError('State/action label alignment failure')
                    np.testing.assert_array_equal(raw['valid'],np.array([5,10,15,20])<=n)
                    np.testing.assert_array_equal(raw['waypoint_valid'],np.arange(2,21,2)<=n)
                    np.testing.assert_array_equal(raw['typed_events'].any(1),flags[:,0])
                    risk,mask,*_=compute_risk_targets(flags)
                    np.testing.assert_array_equal(raw['risk_target'],risk)
                    np.testing.assert_array_equal(raw['risk_mask'],mask)
                    raws.append(raw); commands.append(raw['command'])
                command=torch.from_numpy(np.stack(commands)).to('cuda')
                prediction=model(history[None].expand(len(branches),-1,-1,-1),
                    torch.from_numpy(np.repeat(ego[None],len(branches),0)).to('cuda'),command)
                for ci,(branch,raw) in enumerate(zip(branches,raws)):
                    target=np.zeros((4,64,256),np.float16)
                    if raw['valid'].any():
                        target[raw['valid']]=model.base.encoder(torch.from_numpy(
                            raw['future'][raw['valid']]).to('cuda')).cpu().numpy().astype(np.float16)
                    records.append(dict(group=gi,seed=g['seed'],command=raw['command'],
                        prediction=prediction[ci].cpu().numpy().astype(np.float16),target=target,
                        valid=raw['valid'],risk=raw['risk_target'],risk_mask=raw['risk_mask'],
                        realized=raw['realized'],waypoint_valid=raw['waypoint_valid']))
                    rows.append(dict(group=g['id'],candidate=branch['id'],context=str(context_path),
                        raw=str(root/branch['path']),map_spec=g['map_spec'],seed=g['seed'],
                        anchor_step=g['anchor_step'],context_type=g['context_type']))
                if gi%8==0: print(json.dumps(dict(split=split,context=gi,total=len(items))),flush=True)
            arrays={k:np.asarray([r[k] for r in records]) for k in records[0]}
            arrays.update(histories=np.asarray(histories),egos=np.asarray(egos))
            for k,v in arrays.items():
                path=folder/f'{k}.npy';np.save(path,v);report['array_hashes'][str(path.relative_to(output))]=file_sha256(path)
            write_json(folder/'rows.json',rows);write_json(folder/'raw_hashes.json',raw_hashes)
            y=arrays['risk']; mask=arrays['risk_mask']
            report['splits'][split]=dict(contexts=len(items),samples=len(rows),
                positives=(y*mask).sum(0).astype(int).tolist(),
                negatives=((1-y)*mask).sum(0).astype(int).tolist(),
                censored=(~mask).sum(0).tolist(),seeds=sorted(set(arrays['seed'].tolist())),
                elapsed_s=time.perf_counter()-tick)
            write_json(output/'manifest.json',report)
    report['completed']=True;write_json(output/'manifest.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--motion',type=Path)
    p.add_argument('--collection-audit',type=Path)
    a=p.parse_args();build(a.inputs,a.output,a.smoke,a.motion,a.collection_audit)
