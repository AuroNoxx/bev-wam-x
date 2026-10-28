"""Audit native branch labels and cache frozen WAM features with causal inputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from dataset.action_branches import realized_waypoints
from models.integrated_world import load_integrated_world
from configs.stage7 import Stage7Config
from mini_wa_jepa.provenance import provenance, write_json
from scripts.collect_integrated_navigation import digest
from scripts.preflight_risk_collection import compute_risk_targets


def build(sources, output):
    output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(4)
    if not torch.version.hip or torch.version.cuda:
        raise RuntimeError('Native ROCm required')
    model = load_integrated_world('cuda')
    report = dict(**provenance(), completed=False, stage=7, development_only=True, test_access=False,
        world_active_parameter_count=sum(p.numel() for p in model.parameters()),
        world_checkpoint=Stage7Config().stage6_checkpoint,world_sha256=digest(Stage7Config().stage6_checkpoint),
        world_parameter_count=sum(p.numel() for p in model.world.parameters()), splits={}, sources=[], array_hashes={})
    write_json(output/'manifest.json', report)
    with tarfile.open(output/'source_snapshot.tar.gz','w:gz') as arc:
        for f in report['source_files']:
            arc.add(f,arcname=f)
        arc.add('docs/STAGE7_INTEGRATED_PROTOCOL.md')
    try:
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for source in sources:
                manifest = json.loads((source/'manifest.json').read_text())
                if not manifest['completed'] or not manifest['control_verified']:
                    raise ValueError('Risk source must be complete and control-verified')
                split = {'train':'train','validation':'val','probe':'probe'}[manifest['split']]
                folder=output/split;folder.mkdir()
                report['sources'].append(dict(path=str(source),manifest_sha256=digest(source/'manifest.json')))
                histories=[];egos=[];rows=[];records=[];raw_hashes={};native_readbacks=0
                for gi, group in enumerate(manifest['groups']):
                    context_path=source/group['context']
                    assert digest(context_path)==group['context_sha256']
                    raw_hashes[str(context_path)]=group['context_sha256']
                    with np.load(context_path) as context:
                        history=context['history'];observed=context['states'];ego=context['ego'];commands=context['command']
                        np.testing.assert_array_equal(context['prefix_actions'],context['prefix_applied'])
                    assert [b['name'] for b in group['branches']]==manifest['candidate_names']
                    h=model.encoder(torch.from_numpy(history).to('cuda'))
                    histories.append(h.half().cpu().numpy());egos.append(ego)
                    prediction=model(h[None].expand(len(commands),-1,-1,-1),
                        torch.from_numpy(np.repeat(ego[None],len(commands),0)).to('cuda'),torch.from_numpy(commands).to('cuda'))
                    for ci,branch in enumerate(group['branches']):
                        path=source/branch['path'];assert digest(path)==branch['sha256'];raw_hashes[str(path)]=branch['sha256']
                        with np.load(path) as z:
                            a={k:z[k] for k in z.files}
                        np.testing.assert_array_equal(a['action'],a['applied_action'])
                        np.testing.assert_array_equal(a['observed_history'],history)
                        np.testing.assert_allclose(a['observed_states'],observed,atol=1e-5,rtol=0)
                        np.testing.assert_array_equal(a['command'],commands[ci])
                        np.testing.assert_array_equal(a['typed_events'].any(1),a['events'][:,0])
                        risk,mask,*_=compute_risk_targets(a['events'][:,[0,4]])
                        np.testing.assert_array_equal(a['risk'],risk);np.testing.assert_array_equal(a['risk_mask'],mask)
                        n=len(a['action']);assert len(a['states'])==n+1 and len(a['future_bev'])==n
                        indices=np.array([5,10,15,20]);valid=indices<=n
                        target=np.zeros((4,64,256),np.float16)
                        if valid.any():
                            target[valid]=model.encoder(torch.from_numpy(a['future_bev'][indices[valid]-1]).to('cuda')).half().cpu().numpy()
                        wp=np.arange(2,21,2);way_valid=wp<=n;realized=np.zeros((10,4),np.float32)
                        realized[way_valid]=realized_waypoints(a['states'],0,wp[way_valid])
                        is_flow = bool(branch['name'].startswith('flow_'))
                        is_normal = bool(group.get('kind') == 'open_road')
                        records.append(dict(group=gi,seed=group['seed'],command=commands[ci],prediction=prediction[ci].half().cpu().numpy(),
                            target=target,valid=valid,risk=risk,risk_mask=mask,realized=realized,waypoint_valid=way_valid,
                            is_flow=is_flow,is_normal=is_normal))
                        rows.append(dict(group=f"{group['seed']}_{group['anchor']}",candidate=branch['name'],
                            context=str(context_path.resolve()),raw=str(path.resolve()),seed=group['seed'],
                            anchor_step=group['anchor'],context_type=group['kind'],is_flow=is_flow,is_normal=is_normal))
                        rrd=path.with_suffix('.rrd');assert digest(rrd)==branch['rrd_sha256']
                        with (folder/f"rrd_{gi}_{ci}.txt").open('w') as stream:
                            subprocess.run([sys.executable,'-m','rerun','rrd','print',str(rrd)],stdout=stream,stderr=subprocess.STDOUT,check=True)
                        native_readbacks+=1
                    print(json.dumps(dict(split=split,context=gi+1,total=len(manifest['groups']))),flush=True)
                arrays={k:np.asarray([r[k] for r in records]) for k in records[0]}
                arrays.update(histories=np.asarray(histories),egos=np.asarray(egos))
                for key,value in arrays.items():
                    path=folder/f'{key}.npy';np.save(path,value);report['array_hashes'][str(path.relative_to(output))]=digest(path)
                write_json(folder/'rows.json',rows);write_json(folder/'raw_hashes.json',raw_hashes)
                y=arrays['risk'];mask=arrays['risk_mask'];flow_mask=arrays['is_flow'][:,None]
                report['splits'][split]=dict(contexts=len(histories),samples=len(rows),
                    positives=(y*mask).sum(0).astype(int).tolist(),valid=mask.sum(0).astype(int).tolist(),
                    positives_flow=(y*mask*flow_mask).sum(0).astype(int).tolist(),
                    positives_diag=(y*mask*(~flow_mask)).sum(0).astype(int).tolist(),
                    valid_flow=(mask*flow_mask).sum(0).astype(int).tolist(),
                    valid_diag=(mask*(~flow_mask)).sum(0).astype(int).tolist(),
                    native_rrd_readbacks=native_readbacks)
                write_json(output/'manifest.json',report)
        report['completed']=True
    except BaseException:
        import traceback
        report['error']=traceback.format_exc();raise
    finally:
        write_json(output/'manifest.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.sources,a.output)
