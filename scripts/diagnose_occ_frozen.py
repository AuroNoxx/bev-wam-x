"""Read-only model audit: paired OCC interventions, solver and endpoint probes.

Writes only a new diagnostic directory. No optimizer, fit, test split or new weights.
Input interventions are sensitivity probes, not physically valid counterfactuals.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from lightning_task import ROOT, OccSamples, configure_runtime
from models_v1 import FlowPlanner
from stage1_objective import kinematic_residual


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def groups(state, target):
    speed = target[..., 4] * 30
    xy = target[..., :2] * [50, 10]
    yaw = np.arctan2(target[..., 2], target[..., 3])
    return {
        'all': np.ones(len(state), bool),
        'stationary': (np.abs(state[:, 0]*108) < 1) & (np.abs(speed)*3.6 < 1).all(1) & (np.linalg.norm(xy, axis=-1) < .5).all(1),
        'deceleration': speed[:, -1] - state[:, 0]*30 < -2,
        'strict_cruise': (state[:, 0]*108 >= 5) & (np.abs(speed-state[:, 0, None]*30) < 1).all(1) & (np.abs(xy[..., 1]) < 1).all(1) & (np.abs(yaw) < .1).all(1),
        'acceleration': speed[:, -1] - state[:, 0]*30 > 2,
    }


def summarize(pred, state, target, names, baseline=None):
    # S,N,K,T,D; keep candidate0 and median definitions distinct.
    xy = pred[..., :2] * [50, 10]
    error = np.linalg.norm(xy-target[None, :, None, :, :2]*[50, 10], axis=-1)
    med = np.median(xy, axis=2)
    mederr = np.linalg.norm(med-target[None, :, :, :2]*[50, 10], axis=-1)
    full = np.concatenate([np.zeros_like(xy[..., :1, :]), xy], axis=-2)
    vel = np.diff(full, axis=-2)/.2
    yawdir = pred[..., [3, 2]]
    headvel = yawdir*pred[..., 4:5]*30
    v0 = np.zeros_like(headvel[..., :1, :]); v0[..., 0, 0] = state[None, :, None, 0]*30
    prev = np.concatenate([v0, headvel[..., :-1, :]], axis=-2)
    vals = {
        'meanADE_m': error.mean((2, 3)), 'firstADE_m': error[:, :, 0].mean(-1),
        'firstFDE_m': error[:, :, 0, -1], 'minADE_m': error.mean(-1).min(-1),
        'medianADE_m': mederr.mean(-1), 'medianFDE_m': mederr[..., -1],
        'median_final_x_bias_m': med[..., -1, 0]-target[None, :, -1, 0]*50,
        'median_speed_head_final_bias_mps': np.median(pred[..., -1, 4], axis=2)*30-target[None, :, -1, 4]*30,
        'kinematic_residual_mps': np.linalg.norm(vel-(prev+headvel)/2, axis=-1).mean((2, 3)),
        'candidate_spread_m': np.sqrt(((xy-xy.mean(2, keepdims=True))**2).sum(-1).mean((2, 3))),
    }
    if baseline is not None:
        delta = (pred[..., :2]-baseline[..., :2])*[50, 10]
        vals.update(paired_candidate_displacement_m=np.linalg.norm(delta, axis=-1).mean((2, 3)),
                    paired_mean_final_x_change_m=delta[..., -1, 0].mean(2),
                    paired_mean_final_speed_change_mps=((pred[..., -1, 4]-baseline[..., -1, 4])*30).mean(2))
    routes = np.array([Path(n).parts[-2] for n in names])
    def row(mask):
        return {'windows': int(mask.sum()), 'routes': len(set(routes[mask])),
                **{k: float(v[:, mask].mean()) if mask.any() else None for k, v in vals.items()}}
    return {'groups': {k: row(v) for k, v in groups(state, target).items()},
            'routes': {r: row(routes == r) for r in sorted(set(routes))}}


def intervention(h, mode):
    if mode == 'normal': return h
    out = h.clone()
    if mode == 'vehicles_unfilled': out[out == 4] = 0
    elif mode == 'unknown': out.fill_(255)
    elif mode == 'reverse_history': out = out.flip(1)
    elif mode == 'freeze_common_coverage':
        current = h[:, -1:].expand_as(h)
        mask = (h != 255) & (current != 255)
        out[mask] = current[mask]
    else: raise ValueError(mode)
    return out


def batch_at(dataset, indices):
    samples = [dataset[int(i)] for i in indices]
    return {k: torch.stack([s[k] for s in samples]).cuda() for k in samples[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='artifacts/diagnostics/occ-frozen-audit-20261115')
    args = parser.parse_args()
    out = ROOT/args.out
    out.mkdir(parents=True, exist_ok=False)
    configure_runtime(); started = time.perf_counter()
    checkpoint = ROOT/'artifacts/lightning/flow-v3/best.ckpt'
    manifestpath = ROOT/'data/occ_v3/manifest.json'
    protected = [checkpoint, ROOT/'artifacts/lightning/flow-v3/latest.ckpt', manifestpath,
                 ROOT/'scripts/models_v1.py', ROOT/'scripts/stage1_objective.py',
                 ROOT/'scripts/lightning_task.py', ROOT/'configs/train_v1.json']
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    manifest = json.loads(manifestpath.read_text())
    assert sha(manifestpath) == ck['task_state']['manifest_sha']
    assert manifest['preprocess_version'] == 3 and ck['task_state']['settings']['manifest'] == 'data/occ_v3/manifest.json'
    model = FlowPlanner().cuda().eval()
    model.load_state_dict({k.removeprefix('model.'): v for k, v in ck['state_dict'].items()}, strict=True)
    torch.nn.utils.convert_conv2d_weight_memory_format(model, torch.channels_last)
    model.encoder.channels_last = True
    model.requires_grad_(False)
    ds = OccSamples(manifest, 'validation', 'flow')
    names = np.asarray(manifest['splits']['validation'])
    allstate, alltarget, class_counts, changed = [], [], np.zeros(9, np.int64), []
    for i in range(len(ds)):
        s = ds[i]
        allstate.append(s['state'].numpy()); alltarget.append(s['trajectory'].numpy())
        h = s['history'].numpy()
        class_counts += np.array([(h == c).sum() for c in [0,1,2,3,4,5,6,7,255]])
        common = (h[:-1] != 255) & (h[-1:] != 255)
        changed.append(float(((h[:-1] != h[-1:]) & common).sum()/max(common.sum(), 1)))
    state, target = np.stack(allstate), np.stack(alltarget)
    with np.load(ROOT/'artifacts/diagnostics/20261115-contract/evaluation-labels.npz') as f:
        assert np.array_equal(names, f['names']) and np.array_equal(state, f['state']) and np.array_equal(target, f['target'])
    np.savez_compressed(out/'labels.npz', names=names, state=state, target=target)
    modes = ['normal', 'freeze_common_coverage', 'reverse_history', 'vehicles_unfilled', 'unknown']
    outputs = {m: [] for m in modes}
    solvers = {n: [] for n in [8, 64]}
    seeds = [1908, 2911]
    for seed in seeds:
        pieces = {m: [] for m in modes}; step_parts = {n: [] for n in solvers}
        for bi, start in enumerate(range(0, len(ds), 8)):
            b = batch_at(ds, range(start, min(start+8, len(ds))))
            for mode in modes:
                bb = {**b, 'history': intervention(b['history'], mode)}
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(seed+bi)
                    p = model.sample(bb, k=8, steps=16)
                pieces[mode].append(p.cpu().numpy())
            if seed == seeds[0]:
                for steps in solvers:
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), torch.random.fork_rng(devices=[0]):
                        torch.manual_seed(seed+bi)
                        p = model.sample(b, k=8, steps=steps)
                    step_parts[steps].append(p.cpu().numpy())
            if (bi+1) % 40 == 0: print('VALIDATION', seed, start+len(b['state']), flush=True)
        for mode in modes: outputs[mode].append(np.concatenate(pieces[mode]))
        if seed == seeds[0]:
            for steps in solvers: solvers[steps] = np.concatenate(step_parts[steps])[None]
        print('SEED_DONE', seed, time.perf_counter()-started, flush=True)
    outputs = {k: np.stack(v) for k, v in outputs.items()}
    for mode, p in outputs.items(): np.savez_compressed(out/(mode+'.npz'), prediction=p)
    for steps, p in solvers.items(): np.savez_compressed(out/f'steps{steps}.npz', prediction=p)
    with np.load(ROOT/'artifacts/diagnostics/flow-v3-assessment/flow_v3.npz') as f:
        old = f['prediction'][:2]
        reproduction = {'max_abs_normalized_difference': float(np.max(np.abs(old-outputs['normal']))),
                        'mean_abs_normalized_difference': float(np.mean(np.abs(old-outputs['normal'])))}

    # Teacher-forced bridge contains the target by construction; never a driving score.
    endpoint = {}
    selected = np.unique(np.r_[np.arange(0,len(ds),8), np.flatnonzero(groups(state,target)['stationary']),
                                np.flatnonzero(groups(state,target)['deceleration'])])
    for tt in [.0, .25, .5, .75, .95]:
        ep, residual, fm = [], [], []
        for bi, start in enumerate(range(0, len(selected), 8)):
            b = batch_at(ds, selected[start:start+8]); y = b['trajectory']
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), torch.random.fork_rng(devices=[0]):
                torch.manual_seed(6100+bi); noise = torch.randn_like(y); t = torch.full((len(y),1),tt,device=y.device)
                xt = (1-tt)*noise+tt*y; v = model(b,xt,t).float(); estimate = xt+(1-tt)*v
                ep.append(estimate.cpu().numpy()); residual.append(kinematic_residual(estimate,b['state']).norm(dim=-1).mean(1).cpu().numpy())
                fm.append(((v-(y-noise))**2).mean((1,2)).cpu().numpy())
        pred = np.concatenate(ep)[None,:,None]
        endpoint[str(tt)] = {'metrics': summarize(pred,state[selected],target[selected],names[selected]),
                             'kinematic_residual_raw_mps': float(np.concatenate(residual).mean()),
                             'fm_mse': float(np.concatenate(fm).mean()), 'auxiliary_active': tt>=.5}
    print('ENDPOINT_DONE', flush=True)

    # Spread samples across every training route, plus a few train behavior examples.
    train_ds = OccSamples(manifest,'train','flow'); train_ids=[]
    train_names=np.asarray(manifest['splits']['train']); train_routes=np.array([Path(n).parts[-2] for n in train_names])
    for route in sorted(set(train_routes)):
        ids=np.flatnonzero(train_routes==route)
        train_ids.extend(ids[np.linspace(0,len(ids)-1,min(8,len(ids)),dtype=int)].tolist())
    train_ids=np.array(sorted(set(train_ids))); ts=[];ty=[];tp=[]
    for bi,start in enumerate(range(0,len(train_ids),8)):
        b=batch_at(train_ds,train_ids[start:start+8])
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),torch.random.fork_rng(devices=[0]):
            torch.manual_seed(1908+bi); p=model.sample(b,k=8,steps=16)
        ts.append(b['state'].cpu().numpy());ty.append(b['trajectory'].cpu().numpy());tp.append(p.cpu().numpy())
    ts,ty,tp=np.concatenate(ts),np.concatenate(ty),np.concatenate(tp)[None]
    np.savez_compressed(out/'train_subset.npz', names=train_names[train_ids],state=ts,target=ty,prediction=tp)
    embedding=model.encoder.embedding.weight.detach().float().cpu().numpy()
    after={str(p.relative_to(ROOT)):sha(p) for p in protected}
    assert before==after
    result={'checkpoint_sha256':before[str(checkpoint.relative_to(ROOT))], 'epoch_zero_based':ck['epoch'], 'global_step':ck['global_step'],
            'manifest_sha256':sha(manifestpath),'seeds':seeds,'reproduction':reproduction,
            'validation':{m:summarize(p,state,target,names,outputs['normal'] if m!='normal' else None) for m,p in outputs.items()},
            'solver':{str(s):summarize(p,state,target,names,outputs['normal'][:1]) for s,p in solvers.items()},
            'teacher_forced_endpoint':endpoint,'train_subset':summarize(tp,ts,ty,train_names[train_ids]),
            'input':{'class_counts_labels_0_to7_then255':class_counts.tolist(),'class_fractions':(class_counts/class_counts.sum()).tolist(),
                     'mean_history_change_within_common_coverage':float(np.mean(changed)),
                     'embedding_unknown_to_unfilled_distance':float(np.linalg.norm(embedding[8]-embedding[0])),
                     'history_seconds':.6,'native_xy_voxel_m':.8,'pooled_token_xy_spacing_m':6.4},
            'protected_hashes_before':before,'protected_hashes_after':after,'protected_unchanged':True,
            'test_opened':False,'training_run':False,'elapsed_s':time.perf_counter()-started,
            'limitations':['Validation used for model selection; windows overlap and belong to 9 routes.',
                'OCC replacements are off-distribution sensitivity probes, not retrained ablations or causal traffic responses.',
                'Teacher-forced bridge contains target information; low endpoint error is not deployment performance.',
                'Train subset differs in route and behavior mix; train/validation gap is descriptive, not a matched generalization estimate.',
                'No closed-loop/control or new collision/road-boundary validation.']}
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('DONE',json.dumps({'seconds':result['elapsed_s'],'reproduction':reproduction,'protected':True}),flush=True)


if __name__=='__main__': main()
