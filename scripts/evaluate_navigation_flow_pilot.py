"""Audit development Flow predictions and paired Euler/RK2 component probes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Subset
from mini_wa_jepa.provenance import provenance, write_json
from models.navigation_flow import NavigationFlow
from planning.navigation_reference import navigation_reference
from scripts.train_navigation_pilot import CachedNavigation
from scripts.train_navigation_flow import candidate_metrics, inputs, permission_inputs
from scripts.collect_integrated_navigation import digest
from scripts.report_stage7 import kinematics


def run(root, output, resnet=None):
    output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(4)
    if not torch.version.hip or torch.version.cuda:
        raise RuntimeError('Native ROCm required')
    training = json.loads((root/'report.json').read_text()); assert training['completed']
    cache = Path(training['config']['cache']); data = CachedNavigation(cache/'validation')
    reference = np.load(root/'validation'/'reference.npy', mmap_mode='r')
    indices = np.linspace(0, len(data)-1, 256, dtype=int)
    probes = []; audit = []; rrd_count = 0
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for trained in training['models']:
            folder = root/f"flow_{trained['seed']}"
            assert digest(folder/'best.pt') == trained['checkpoint_sha256']
            selected = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
            model = NavigationFlow(**selected['arch'], residual_std=selected['residual_std']).to('cuda').eval()
            model.load_state_dict(selected['model'])
            with np.load(folder/'validation.npz') as raw:
                candidates = raw['candidates']; target = raw['target']
                np.testing.assert_array_equal(target, data.arrays['trajectory'][data.arrays['anchors']])
                baselines = {}
                for name, base in [('joint_mse', cache), ('resnet_mse', resnet or root.parent/'resnet_navigation_pilot_005')]:
                    with np.load(base/f"head_{trained['seed']}"/'validation.npz') as other:
                        np.testing.assert_array_equal(target, other['target'])
                        error = np.linalg.norm(other['prediction'][..., :2]-target[..., :2], axis=-1)
                        baselines[name] = dict(ade_m=float(error.mean()), fde_m=float(error[:, -1].mean()))
                metrics = candidate_metrics(candidates, target); kin = kinematics(candidates)
                audit.append(dict(seed=trained['seed'], validation=metrics, kinematics=kin,
                    matched_baselines=baselines, validation_only_checks={
                        'ade_vs_joint':metrics['nominal_ade_m']<=1.25*baselines['joint_mse']['ade_m'],
                        'fde_vs_joint':metrics['nominal_fde_m']<=1.25*baselines['joint_mse']['fde_m'],
                        'ade_vs_resnet':metrics['nominal_ade_m']<=1.25*baselines['resnet_mse']['ade_m'],
                        'fde_vs_resnet':metrics['nominal_fde_m']<=1.25*baselines['resnet_mse']['fde_m'],
                        'diversity':metrics['pairwise_diversity_m']>=.1,
                        'kinematics':kin['pointwise_valid_fraction']>=.95},
                    raw_sha256=digest(folder/'validation.npz')))
            for solver, steps in [('euler', 10), ('euler', 3), ('midpoint', 3)]:
                plans = []; truths = []
                for batch in DataLoader(Subset(data, indices), batch_size=32, shuffle=False):
                    batch['reference'] = torch.from_numpy(np.array(reference[batch['sample'].numpy()]))
                    pred = model.sample(*inputs(batch), candidates=8, steps=steps, solver=solver, seed=20261025,
                                        **permission_inputs(model,batch))
                    plans.append(pred.cpu().numpy()); truths.append(batch['trajectory'].numpy())
                plans = np.concatenate(plans); target = np.concatenate(truths)
                np.savez_compressed(output/f"{trained['seed']}_{solver}{steps}.npz", indices=indices,
                                    candidates=plans, target=target)
                latency = []; with_prior = []
                # Sequentially measure one scene / eight candidates; warm up8.
                for count, index in enumerate(indices[:72]):
                    item = data[int(index)]
                    batch = {k: torch.from_numpy(v[None]) for k, v in item.items() if isinstance(v, np.ndarray)}
                    batch['reference'] = torch.from_numpy(np.array(reference[int(index)][None]))
                    observed = inputs(batch)
                    torch.cuda.synchronize(); tick = time.perf_counter()
                    model.sample(*observed, candidates=8, steps=steps, solver=solver, seed=20261025,
                                 **permission_inputs(model,batch))
                    torch.cuda.synchronize(); elapsed = (time.perf_counter()-tick)*1000
                    tick = time.perf_counter()
                    ref = navigation_reference(*[batch[k].numpy() for k in ('ego', 'route_points', 'route_valid', 'vehicle')])
                    model.sample(*observed[:-1], torch.from_numpy(ref).to('cuda'), candidates=8,
                                 steps=steps, solver=solver, seed=20261025,**permission_inputs(model,batch))
                    torch.cuda.synchronize(); elapsed_with_prior = (time.perf_counter()-tick)*1000
                    if count >= 8:
                        latency.append(elapsed); with_prior.append(elapsed_with_prior)
                probes.append(dict(seed=trained['seed'], solver=solver, steps=steps,
                    vector_field_evaluations=steps*(2 if solver=='midpoint' else 1),
                    metrics=candidate_metrics(plans, target), kinematics=kinematics(plans),
                    head_p95_ms=float(np.percentile(latency, 95)),
                    reference_plus_head_p95_ms=float(np.percentile(with_prior, 95))))
    for path in root.glob('flow_*/epoch_*/predictions.rrd'):
        with (output/f'{path.parent.parent.name}_{path.parent.name}_rrd.txt').open('w') as stream:
            subprocess.run([sys.executable, '-m', 'rerun', 'rrd', 'print', str(path)],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        rrd_count += 1
    write_json(output/'audit.json', dict(**provenance(), completed=True, stage7_acceptance=False,
               training_report_sha256=digest(root/'report.json'), full_validation=audit,
               compression_probes=probes, training_rrd_readbacks=rrd_count,
               timing_scope='cached observations/reference+Flow head only; not Full decision latency',
               periodic_allocator_memory_trend='recorded_in_epochs' if any('memory' in r for r in json.loads((root/'flow_20261023'/'epochs.json').read_text())) else 'not_recorded_keep_missing'))
    lines = ['# 导航Flow组件开发结果（非阶段验收）', '',
             '| 训练种子 | 候选0 ADE/FDE(m) | 多样性(m) | 运动学有效点 |', '|---|---|---|---|']
    for row in audit:
        m = row['validation']; k = row['kinematics']
        lines.append(f"| {row['seed']} | {m['nominal_ade_m']:.4f}/{m['nominal_fde_m']:.4f} | {m['pairwise_diversity_m']:.4f} | {k['pointwise_valid_fraction']:.2%} |")
    lines += ['', f'使用{cache.name}的实际观察/目标缓存及声明路线先验；两训练种子均完成25epoch，检查点按固定验证Flow loss选择。完整验证标签/索引与Joint-MSE、ResNet逐元素对齐，{rrd_count}份周期训练RRD原生读回。先验在训练/推断完全相同，不混入未经训练的候选动作先验。', '',
              '| 种子 | 积分器 | VF前向次数 | 256锚点候选0 ADE(m) | 多样性(m) | head P95(ms) | 先验+head P95(ms) |', '|---|---|---|---|---|---|---|']
    for r in probes:
        m = r['metrics']
        lines.append(f"| {r['seed']} | {r['solver']}{r['steps']} | {r['vector_field_evaluations']} | {m['nominal_ade_m']:.4f} | {m['pairwise_diversity_m']:.4f} | {r['head_p95_ms']:.2f} | {r['reference_plus_head_p95_ms']:.2f} |")
    lines += ['', '只在重复使用的开发验证集检查精度/多样性/物理指标；不是科学盲测Gate1，也没有Full/WAM闭环/到达/风险增益证据。组件计时不含BEV/导航构造/编码器/几何/世界模型/风险头/控制器，不能作为≤50ms全系统验收。',
              '运动学采用历史冻结点级定义：首点只查有限值和速度，后续0.1m以上位移检查航向；首点相对实际origin/速度的更严格诊断仍需单独完成。资源趋势按实际epoch日志读取，旧006缺失趋势保持缺失。']
    (output/'PROGRESS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(audit=audit, probes=probes)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True);p.add_argument('--resnet',type=Path)
    args = p.parse_args(); run(args.root, args.output,args.resnet)
