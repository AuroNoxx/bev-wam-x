"""Read-only report for the matched v2 validation loss experiment."""
import argparse
import json
from pathlib import Path

import numpy as np

from mini_wa_jepa.provenance import write_json


def read(path):
    return json.loads(Path(path).read_text())


def paired_episode_interval(metric, control, episodes, seed):
    """Paired episode cluster bootstrap; no independent-anchor assumption."""
    difference = np.asarray(metric, dtype=np.float64) - control
    groups = np.unique(episodes)
    sums = np.array([difference[episodes == group].sum() for group in groups])
    counts = np.array([(episodes == group).sum() for group in groups])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(2000, len(groups)))
    samples = sums[draws].sum(-1) / counts[draws].sum(-1)
    return dict(delta=float(difference.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(),
                episode_clusters=len(groups), bootstrap_samples=2000)


def make_report(root):
    data = read(root / 'summary.json'); config = data['config']; source = read(root / 'provenance.json')
    if not data['completed']: raise ValueError('Training is incomplete')
    if config['smoke']: raise ValueError('Do not report smoke as the full experiment')
    rows = []; comparisons = {}; supported = {}; all_pass = {}
    for seed in config['training_seeds']:
        for arm in ('control', 'metric'):
            run = data['runs'][f'{arm}_{seed}']
            if not run['completed']: raise ValueError('Missing completed matched run')
            for tau, metrics in run['validation']['temperatures'].items():
                rows.append(dict(seed=seed, arm=arm, tau=float(tau), **metrics))
        comparisons[str(seed)] = {}
        for tau in config['temperatures']:
            control = np.load(root / f'control_{seed}/validation/tau_{tau:g}.npz')
            metric = np.load(root / f'metric_{seed}/validation/tau_{tau:g}.npz')
            for field in ('anchors', 'episodes', 'target', 'mse'):
                if not np.array_equal(control[field], metric[field]):
                    raise AssertionError(f'Matched data differs: {field}')
            comparisons[str(seed)][str(tau)] = {field: paired_episode_interval(
                metric[field], control[field], metric['episodes'], seed)
                for field in ('sample1_ade_m', 'sample1_fde_m', 'diversity_m')}
    for tau in config['temperatures']:
        subset = [r for r in rows if r['tau'] == tau]
        all_pass[str(tau)] = all(r['all_gates_passed'] for r in subset if r['arm'] == 'metric')
        supported[str(tau)] = all(
            comparisons[str(seed)][str(tau)]['sample1_ade_m']['delta'] < 0
            and comparisons[str(seed)][str(tau)]['sample1_fde_m']['delta'] < 0
            and next(r for r in subset if r['seed'] == seed and r['arm'] == 'metric')['diversity_m'] >= .1
            for seed in config['training_seeds'])
    result = dict(experiment_completed=True, validation_only=True,
                  supports_better_precision_with_diversity_by_tau=supported,
                  both_seed_validation_gates_passed_by_tau=all_pass,
                  comparisons= comparisons, rows=rows,
                  note='Exploratory paired episode intervals; two training seeds, reused validation cohort; no test or closed-loop acceptance')
    write_json(root / 'comparison.json', result)

    # Dashboard contract: only measured open-loop validation metrics are present.
    runs = []
    for row in rows:
        runs.append(dict(run_id=f'metric_loss_v2_{row["arm"]}_{row["seed"]}_tau{row["tau"]:g}',
            model='TransformerKinematicFlowMatcher-v2', ablation=f'{row["arm"]}; tau={row["tau"]:g}; validation only',
            split='val', seed=row['seed'], training_seed=row['seed'],
            evaluation_kind='validation_loss_ablation', metrics={k:row[k] for k in ('ade_m','fde_m','diversity_m')}))
    write_json(root / 'metrics.json', dict(visualization_schema_version=1, spec_version=2, stage=7,
        config=config, seeds=config['training_seeds'], git_commit=source['git_commit'],
        goal_sha256=source['goal_sha256'], runs=runs))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for j, seed in enumerate(config['training_seeds']):
        for arm, style in [('control','o--'),('metric','s-')]:
            these = sorted([r for r in rows if r['seed'] == seed and r['arm'] == arm], key=lambda r:r['tau'])
            axes[0,j].plot([r['diversity_m'] for r in these], [r['ade_m'] for r in these], style, label=arm)
            axes[1,j].plot([r['diversity_m'] for r in these], [r['fde_m'] for r in these], style, label=arm)
            for row in these:
                axes[0,j].annotate(f'{row["tau"]:g}', (row['diversity_m'],row['ade_m']))
        ref = data['runs'][f'control_{seed}']['validation']['mse_reference']
        for ax, metric in [(axes[0,j],'ade_m'),(axes[1,j],'fde_m')]:
            ax.axvline(.1, color='grey', ls=':'); ax.axhline(1.25*ref[metric], color='red', ls=':')
            ax.set(xlabel='Pairwise diversity (m)', ylabel=metric, title=f'Validation | seed {seed}')
            ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(root / 'precision_diversity.png'); plt.close(fig)

    if any(all_pass.values()):
        verdict = '新增损失在至少一个统一温度下，使两个训练种子同时通过全部验证门槛；这仍然只是验证集证据。'
    elif any(supported.values()):
        verdict = '新增损失在至少一个统一温度下，使两个训练种子的实际路线和终点误差都下降，同时保住多样性；但没有同时通过全部验证门槛，问题尚未解决。'
    else:
        verdict = '本轮新增损失没有稳定做到两个训练种子同时改善路线/终点误差并保住多样性。不能把这项改动当成已经奏效的修复。'
    lines = ['# 阶段 7 第二版：实际路线/终点损失消融', '', verdict, '',
        '本轮为 train/val 对照实验，不改写原阶段7验收；没有加载 test，也没有新增闭环成绩。', '',
        f'两种子同时保持多样性且改善 ADE/FDE：`{supported}`。',
        f'两种子同时通过全部验证门槛：`{all_pass}`。', '',
        '## 实测结果', '',
        '| 种子 | 目标 | 温度 | ADE(m) | FDE(m) | 多样性(m) | 路点/整条通过率 | 头部P95(ms) | 全部门槛 |',
        '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f'| {r["seed"]} | {r["arm"]} | {r["tau"]:g} | {r["ade_m"]:.5f} | {r["fde_m"]:.5f} | {r["diversity_m"]:.5f} | '
            f'{r["pointwise_valid_fraction"]:.2%}/{r["all_points_valid_candidate_fraction"]:.2%} | {r["head_p95_ms"]:.2f} | {r["all_gates_passed"]} |')
    lines += ['', '## 相同随机性的逐种子比较', '']
    for seed in config['training_seeds']:
        for tau in (.8, 1.):
            a = next(r for r in rows if r['seed']==seed and r['tau']==tau and r['arm']=='control')
            b = next(r for r in rows if r['seed']==seed and r['tau']==tau and r['arm']=='metric')
            lines.append(f'- 种子{seed}，温度{tau:g}：ADE {a["ade_m"]:.5f}→{b["ade_m"]:.5f}m，'
                         f'FDE {a["fde_m"]:.5f}→{b["fde_m"]:.5f}m，'
                         f'多样性 {a["diversity_m"]:.5f}→{b["diversity_m"]:.5f}m。')
    diagnostic = root / 'train_residual_geometry.json'
    if diagnostic.exists():
        d = read(diagnostic)
        row = next(r for r in d['rows'] if r['dimensions']==8)
        lines += ['', '仅训练集的附加诊断：40维标准残差的前8个主成分解释'
                  f'{row["variance_explained"]:.2%}方差。这支持下一步检验沿整条路线的相关扰动。'
                  '该数值使用真实训练残差做投影，不能当作生成器精度或验证成绩。']
    lines += ['', '## 对照方式与解释边界', '',
        '- control 为第二版原向量场损失；metric 增加 0.25/m×重建ADE(m) 与 0.5/m×重建FDE(m)。结构、数据、初始化、训练预算和选点规则匹配。',
        '- 两臂都按固定验证向量场 MSE 选 checkpoint，实际采样结果不用来挑 epoch。权重和温度网格在本轮训练前冻结。',
        '- 重建损失作用在线性路径的干净终点估计，不是实际积分输出；最终表格来自真实10步Euler采样。辅助损失本质上改变回归权重，不能宣称引入了新的独立标签。',
        '- 8候选都使用同一温度，candidate1保持原始随机样本；没有用专家选候选或混入零噪声主候选。',
        '- 路点通过率使用原部署严格口径 >0.1m/segment；旧第二版温度脚本的 >0.2m/segment 口径不可直接混比。',
        '- 逐episode配对置信区间见 comparison.json；不把大量重叠锚点当独立样本。仅两个训练种子，不能据此宣称普遍有效。',
        '- 本轮沿用已被观察的验证集，test历史也已被前序实验观察；不能把任何旧test重跑称作新的盲测。',
        '- 头部延迟为batch1、8候选，20次预热后64次，包含上下文融合与采样，不含编码器/模拟器；原始时间保存在serial_latency.json。',
        '- 本轮没有世界潜变量预测、实际执行或闭环收益，相关指标保持缺失。阶段8没有开始。', '',
        '## 复现与交付', '',
        f'- source commit `{source["git_commit"]}`；GOAL SHA256 `{source["goal_sha256"]}`；训练源码 SHA256 `{config["source_tree_sha256"]}`。',
        '- 配置、cohort、源码快照、既有checkpoint哈希、逐epoch损失/资源、TensorBoard、周期PNG/NPZ/RRD与最终每温度原始候选均在本目录。',
        '- metrics.json 已按看板契约提供逐种子/温度验证指标；不伪造未测量闭环指标。precision_diversity.png 显示完整权衡曲线。',
        '- 本轮报告后停止，后续运动模式或结构化噪声实验需依据本轮结果另行冻结设计。', '']
    reproduction = root / 'checks/v2_control_reproduction.json'
    if reproduction.exists():
        r = read(reproduction)
        lines.append(f'第二版复现：种子20261021对照的选中checkpoint与原第二版checkpoint逐张量最大差'
                     f'{r["maximum_state_difference"]}，逐值完全一致={r["all_state_values_identical"]}。')
    delivery = root / 'checks/delivery.json'
    if delivery.exists() and read(delivery).get('completed'):
        d = read(delivery)
        lines += ['', f'交付复核：{d["tests"]}项测试通过；{d["native_rrd_files"]}个周期RRD原生读回通过；'
                  f'{d["dashboard_rows"]}行新验证指标看板解析及运行时HTML导出通过；'
                  f'{d["protected_checkpoints"]}个既有best.pt哈希未变；训练核心源码哈希与冻结快照一致。'
                  '硬件profile中的FLASH/EFFICIENT为探测能力，本轮执行显式限定SDPA MATH。']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); report = make_report(args.root)
    print(json.dumps({k:v for k,v in report.items() if k not in ('comparisons','rows')}, ensure_ascii=False))
