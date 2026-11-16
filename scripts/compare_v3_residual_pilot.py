"""Compare prespecified latest/6620 checkpoints; no test or new training."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/diagnostics/v3-residual-pilot'


def main():
    plan = json.loads((OUT / 'experiment-plan.json').read_text())
    results = {arm: json.loads((OUT / arm / 'diagnostics/report.json').read_text()) for arm in plan['arms']}
    checks = {}
    settings = []
    for arm in plan['arms']:
        folder = ROOT / ('artifacts/lightning/flow-v3-' + arm + '-pilot10')
        ck = torch.load(folder / 'latest.ckpt', map_location='cpu', weights_only=False)
        assert ck['global_step'] == 6620 and ck['task_state']['manifest_sha'] == plan['manifest_sha256']
        assert ck['task_state']['settings']['trajectory_parameterization'] == arm
        optimizer_steps = {int(s['step']) for s in ck['optimizer_states'][0]['state'].values()}
        assert optimizer_steps == {6620}
        assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
        assert 'lr_schedulers' in ck and 'loops' in ck
        assert ck['task_state']['rng'] and ck['loops']
        settings.append(ck['task_state']['settings'])
        checks[arm] = {'step': ck['global_step'], 'epoch': ck['epoch'], 'optimizer_step': 6620,
                       'finite_parameters': True, 'checkpoint_files': [p.name for p in folder.glob('*.ckpt')]}
        assert set(checks[arm]['checkpoint_files']) == {'best.ckpt', 'latest.ckpt'}
        del ck
    differences = [k for k in set(settings[0]) | set(settings[1]) if settings[0].get(k) != settings[1].get(k)]
    assert set(differences) == {'run_name', 'trajectory_parameterization'}
    for name, expected in plan['protected_v3_checkpoint_sha256'].items():
        assert hashlib.sha256((ROOT / 'artifacts/lightning/flow-v3' / name).read_bytes()).hexdigest() == expected
    for path, expected in plan['implementation_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    a = results['absolute']['groups']; b = results['cv_residual']['groups']
    method = 'median_tangent'
    dec_gain = 1 - b['deceleration']['methods'][method]['ADE_m'] / a['deceleration']['methods'][method]['ADE_m']
    all_regress = b['all']['methods'][method]['ADE_m'] / a['all']['methods'][method]['ADE_m'] - 1
    bbox_delta = b['all']['methods'][method]['bbox_overlap'] - a['all']['methods'][method]['bbox_overlap']
    stop_delta = b['stationary']['methods'][method]['final_x_bias_m'] - a['stationary']['methods'][method]['final_x_bias_m']
    gates = {'deceleration_ADE_reduction_at_least10pct': dec_gain >= .10,
             'overall_ADE_regression_at_most5pct': all_regress <= .05,
             'bbox_regression_at_most2percentage_points': bbox_delta <= .02,
             'stationary_forward_bias_not_increased': stop_delta <= 0}
    comparison = {}
    for bucket in ['all', 'strict_moving_cruise', 'stationary', 'deceleration', 'acceleration', 'lateral_motion']:
        comparison[bucket] = {'windows': a[bucket]['windows'], 'routes': a[bucket]['routes'],
                              **{arm: report['groups'][bucket]['methods'][method] for arm, report in results.items()}}
    robustness = {name: {arm: results[arm]['groups']['all']['methods'][name]['bbox_overlap']
                        for arm in plan['arms']} for name in ['first', 'median_separate_heads', 'mean_tangent', 'medoid']}
    result = {'checkpoint_checks': checks, 'settings_differences': differences, 'protected_v3_unchanged': True,
              'code_unchanged_during_pilot': True, 'gate_method': method, 'gates': gates, 'all_gates_pass': all(gates.values()),
              'deceleration_ADE_relative_reduction': dec_gain, 'overall_ADE_relative_regression': all_regress,
              'bbox_absolute_delta': bbox_delta, 'stationary_forward_bias_delta_m': stop_delta,
              'comparison': comparison, 'risk_by_other_output_rules': robustness, 'test_opened': False,
              'limitations': ['One training seed and 10epochs per arm; not converged full training or statistical generalisation proof',
                              'Both arms evaluated at same6620steps, same60epoch LR schedule, not each best',
                              'Three sampling seeds on9routes; windows overlap and development validation already used for model selection',
                              'Diagnostic bbox uses logged actors, no continuous collision/lane/control closed-loop acceptance',
                              'No automatic promotion or extension beyond10epochs even if exploratory gates pass']}
    (OUT / 'comparison.json').write_text(json.dumps(result, indent=2))
    old = json.loads((ROOT / 'artifacts/diagnostics/v3-aggregation-feasibility/report.json').read_text())
    lines = ['# v3 匀速参考残差：10轮单因素对照结果', '',
             '2026-11-15。两组从头训练，各10轮/6620步；同种子、数据、模型规模、损失、学习率和60轮调度，只改变Flow变量是否减去当前速度匀速参考。比较相同步数latest，不挑best。', '',
             '| 分组 | 窗口/路线 | 原60轮v3 ADE | 对照10轮ADE | 残差10轮ADE | 对照框重叠率 | 残差框重叠率 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    labels = {'all':'全量','strict_moving_cruise':'严格移动巡航','stationary':'全程停车','deceleration':'减速','acceleration':'加速','lateral_motion':'横向运动代理'}
    for group, row in comparison.items():
        aa = row['absolute']; bb = row['cv_residual']; baseline = old['groups'][group]['methods'][method]
        lines.append(f"| {labels[group]} | {row['windows']}/{row['routes']} | {baseline['ADE_m']:.3f} | {aa['ADE_m']:.3f} | {bb['ADE_m']:.3f} | {aa['bbox_overlap']:.2%} | {bb['bbox_overlap']:.2%} |")
    lines += ['', '全部为每种子8候选中位数XY、切线航向/速度的单条结果；原60轮列仅供背景参考，不能与10轮差值归因于参数化。', '',
              '预登记继续门槛：', *[f'- {key}: {value}' for key, value in gates.items()], '',
              f'减速ADE相对改善 {dec_gain:.2%}；全量ADE相对退化 {all_regress:.2%}；全量框重叠变化 {bbox_delta*100:+.2f} 个百分点；停车末纵向偏差变化 {stop_delta:+.3f}m。', '',
              ('全部探索性门槛通过，可作为下一轮充分训练验证的候选；不是驾驶验收。' if all(gates.values()) else '未通过全部探索性门槛。停止本轮，不自动延长或替换原v3，不能宣称参数化已解决问题。'), '',
              '## 对失败原因的限定解释', '',
              '生成侧确有缺陷，不能只靠挑选候选解决；但本轮不支持“改成匀速参考残差就能修好减速”。它改善了总体平均误差和停车偏差，同时使加速/横向运动误差与总体风险代理恶化。只能判断该10轮方案未通过预登记门槛，不能外推所有残差/积分参数化都无效。', '',
              '风险上升不限于切线航向重建：',
              *[f"- {name}：{v['absolute']:.2%} → {v['cv_residual']:.2%}" for name, v in robustness.items()], '',
              f"原始位置差分的平均jerk：{a['all']['methods'][method]['mean_jerk_mps3']:.2f} → {b['all']['methods'][method]['mean_jerk_mps3']:.2f} m/s³。未平滑、未标定，不能解释成乘员真实受力或硬安全阈值，但同口径结果不支持平顺性改善。", '',
              '下一项建议：保留原默认参数化，在匹配训练预算下检查有/无OCC的信息贡献，并检查历史运动线索与局部终点辅助损失到完整采样轨迹之间的差距；本轮没有继续改损失或编码器。', '',
              '检查：两组latest优化器步数均6620、参数有限、调度器/循环/随机状态存在；每组仅best/latest；两组配置只差run_name和trajectory_parameterization；原v3两个断点与试验代码哈希未变。', '',
              '局限：每组单次初始化、仅10轮；验证集已用于选模，9路线且窗口重叠；巡航10窗口/2路线。无最终测试、连续扫掠、完整静态障碍、合法车道或控制器闭环。全部保留旧结果，不更改默认absolute参数化。', '',
              '对照重跑与历史v3前5轮指标相同，后5轮存在小幅差异，前10轮meanADE最大差0.069m，见control-reproduction.json；短期单次训练不足以给出严格泛化置信区间。', '',
              '原始结果：`artifacts/diagnostics/v3-residual-pilot/comparison.json`；每组完整分组和候选覆盖见对应`diagnostics/report.json`。']
    (ROOT / 'docs/V3_RESIDUAL_PILOT_RESULTS_20261115.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ['comparison', 'limitations']}))


if __name__ == '__main__': main()
