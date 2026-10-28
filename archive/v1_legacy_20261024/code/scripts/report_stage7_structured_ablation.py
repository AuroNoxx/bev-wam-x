"""Report real coefficient-flow validation predictions against immutable v2 controls."""
import argparse
import json
from pathlib import Path

import numpy as np

from mini_wa_jepa.provenance import write_json
from scripts.report_stage7_metric_ablation import paired_episode_interval


def read(path):return json.loads(Path(path).read_text())


def make_report(root):
    data=read(root/'summary.json');cfg=data['config'];source=read(root/'provenance.json')
    if not data['completed'] or cfg['smoke']:raise ValueError('Full completed experiment required')
    reference=Path(cfg['baseline_root']);old=read(reference/'summary.json')
    rows=[];pairs={};gates={};dimensions=cfg['dimensions']
    for seed in cfg['training_seeds']:
        pairs[str(seed)]={}
        for arm in ('control','metric','structured'):
            run=(data if arm=='structured' else old)['runs'][f'{arm}_{seed}']
            if not run['completed']:raise ValueError('Incomplete matched run')
            for tau,value in run['validation']['temperatures'].items():
                path=(root if arm=='structured' else reference)/f'{arm}_{seed}/validation/tau_{float(tau):g}.npz'
                with np.load(path) as raw:
                    rough=float(np.linalg.norm(np.diff(raw['candidates'][...,:2],n=2,axis=2),axis=-1).mean())
                rows.append(dict(value,seed=seed,arm=arm,second_difference_position_m=rough))
        if cfg.get('previous_structured_root'):
            previous=read(Path(cfg['previous_structured_root'])/'summary.json')['runs'][f'structured_{seed}']
            for value in previous['validation']['temperatures'].values():
                rows.append(dict(value,seed=seed,arm='structured12'))
        for tau in cfg['temperatures']:
            with (np.load(root/f'structured_{seed}/validation/tau_{tau:g}.npz') as new,
                  np.load(reference/f'metric_{seed}/validation/tau_{tau:g}.npz') as baseline):
                for field in ('target','mse','anchors','episodes'):
                    if not np.array_equal(new[field],baseline[field]):raise AssertionError(f'Cohort mismatch: {field}')
                pairs[str(seed)][str(tau)]={field:paired_episode_interval(new[field],baseline[field],new['episodes'],seed)
                    for field in ('sample1_ade_m','sample1_fde_m','diversity_m')}
    for tau in cfg['temperatures']:
        gates[str(tau)]=all(r['all_gates_passed'] for r in rows if r['arm']=='structured' and r['temperature']==tau)
    passing=[float(t) for t,v in gates.items() if v]
    result=dict(completed=True,validation_only=True,uniform_temperature_both_seed_gates=gates,
                passing_temperatures=passing,comparisons_to_metric=pairs,rows=rows,
                noise_contract='Same seed and scenes; different coefficient dimensionality, not elementwise-identical noise')
    write_json(root/'comparison.json',result)
    runs=[]
    for r in rows:
        if r['arm']!='structured':continue
        runs.append(dict(run_id=f'structured{dimensions}_v2_{r["seed"]}_tau{r["temperature"]:g}',
             model='Whole-trajectory coefficient Flow',ablation=f'train-only PCA{dimensions}; tau={r["temperature"]:g}; validation only',
             split='val',seed=r['seed'],training_seed=r['seed'],evaluation_kind='validation_structured_ablation',
             metrics={k:r[k] for k in ('ade_m','fde_m','diversity_m')}))
    write_json(root/'metrics.json',dict(visualization_schema_version=1,spec_version=2,stage=7,config=cfg,
          seeds=cfg['training_seeds'],git_commit=source['git_commit'],goal_sha256=source['goal_sha256'],runs=runs))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for col,seed in enumerate(cfg['training_seeds']):
        styles=[('control','o--'),('metric','s--')]
        if cfg.get('previous_structured_root'):styles.append(('structured12','x--'))
        styles.append(('structured','^-'))
        for arm,style in styles:
            selected=sorted([r for r in rows if r['seed']==seed and r['arm']==arm],key=lambda r:r['temperature'])
            for ax,key in [(axes[0,col],'ade_m'),(axes[1,col],'fde_m')]:
                ax.plot([r['diversity_m'] for r in selected],[r[key] for r in selected],style,label=arm)
                if arm=='structured':
                    for r in selected:ax.annotate(f'{r["temperature"]:g}',(r['diversity_m'],r[key]))
        ref=data['runs'][f'structured_{seed}']['validation']['mse_reference']
        for ax,key in [(axes[0,col],'ade_m'),(axes[1,col],'fde_m')]:
            ax.axvline(.1,color='grey',ls=':');ax.axhline(1.25*ref[key],color='red',ls=':')
            ax.set(xlabel='Pairwise diversity (m)',ylabel=key,title=f'Validation | seed {seed}')
            ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(root/'precision_diversity.png');plt.close(fig)

    verdict=(f'两个训练种子在统一温度{passing}同时通过本轮全部验证门槛；尚未检验闭环驾驶。'
             if passing else '本轮整条轨迹系数生成尚未使两个训练种子同时通过全部验证门槛。不能宣称修复完成。')
    basis=read(root/'basis_summary.json')
    lines=['# 阶段7第二版：整条路线变化的生成实验','',verdict,'',
        '复用原有20,000条专家训练锚点与2,048条验证锚点，没有重新采集专家、重跑编码器或训练世界模型。',
        f'输出仍是10个(x,y,v,heading)路点；模型先生成{dimensions}个描述整条路线的系数，再展开为完整轨迹。','',
        f'训练集拟合的{dimensions}维基底保留标准残差方差{basis["explained_variance"]:.2%}。所有误差都对原始完整专家路线计算。','',
        '| 种子 | 方案 | 温度 | ADE(m) | FDE(m) | 多样性(m) | 路点/整条通过率 | 头部P95(ms) | 全部门槛 |',
        '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f'| {r["seed"]} | {r["arm"]} | {r["temperature"]:g} | {r["ade_m"]:.5f} | {r["fde_m"]:.5f} | {r["diversity_m"]:.5f} | '
           f'{r["pointwise_valid_fraction"]:.2%}/{r["all_points_valid_candidate_fraction"]:.2%} | {r["head_p95_ms"]:.2f} | {r["all_gates_passed"]} |')
    lines+=['','## 对照与解释','',
        f'- control 为第二版原损失，metric 为上轮米制路线/终点损失，structured 保留米制损失并改成{dimensions}维整条轨迹生成。旧对照直接读取已完成原始结果，未重新训练/改写。',
        '- 两种子各25epoch，20,000训练锚点、相同训练预算和Transformer主体；输出头尺寸随系数表示变化，不能声称结构完全相同。',
        f'- 在每个变体训练前固定{dimensions}维、损失权重和温度网格。20维为12维结果之后的细节保留对照，不能称作最初即冻结的单次试验。PCA均值/方差/基底只在train拟合；validation和test不参与拟合。',
        f'- 噪声来自{dimensions}维系数，训练与推理使用同一表示；每条采样都来自完整ODE积分。8条候选使用统一温度，首条保持随机样本。',
        '- 维数与几何改变后，相同温度不代表相同米制扰动大小；比较完整精度—多样性曲线，不仅比较某个相同温度的数值。',
        '- PCA投影不能保证道路合法、物理可执行或存在不同驾驶意图；仍以原严格运动学检查与后续闭环验证为准。',
        '- 原始GT未投影后替换。单列的projection_diagnostic使用真实验证系数，仅表示表示损失，不能当作生成精度。',
        '- 向量场损失在12维/40维不同空间，不比较绝对值。各自checkpoint均按固定目标验证向量场MSE选择，不用实际采样成绩挑epoch。',
        '- comparisons_to_metric中的置信区间按episode配对聚类，2000次重采样；验证集反复使用、仅两个训练种子，结论仍属探索。',
        '- moving>0.1m/segment；头部P95为batch1、8候选、预热20后测64次，不包括编码器、模拟器和控制器。',
        '- 本轮不加载test、不新增闭环成绩或世界潜变量预测；没有阶段7整体通过或阶段8验收结论。','',
        '## 复现与证据','',
        f'- source commit `{source["git_commit"]}`；GOAL SHA256 `{source["goal_sha256"]}`；训练源码 SHA256 `{cfg["source_tree_sha256"]}`。',
        '- trajectory_basis.npz 保存训练集基底、均值/尺度和谱；source.tar.gz、cohort.json、config.json 保存可复核来源。',
        '- 每epoch保存TensorBoard与资源；固定样本周期PNG/NPZ/RRD与实际完整验证候选全部保留。指标已接入原Marimo看板，未测量闭环指标保持缺失。']
    if cfg.get('previous_structured_root'):
        detail=['','## 这轮实际解决了什么','',
                '整体生成使相邻路段变化更平缓，但这没有稳定换来更低的专家路线误差。保留上一轮米制损失版本作为实验对照，不替换现有驾驶策略。',
                '统一温度1.0的随机候选终点误差（均保持多样性过线）：','',
                '| 种子 | 上轮米制损失 | 12维整体生成 | 20维整体生成 |',
                '|---|---|---|---|']
        for seed in cfg['training_seeds']:
            def value(arm):return next(r for r in rows if r['seed']==seed and r['arm']==arm and r['temperature']==1.)['fde_m']
            detail.append(f'| {seed} | {value("metric"):.4f}m | {value("structured12"):.4f}m | {value("structured"):.4f}m |')
        detail+=['','20维显著减少了标签投影损失，实际生成精度仍不稳定；因此增加保留细节本身不足以完成修复。上述比较不能证明唯一根因。']
        lines[2:2]=detail+['']
    for seed in cfg['training_seeds']:
        run=data['runs'][f'structured_{seed}'];floor=run['projection_diagnostic']
        lines.append(f'- 种子{seed}：head参数{run["head_parameters"]:,}，完整策略参数{run["full_policy_parameters"]:,}；'
                     f'选中epoch{run["selected_epoch"]}；真实验证标签的投影损失ADE/FDE '
                     f'{floor["projection_ade_m"]:.5f}/{floor["projection_fde_m"]:.5f}m（不是预测成绩）。')
    if (root/'effective_config.json').exists():
        lines.append('- 元数据勘误：原config.json的source文字说明复制了I_12；执行的dimensions、40×20基底和checkpoint均为20维。effective_config.json更正说明文字，原始配置及源码快照保持不动。')
    delivery=root/'checks/delivery.json'
    if delivery.exists() and read(delivery).get('completed'):
        d=read(delivery)
        lines+=['',f'交付复核：{d["tests"]}项测试通过；{d["native_rrd_files"]}个周期RRD原生读回；'
                f'{d["protected_checkpoints"]}个既有检查点哈希未改；原始指标复算与Marimo运行时导出通过。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();r=make_report(a.root);print(json.dumps({k:v for k,v in r.items() if k not in ('comparisons_to_metric','rows')},ensure_ascii=False))
