"""Audit the frozen Stage 7 experiments without model selection or test retuning."""
import argparse
import json
from pathlib import Path
import tarfile

import numpy as np

from configs.stage7 import Stage7Config
from evaluation.risk import paired_seed_interval
from mini_wa_jepa.provenance import provenance, write_json
from training.flow_matching import file_sha256


def read(path):
    return json.loads(Path(path).read_text())


def number(value,places=4):
    if value is None:return '缺失'
    return f'{value:.{places}g}' if 0<abs(value)<1e-4 else f'{value:.{places}f}'


def kinematics(candidates):
    """Pointwise gate; first point has no earlier predicted derivative to test."""
    a = np.asarray(candidates)
    good = np.isfinite(a).all(-1) & (a[..., 2] >= 0) & (a[..., 2] <= 15)
    accel = np.abs(np.diff(a[..., 2], axis=-1)) / .2
    good[..., 1:] &= accel <= 8
    delta = np.diff(a[..., :2], axis=-2)
    moving = np.linalg.norm(delta, axis=-1) > .1
    direction = np.arctan2(delta[..., 1], delta[..., 0])
    error = np.abs((direction-a[..., 1:, 3]+np.pi) % (2*np.pi)-np.pi)
    good[..., 1:] &= (~moving) | (error <= .35)
    return dict(pointwise_valid_fraction=float(good.mean()),
                all_points_valid_candidate_fraction=float(good.all(-1).mean()),
                moving_heading_valid_fraction=float((error[moving] <= .35).mean()) if moving.any() else None,
                derivative_convention='Later endpoint; first point checks finite fields and speed only; moving >0.1m/segment')


def compare_repeated_predictions(current,previous):
    if set(current.keys())!=set(previous.keys()):raise AssertionError('Repeated fields changed')
    prediction={};timing={}
    for key in current.keys():
        value=float(np.max(np.abs(current[key].astype(float)-previous[key].astype(float))))
        (timing if key in ('fm_latencies_ms','mse_latencies_ms') else prediction)[key]=value
    if not prediction or max(prediction.values())>1e-5:raise AssertionError('Repeated fixed predictions changed')
    return prediction,timing


def flow_audit(root, seeds):
    rows = {}
    for seed in seeds:
        final = root/f'flow_eval_final_{seed}'
        s = read(final/'summary.json')
        assert s['completed'] and s['evaluated_anchors'] == 2048
        with np.load(final/'raw_eval.npz') as z, np.load(root/f'flow_eval_{seed}/raw_eval.npz') as old:
            differences,timing_differences=compare_repeated_predictions(z,old)
            kin = kinematics(z['candidates'])
        fm = s['flow_matching_sample1']; mse = s['mse_baseline']
        gates = dict(ade=fm['mean_ade_m'] <= 1.25*mse['mean_ade_m'],
                     fde=fm['mean_fde_m'] <= 1.25*mse['mean_fde_m'],
                     diversity=s['diversity']['mean_pairwise_distance_m'] >= .1,
                     kinematics=kin['pointwise_valid_fraction'] >= .95,
                     head_latency=s['latency']['fm_head_p95_ms'] <= 50)
        rows[str(seed)] = dict(summary=s,kinematics=kin,gates=gates,
                               repeat_max_abs_differences=differences,
                               repeat_latency_max_abs_changes_ms=timing_differences)
    return rows


def closed_audit(root, cfg):
    expected = {(m, base+i) for m, base in zip(('SCS','CSC'), cfg.closed_loop_seeds)
                for i in range(cfg.closed_loop_seeds_per_map)}
    reports = {}; baselines = {}; initial_errors = []
    for training_seed in cfg.training_seeds:
        for arm in ('mse','geometry','full'):
            name = f'closed_{arm}_{training_seed}'; r = read(root/name/'report.json')
            assert r['completed'] and not r['smoke'] and r['max_steps'] == 1000
            assert len(r['episodes']) == 8
            assert {(e['map_spec'],e['seed']) for e in r['episodes']} == expected
            for row in r['episodes']:
                assert file_sha256(row['raw']) == row['raw_sha256']
                with np.load(row['raw']) as raw:
                    actual=raw['states'][:,:2]
                    row['travelled_distance_m']=float(np.linalg.norm(np.diff(actual,axis=0),axis=1).sum())
                    row['net_displacement_m']=float(np.linalg.norm(actual[-1]-actual[0]))
                key = row['map_spec'],row['seed']; ref = baselines.setdefault(key,row)
                state_error = float(np.max(np.abs(np.asarray(ref['initial_state'])-row['initial_state'])))
                assert np.shape(ref['initial_scene']) == np.shape(row['initial_scene'])
                scene_error = float(np.max(np.abs(np.asarray(ref['initial_scene'])-row['initial_scene']),initial=0))
                assert max(state_error,scene_error) <= 1e-5
                assert ref['initial_bev_sha256'] == row['initial_bev_sha256']
                initial_errors.append(max(state_error,scene_error))
            reports[name] = r
    comparisons = {}; gates = {}
    for training_seed in cfg.training_seeds:
        full = sorted(reports[f'closed_full_{training_seed}']['episodes'],key=lambda e:e['seed'])
        pairs = {}; gs = {}
        for arm in ('mse','geometry'):
            control = sorted(reports[f'closed_{arm}_{training_seed}']['episodes'],key=lambda e:e['seed'])
            pair = {metric: paired_seed_interval([float(e[metric]) for e in full],
                    [float(e[metric]) for e in control], [e['seed'] for e in full],cfg.bootstrap_samples,training_seed)
                    for metric in ('route_completion','collision','out_of_road')}
            route = pair['route_completion']
            gs[arm] = bool(route['delta'] >= .02 and route['ci95'][0] > 0 and
                          pair['collision']['delta'] <= 0 and pair['out_of_road']['delta'] <= 0)
            pairs[arm] = pair
        comparisons[str(training_seed)] = pairs; gates[str(training_seed)] = gs
    return dict(runs=reports,paired=comparisons,gates=gates,
                initial_max_abs_error=max(initial_errors),unique_scene_seeds=8,
                note='MSE repeats are duplicate deterministic controls, not 16 independent scene seeds')


def resources(path):
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    if not rows:
        return dict(samples=0)
    devices = {}
    for row in rows:
        for d in row['gpu']:
            devices.setdefault(d['path'],[]).append(d['used_bytes'])
    complete_pss=[r for r in rows if all('pss_bytes' in p for p in r['processes'])]
    return dict(samples=len(rows),duration_s=rows[-1]['timestamp']-rows[0]['timestamp'],
                minimum_available_ram_bytes=min(r['available_ram_bytes'] for r in rows),
                maximum_sum_worker_pss_bytes=max((sum(p['pss_bytes'] for p in r['processes']) for r in complete_pss),default=None),
                complete_pss_samples=len(complete_pss),
                missing_pss_process_records=sum('pss_bytes' not in p for r in rows for p in r['processes']),
                devices={k:dict(min_used_bytes=min(v),max_used_bytes=max(v),first_used_bytes=v[0],last_used_bytes=v[-1])
                         for k,v in devices.items()},
                caveat='DRM includes desktop and other processes; PSS peak uses only complete collector/cache/risk-trainer/evaluator samples; missing early PSS is never replaced by RSS or zero')


def metrics_report(root, summary):
    runs = []
    for seed, row in summary['flow'].items():
        s = row['summary']
        for model,key in (('MSE','mse_baseline'),('Flow','flow_matching_sample1')):
            runs.append(dict(run_id=f'open_{model}_{seed}',model=model,ablation='sample1' if model=='Flow' else 'frozen_stage4',
                split='test',seed=int(seed),seed_kind='paired_flow_run' if model=='MSE' else 'training',evaluation_kind='expert_open_loop',
                metrics=dict(ade_m=s[key]['mean_ade_m'],fde_m=s[key]['mean_fde_m'],
                    diversity_m=s['diversity']['mean_pairwise_distance_m'] if model=='Flow' else None,
                    inference_ms=s['latency']['fm_head_mean_ms' if model=='Flow' else 'mse_head_mean_ms'])))
    for name,row in summary['risk']['models'].items():
        kind,seed = name.split('_');m = row['metrics']
        runs.append(dict(run_id=f'risk_{name}',model='Risk',ablation=kind,split='test',seed=int(seed),
            seed_kind='training',evaluation_kind='fixed_command_risk',metrics={f'{event}_{metric}':m[event][metric]
                for event in ('crash','offroad') for metric in ('auroc','brier')}))
    for name,r in summary['closed']['runs'].items():
        for row in r['episodes']:
            runs.append(dict(run_id=f'{name}_{row["seed"]}',model=r['arm'],ablation=r['arm'],
                split='test',seed=row['seed'],seed_kind='simulation',training_seed=r['training_seed'],
                evaluation_kind='closed_loop',metrics=dict(route_completion=row['route_completion'],
                    success_rate=float(row['arrive_dest']),collision_rate=float(row['collision']),
                    out_of_road_rate=float(row['out_of_road']),lateral_error_m=row['lateral_error_m'],
                    inference_ms=row['inference_ms'],pipeline_ms_p95=row['pipeline_p95_ms'],
                    fallback_fraction=row['fallback_fraction'],
                    replanned_execution_ade_m=row['replanned_execution_ade_m'])))
    write_json(root/'metrics.json',dict(visualization_schema_version=1,spec_version=2,stage=7,
        **{k:summary[k] for k in ('config','seeds','git_commit','goal_sha256')},runs=runs))


def plots(root, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'figures';out.mkdir(exist_ok=True)
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    names=[];ade=[];feasible=[];div=[]
    for seed,r in summary['flow'].items():
        names.append(seed);ade.append(r['summary']['flow_matching_sample1']['mean_ade_m'])
        feasible.append(r['kinematics']['pointwise_valid_fraction']);div.append(r['summary']['diversity']['mean_pairwise_distance_m'])
    ax[0].bar(names,ade);ax[0].axhline(next(iter(summary['flow'].values()))['summary']['mse_baseline']['mean_ade_m'],c='orange',label='Frozen MSE')
    ax[0].legend();ax[0].set_title('Non-oracle sample 1 ADE (m)')
    ax[1].bar(names,feasible);ax[1].axhline(.95,c='red');ax[1].set_ylim(0,1);ax[1].set_title('Pointwise kinematic validity')
    ax[2].bar(names,div);ax[2].set_title('Pairwise diversity (m)')
    fig.savefig(out/'generation.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for seed in summary['seeds']:
        epochs=read(root/f'flow_{seed}/manifest.json')['history']
        # Flow epoch records use their recorded validation field, never reconstructed losses.
        field=next((k for k in ('val_loss','val_mse','val') if k in epochs[0]),None)
        if field:axes[0].plot([e['epoch'] for e in epochs],[e[field] for e in epochs],label=str(seed))
        for kind in ('world','zero'):
            ep=read(root/f'risk_runs/{kind}_{seed}/report.json')['epochs']
            axes[1].plot([e['epoch'] for e in ep],[e['val_bce'] for e in ep],label=f'{kind}_{seed}')
    axes[0].set_title('Flow validation loss');axes[1].set_title('Risk validation masked BCE')
    for ax in axes:ax.set_xlabel('Epoch');ax.legend(fontsize=8)
    fig.savefig(out/'validation_losses.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,event in zip(axes,('crash','offroad')):
        for name in summary['risk']['models']:
            with np.load(root/'risk_eval'/f'{name}_raw.npz') as z:
                k=('crash','offroad').index(event);mask=z['valid'][:,k];p=z['probability'][mask,k];y=z['target'][mask,k]
                bins=np.minimum((p*10).astype(int),9);xs=[];ys=[]
                for b in range(10):
                    if np.any(bins==b):xs.append(p[bins==b].mean());ys.append(y[bins==b].mean())
                ax.plot(xs,ys,'o-',label=name)
        ax.plot([0,1],[0,1],'--',c='gray');ax.set(xlabel='Predicted probability',ylabel='Observed event frequency',title=f'{event} calibration')
        ax.legend(fontsize=7)
    fig.savefig(out/'risk_calibration.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    names=[];deltas=[];bounds=[]
    for seed,pairs in summary['risk']['paired'].items():
        for event,pair in pairs.items():
            names.append(f'{seed} {event}');deltas.append(pair['delta']);bounds.append(pair['ci95'])
    for i,(name,delta,bound) in enumerate(zip(names,deltas,bounds)):
        if bound is not None:axes[0].hlines(i,*bound,color='steelblue')
        if delta is not None:axes[0].plot(delta,i,'o',color='steelblue')
    axes[0].axvline(0,color='gray',ls='--');axes[0].set_yticks(range(len(names)),names)
    axes[0].set_title('World minus zero Brier; seed-cluster 95% CI')
    names=list(summary['risk']['models'])
    for i,event in enumerate(('crash','offroad')):
        axes[1].bar(np.arange(len(names))+(i-.5)*.35,
                    [summary['risk']['models'][n]['metrics'][event]['brier'] for n in names],width=.35,label=event)
    axes[1].set_xticks(range(len(names)),names,rotation=20);axes[1].set_title('Held-out Brier (lower is better)');axes[1].legend()
    fig.savefig(out/'risk_brier.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for ax,seed in zip(axes,summary['seeds']):
        for arm in ('mse','geometry','full'):
            rows=sorted(summary['closed']['runs'][f'closed_{arm}_{seed}']['episodes'],key=lambda e:e['seed'])
            ax.plot(range(8),[r['route_completion'] for r in rows],'o-',label=arm)
        ax.set(title=f'Training seed {seed}',xlabel='Same eight simulation seeds',ylabel='Route completion',ylim=(0,1));ax.legend()
    fig.savefig(out/'closed_loop.png');plt.close(fig)


def report(root):
    cfg=Stage7Config();pipeline=read(root/'remaining_status.json')
    assert pipeline['completed'], 'Required experiment pipeline has not finished'
    s=dict(**provenance(cfg),stage=7,seeds=list(cfg.training_seeds),completed=True,stage8_started=False,
           retained_policy='frozen Stage4 MSE',flow=flow_audit(root,cfg.training_seeds),
           risk=read(root/'risk_eval/report.json'),closed=closed_audit(root,cfg),
           resources=resources(root/'resources.jsonl'))
    partitions=read(root/'risk_features_001/manifest.json')['inputs']
    s['collection']={Path(p['path']).name:read(Path(p['path'])/'manifest.json')['summary_counts'] for p in partitions}
    motion=read(root/'risk_motion_001/manifest.json')
    assert motion['completed'] and not motion['smoke'] and len(motion['groups'])==192
    s['motion_recovery']=dict(manifest_sha256=file_sha256(root/'risk_motion_001/manifest.json'),
        contexts=len(motion['groups']),max_state_error=max(r['state_max_abs_error'] for r in motion['groups'].values()),
        max_scene_error=max(r['scene_max_abs_error'] for r in motion['groups'].values()),
        different_bev_values=sum(r['bev_different_values'] for r in motion['groups'].values()))
    assert all(read(Path(p['path'])/'manifest.json')['completed'] for p in partitions)
    prior=read(root/'codex_review/prior_checkpoints.json')
    s['prior_checkpoints_unchanged']=all(Path(p).stat().st_size==v['size'] and file_sha256(p)==v['sha256']
                                         for p,v in prior.items())
    s['scientific_gates']=dict(generation=all(v for r in s['flow'].values() for v in r['gates'].values()),
        risk=s['risk']['scientific_passed'],planning=all(v for r in s['closed']['gates'].values() for v in r.values()))
    s['scientific_passed']=all(s['scientific_gates'].values())
    tests=read(root/'codex_review/tests_result.json');browser=read(root/'codex_review/browser_checks.json')
    rrd=read(root/'codex_review/rrd_checks.json');checks=read(root/'codex_review/marimo_checks.json')
    export_path=root/'codex_review/html_exports.json'
    exports=read(export_path) if export_path.exists() else dict(passed=False,status='pending_exports')
    delivery=bool(tests['passed'] and browser['passed'] and rrd['passed'] and checks['passed'] and exports['passed'] and s['prior_checkpoints_unchanged'])
    s['delivery_passed']=delivery;s['validation']=dict(tests=tests,browser=browser,rrd=rrd,marimo=checks,exports=exports)
    plots(root,s);metrics_report(root,s)
    write_json(root/'summary.json',s)
    write_json(root/'acceptance.json',dict(stage=7,completed=True,scientific_passed=s['scientific_passed'],
        passed=s['scientific_passed'] and delivery,delivery_passed=delivery,gates=s['scientific_gates'],
        stage8_started=False,retained_policy=s['retained_policy'],validation=s['validation']))
    with tarfile.open(root/'final_source_snapshot.tar.gz','w:gz') as arc:
        for p in s['source_files']:arc.add(p,arcname=p)
        for p in ('GOAL.md','AGENTS.md','docs/STAGE7_PROTOCOL.md','docs/VISUALIZATION.md','TASKS.md',
                  'README.md','chapters/README.md','chapters/chapter7_candidate_planning/README.md'):arc.add(p,arcname=p)
    lines=['# 阶段 7：候选轨迹、风险预测与闭环规划', '',
           f'实验已完成；科学门槛通过：**{s["scientific_passed"]}**；工程交付检查：**{delivery}**。保留原 MSE 策略，不进入阶段 8。',
           '', '## 生成结果', '',
           '| 训练种子 | MSE ADE/FDE (m) | Flow 非 oracle 候选1 ADE/FDE (m) | 多样性 (m) | 路点运动学通过率 | 头部 P95 (ms) |',
           '|---|---|---|---|---|---|']
    for seed,r in s['flow'].items():
        q=r['summary'];m=q['mse_baseline'];f=q['flow_matching_sample1']
        lines.append(f'| {seed} | {m["mean_ade_m"]:.4f}/{m["mean_fde_m"]:.4f} | {f["mean_ade_m"]:.4f}/{f["mean_fde_m"]:.4f} | {q["diversity"]["mean_pairwise_distance_m"]:.4f} | {r["kinematics"]["pointwise_valid_fraction"]:.2%} | {q["latency"]["fm_head_p95_ms"]:.2f} |')
    lines += ['', '多样性增加不能说明路线可执行。原始输出没有做测试集后处理或重训。最终计时在采集空闲后重复，原始预测与初测逐元素核对；初测共享负载记录保留。头部计时不含编码器/模拟器，MSE 计时调用原 Joint 的两个头，不能解释为单独回归层耗时。',
              '', '## 风险预测', '',
              '| 模型/种子 | 事件 | 正/负样本 | AUROC | AP/测试阳性率 | Brier/训练阳性率常数基线 |', '|---|---|---|---|---|---|']
    for name,r in s['risk']['models'].items():
        for event,m in r['metrics'].items():
            lines.append(f'| {name} | {event} | {m["positives"]}/{m["negatives"]} | {number(m["auroc"])} | {number(m["average_precision"])}/{number(m["prevalence"])} | {number(m["brier"],5)}/{number(m["constant_train_prevalence_brier"],5)} |')
    lines += ['', '世界条件相对零世界的 Brier 差（负数更好）；CI 按仿真 seed 聚类，2000 次配对重采样：', '']
    for seed,r in s['risk']['paired'].items():
        for event,p in r.items():lines.append(f'- {seed} {event}：相对改善 {p["relative_brier_gain"]:.2%}，差 {p["delta"]:.6g}，95% CI {p["ci95"]}。')
    typed={k:sum(v['typed_events_seen'].get(k,0) for v in s['collection'].values())
           for k in ('crash_vehicle','crash_object','crash_building','crash_human','crash_sidewalk')}
    lines += ['',f'原生事件分布（分支计数）：`{json.dumps(typed)}`。路沿碰撞属于原生 crash 标签，但当前样本不能证明车辆动态避碰有效。早停未观察到的事件按删失 mask 排除，没有当作安全负例。数据为128/32/32个 train/val/test 上下文，每个13条命令，地图与 seed 隔离；风险测试8个 seed 与闭环8个 seed 不同。',
              '', '风险模型的判别能力与世界潜变量的增益分别验收；不能只用接近1的 AUROC 宣称世界模型有效。当前 crash Brier 已接近数值饱和，较大的相对百分比对应的绝对误差变化可能非常小。两个训练种子的世界增益稳定性均须通过冻结 CI 门槛。',
              '', '## 三臂闭环', '', '| 训练/配对种子 | 策略 | 平均路线完成率 | 实际累计行驶 (m) | 碰撞率 | 越界率 | 到达率 | 平均规则刹停占比 |', '|---|---|---|---|---|---|---|---|']
    for name,r in s['closed']['runs'].items():
        e=r['episodes'];avg=lambda k:float(np.mean([x[k] for x in e]))
        lines.append(f'| {r["training_seed"]} | {r["arm"]} | {avg("route_completion"):.4f} | {avg("travelled_distance_m"):.2f} | {avg("collision"):.2%} | {avg("out_of_road"):.2%} | {avg("arrive_dest"):.2%} | {avg("fallback_fraction"):.2%} |')
    lines += ['', 'Full 相对对照的路线完成率差（正数更好），同一8个场景配对 CI：', '']
    for seed,r in s['closed']['paired'].items():
        for arm,p in r.items():
            q=p['route_completion'];lines.append(f'- {seed} vs {arm}：差 {q["delta"]:.5f}，95% CI {q["ci95"]}；冻结门槛 {s["closed"]["gates"][seed][arm]}。')
    lines += ['', '48 个实际 episode，控制间隔0.1秒，最多1000步或原生终止。各臂初始完整场景/状态误差≤1e-5且 BEV 字节一致；MSE 两次是重复对照，不计为16个独立场景。STOP 是候选全部拒绝后的规则刹停，不是 Flow 生成成功。路线完成率包含出生位置已有偏移；刹停时的小比例不能算作主动推进，实际累计行驶距离从原始位置序列另行计算。低碰撞率若依赖不行驶，不能说明规划更好。闭环 plan-vs-replanned-execution ADE 是与随后重新规划的执行轨迹比较，不能混入专家开环 ADE。不同潜变量预测 horizon 不跨模型比较。',
              '', '## 交付与限制', '',
              f'- 参数（包括冻结模块）：MSE {s["closed"]["runs"]["closed_mse_20261021"]["parameter_count"]:,}；几何臂 {s["closed"]["runs"]["closed_geometry_20261021"]["parameter_count"]:,}；完整臂 {s["closed"]["runs"]["closed_full_20261021"]["parameter_count"]:,}，均在15–30M内。',
              f'- 全套测试 {tests["tests_run"]} 项，失败 {tests["failures"]} / 错误 {tests["errors"]}；浏览器实际键盘滑块、页签切换、评分表与逐步回放检查见 `codex_review/`。Rerun 原生读回见 `rrd_checks.json`。',
              f'- 原10个已有 best.pt 哈希未改：{s["prior_checkpoints_unchanged"]}。ROCm 原生 PyTorch / SDPA MATH；硬件自动检测与校准见各训练报告，资源趋势见 `resources.jsonl`。无14GB人为上限。',
              f'- RAM 最小可用 {s["resources"]["minimum_available_ram_bytes"]/2**30:.2f} GiB，监控 {s["resources"]["samples"]} 次；DRM 显存包含桌面及其他进程，不能冒充单模型占用。各运行保留 allocator 峰值。',
              '- 复查发现原采集器把尚未计算的零运动历史写入 context。原文件保持不动：`risk_motion_001/` 用每条保存的 prefix_actions 完整重放192个上下文，核对4帧 BEV 字节、实际状态与全部车辆状态一致后，从5个真实观测状态恢复4帧速度/转向历史。正式特征、世界预测和风险模型只用已核对的恢复结果；较早的零历史冒烟缓存/训练保留作失效证据，不计正式成绩。',
              '- 原 train_b/test 采集各因1个边界像素不一致而停止，失败分区和 MRE 原样保留。原生 RGB 在地形/路沿类别间切换，物理状态、双精度位姿与相机矩阵一致；重建环境、24位深度请求均未消除。仅固定语义相机绘制顺序后，两失败起点各16次重放零差。continuation 分区复用完整上下文并补齐原固定 cohort，无删样本或放宽像素门槛。原/修复渲染版本逐上下文标注，运动恢复分别按原版本重放；三臂正式闭环统一使用修复版。这种混合渲染来源限制风险数据的完全同质性，审计哈希与源码差异见 `codex_review/determinism_repair.*`。',
              '- 配置中的 flow_num_layers/flow_num_heads 是未使用的初稿字段；实际生成器是保留64个空间位置的线性时序融合及两隐藏层 MLP，模型结构以检查点 model_arch 和冻结源码为准。risk_num_layers=3 指分类器三层 Linear。没有额外 Transformer 或隐含注意力优化。',
              '- 保留全部原始 NPZ、TensorBoard、周期预测/真值 PNG/RRD、失败 MRE、冒烟数据、初测与最终计时、源码快照。`flow_paths/` 是11个真实积分状态，`closed_*/<map_seed>/actual.npz` 是同一条实际驾驶记录。未执行候选的反事实真值不可得，保持缺失。',
              '- 中文交互入口：项目根目录运行 `.venv/bin/marimo run app.py`，选 Chapter 7；阶段指标在 `scripts/dashboard_marimo.py` 选择7。静态HTML是快照，不能替代运行时交互。没有新增视频或启动已知失败的桌面路径。',
              '- 20k 专家锚点、单演示监督、冻结表征和较少独立地图/seed 限制结论；生成误差原因尚未由消融证明，不归因于某一算法。风险训练使用人工命令压力分布，Flow 部署候选可能存在分布偏移。',
              f'- 科学门槛：`{json.dumps(s["scientific_gates"])}`。科学失败与实验链路交付分别记录，阶段8尚未开始。',
              '', f'源码 commit `{s["git_commit"]}`，dirty tree SHA256 `{s["source_tree_sha256"]}`；GOAL SHA256 `{s["goal_sha256"]}`。各运行另有独立源码哈希与快照；最终报告不反写历史来源。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(scientific_passed=s['scientific_passed'],delivery_passed=delivery,gates=s['scientific_gates'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('artifacts/restart_stage7'))
    report(p.parse_args().root)
