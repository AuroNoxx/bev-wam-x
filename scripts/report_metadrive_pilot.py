"""Audit, summarize and visualize the measured native-PG pilot recordings."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mini_wa_jepa.provenance import write_json
from scripts.benchmark_metadrive_pilot import EVENTS, sha

ARMS = ('idm', 'mse', 'geometry', 'full')


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [float(center-half), float(center+half)]


def summarize(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    expected = protocol['evaluated_seeds']
    reports = {}
    summary = dict(benchmark=protocol['name'], pilot=True, protocol=protocol,
                   completed=False, arms={}, audit=dict(raw_files=0, rrd_files=0,
                                                        initial_max_abs_error=0., initial_bev_equal=True))
    refs = {}
    audit = root / 'audit'
    audit.mkdir(exist_ok=True)
    for arm in ARMS:
        r = json.loads((root / arm / 'report.json').read_text())
        assert r['completed'] and not r['smoke']
        assert [e['seed'] for e in r['episodes']] == expected
        reports[arm] = r
        all_ms = []
        for e in r['episodes']:
            raw = ROOT / e['raw']
            assert sha(raw) == e['raw_sha256']
            with np.load(raw) as z:
                assert len(z['rewards']) == e['steps'] == len(z['actions'])
                assert len(z['states']) == e['steps'] + 1
                assert np.isclose(z['rewards'].sum(), e['episode_reward'])
                assert np.isclose(z['costs'].sum(), e['total_cost'])
                assert np.isclose(z['route'][-1], e['route_completion'])
                for j, event in enumerate(EVENTS):
                    assert bool(z['events'][:, j].any()) == e[event]
                if arm != 'idm':
                    assert z['candidates'].shape[1] == (1 if arm == 'mse' else 8)
                    all_ms.extend(z['decision_ms'][20:].tolist())
            summary['audit']['raw_files'] += 1
            ref = refs.setdefault(e['seed'], e)
            error = float(np.max(np.abs(np.asarray(ref['initial_scene'])-np.asarray(e['initial_scene']))))
            assert error <= 1e-5
            assert ref['initial_bev_sha256'] == e['initial_bev_sha256']
            summary['audit']['initial_max_abs_error'] = max(summary['audit']['initial_max_abs_error'], error)
            rrd = ROOT / e['rerun']
            assert sha(rrd) == e['rerun_sha256']
            with (audit / f'{arm}_{e["seed"]:03d}_rrd.txt').open('w') as f:
                subprocess.run([sys.executable, '-m', 'rerun', 'rrd', 'print', str(rrd)],
                               stdout=f, stderr=subprocess.STDOUT, check=True)
            summary['audit']['rrd_files'] += 1
        es = r['episodes']
        n = len(es)
        successes = sum(e['arrive_dest'] for e in es)
        summary['arms'][arm] = dict(episodes=n, successes=successes, success_rate=successes/n,
            success_rate_wilson95=wilson(successes,n),
            **{event+'_rate':float(np.mean([e[event] for e in es])) for event in EVENTS if event != 'arrive_dest'},
            mean_route_completion=float(np.mean([e['route_completion'] for e in es])),
            mean_episode_reward=float(np.mean([e['episode_reward'] for e in es])),
            mean_episode_cost=float(np.mean([e['total_cost'] for e in es])),
            mean_distance_m=float(np.mean([e['travelled_distance_m'] for e in es])),
            decision_p95_ms=float(np.percentile(all_ms,95)) if all_ms else None,
            decision_step_count=len(all_ms),
            parameter_count=r.get('parameter_count'),
            report_sha256=sha(root/arm/'report.json'))
    summary['completed'] = True
    write_json(root/'summary.json',summary)
    lines=['# MetaDrive程序生成单车泛化：10场景试跑', '',
           '固定测试seed0–9，每场景一局；4臂共40局。主指标为原生到达成功率，不把路线完成率当成功率。', '',
           '| 策略 | 到达SR | 车辆碰撞率 | 全部碰撞率 | 越界率 | 超时率 | 平均完成率 | 平均奖励 | 平均cost | 决策P95(ms) |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for arm, s in summary['arms'].items():
        latency='—' if s['decision_p95_ms'] is None else f'{s["decision_p95_ms"]:.2f}'
        lines.append(f'| {arm} | {s["successes"]}/{s["episodes"]} ({s["success_rate"]:.0%}) | '
                     f'{s["crash_vehicle_rate"]:.0%} | {s["crash_rate"]:.0%} | {s["out_of_road_rate"]:.0%} | '
                     f'{s["max_step_rate"]:.0%} | {s["mean_route_completion"]:.2%} | '
                     f'{s["mean_episode_reward"]:.2f} | {s["mean_episode_cost"]:.2f} | {latency} |')
    lines += ['', '## 评测口径', '',
        '- 沿用官方文档的测试seed池0–199及随机车道宽度、车道数和车型；本轮只跑前10个，map=3、traffic_density=0.1、horizon=1000、control_dt=0.1s。这里3表示三个生成路段，实际另含原生起始路段。',
        '- 使用原生MetaDrive reward/cost/done及默认连续线越界判定；没有启用项目FullOffroadRule，没有放宽碰撞或越界终止。每步原生reward、cost和分类事件均落盘。',
        '- 保留项目128×128四通道语义BEV和固定绘制顺序适配。神经策略只看实际BEV及自车历史，不提供导航检查点/真实未来；IDM使用原生导航和状态，属于特权参考。',
        '- 官方示例的随机路段包含T/X路口与匝道；本项目专家训练地图仅S/C/SC/CS。神经输入也没有原论文基线使用的导航检查点，因此本轮同时测试道路/车型分布变化及导航意图缺失，不能把与IDM或原论文成绩的差异归因于单一模型因素。',
        '- MSE/Geometry/Full固定为主报告训练种子20261021，规划器和评分器从主报告源码快照加载；不包含后加的12候选超车版，没有重训、优化或根据本轮结果调参。',
        '- 这是官方示例启发的适配pilot，不是原论文PPO/SAC训练协议复现，也不是官方排行榜提交。仅10个独立场景，成功率Wilson区间保存在summary.json；不据此作普遍优劣或阶段8验收结论。',
        '- 延迟是当前AMD/ROCm上的BEV就绪到神经控制量输出，不含仿真、日志和离线可视化；前20步预热排除，表中P95从各臂实际决策样本合并计算。IDM决策在sim.step内部，单独决策耗时保持缺失。',
        '- 使用同步固定步长仿真：0.1s为仿真控制周期，模型运行慢不会自动注入额外物理行驶延迟；视频按仿真时间10fps播放，不代表墙钟实时运行。',
        '- 原生crash聚合包括路沿等事件，官方论文的车辆crash指标另列crash_vehicle。默认cost不覆盖所有crash种类，故cost低不等于无碰撞。',
        f'- 已核对{summary["audit"]["raw_files"]}个原始NPZ哈希、原生reward/cost累计及所有事件；{summary["audit"]["rrd_files"]}个RRD通过原生CLI读回。同seed各臂完整初始车辆场景最大差{summary["audit"]["initial_max_abs_error"]:.3g}，初始BEV字节哈希一致。',
        '- preflight_idm_001的JSON序列化失败与MRE保留；复制运行时环境配置后，IDM/Full短程预检通过。预检不计正式成绩。',
        '', '完整来源、配置、种子、GOAL/源码/检查点哈希见protocol.json与各臂report.json。原始轨迹和原生评分见各seed目录actual.npz；同源Rerun记录为planning.rrd。',
        '', '官方参考：[泛化环境示例](https://metadrive-simulator.readthedocs.io/en/latest/rl_environments.html#generalization-environment)。']
    lines += ['', '## 逐场景到达结果', '',
              '| seed | 实际路段（I为起始段） | 车道数/宽度m | 自车 | IDM | MSE | Geometry | Full |',
              '|---|---|---|---|---|---|---|---|']
    for i, seed in enumerate(expected):
        e=reports['idm']['episodes'][i]
        outcomes=['到达' if reports[a]['episodes'][i]['arrive_dest'] else '未到达' for a in ARMS]
        lines.append(f'| {seed} | {"/".join(e["map_blocks"])} | {e["lane_num"]}/{e["lane_width_m"]:.2f} | '
                     f'{e["vehicle_model"]} | '+ ' | '.join(outcomes)+' |')
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary


def render(root, summary):
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    from mini_wa_jepa.visualization.rerun_logger import colorize_bev_4ch
    from mini_wa_jepa.geometry import ego_to_pixels
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data={}
    for arm in ARMS:
        with np.load(root/arm/'seed_000/actual.npz') as z:
            keys=['bev','states','route','events','trajectory_target','trajectory_valid']
            if arm != 'idm':keys += ['candidates','plan','selected_ids']
            data[arm]={k:z[k] for k in keys}
    font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font=ImageFont.truetype(font_path,19)
    small=ImageFont.truetype(font_path,16)
    def frame(step):
        canvas=Image.new('RGB',(1280,880),'#111827')
        draw=ImageDraw.Draw(canvas)
        draw.text((20,10),'MetaDrive泛化benchmark试跑 | 同一seed 0真实记录 | 车流0.1',font=font,fill='white')
        for index,arm in enumerate(ARMS):
            x=(index%2)*640+14;y=(index//2)*400+52
            d=data[arm];n=len(d['bev']);i=min(step,n-1)
            panel=Image.fromarray(colorize_bev_4ch(d['bev'][i])).resize((336,336),Image.Resampling.NEAREST)
            canvas.paste(panel,(x,y+30));draw=ImageDraw.Draw(canvas)
            draw.text((x,y),f'{arm.upper()} | seed 0 | t={i*.1:.1f}s',font=font,fill='#e2e8f0')
            def points(a):return [tuple(p) for p in (ego_to_pixels(a[:,:2],128,64)+.5)*(336/128)+np.array([x,y+30])]
            if arm!='idm':
                for a in d['candidates'][i]:draw.line(points(a),fill='#74849a',width=1)
                draw.line(points(d['plan'][i]),fill='#22d3ee',width=3)
            truth=d['trajectory_target'][i][d['trajectory_valid'][i]]
            if len(truth)>1:draw.line(points(truth),fill='#fb923c',width=2)
            s=summary['arms'][arm]
            labels=[f'进度 {d["route"][i]:.1%}',f'车速 {d["states"][i,3]:.2f} m/s',
                    '全部10场景成绩：',f'到达 {s["successes"]}/10',f'车辆碰撞 {s["crash_vehicle_rate"]:.0%}',
                    f'越界 {s["out_of_road_rate"]:.0%}',
                    'IDM：使用导航/状态' if arm=='idm' else '蓝：计划 / 橙：后来实驶']
            if step>=n:
                flags=dict(zip(EVENTS,d['events'].any(0)))
                reason='到达' if flags['arrive_dest'] else '越界' if flags['out_of_road'] else '碰撞' if flags['crash'] else '超时'
                labels.append(f'本局终止：{reason}（末帧）')
            for k,label in enumerate(labels):draw.text((x+350,y+43+k*33),label,font=small,fill='#e2e8f0')
        draw.text((20,852),'官方示例的10/200场景适配pilot；非论文复现。完整轨迹见RRD；原始分数见报告。',font=small,fill='#cbd5e1')
        return np.asarray(canvas)
    path=root/'benchmark_preview.mp4'
    if path.exists():raise FileExistsError(path)
    writer=imageio_ffmpeg.write_frames(str(path),(1280,880),fps=10,codec='libx264',quality=7,
                                      macro_block_size=16,output_params=['-movflags','+faststart'])
    writer.send(None)
    n=max(len(d['bev']) for d in data.values())
    snapshots=root/'frames'
    snapshots.mkdir(exist_ok=True)
    try:
        for step in range(n):
            rgb=frame(step)
            writer.send(np.ascontiguousarray(rgb))
            if step%100==0 or step==n-1:
                Image.fromarray(rgb).save(snapshots/f'step_{step:05d}.png')
    finally:writer.close()
    Image.fromarray(frame(min(180,n-1))).save(root/'preview.png')
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for ax,key,title in zip(axes,['success_rate','crash_vehicle_rate','out_of_road_rate'],
                            ['Success rate ↑','Vehicle crash rate ↓','Out-of-road rate ↓']):
        values=[summary['arms'][a][key]*100 for a in ARMS]
        bars=ax.bar(ARMS,values,color=['#94a3b8','#60a5fa','#34d399','#fbbf24'])
        ax.bar_label(bars,fmt='%.0f%%');ax.set_ylim(0,115);ax.set_title(title);ax.set_ylabel('% of 10 scenarios')
    fig.suptitle('MetaDrive PG pilot | seeds 0–9 | density 0.1 | native termination')
    fig.tight_layout();fig.savefig(root/'benchmark_scores.png',dpi=160);plt.close(fig)
    write_json(root/'video_manifest.json',dict(video=str(path.relative_to(ROOT)),sha256=sha(path),
               fps=10,frames=n,seed=0,source_raw={a:str((root/a/'seed_000/actual.npz').relative_to(ROOT)) for a in ARMS},
               semantics='actual recorded first-seed BEV/plan and subsequently executed trajectory, same 0.1s timeline; ended episodes hold labeled final frame; IDM has no neural plan'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    args=p.parse_args();root=Path(args.root).resolve()
    s=summarize(root);render(root,s)
    print(json.dumps(s['arms'],indent=2))
