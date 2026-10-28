"""Audited interim Stage7 progress, actual rollout preview; no stage acceptance."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
from mini_wa_jepa.provenance import write_json
from scripts.collect_integrated_navigation import digest, EVENTS


def report(root, version='002'):
    training = root/(f'resnet_navigation_pilot_{version}' if version == '005' else f'navigation_head_pilot_{version}')
    closed = root/(f'resnet_navigation_closed_{version}' if version == '005' else f'navigation_closed_validation_{version}')
    tr = json.loads((training/'report.json').read_text())
    cl = json.loads((closed/'report.json').read_text())
    assert tr['completed'] and cl['completed'] and tr['control_verified'] and cl['control_verified']
    output = root/f'startup_progress_{version}'; output.mkdir(exist_ok=False)
    episodes = cl['episodes']; timings = []; controls = 0; recordings = 0
    for ep in episodes:
        folder = closed/f"seed_{ep['seed']}"
        assert digest(folder/'actual.npz') == ep['raw_sha256']
        assert digest(folder/'rollout.rrd') == ep['rerun_sha256']
        with np.load(folder/'actual.npz', allow_pickle=False) as raw:
            np.testing.assert_allclose(raw['action'], raw['applied_action'], atol=1e-7, rtol=0)
            assert len(raw['states']) == len(raw['plan'])+1
            np.testing.assert_array_equal(raw['events'][-1], np.array([ep['outcome'][k] for k in EVENTS]))
            controls += len(raw['action']); timings.extend(raw['decision_ms'][20:].tolist())
        with (output/f"seed_{ep['seed']}_rrd.txt").open('w') as stream:
            subprocess.run([sys.executable, '-m', 'rerun', 'rrd', 'print', str(folder/'rollout.rrd')],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        recordings += 1
    for rrd in sorted(training.glob('head_*/epoch_*/predictions.rrd')):
        with (output/(rrd.parent.parent.name+'_'+rrd.parent.name+'_rrd.txt')).open('w') as stream:
            subprocess.run([sys.executable, '-m', 'rerun', 'rrd', 'print', str(rrd)],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        recordings += 1
    summary = dict(completed=True, interim_only=True, stage7_acceptance=False,
                   source_reports_sha256=dict(training=digest(training/'report.json'), closed=digest(closed/'report.json')),
                   models=tr['models'], validation_scenes=len(episodes),
                   arrivals=sum(e['outcome']['arrive_dest'] for e in episodes),
                   crashes=sum(e['outcome']['crash'] for e in episodes),
                   offroad=sum(e['outcome']['out_of_road'] for e in episodes),
                   timeouts=sum(e['outcome']['max_step'] for e in episodes),
                   mean_route=float(np.mean([e['route_completion'] for e in episodes])),
                   navigation_MSE_decision_p95_ms=float(np.percentile(timings, 95)),
                   verified_control_transitions=controls, verified_training_and_closed_rrd=recordings,
                   invalid_runs_preserved='All001 datasets/models/rollouts are invalidated by INVALID_CONTROL_WIRING.json',
                   pending=['ResNet_navigation', 'traffic_expert_repair', 'recovery_and_passing_data',
                            'navigation_Flow', 'WAM_risk_training', 'paired_scientific_acceptance', 'Stage8', 'Stage9'])
    if version == '005':
        summary['pending'].remove('ResNet_navigation')
        summary['loaded_navigation_parameters'] = cl['loaded_navigation_parameters']
        summary['comparison_required'] = 'Read prior_contribution_audit_004: rule-only already6/6; arrival alone is not learned/WAM gain.'
    write_json(output/'summary.json', summary)
    lines = ['# 阶段7补充：首轮执行进展（不作阶段验收）', '',
             '已实现地图辅助导航接口、schema3逐步动作核对、严格候选选择、曲率制动及实际车型批量物理先验；旧策略/检查点保留。', '',
             '**首轮001无效：** 默认IDM覆盖了请求动作。所有001数据、训练权重、6/6闭环和制动记录保留并标无效，不用于收益结论或新训练。002显式EnvInputPolicy，每步核对请求/策略/车辆实际动作，重新采集并训练。', '',
             f'## {version}导航MSE试训练（冻结BEV编码器，WAM未训练/未参与该闭环）', '',
             '| 训练种子 | 最优epoch | 验证ADE(m) | 验证FDE(m) | 导航置零ADE(m，仅依赖诊断) |',
             '|---|---|---|---|---|']
    for model in tr['models']:
        lines.append(f"| {model['seed']} | {model['selected_epoch']} | {model['validation']['ade_m']:.4f} | "
                     f"{model['validation']['fde_m']:.4f} | {model['navigation_zeroed_diagnostic']['ade_m']:.4f} |")
    lines += ['', f"导航头20261023在预登记的6个验证场景到达{summary['arrivals']}/6，碰撞{summary['crashes']}、越界{summary['offroad']}、超时{summary['timeouts']}；平均路线完成{summary['mean_route']:.2%}。",
              f"该小基线完整决策P95={summary['navigation_MSE_decision_p95_ms']:.2f}ms，含导航/编码器/轨迹头/跟踪器，不含仿真/日志；不是Full/WAM延迟。",
              '这些场景属于参与模型选择的开发验证集，固定车型与车道、车流0.1、地图辅助导航；不是盲测benchmark、不是旧无导航MSE的配对提升证明，也不是完整融合驾驶验收。', '',
              '## 当前仍需完成', '',
              '交通专家/安全超车与回归数据、导航Flow、世界/零世界风险头和配对科学验收仍待完成。WAM第一轮保持原权重；阶段8/9尚未启动。', '',
              f"审计通过{controls}个实际控制转移及{recordings}个训练/闭环RRD原生读回；训练/验证采集审计见各002目录audit。首个路口模板4300002导出视频，展示原始BEV、导航、模型计划及后续实际轨迹。"]
    if version == '005':
        lines += ['', '本005为冻结项目自训ResNet＋同输入导航头，复用003恢复数据、004路线物理先验。',
                  f"实际仅加载编码器和新导航头，共{cl['loaded_navigation_parameters']:,}参数；旧ResNet轨迹头不加载。该基线不适用Full系统的50M下限，也不能替代Full参数/延迟检查。",
                  '004仅路线规则先验已6/6到达，学习残差版也6/6；请合并阅读prior_contribution_audit_004/COMPARISON.md，不能将相同到达数算作学习或WAM增益。']
    (output/'PROGRESS.md').write_text('\n'.join(lines)+'\n')
    render(closed/'seed_4300002'/'actual.npz', output)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def render(path, output):
    from PIL import Image, ImageDraw, ImageFont
    import imageio_ffmpeg
    from mini_wa_jepa.geometry import ego_to_pixels
    from mini_wa_jepa.visualization import RerunLogger
    from mini_wa_jepa.visualization.rerun_logger import colorize_bev_4ch
    with np.load(path, allow_pickle=False) as raw:
        data = {k:raw[k] for k in raw.files}
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 20)
    small = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 16)
    movie = output/'navigation_validation.mp4'
    writer = imageio_ffmpeg.write_frames(str(movie), (960, 640), fps=10, codec='libx264',
                                        quality=7, macro_block_size=16, output_params=['-movflags', '+faststart'])
    writer.send(None); n = len(data['bev'])
    try:
        for i in range(n):
            canvas = Image.new('RGB', (960, 640), '#111827'); draw = ImageDraw.Draw(canvas)
            draw.text((20, 10), '导航轨迹基线开发验证 | 实际外部动作已逐步核对', font=font, fill='white')
            panel = Image.fromarray(colorize_bev_4ch(data['bev'][i])).resize((512, 512), Image.Resampling.NEAREST)
            canvas.paste(panel, (20, 65))
            def poly(points, color):
                if len(points) > 1:
                    pixels = (ego_to_pixels(points[:, :2], 128, 64)+.5)*4+np.array([20, 65])
                    draw.line([tuple(p) for p in pixels], fill=color, width=3)
            poly(data['route_points'][i][data['route_valid'][i]], '#c084fc')
            poly(data['plan'][i], '#22d3ee'); poly(data['target'][i][data['valid'][i]], '#fbbf24')
            text = [f'seed4300002 | TXC | {i*.1:.1f}s', f'路线完成 {data["route_completion"][i]:.1%}',
                    f'速度 {data["states"][i,3]:.2f}m/s', '紫：当前地图辅助导航', '蓝：模型计划',
                    '黄：后来实际轨迹（离线）', '固定车型 / 车流0.1', '开发验证集 / 非正式验收', '此基线尚未使用WAM']
            for j, line in enumerate(text):
                draw.text((555, 80+j*47), line, font=small, fill='white')
            draw.text((20, 600), '按仿真时间10fps播放；完整轨迹与动作见同源Rerun记录。', font=small, fill='#cbd5e1')
            rgb = np.asarray(canvas); writer.send(np.ascontiguousarray(rgb))
            if i == 0:
                canvas.save(output/'preview.png')
    finally:
        writer.close()
    with RerunLogger(app_id='navigation-progress-preview', spawn=False, save_path=output/'preview.rrd') as log:
        log.set_step(0, 0.)
        with Image.open(output/'preview.png') as im:
            log.log_camera(np.asarray(im.convert('RGB')), name='preview/actual_recording')
    write_json(output/'video_manifest.json', dict(source_raw_sha256=digest(path), video_sha256=digest(movie),
                                                 frames=n, fps=10, seed=4300002, actual_control_verified=True))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=('002', '003', '004', '005'), default='002')
    args = parser.parse_args()
    report(Path('artifacts/stage7_integrated'), args.version)
