"""Recompute actual maneuver outcomes and export a three-case raw BEV video."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from mini_wa_jepa.provenance import provenance, write_json
from scripts.collect_integrated_navigation import digest, EVENTS
from dataset.recording import future_labels_v2


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    report = json.loads((source/'report.json').read_text())
    assert report['completed'] and report['control_verified']
    results = []; data = {}
    for ep in report['episodes']:
        folder = source/ep['case']
        assert digest(folder/'actual.npz') == ep['raw_sha256']
        assert digest(folder/'traffic.rrd') == ep['rerun_sha256']
        with np.load(folder/'actual.npz', allow_pickle=False) as raw:
            a = {k: raw[k] for k in raw.files}
        data[ep['case']] = a
        np.testing.assert_array_equal(a['action'], a['applied_action'])
        assert not a['events'][:, [0, 4]].any()
        assert a['bev'][0, 2].any(), 'Stale spawn-before-NPC initial frame'
        assert len(a['states']) == len(a['action'])+1
        target, valid, _ = future_labels_v2(a['states'], np.zeros(len(a['states']), bool), 10, 2)
        np.testing.assert_allclose(a['target'], target[:-1], atol=1e-6)
        np.testing.assert_array_equal(a['valid'], valid[:-1])
        np.testing.assert_array_equal(a['events'][-1], [ep['outcome'][k] for k in EVENTS])
        safe = not a['events'][:, [0, 4]].any()
        if ep['case'] == 'allowed':
            phases = a['phase'].tolist()
            sequence = [phases.index(k) for k in ('FOLLOW', 'CHANGE_OUT', 'PASS', 'CHANGE_BACK', 'COMPLETE')]
            assert sequence == sorted(sequence)
            assert abs(a['home_lateral_m'][-1]) < .35 and abs(a['heading_error_rad'][-1]) < .08
            assert a['lead_rear_clearance_m'][-1] > 10
            assert a['phase'][-1] == 'COMPLETE'
        else:
            assert set(a['phase']) == {'FOLLOW'}
            assert np.all((a['states'][-21:-1, 3] >= .5) & (a['states'][-21:-1, 3] <= 3))
            assert np.all((a['front_clearance_m'][-20:] >= 3) & (a['front_clearance_m'][-20:] <= 18))
        speed = a['states'][:, 3]
        yaw = np.diff(np.unwrap(a['states'][:, 2]))/.1
        results.append(dict(case=ep['case'], seed=ep['seed'], steps=len(a['action']),
            passing_and_return=ep['case']=='allowed', teacher_only=True, safe=bool(safe),
            final_home_lateral_m=float(a['home_lateral_m'][-1]),
            max_measured_lateral_acceleration_m_s2=float(np.max(np.abs(yaw*speed[:-1]))),
            initial_actor_pixels=int((a['bev'][0, 2] > 0).sum()),
            actual_arrive_dest=ep['outcome']['arrive_dest']))
        with (output/f"{ep['case']}_rrd.txt").open('w') as stream:
            subprocess.run([sys.executable, '-m', 'rerun', 'rrd', 'print', str(folder/'traffic.rrd')],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
    write_json(output/'audit.json', dict(**provenance(), completed=True, source_report_sha256=digest(source/'report.json'),
               results=results, raw_controls_verified=sum(r['steps'] for r in results),
               native_rrd_readbacks=3, stage7_acceptance=False))
    render(data, output, source)
    lines = ['# 交通老师开发探针（非模型/阶段验收）', '',
             '| 场景 | 实际步数 | 行为 | 原生碰撞/越界 |', '|---|---|---|---|']
    labels = {'allowed': '可超车', 'no_adjacent_lane': '单车道', 'blocked': '邻道阻塞'}
    for row in results:
        lines.append(f"| {labels[row['case']]} | {row['steps']} | {'完整超越并回归' if row['passing_and_return'] else '保持跟车至少20步'} | 0/0 |")
    lines += ['', '三局均达到预登记探针目标，提前停止被明确标为probe_complete，实际arrive_dest均false；不能统计为到达率。背景车流0、受控NPC分别1/1/3个，非密度0.3。',
              '老师拥有真实车道与交通对象状态权限，学习输入没有这些真值。只验证240m直路上的跟车、变道、超越和回归；弯道/路口/突然插入/长期跟车尚未由此覆盖。',
              '哈希、实际动作、终止事件、未来标签、首帧NPC可见及3份RRD原生读回已核对。001/002构造失败和003首帧滞后版本完整保留，不用于训练。',
              '实际BEV/未来轨迹视频见traffic_teacher.mp4，未结束场景后续画面显式标为结束帧保持；不是模型预测。']
    (output/'PROGRESS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(results), flush=True)


def render(data, output, source):
    from PIL import Image, ImageDraw, ImageFont
    import imageio_ffmpeg
    from mini_wa_jepa.geometry import ego_to_pixels
    from mini_wa_jepa.visualization import RerunLogger, colorize_bev_4ch
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 20)
    small = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 17)
    labels = ['允许超车：完整超过并回归', '单车道：保持跟车', '相邻车道阻塞：保持跟车']
    n = max(len(a['bev']) for a in data.values())
    movie = output/'traffic_teacher.mp4'
    writer = imageio_ffmpeg.write_frames(str(movie), (1200, 656), fps=10, codec='libx264',
                quality=7, macro_block_size=16, output_params=['-movflags', '+faststart'])
    writer.send(None)
    try:
        for step in range(n):
            canvas = Image.new('RGB', (1200, 656), '#111827'); draw = ImageDraw.Draw(canvas)
            draw.text((15, 8), '交通老师真实执行探针 | 特权规则老师；不是学习模型或WAM成绩', font=font, fill='white')
            for j, a in enumerate(data.values()):
                index = min(step, len(a['bev'])-1); x = 12+j*398; y = 78
                panel = Image.fromarray(colorize_bev_4ch(a['bev'][index])).resize((384, 384), Image.Resampling.NEAREST)
                canvas.paste(panel, (x, y)); draw.text((x, 47), labels[j], font=small, fill='white')
                for points, color in [(a['route_points'][index][a['route_valid'][index]], '#c084fc'),
                                      (a['target'][index][a['valid'][index], :2], '#fbbf24')]:
                    if len(points) > 1:
                        pixels = (ego_to_pixels(points[:, :2], 128, 64)+.5)*3+np.array([x, y])
                        draw.line([tuple(p) for p in pixels], fill=color, width=3)
                text = [f"仿真{index*.1:.1f}s | {a['phase'][index]}",
                        f"速度{a['states'][index,3]:.2f}m/s | 回归偏差{a['home_lateral_m'][index]:.2f}m",
                        '已达到探针目标：保持结束帧' if step >= len(a['bev']) else '实际动作已核对 / 原生安全事件保留']
                for k, line in enumerate(text):
                    draw.text((x, 475+k*32), line, font=small, fill='#fbbf24' if k==2 and step>=len(a['bev']) else 'white')
            draw.text((15, 595), '紫：固定地图路线；黄：后来实际轨迹（离线真值） | 背景车流0＋显式慢车；非到达率验收', font=small, fill='#cbd5e1')
            writer.send(np.ascontiguousarray(np.asarray(canvas)))
            if step == min(80, n-1):
                canvas.save(output/'preview.png')
    finally:
        writer.close()
    with RerunLogger(app_id='traffic-teacher-preview', spawn=False, save_path=output/'preview.rrd') as log:
        log.set_step(0, 0.)
        with Image.open(output/'preview.png') as im:
            log.log_camera(np.asarray(im.convert('RGB')), name='preview/actual_traffic_cases')
    write_json(output/'video_manifest.json', dict(frames=n, fps=10, sha256=digest(movie),
               raw_sources={case:digest(source/case/'actual.npz') for case in data}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args(); run(args.source, args.output)
