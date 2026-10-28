"""Audit matched development rollouts and export actual comparison media."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from mini_wa_jepa.provenance import provenance,write_json
from scripts.collect_integrated_navigation import digest,EVENTS


def report(sources,output):
    output.mkdir(parents=True,exist_ok=False); results=[]; initials={}; media={}; total=0
    for source in sources:
        r=json.loads((source/'report.json').read_text());assert r['completed'] and r['control_verified']
        key=source.name;initials[key]={};timings=[];safe_fallback=0;unsafe_fallback=0
        for ep in r['episodes']:
            folder=source/f"seed_{ep['seed']}";assert digest(folder/'actual.npz')==ep['raw_sha256']
            assert digest(folder/'driving.rrd')==ep['rrd_sha256']
            initial=json.loads((folder/'initial_scene.json').read_text());initials[key][ep['seed']]=initial
            with np.load(folder/'actual.npz') as a:
                np.testing.assert_array_equal(a['action'],a['applied_action'])
                np.testing.assert_array_equal(a['events'][-1],[ep['outcome'][k] for k in EVENTS])
                assert len(a['states'])==len(a['action'])+1
                assert abs(float(a['reward'].sum())-ep['reward'])<1e-8
                assert abs(float(a['cost'].sum())-ep['cost'])<1e-8
                total+=len(a['action']);timings.extend(a['decision_ms'][20:].tolist())
                mask=a['fallback'];safe_fallback+=int((mask&(a['fallback_geometry_safe']==1)).sum())
                unsafe_fallback+=int((mask&(a['fallback_geometry_safe']==0)).sum())
                if r['arm'] in ('full','zero'):
                    assessed=a['risk_assessed'];chosen=a['selected_index']
                    for i,index in enumerate(chosen):
                        if index>=0:assert assessed[i,index]
                    assert (assessed.sum(1)<=4).all()
                    assert np.isnan(a['risk_probability'][~assessed]).all()
                if ep['seed']==4300301:
                    media[key]={k:a[k] for k in a.files}
            with (output/f"{key}_{ep['seed']}_rrd.txt").open('w') as stream:
                subprocess.run([sys.executable,'-m','rerun','rrd','print',str(folder/'driving.rrd')],
                    stdout=stream,stderr=subprocess.STDOUT,check=True)
        nav=[e for e in r['episodes'] if 'case' not in e];traffic=[e for e in r['episodes'] if 'case' in e]
        results.append(dict(source=key,arm=r['arm'],parameters=r['loaded_parameters'],
            source_report_sha256=digest(source/'report.json'),ordinary_scenes=len(nav),traffic_scenes=len(traffic),
            navigation_arrivals=sum(e['outcome']['arrive_dest'] for e in nav),
            traffic_maneuvers=sum(e['maneuver_complete'] for e in traffic),
            crash_episodes=sum(e['outcome']['crash'] for e in r['episodes']),
            offroad_episodes=sum(e['outcome']['out_of_road'] for e in r['episodes']),
            timeout_episodes=sum(e['outcome']['max_step'] for e in r['episodes']),
            controlled_actor_removed=sum(bool(e['removed_npc_ids']) for e in r['episodes']),
            native_reward=float(sum(e['reward'] for e in r['episodes'])),native_cost=float(sum(e['cost'] for e in r['episodes'])),
            decision_p95_ms=float(np.percentile(timings,95)),safe_brake_fallback_steps=safe_fallback,
            unsafe_brake_fallback_steps=unsafe_fallback,episodes=r['episodes']))
    first=next(iter(initials.values()))
    for current in initials.values():
        assert current.keys()==first.keys()
        for seed in first:
            np.testing.assert_allclose(current[seed]['actor_states'],first[seed]['actor_states'],atol=1e-5,rtol=0)
            assert current[seed]['bev_sha256']==first[seed]['bev_sha256']
            assert current[seed]['vehicle_parameters']==first[seed]['vehicle_parameters']
            assert current[seed]['map_hash']==first[seed]['map_hash']
    summary=dict(**provenance(),completed=True,interim_only=True,stage7_acceptance=False,test_access=False,
        sources=results,verified_actions=total,native_rrd_readbacks=sum(len(r['episodes']) for r in results),
        paired_initial_scene_bev_vehicle_map=True,timing='development observations; concurrent jobs possible; not isolated Full acceptance')
    write_json(output/'audit.json',summary)
    lines=['# 融合驾驶开发闭环对照（阶段7尚未通过）','','| 模型 | 普通导航到达 | 交通目标完成 | 碰撞/越界 | 超时 | NPC移除 | 决策P95(ms) |','|---|---|---|---|---|---|---|']
    for r in results:
        lines.append(f"| {r['source']} | {r['navigation_arrivals']}/{r['ordinary_scenes']} | {r['traffic_maneuvers']}/{r['traffic_scenes']} | {r['crash_episodes']}/{r['offroad_episodes']} | {r['timeout_episodes']} | {r['controlled_actor_removed']} | {r['decision_p95_ms']:.2f} |")
    lines+=['','普通导航6局为密度0.1、固定车型、地图辅助路线；受控交通4局另有1/2/3/1个明确NPC，背景车流0。交通目标是完整超越回归或稳定跟车，不能把提前完成目标计成原生arrive_dest。表中碰撞/越界是全部10局原生ego事件，NPC移除单独列出，不偷偷算到达或安全完成。',
        '所有臂使用同一真实观察权限与固定第一训练种子。初始全交通/自车状态tol1e-5、BEV逐元素hash、车型和地图hash配对一致，动作/标签/成本及全部RRD已审计。当前计时可能存在开发并发，不作为完整Full≤50ms或10Hz真实调度验收。',
        'Joint-MSE的交通失败说明低开环误差不保证闭环行为。Geometry的终点前拒绝/停车、静态许可与候选覆盖问题保留；没有通过放松硬约束或重新启用危险候选来提高完成率。当前不能宣布WAM增益或阶段7通过；阶段8/9未启动。']
    (output/'PROGRESS.md').write_text('\n'.join(lines)+'\n')
    render(media,output)
    print(json.dumps({k:v for k,v in summary.items() if k in ('completed','verified_actions','native_rrd_readbacks','paired_initial_scene_bev_vehicle_map')}),flush=True)


def render(data, output):
    from PIL import Image, ImageDraw, ImageFont
    import imageio_ffmpeg
    from mini_wa_jepa.geometry import ego_to_pixels
    from mini_wa_jepa.visualization import RerunLogger, colorize_bev_4ch

    num_arms = len(data)
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 18)
    small = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 15)
    n = max(len(a['bev']) for a in data.values())
    movie = output / 'actual_traffic_comparison.mp4'

    # Support arbitrary number of arms (e.g. 3 baselines or full 5 arms)
    if num_arms <= 3:
        canvas_w, canvas_h = 1200, 656
        panel_size = 384
        x_spacing = 398
        x_offset = 12
    else:
        # 5-arm layout: 1520 x 640 (macro-block 16 aligned)
        canvas_w, canvas_h = 1520, 640
        panel_size = 280
        x_spacing = 298
        x_offset = 16

    writer = imageio_ffmpeg.write_frames(
        str(movie), (canvas_w, canvas_h), fps=10, codec='libx264', quality=7, macro_block_size=16,
        output_params=['-movflags', '+faststart']
    )
    writer.send(None)
    try:
        for step in range(n):
            canvas = Image.new('RGB', (canvas_w, canvas_h), '#111827')
            draw = ImageDraw.Draw(canvas)
            title = f"同输入开发闭环 | seed4300301右侧可绕行 | 五臂真实事件对比（无老师干预）" if num_arms >= 5 else "同输入开发闭环 | seed4300301右侧可绕行 | 原生事件与失败完整保留"
            draw.text((15, 9), title, font=font, fill='white')
            for j, (name, a) in enumerate(data.items()):
                i = min(step, len(a['bev']) - 1)
                x = x_offset + j * x_spacing
                y = 74
                panel = Image.fromarray(colorize_bev_4ch(a['bev'][i])).resize((panel_size, panel_size), Image.Resampling.NEAREST)
                overlay = ImageDraw.Draw(panel)
                scale = panel_size / 128.0
                for points, color in [
                    (a['route_points'][i][a['route_valid'][i]], '#c084fc'),
                    (a['plan'][i, :, :2], '#22d3ee'),
                    (a['target'][i][a['valid'][i], :2], '#fbbf24'),
                ]:
                    if len(points) > 1:
                        pixels = (ego_to_pixels(points[:, :2], 128, 64) + 0.5) * scale
                        overlay.line([tuple(p) for p in pixels], fill=color, width=3)
                canvas.paste(panel, (x, y))
                clean_name = name.replace('mixed_', '').replace('_closed_012', '').replace('_closed_014', '').replace('_closed_015', '')
                draw.text((x, 44), clean_name, font=font, fill='#38bdf8' if 'full' in clean_name else ('#a78bfa' if 'zero' in clean_name else 'white'))
                ended = step >= len(a['bev'])
                unsafe = bool(a['events'][i, [0, 4]].any())
                telemetry_y = y + panel_size + 12
                lines = [
                    f"仿真 {i * 0.1:.1f}s | 速度 {a['states'][i, 3]:.2f}m/s",
                    f"目标 {bool(a['diagnostic_goal_complete'][i])} | 制动回退 {bool(a['fallback'][i])}",
                    ('已碰撞/越界；保持最后帧' if unsafe else '回合已结束；保持最后帧') if ended else ('原生碰撞/越界' if unsafe else '真实动作/无干预'),
                ]
                for k, line in enumerate(lines):
                    draw.text((x, telemetry_y + k * 26), line, font=small, fill='#f87171' if unsafe else '#e2e8f0')
            draw.text((15, canvas_h - 32), '紫：地图辅助路线；蓝：所选计划；黄：实际轨迹真值 | 开发验证，非阶段验收', font=small, fill='#94a3b8')
            writer.send(np.ascontiguousarray(np.asarray(canvas)))
            if step == min(80, n - 1):
                canvas.save(output / 'preview.png')
    finally:
        writer.close()
    with RerunLogger(app_id='integrated-driving-preview', spawn=False, save_path=output / 'preview.rrd') as log:
        log.set_step(0, 0.0)
        with Image.open(output / 'preview.png') as im:
            log.log_camera(np.asarray(im.convert('RGB')), name='preview/actual_comparison')
    write_json(output / 'video_manifest.json', dict(
        seed=4300301, frames=n, fps=10, video_sha256=digest(movie),
        raw_sources=list(data), ended_frames_explicitly_held=True, num_arms=num_arms
    ))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();report(a.sources,a.output)
