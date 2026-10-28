"""Generate comprehensive visual assets for BEV-JEPA-X success and failure case analysis."""
import os
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from mini_wa_jepa.geometry import ego_to_pixels, world_to_ego, box_corners
from mini_wa_jepa.visualization.rerun_logger import colorize_bev_4ch
from planning.candidates import actor_tracks, TERM_NAMES

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

CANDIDATE_COLORS = [
    '#00d4ff', '#ff5252', '#69f0ae', '#ffd740',
    '#e040fb', '#ff6e40', '#18ffff', '#b0bec5'
]

CANDIDATE_NAMES = [
    'A (直行巡航 a=0)',
    'B (轻踩刹车 a=-2)',
    'C (紧急重刹 a=-4.5)',
    'D (左转巡航 δ=+4.6°)',
    'E (右转巡航 δ=-4.6°)',
    'F (左转减速 a=-2.5)',
    'G (右转减速 a=-2.5)',
    'H (直行加速 a=+1.0)'
]


def render_bev_decision_axes(ax, data, step, title=None, show_legend=False):
    bev = data['bev'][step]
    rgb = colorize_bev_4ch(bev)
    ax.imshow(rgb, extent=[-32, 32, -32, 32], origin='lower')
    
    # Draw ego vehicle footprint
    length, width = 4.5, 2.0
    corners = box_corners(np.zeros(2), 0.0, length, width)
    # in local ego frame: x is forward (vertical in plot? wait: check ego_to_pixels)
    # In ego_to_pixels: u = 63.5 - y*2, v = 63.5 - x*2.
    # When displaying image with imshow(rgb):
    # extent=[-32, 32, -32, 32] where X axis is y_ego (lateral, right is positive or left positive?)
    # Let's use ego_to_pixels directly on pixel grid for 100% exact alignment with existing tools!
    pass


def render_bev_decision_pixels(ax, data, step, title=None, show_legend=False):
    bev = data['bev'][step]
    rgb = colorize_bev_4ch(bev)
    ax.imshow(rgb)
    
    selected_idx = int(data['selected_index'][step])
    candidates = data['candidates'][step]
    state = data['states'][step]
    
    # 1. Past trajectory
    past_states = data['states'][max(0, step - 30):step + 1]
    if len(past_states) > 1:
        past_ego = world_to_ego(past_states[:, :2], state[:2], state[2])
        past_px = ego_to_pixels(past_ego, 128, 64)
        ax.plot(past_px[:, 0], past_px[:, 1], color='#e2e8f0', lw=2.5, linestyle='-', alpha=0.7, label='行驶历史轨迹')
    
    # 2. Candidate trajectories
    for k, path in enumerate(candidates):
        xy = ego_to_pixels(path[:, :2], 128, 64)
        name = chr(65 + k)
        is_sel = (k == selected_idx)
        color = CANDIDATE_COLORS[k % len(CANDIDATE_COLORS)]
        alpha = 1.0 if is_sel else 0.45
        lw = 3.5 if is_sel else 1.2
        ls = '-' if is_sel else '--'
        ax.plot(xy[:, 0], xy[:, 1], ls, color=color, lw=lw, alpha=alpha,
                label=f'{name}: {CANDIDATE_NAMES[k]}' if show_legend else None)
        if not is_sel:
            ax.text(xy[-1, 0], xy[-1, 1], name, color=color, fontsize=8, weight='bold', alpha=0.8)
    
    # 3. Selected plan
    plan_px = ego_to_pixels(data['plans'][step, :, :2], 128, 64)
    ax.plot(plan_px[:, 0], plan_px[:, 1], color='#00ffcc', lw=3.0, label='最终执行规划' if show_legend else None)
    
    # 4. Subsequent actual trajectory if valid
    actual = data['trajectory_target'][step][data['trajectory_valid'][step]]
    if len(actual) > 0:
        actual_px = ego_to_pixels(actual[:, :2], 128, 64)
        ax.plot(actual_px[:, 0], actual_px[:, 1], color='#ffb703', lw=2.0, linestyle=':', label='随后真实行驶' if show_legend else None)
    
    # 5. Ego vehicle marker
    ax.scatter([63.5], [63.5], marker='^', c='#ffffff', s=120, edgecolors='#000000', linewidths=1.5, zorder=15)
    
    # 6. Obstacle velocity vectors & tracking
    bevs = data['bev'][max(0, step - 1):step + 1]
    states = data['states'][max(0, step - 1):step + 1]
    pos, vel, rad, matched = actor_tracks(bevs, states)
    for p, v, r, m in zip(pos, vel, rad, matched):
        dist = np.hypot(p[0], p[1])
        if dist < 28:
            px = ego_to_pixels(p[None, :2], 128, 64)[0]
            # Draw circle or box
            r_px = r * 2.0
            circle = plt.Circle((px[0], px[1]), r_px, fill=False, color='#ff0055', lw=1.5, linestyle='-', alpha=0.85)
            ax.add_patch(circle)
            if np.hypot(v[0], v[1]) > 0.3:
                v_end = p[:2] + v[:2] * 0.8
                v_px = ego_to_pixels(v_end[None, :2], 128, 64)[0]
                ax.annotate('', xy=(v_px[0], v_px[1]), xytext=(px[0], px[1]),
                            arrowprops=dict(arrowstyle="->", color='#ff0055', lw=1.5))
    
    # Limits & overlay text
    ax.set_xlim(10, 118)
    ax.set_ylim(112, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    sel_id = data['selected_ids'][step]
    speed = state[3]
    route = data['route'][step] * 100
    crash_prob = data['probability'][step, selected_idx, 0] if 'probability' in data and data['probability'] is not None else 0.0
    
    # Information badge
    info_str = f"步数: {step} ({step*0.1:.1f}s) | 速度: {speed:.1f} m/s\n路径完成: {route:.1f}% | 优选: {sel_id}\n碰撞风险: {crash_prob*100:.1f}%"
    ax.text(0.03, 0.05, info_str, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f172a', alpha=0.8, edgecolor='#38bdf8'))
    
    if title:
        ax.set_title(title, fontsize=11, weight='bold', pad=6)


def generate_success_progression(data1, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), layout='constrained')
    
    stages = [
        (100, "阶段 1: 初始开阔段加速巡航\n(步数 100 · 速度 5.9 m/s · 选 H 加速)"),
        (282, "阶段 2: 高密度车流主动减速跟车\n(步数 282 · 发现前车降速 · 选 A 控速)"),
        (500, "阶段 3: 弯道多车博弈平顺过弯\n(步数 500 · 速度 2.4 m/s · 选 D 转向)"),
        (978, "阶段 4: 终点安全平稳抵达\n(步数 978 · 完成度 97.9% · 零碰撞零违规)")
    ]
    
    for ax, (step, title) in zip(axes, stages):
        render_bev_decision_pixels(ax, data1, step, title=title)
    
    fig.suptitle("【成功案例】BEV-JEPA-X 闭环全通场景 (SCS_1010001 · 97.9% 完成度 · 零碰撞抵达)",
                 fontsize=14, weight='bold', y=1.04)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_failure_progression(data2, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), layout='constrained')
    
    stages = [
        (300, "阶段 1: 前车拥堵减速跟随\n(步数 300 · 速度 0.7 m/s · 路径 58.8%)"),
        (335, "阶段 2: 拥堵缓行静止等待\n(步数 335 · 速度 0.06 m/s · 蠕行排队)"),
        (340, "阶段 3: 右前方切入车动态挤压\n(步数 340 · 速度 1.3 m/s · 间隙逼仄)"),
        (341, "阶段 4: 起步加速遭遇右侧擦碰\n(步数 341 · 速度 1.6 m/s · 第342步触发接触)")
    ]
    
    for ax, (step, title) in zip(axes, stages):
        render_bev_decision_pixels(ax, data2, step, title=title)
        
    fig.suptitle("【失败案例】BEV-JEPA-X 唯一碰撞场景深度复盘 (SCS_1010002 · 60.3% 完成度 · 第342步侧前方擦碰)",
                 fontsize=14, weight='bold', y=1.04)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_telemetry_comparison(data1, data2, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), layout='constrained')
    
    # 1. Speed profile comparison
    t1 = np.arange(len(data1['route'])) * 0.1
    t2 = np.arange(len(data2['route'])) * 0.1
    
    v1 = data1['states'][:-1, 3] if len(data1['states']) > len(data1['route']) else data1['states'][:, 3]
    v2 = data2['states'][:-1, 3] if len(data2['states']) > len(data2['route']) else data2['states'][:, 3]
    
    ax0 = axes[0, 0]
    ax0.plot(t1, v1[:len(t1)], color='#10b981', lw=1.8, label='成功案例 SCS_1010001 (全通 97.9%)')
    ax0.plot(t2, v2[:len(t2)], color='#ef4444', lw=2.0, label='失败案例 SCS_1010002 (第34.2s碰撞)')
    ax0.axvline(x=len(t2)*0.1, color='#ef4444', linestyle='--', alpha=0.7, label='SCS_1010002 碰撞时刻')
    ax0.set_title('车速演化曲线对比 (Speed m/s)', fontsize=12, weight='bold')
    ax0.set_xlabel('时间 (s)')
    ax0.set_ylabel('纵向速度 (m/s)')
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc='upper right', fontsize=9)
    
    # 2. Route completion comparison
    ax1 = axes[0, 1]
    ax1.plot(t1, data1['route'] * 100, color='#10b981', lw=2.0, label='成功案例 SCS_1010001 (抵达 97.9%)')
    ax1.plot(t2, data2['route'] * 100, color='#ef4444', lw=2.0, label='失败案例 SCS_1010002 (止步 60.3%)')
    ax1.set_title('全局导航路径完成度对比 (Route Completion %)', fontsize=12, weight='bold')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('路径完成度 (%)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=9)
    
    # 3. Cost terms around crash in SCS_1010002
    ax2 = axes[1, 0]
    crash_window = range(310, 342)
    cw_t = np.array(list(crash_window)) * 0.1
    sel_indices = data2['selected_index'][crash_window]
    
    dyn_cost = [data2['terms'][t, sel_indices[i], 5] for i, t in enumerate(crash_window)]
    stat_cost = [data2['terms'][t, sel_indices[i], 4] for i, t in enumerate(crash_window)]
    spd_cost = [data2['terms'][t, sel_indices[i], 1] for i, t in enumerate(crash_window)]
    
    ax2.plot(cw_t, dyn_cost, label='动态障碍物惩罚 (Dynamic Cost)', color='#f59e0b', lw=2.0)
    ax2.plot(cw_t, stat_cost, label='道路边界静态惩罚 (Static Cost)', color='#3b82f6', lw=1.8)
    ax2.plot(cw_t, spd_cost, label='期望速度偏差惩罚 (Speed Cost)', color='#8b5cf6', lw=1.8)
    ax2.axvline(x=34.1, color='#ef4444', linestyle='--', label='碰撞接触前 100ms')
    ax2.set_title('失败案例最后 3 秒核心规划代价项演化 (SCS_1010002)', fontsize=12, weight='bold')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('归一化代价')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=9)
    
    # 4. Candidate distribution comparison
    ax3 = axes[1, 1]
    from collections import Counter
    c1 = Counter(data1['selected_ids'])
    c2 = Counter(data2['selected_ids'])
    keys = [chr(65+i) for i in range(8)]
    x = np.arange(len(keys))
    w = 0.35
    ax3.bar(x - w/2, [c1.get(k, 0) / len(data1['selected_ids']) * 100 for k in keys],
            width=w, label='成功案例 (SCS_1010001)', color='#10b981', alpha=0.85)
    ax3.bar(x + w/2, [c2.get(k, 0) / len(data2['selected_ids']) * 100 for k in keys],
            width=w, label='失败案例 (SCS_1010002)', color='#ef4444', alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{k}\n({CANDIDATE_NAMES[i].split()[1]})' for i, k in enumerate(keys)], fontsize=8)
    ax3.set_title('物理候选动作选择频次分布对比 (%)', fontsize=12, weight='bold')
    ax3.set_ylabel('选择占比 (%)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=9)
    
    fig.suptitle('BEV-JEPA-X 闭环行为与遥测特征深度对比分析', fontsize=14, weight='bold')
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_side_by_side_contrast(data1, data2, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout='constrained')
    
    # Left: Success moment (step 282, proactive following)
    render_bev_decision_pixels(axes[0], data1, 282,
                              title="【成功案例】步数 282 (28.2s)：超前跟车控速\n识别前方减速慢车，选择制动候选B/A，从 6.0m/s 平稳降至 4.5m/s",
                              show_legend=True)
    axes[0].legend(loc='lower left', fontsize=7.5, framealpha=0.85)
    
    # Right: Failure moment (step 341, right cut-in pinch)
    render_bev_decision_pixels(axes[1], data2, 341,
                              title="【失败案例】步数 341 (34.1s)：盲区多车夹缝挤压\n右前侧近邻慢车切入，多候选遭航向约束剔除，起步加速发生擦碰",
                              show_legend=False)
    
    fig.suptitle('关键决策瞬态对比：超前跟车控速 (成功) vs 密闭车流夹缝突围 (碰撞)',
                 fontsize=13, weight='bold', y=1.03)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_animated_gif(data, start_step, end_step, out_path, title_prefix=""):
    frames = []
    fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')
    for step in range(start_step, end_step + 1):
        ax.clear()
        title = f"{title_prefix} · 步数 {step} ({step*0.1:.1f}s)"
        render_bev_decision_pixels(ax, data, step, title=title)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        im = Image.fromarray(rgba).convert('RGB')
        frames.append(im)
    plt.close(fig)
    if frames:
        frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        print(f"Saved GIF: {out_path} ({len(frames)} frames)")


def main():
    root = Path('artifacts/restart_stage7')
    fig_dir = root / 'figures'
    fig_dir.mkdir(exist_ok=True, parents=True)
    
    d1 = np.load(root / 'closed_full_20261021/SCS_1010001/actual.npz')
    d2 = np.load(root / 'closed_full_20261021/SCS_1010002/actual.npz')
    
    p1 = fig_dir / 'case_success_scs1010001_progression.png'
    p2 = fig_dir / 'case_failure_scs1010002_progression.png'
    p3 = fig_dir / 'case_telemetry_comparison.png'
    p4 = fig_dir / 'case_side_by_side_contrast.png'
    g1 = fig_dir / 'success_car_following.gif'
    g2 = fig_dir / 'failure_crash_moment.gif'
    
    print("Generating progression figures...")
    generate_success_progression(d1, p1)
    generate_failure_progression(d2, p2)
    generate_telemetry_comparison(d1, d2, p3)
    generate_side_by_side_contrast(d1, d2, p4)
    
    print("Generating animated GIFs...")
    # Step 275 to 295 for car following
    generate_animated_gif(d1, 275, 295, g1, title_prefix="成功跟车减速 (SCS_1010001)")
    # Step 330 to 341 for failure
    generate_animated_gif(d2, 330, 341, g2, title_prefix="碰撞挤压瞬态 (SCS_1010002)")
    
    # Also copy to artifact directory
    artifact_dir = Path('/home/auronox/.gemini/antigravity/brain/cd91365c-b14d-4381-9b9c-2c06061c55ba')
    if artifact_dir.exists():
        for f in [p1, p2, p3, p4, g1, g2]:
            shutil.copy(f, artifact_dir / f.name)
        print(f"Copied all visual assets to artifact directory: {artifact_dir}")

if __name__ == '__main__':
    main()
