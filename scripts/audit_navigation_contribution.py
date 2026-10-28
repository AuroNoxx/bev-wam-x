"""Audit the explicit map-prior control against the learned residual pilot."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from mini_wa_jepa.provenance import provenance, write_json
from scripts.collect_integrated_navigation import digest, EVENTS


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    root = output.parent
    reports = {}; results = {}; initials = {}
    for arm, name in [('learned', 'navigation_closed_validation_004'),
                      ('prior', 'navigation_reference_validation_004')]:
        folder = root/name; report = json.loads((folder/'report.json').read_text())
        assert report['completed'] and report['control_verified']
        assert report['prior_only'] == (arm == 'prior')
        reports[arm] = digest(folder/'report.json'); times = []; initials[arm] = {}; controls = 0
        rows = []
        for ep in report['episodes']:
            scene = folder/f"seed_{ep['seed']}"
            assert digest(scene/'actual.npz') == ep['raw_sha256']
            assert digest(scene/'rollout.rrd') == ep['rerun_sha256']
            with np.load(scene/'actual.npz', allow_pickle=False) as raw:
                np.testing.assert_allclose(raw['action'], raw['applied_action'], atol=1e-7, rtol=0)
                np.testing.assert_array_equal(raw['events'][-1], [ep['outcome'][k] for k in EVENTS])
                if arm == 'prior':
                    np.testing.assert_array_equal(raw['plan'], raw['reference'])
                times.extend(raw['decision_ms'][20:].tolist()); controls += len(raw['action'])
                initials[arm][ep['seed']] = (raw['states'][0].copy(), raw['bev'][0].copy())
                rows.append(dict(seed=ep['seed'], steps=ep['steps'], outcome=ep['outcome'],
                                 route_completion=ep['route_completion'],
                                 mean_residual_position_m=float(np.linalg.norm(raw['plan'][..., :2]-raw['reference'][..., :2], axis=-1).mean())))
            with (output/f"{arm}_{ep['seed']}_rrd.txt").open('w') as stream:
                subprocess.run([sys.executable, '-m', 'rerun', 'rrd', 'print', str(scene/'rollout.rrd')],
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
        results[arm] = dict(episodes=rows, arrivals=sum(r['outcome']['arrive_dest'] for r in rows),
                           crashes=sum(r['outcome']['crash'] for r in rows),
                           offroad=sum(r['outcome']['out_of_road'] for r in rows),
                           mean_steps=float(np.mean([r['steps'] for r in rows])),
                           mean_route=float(np.mean([r['route_completion'] for r in rows])),
                           decision_p95_ms=float(np.percentile(times, 95)), verified_controls=controls)
    assert initials['learned'].keys() == initials['prior'].keys()
    for seed in initials['learned']:
        for a, b in zip(initials['learned'][seed], initials['prior'][seed]):
            np.testing.assert_array_equal(a, b)
    write_json(output/'audit.json', dict(**provenance(), completed=True, arms=results,
               source_reports_sha256=reports, initial_ego_and_bev_equal=True,
               full_initial_actor_state_equality='not_recorded', stage7_acceptance=False,
               conclusion='Both6/6; learned residual has no demonstrated arrival benefit and takes more steps. WAM not used.'))
    lines = ['# 地图路线先验贡献对照（开发验证，非阶段验收）', '',
             '| 控制版本 | 到达 | 碰撞/越界 | 平均步数 | 平均路线完成 |', '|---|---|---|---|---|']
    for arm, label in [('prior', '仅地图物理路线先验'), ('learned', '同先验＋BEV学习残差')]:
        r = results[arm]
        lines.append(f"| {label} | {r['arrivals']}/6 | {r['crashes']}/{r['offroad']} | {r['mean_steps']:.1f} | {r['mean_route']:.2%} |")
    lines += ['', '两臂初始自车状态和BEV逐元素一致；完整初始交通对象状态未记录，保持缺失。逐步实际动作、原始哈希、最终事件及全部12份RRD原生读回通过。', '',
              '这6个重复使用的开发场景是固定车型、车流0.1、精确地图辅助路线输入。学习版没有提高到达数，平均行驶步数反而更多；基础循迹成功主要由规则先验支撑，不能计为学习或WAM增益。两个版本均未使用WAM。', '',
              '规则臂保留公共BEV编码/历史记录，其计时不是纯规则最优实现；所有延迟都不是完整Full系统的验收数据。原004进展文件保留，本文件补充其缺失的贡献对照。']
    (output/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(results, ensure_ascii=False))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
