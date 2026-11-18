"""Paired summaries of the final Flow replay; route and frame IDs remain auditable."""
import json
from pathlib import Path
from statistics import mean

import numpy as np


def summarize(rows):
    if not rows:
        return {'windows': 0}
    def avg(key):
        return mean(row[key] for row in rows)
    result = {
        'windows': len(rows), 'routes': len({row['route'] for row in rows}),
        'model_wins_ade': sum(row['excess_ade'] < 0 for row in rows),
        'mean_absolute_model_5s_x_error': mean(abs(row['model_dx5']) for row in rows),
        'mean_absolute_model_5s_y_error': mean(abs(row['model_dy5']) for row in rows),
    }
    for key in ('model_ade','cv_ade','excess_ade','clip','clip_ax','clip_ay','raw_replay','bounded_replay',
                'model_dx5','cv_dx5','model_jerk_x_p95','model_jerk_y_p95','expert_jerk_x_p95','expert_jerk_y_p95'):
        result[key] = avg(key)
    for method in ('model', 'cv'):
        result[method+'_fde_1_3_5'] = [avg(f'{method}_fde{h}') for h in (1,3,5)]
    return result


def main():
    folder = Path(__file__).resolve().parents[1]/'artifacts/v5-qwen-drive-world/subset2000/final-window-audit'
    rows = json.loads((folder/'windows.json').read_text())
    groups = {
        'all': rows,
        'hold_max_displacement_lt_0.5m': [r for r in rows if r['max_expert_displacement'] < .5],
        'other_motion': [r for r in rows if r['max_expert_displacement'] >= .5],
        'slow_start_v0_le_3_kmh_moves': [r for r in rows if r['speed_kmh'] <= 3 and r['max_expert_displacement'] >= .5],
        'deceleration_ax3_lt_minus0.5': [r for r in rows if r['ax_3s'] < -.5],
        'lateral_abs_y3_ge_1.2m': [r for r in rows if abs(r['y_3s']) >= 1.2],
        'zero_clip': [r for r in rows if not r['clip']],
        'clip_0_to_20pct_exclusive': [r for r in rows if 0 < round(100*r['clip']) < 20],
        'clip_at_least_20pct': [r for r in rows if round(100*r['clip']) >= 20],
    }
    route_stats = {route:summarize([r for r in rows if r['route']==route]) for route in sorted({r['route'] for r in rows})}
    result = {
        'groups_overlap': True, 'groups': {k:summarize(v) for k,v in groups.items()},
        'routes': route_stats,
        'equal_route_mean_excess_ade': mean(r['excess_ade'] for r in route_stats.values()),
        'clip_excess_ade_pearson': float(np.corrcoef([r['clip'] for r in rows],[r['excess_ade'] for r in rows])[0,1]),
        'replay_max_discrepancy_m': max(max(abs(r['model_ade']-r['logged_ade']),abs(r['model_fde5']-r['logged_fde'])) for r in rows)
    }
    (folder/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for name, stats in result['groups'].items():
        print(name,stats['windows'],*(round(stats[k],3) for k in ('model_ade','cv_ade','model_dx5','cv_dx5','clip')),
              [round(v,3) for v in stats['model_fde_1_3_5']])
    print('route-balanced gap:',result['equal_route_mean_excess_ade'])


if __name__=='__main__':
    main()
