"""Audit the v3 data/model scope limits that bound 0-130 km/h highway claims.

Read-only over data/occ_v3. Computes window speed distribution, 5 s travel
distance, forward occupancy coverage and trajectory validity. Writes
artifacts/diagnostics/v3-limits/v3-limits.json.
"""
from pathlib import Path
import json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/diagnostics/v3-limits'
MANIFEST = json.loads((ROOT / 'data/occ_v3/manifest.json').read_text(encoding='utf-8'))
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 1

def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else None

def bands(forward_frac):
    labels = ['behind', '0-25.6', '25.6-51.2', '51.2-76.8', '76.8-102.4']
    edges = [0, 32, 64, 96, 128, 160]
    return {lab: float(forward_frac[a:b].mean()) for lab, a, b in zip(labels, edges[:-1], edges[1:])}

route_meta = {r['route']: {'split': r['split'], 'family': r['family'],
                           'max_speed_kmh': r.get('max_speed_kmh'),
                           'windows': r.get('windows')} for r in MANIFEST['routes']}
rows = []
for split in ('train', 'validation', 'test'):
    for rel in MANIFEST['splits'][split]:
        parts = Path(rel).parts
        route_name = parts[-2]
        rows.append((split, route_meta.get(route_name, {}).get('family', 'unknown'), rel))

print('windows', len(rows), flush=True)
S = {'speed_kmh': [], 'endpoint_m': [], 'future_unknown_frac': [], 'history_unknown_frac': [],
     'future_forward_frac': [], 'history_forward_frac': [], 'speed_end_kmh': [],
     'speed_change_mps': [], 'traj_unit_err': [], 'family_speed': {}, 'split_counts': {},
     'future_xy_extent': [], 'history_xy_extent': []}
band_edges = np.array([-25.6, 0, 25.6, 51.2, 76.8, 102.4])
for i, (split, family, rel) in enumerate(rows):
    if i % STEP:
        continue
    with np.load(ROOT / 'data/occ_v3' / rel, allow_pickle=False) as a:
        state = a['state']; traj = a['trajectory'].astype(np.float64)
        history = a['history']; future = a['future']
    speed = float(state[0] * 30.0)
    xy = traj[:, :2] * np.array([50.0, 10.0])
    S['speed_kmh'].append(speed * 3.6)
    S['speed_end_kmh'].append(float(traj[-1, 4] * 30.0) * 3.6)
    S['speed_change_mps'].append(float(traj[-1, 4] * 30.0) - speed)
    S['endpoint_m'].append(float(np.linalg.norm(xy[-1])))
    S['traj_unit_err'].append(float(np.abs(np.hypot(traj[:, 2], traj[:, 3]) - 1).max()))
    S['future_unknown_frac'].append(float((future == 255).mean()))
    S['history_unknown_frac'].append(float((history == 255).mean()))
    S['future_forward_frac'].append(bands((future != 255).mean(axis=(0, 1, 2))))
    S['history_forward_frac'].append(bands((history != 255).mean(axis=(0, 1, 2))))
    for tag, grid in (('history', history), ('future', future)):
        S.setdefault(tag + '_classes', []).append([
            float((grid == 255).mean()), float((grid == 0).mean()),
            float(((grid != 0) & (grid != 255)).mean())])
    fk = (future != 255).mean(axis=(0, 2, 3))  # per future step, fraction known
    S['future_xy_extent'].append([float(v) for v in fk])
    S['family_speed'].setdefault(family, []).append(speed * 3.6)
    S['split_counts'][split] = S['split_counts'].get(split, 0) + 1
    if len(S['speed_kmh']) % 2000 == 0:
        print('processed', len(S['speed_kmh']), flush=True)

def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {'n': int(arr.size), 'mean': float(arr.mean()), 'p10': pct(arr, 10), 'median': pct(arr, 50),
            'p90': pct(arr, 90), 'p99': pct(arr, 99), 'max': float(arr.max()), 'min': float(arr.min()),
            'fraction_ge_kmh': {str(t): float((arr >= t).mean()) for t in (40, 60, 80, 100, 120, 130)}}

speed = np.asarray(S['speed_kmh']); endpoint = np.asarray(S['endpoint_m'])
bins = [0, 20, 40, 60, 80, 100, 120, 999]
report = {
    'manifest': 'data/occ_v3/manifest.json',
    'preprocess_version': MANIFEST.get('preprocess_version'),
    'windows_total': len(rows), 'windows_sampled': int(speed.size), 'sample_step': STEP,
    'split_counts': S['split_counts'],
    'window_speed_kmh': summarize(speed),
    'window_speed_histogram': {f'{a}-{b}': int(((speed >= a) & (speed < b)).sum()) for a, b in zip(bins[:-1], bins[1:])},
    'window_end_speed_kmh': summarize(S['speed_end_kmh']),
    'speed_change_over_5s_mps': summarize(S['speed_change_mps']),
    'endpoint_distance_5s_m': summarize(endpoint),
    'endpoint_distance_histogram': {f'{a}-{b}': int(((endpoint >= a) & (endpoint < b)).sum()) for a, b in zip([0, 25.6, 51.2, 76.8, 102.4, 150, 200], [25.6, 51.2, 76.8, 102.4, 150, 200, 9999])},
    'future_unknown_fraction': summarize(S['future_unknown_frac']),
    'history_unknown_fraction': summarize(S['history_unknown_frac']),
    'future_known_fraction_by_forward_band': {k: float(np.mean([e[k] for e in S['future_forward_frac']])) for k in S['future_forward_frac'][0]},
    'history_known_fraction_by_forward_band': {k: float(np.mean([e[k] for e in S['history_forward_frac']])) for k in S['history_forward_frac'][0]},
    'future_known_fraction_per_step_mean': [float(v) for v in np.mean(np.asarray(S['future_xy_extent']), axis=0)],
    'class_composition': {'order': ['out_of_range_255', 'unfilled_0_not_verified_free', 'evidenced_semantic'],
                          'history': [float(v) for v in np.mean(np.asarray(S['history_classes']), axis=0)],
                          'future': [float(v) for v in np.mean(np.asarray(S['future_classes']), axis=0)]},
    'trajectory_unit_heading_max_abs_error': summarize(S['traj_unit_err']),
    'family_window_speed_kmh': {k: {'n': len(v), 'mean': float(np.mean(v)), 'max': float(np.max(v)), 'p90': float(np.percentile(v, 90))} for k, v in sorted(S['family_speed'].items())},
    'route_max_speed_kmh': {k: v['max_speed_kmh'] for k, v in route_meta.items()},
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'v3-limits.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k != 'route_max_speed_kmh'}, indent=2))
