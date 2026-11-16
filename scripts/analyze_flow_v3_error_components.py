"""Break the frozen v3 validation error down by manoeuvre proxy.

Uses the stored predictions from artifacts/diagnostics/flow-v3-assessment and the
v3 labels; no new inference, no test data. Answers "where does the 3.6 m come from"
and how a constant-velocity baseline compares in each bucket.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'data/occ_v3/manifest.json').read_text(encoding='utf-8'))
PRED = ROOT / 'artifacts/diagnostics/flow-v3-assessment/flow_v3.npz'
BASELINE = ROOT / 'artifacts/diagnostics/flow-v3-assessment/constant_velocity.npz'
OUT = ROOT / 'artifacts/diagnostics/v3-limits/error-by-maneuver.json'

names = np.asarray(MANIFEST['splits']['validation'])
state, target = [], []
for name in names:
    with np.load(ROOT / 'data/occ_v3' / name, allow_pickle=False) as a:
        state.append(a['state']); target.append(a['trajectory'])
state = np.stack(state); target = np.stack(target)
pred = np.load(PRED)['prediction']            # seeds, windows, candidates, 25, 5
cv = np.load(BASELINE)['prediction'][0, :, 0]  # windows, 25, 5

scale = np.array([50.0, 10.0])
truth_xy = target[..., :2] * scale
speed0 = state[:, 0] * 30.0
speed_end = target[:, -1, 4] * 30.0
dv = speed_end - speed0
dy = float(0)
lateral = np.abs(truth_xy[:, -1, 1] - truth_xy[:, 0, 1])
path_len = np.linalg.norm(np.diff(truth_xy, axis=1), axis=-1).sum(1)

def ade_of(p, sel=None):
    xy = p[..., :2] * scale
    err = np.linalg.norm(xy - truth_xy[:, None], axis=-1)   # windows, candidates, 25
    if sel is not None:
        err = err[sel]
    return err

err = ade_of(pred[:, :, :, :])            # seeds, windows, candidates, 25
err_cv = ade_of(cv[:, None])              # windows, 1, 25

buckets = {
    'all': np.ones(len(state), bool),
    'cruise_steady': (np.abs(dv) < 1.0) & (lateral < 1.0),
    'decel_gt2': dv < -2.0,
    'accel_gt2': dv > 2.0,
    'lane_change_lateral_gt2m': lateral > 2.0,
    'stop_in_window_speed_lt5': (speed0 < 5 / 3.6) | (speed_end < 5 / 3.6),
    'long_travel_gt45m': path_len > 45.0,
}
report = {'windows': int(len(state)), 'seeds': int(pred.shape[0]), 'candidates': int(pred.shape[2]),
          'derived_from': ['artifacts/diagnostics/flow-v3-assessment/flow_v3.npz',
                           'artifacts/diagnostics/flow-v3-assessment/constant_velocity.npz'],
          'buckets': {}}
for label, mask in buckets.items():
    if not mask.any():
        continue
    e = err[:, mask]                        # seeds, n, candidates, 25
    ecv = err_cv[mask]
    # first candidate, mean over seeds -> the deployable single-output number
    first = e[:, :, 0].mean(0)
    # Oracle *candidate* selection: min over candidates of the whole-window ADE,
    # which one fixed candidate must be chosen before the future is known.
    best = e.mean(-1).min(2)
    signed_long = ((pred[:, mask, :, :, :2] - target[None, mask, None, :, :2]) * scale)[..., 0]
    signed_lat = ((pred[:, mask, :, :, :2] - target[None, mask, None, :, :2]) * scale)[..., 1]
    report['buckets'][label] = {
        'windows': int(mask.sum()), 'share': float(mask.mean()),
        'first_ADE_m': float(first.mean(-1).mean()),
        'first_FDE_m': float(first[:, -1].mean()),
        'oracle_minADE8_m': float(best.mean()),
        'constant_velocity_ADE_m': float(ecv.mean()),
        'signed_longitudinal_bias_m_final': float(signed_long[..., -1].mean()),
        'abs_longitudinal_error_m': float(np.abs(signed_long).mean()),
        'abs_lateral_error_m': float(np.abs(signed_lat).mean()),
        'end_speed_error_mps': float(((pred[:, mask, :, -1, 4] * 30.0) - (target[None, mask, None, -1, 4] * 30.0)).mean()),
        'mean_speed_kmh': float(speed0[mask].mean() * 3.6),
        'mean_abs_dv_mps': float(np.abs(dv[mask]).mean()),
    }
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
