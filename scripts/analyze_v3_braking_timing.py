"""Post-hoc sustained speed-drop timing; no control or training input."""
import json
from pathlib import Path
import numpy as np
from diagnose_v3_aggregation import ROOT, from_xy


def onset(speed, initial):
    below = speed < initial[..., None] - .5
    sustained = below[..., :-2] & below[..., 1:-1] & below[..., 2:]
    return np.where(sustained.any(-1), (sustained.argmax(-1) + 1) * .2, np.nan)


def summary(v):
    v = np.asarray(v); valid = v[np.isfinite(v)]
    return {'n': int(v.size), 'finite_n': int(valid.size), 'missing_fraction': float((~np.isfinite(v)).mean()),
            'mean': float(valid.mean()) if len(valid) else None,
            'p10_p50_p90': np.percentile(valid, [10, 50, 90]).tolist() if len(valid) else None}


def main():
    folder = ROOT / 'artifacts/diagnostics/v3-aggregation-feasibility'
    manifest = json.loads((ROOT / 'data/occ_v3/manifest.json').read_text()); states = []; targets = []
    for name in manifest['splits']['validation']:
        with np.load(ROOT / 'data/occ_v3' / name) as z: states.append(z['state']); targets.append(z['trajectory'])
    state = np.array(states); target = np.array(targets)
    v0 = state[:, 0] * 30; mask = target[:, -1, 4] * 30 - v0 < -2
    target_onset = onset(target[..., 4] * 30, v0)
    p = np.load(ROOT / 'artifacts/diagnostics/flow-v3-assessment/flow_v3.npz')['prediction']
    median_q = from_xy(np.median(p[..., :2] * [50, 10], axis=2))
    speeds = {'first_head': p[:, :, 0, :, 4] * 30,
              'median_tangent': median_q[..., 4] * 30,
              'median_speed_head': np.median(p[..., 4], axis=2) * 30}
    result = {'definition': 'First future time with speed below current speed minus0.5m/s for >=3 consecutive0.2s samples; a sustained speed-drop proxy, not brake-pedal onset',
              'windows': int(mask.sum()), 'expert_drop_time_s': summary(target_onset[mask]),
              'current_ax_mps2': summary(state[mask, 1] * 10),
              'already_ax_below_minus1_fraction': float((state[mask, 1] * 10 < -1).mean()), 'methods': {}}
    for name, speed in speeds.items():
        timing = onset(speed, v0[None])[:, mask]
        truth_valid = np.isfinite(target_onset[mask])
        lag = timing[:, truth_valid] - target_onset[None, mask][:, truth_valid]
        result['methods'][name] = {'drop_time_s': summary(timing), 'lag_vs_expert_s_detected_only': summary(lag),
                                   'paired_expert_event_windows': int(truth_valid.sum()),
                                   'missing_or_lag_over1s': float(((~np.isfinite(lag)) | (lag > 1)).mean())}
    leads = json.loads((folder / 'deceleration-leads.json').read_text())
    found = [v for v in leads if v['lead_corridor_found']]
    result['current_lead_center_distance_m'] = summary([v['lead_center_x_m'] for v in found])
    result['current_closing_speed_mps'] = summary([v['ego_speed_mps'] - v['lead_speed_mps'] for v in found])
    result['limitations'] = ['Only83 overlapping windows,5routes; posthoc truth-defined group',
                            'Speed-drop event is not actuator brake onset, may differ with coasting or speed oscillation',
                            'Missing event excluded from finite lag mean but reported explicitly',
                            'Cannot establish causality between current actor evidence and model behaviour']
    (folder / 'braking-timing.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result))


if __name__ == '__main__': main()
