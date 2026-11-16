"""Frozen v3 geometry/behaviour diagnostics; validation only, no training.

Online rules never use future truth. Truth is used only for reporting buckets
and clearly labelled oracle diagnostics. Actor responses are logged, not causal.
"""
import csv
import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from audit_logged_collision import overlap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/diagnostics/v3-aggregation-feasibility'
DT = .2
SCALE = np.array([50., 10.])


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def from_xy(xy):
    """Tangent heading and central-difference point speed, no smoothing/clipping.

    Append the observed origin, use centred differences for future interior
    points and a backward difference at the endpoint. Below .3m/s hold the
    previous heading (initial relative heading 0). This is a convention, not
    proof of physically executable continuous motion.
    """
    full = np.concatenate([np.zeros_like(xy[..., :1, :]), xy], axis=-2)
    velocity = np.gradient(full, DT, axis=-2)[..., 1:, :]
    speed = np.linalg.norm(velocity, axis=-1)
    yaw = np.arctan2(velocity[..., 1], velocity[..., 0])
    previous = np.zeros_like(yaw[..., 0])
    for t in range(xy.shape[-2]):
        yaw[..., t] = np.where(speed[..., t] >= .3, yaw[..., t], previous)
        previous = yaw[..., t]
    return np.concatenate([xy / SCALE, np.sin(yaw)[..., None],
                           np.cos(yaw)[..., None], speed[..., None] / 30.], -1)


def overlap_batch(centers, angles, extent, other_centers, other_angles, other_extents):
    """Vectorised SAT; returns one flag per ego candidate."""
    if not len(other_centers):
        return np.zeros(len(centers), bool)
    u = np.stack([np.cos(angles), np.sin(angles)], -1)
    v = np.stack([-u[:, 1], u[:, 0]], -1)
    ou = np.stack([np.cos(other_angles), np.sin(other_angles)], -1)
    ov = np.stack([-ou[:, 1], ou[:, 0]], -1)
    delta = other_centers[None] - centers[:, None]
    result = np.ones(delta.shape[:2], bool)
    for axis in [u[:, None], v[:, None], ou[None], ov[None]]:
        proj = np.abs((delta * axis).sum(-1))
        radius = (extent[0] * np.abs((u[:, None] * axis).sum(-1)) +
                  extent[1] * np.abs((v[:, None] * axis).sum(-1)) +
                  other_extents[None, :, 0] * np.abs((ou[None] * axis).sum(-1)) +
                  other_extents[None, :, 1] * np.abs((ov[None] * axis).sum(-1)))
        result &= proj <= radius
    return result.any(-1)


def self_test():
    rng = np.random.default_rng(99)
    c = rng.normal(size=(25, 2)) * 5; a = rng.normal(size=25)
    oc = rng.normal(size=(11, 2)) * 5; oa = rng.normal(size=11)
    oe = rng.uniform(.5, 2, (11, 2)); e = np.array([2.3, 1.])
    expected = [any(overlap(x, y, e, xx, yy, ee) for xx, yy, ee in zip(oc, oa, oe))
                for x, y in zip(c, a)]
    assert np.array_equal(expected, overlap_batch(c, a, e, oc, oa, oe))
    xy = np.stack([np.arange(1, 26) * DT * 7, np.zeros(25)], -1)
    q = from_xy(xy)
    assert np.allclose(q[:, 4] * 30, 7) and np.allclose(q[:, 3], 1)
    assert np.allclose(from_xy(np.zeros((25, 2)))[:, 3], 1)


def geometry(q, state, target):
    xy = q[..., :2] * SCALE
    err = np.linalg.norm(xy - target[None, :, :, :2] * SCALE, axis=-1)
    pos = np.concatenate([np.zeros_like(xy[..., :1, :]), xy], -2)
    vel = np.diff(pos, axis=-2) / DT
    initial = np.zeros_like(vel[..., :1, :]); initial[..., 0, 0] = state[None, :, 0] * 30
    acc = np.diff(np.concatenate([initial, vel], -2), axis=-2) / DT
    jerk = np.diff(acc, axis=-2) / DT
    yaw = np.arctan2(q[..., 2], q[..., 3])
    heading_velocity = np.stack([np.cos(yaw), np.sin(yaw)], -1) * q[..., 4:5] * 30
    prev_v = np.concatenate([initial, heading_velocity[..., :-1, :]], -2)
    residual = np.linalg.norm(vel - (prev_v + heading_velocity) / 2, axis=-1)
    delta_yaw = np.angle(np.exp(1j * np.diff(np.concatenate([np.zeros_like(yaw[..., :1]), yaw], -1), axis=-1)))
    norm_a = np.linalg.norm(acc, axis=-1); norm_j = np.linalg.norm(jerk, axis=-1)
    return {
        'ADE_m': err.mean(-1), 'FDE_m': err[..., -1],
        'final_x_bias_m': xy[..., -1, 0] - target[None, :, -1, 0] * 50,
        'final_speed_bias_mps': (q[..., -1, 4] - target[None, :, -1, 4]) * 30,
        'mean_accel_mps2': norm_a.mean(-1), 'mean_jerk_mps3': norm_j.mean(-1),
        'max_accel_mps2': norm_a.max(-1), 'max_jerk_mps3': norm_j.max(-1),
        'kinematic_residual_mps': residual.mean(-1),
        'initial_segment_speed_error_mps': np.linalg.norm(vel[..., 0, :] - initial[..., 0, :], axis=-1),
        'any_backward_segment': (vel[..., 0] < -.5).any(-1),
        'any_accel_over6': (norm_a > 6).any(-1),
        'any_jerk_over10': (norm_j > 10).any(-1),
        'any_yaw_rate_over1radps': (np.abs(delta_yaw / DT) > 1).any(-1),
    }


def annotations_and_collisions(names, methods, target, decel_mask):
    layout = {}; chunks = []; start = 0
    for key, value in methods.items():
        layout[key] = slice(start, start + len(value)); start += len(value); chunks.append(value)
    allq = np.concatenate(chunks)
    hits = np.zeros((len(allq), len(names), 25), bool)
    info = []; cached_route = None; box_count = 0
    for n, name in enumerate(names):
        route = Path(name).parts[-2]; frame = int(Path(name).stem)
        if route != cached_route:
            folder = ROOT / 'data/public/bench2drive/clips' / route / 'anno'
            ann = {int(p.name.split('.')[0]): json.load(gzip.open(p, 'rt')) for p in folder.glob('*.gz')}
            cached_route = route
        ego = next(b for b in ann[frame]['bounding_boxes'] if b['class'] == 'ego_vehicle')
        w = np.asarray(ego['world2ego']); yaw0 = np.deg2rad(ego['rotation'][2])
        offset = np.asarray(ego['bbx_loc'][:2]) * [1, -1]
        if decel_mask[n]:
            leads = []
            for b in ann[frame]['bounding_boxes']:
                if b['class'] != 'vehicle': continue
                center = (w @ np.r_[b['center'], 1])[:3] * [1, -1, 1]
                dyaw = np.angle(np.exp(1j * (yaw0 - np.deg2rad(b['rotation'][2]))))
                if (0 < center[0] < 60 and abs(center[1]) < ego['extent'][1] + b['extent'][1]
                    and abs(dyaw) < np.deg2rad(20) and abs(center[2] - ego['bbx_loc'][2]) < 3):
                    leads.append((center[0], b, center))
            row = {'window': str(name), 'route': route, 'frame': frame, 'lead_corridor_found': bool(leads)}
            if leads:
                _, b, center = min(leads, key=lambda v: v[0])
                row.update(lead_actor_id=b['id'], lead_center_x_m=float(center[0]),
                           lead_center_y_m=float(center[1]), lead_speed_mps=float(b['speed']),
                           ego_speed_mps=float(ann[frame]['speed']),
                           lead_inside_original_xy=bool(abs(center[0]) <= 25.6 and abs(center[1]) <= 25.6))
                with np.load(ROOT / 'data/occ_v3' / name) as sample:
                    grid = sample['history'][-1]
                    lo = np.floor((center[:2] - np.asarray(b['extent'][:2]) + 25.6) / .8).astype(int)
                    hi = np.ceil((center[:2] + np.asarray(b['extent'][:2]) + 25.6) / .8).astype(int)
                    lo = np.maximum(lo, 0); hi = np.minimum(hi, [160, 64])
                    roi = grid[:, lo[1]:hi[1], lo[0]:hi[0]]
                    row['vehicle_class_voxels_in_axis_aligned_lead_region'] = int((roi == 4).sum())
                    row['unknown_fraction_in_lead_region'] = float((roi == 255).mean()) if roi.size else None
                series = []
                for ti in range(26):
                    aa = ann[frame + 2 * ti]
                    bb = next((bb for bb in aa['bounding_boxes'] if bb.get('id') == b['id'] and bb['class'] == 'vehicle'), None)
                    series.append({'t_s': ti * DT, 'ego_speed_mps': float(aa['speed']),
                                   'lead_speed_mps': float(bb['speed']) if bb else None})
                row['logged_speed_series'] = series
            info.append(row)
        for ti in range(25):
            boxes = []
            for b in ann[frame + 2 * (ti + 1)]['bounding_boxes']:
                if b['class'] not in ['vehicle', 'walker'] or 'extent' not in b or 'center' not in b: continue
                c = (w @ np.r_[b['center'], 1])[:3]; c[1] *= -1
                if abs(c[2] - ego['bbx_loc'][2]) > b['extent'][2] + ego['extent'][2] + .3: continue
                boxes.append((c[:2], yaw0 - np.deg2rad(b['rotation'][2]), np.asarray(b['extent'][:2])))
            box_count += len(boxes)
            if not boxes: continue
            q = allq[:, n, ti]; angle = np.arctan2(q[:, 2], q[:, 3])
            co, si = np.cos(angle), np.sin(angle)
            center = q[:, :2] * SCALE + np.stack([co * offset[0] - si * offset[1], si * offset[0] + co * offset[1]], -1)
            oc, oa, oe = [np.asarray(v) for v in zip(*boxes)]
            hits[:, n, ti] = overlap_batch(center, angle, ego['extent'], oc, oa, oe)
        if (n + 1) % 250 == 0: print('COLLISION_WINDOWS', n + 1, flush=True)
    return {k: hits[v] for k, v in layout.items()}, info, box_count


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction-directory', default='artifacts/diagnostics/flow-v3-assessment')
    parser.add_argument('--checkpoint', default='artifacts/lightning/flow-v3/best.ckpt')
    parser.add_argument('--out', default='artifacts/diagnostics/v3-aggregation-feasibility')
    args = parser.parse_args(); OUT = ROOT / args.out
    started = time.perf_counter(); self_test(); OUT.mkdir(parents=True, exist_ok=True)
    ck = ROOT / args.checkpoint; cksha = sha(ck)
    manifest_path = ROOT / 'data/occ_v3/manifest.json'; manifest = json.loads(manifest_path.read_text())
    names = np.array(manifest['splits']['validation']); states = []; truths = []
    for name in names:
        with np.load(ROOT / 'data/occ_v3' / name) as z:
            states.append(z['state']); truths.append(z['trajectory'])
    state = np.array(states); truth = np.array(truths)
    with np.load(ROOT / 'artifacts/diagnostics/20261115-contract/evaluation-labels.npz') as z:
        assert np.array_equal(names, z['names']) and np.array_equal(state, z['state']) and np.array_equal(truth, z['target'])
    assessment = json.loads((ROOT / args.prediction_directory / 'assessment.json').read_text())
    assert assessment['checkpoint_sha256'] == cksha and assessment['manifest_sha256'] == sha(manifest_path)
    p = np.load(ROOT / args.prediction_directory / 'flow_v3.npz')['prediction'].astype(np.float64)
    assert p.shape == (3, len(names), 8, 25, 5) and np.isfinite(p).all()
    xy = p[..., :2] * SCALE; median_xy = np.median(xy, axis=2)
    distance = np.linalg.norm(xy[:, :, :, None] - xy[:, :, None, :], axis=-1).mean(-1)
    medoid_index = distance.sum(-1).argmin(-1)
    medoid = np.take_along_axis(p, medoid_index[:, :, None, None, None], axis=2)[:, :, 0]
    median_heads = np.median(p, axis=2)
    median_heads[..., 2:4] /= np.maximum(np.linalg.norm(median_heads[..., 2:4], axis=-1, keepdims=True), 1e-8)
    methods = {'first': p[:, :, 0], 'median_tangent': from_xy(median_xy),
               'median_separate_heads': median_heads, 'mean_tangent': from_xy(xy.mean(2)),
               'medoid': medoid, 'first_tangent': from_xy(xy[:, :, 0]),
               'expert': truth[None], 'expert_tangent': from_xy(truth[None, ..., :2] * SCALE),
               'constant_velocity': np.load(ROOT / 'artifacts/diagnostics/flow-v3-assessment/constant_velocity.npz')['prediction'][:, :, 0],
               'ridge': np.load(ROOT / 'artifacts/diagnostics/flow-v3-assessment/ridge_state_nav.npz')['prediction'][:, :, 0],
               'candidate_pool': p.transpose(0, 2, 1, 3, 4).reshape(24, len(names), 25, 5)}
    speed = np.concatenate([state[:, :1] * 30, truth[..., 4] * 30], -1)
    truthxy = truth[..., :2] * SCALE; v0 = speed[:, 0]
    yaw = np.arctan2(truth[..., 2], truth[..., 3])
    legacy_cruise = (np.abs(speed[:, -1] - v0) < 1) & (np.abs(truthxy[:, -1, 1] - truthxy[:, 0, 1]) < 1)
    strict_cruise = ((np.abs(speed - v0[:, None]).max(-1) < 1) & (np.abs(truthxy[..., 1]).max(-1) < 1)
                     & (np.abs(yaw).max(-1) < .1) & (v0 >= 5 / 3.6))
    masks = {'all': np.ones(len(names), bool), 'legacy_cruise_proxy': legacy_cruise,
             'strict_moving_cruise': strict_cruise,
             'stationary': (np.abs(speed).max(-1) < 1 / 3.6) & (np.linalg.norm(truthxy, axis=-1).max(-1) < .5),
             'deceleration': speed[:, -1] - v0 < -2,
             'acceleration': speed[:, -1] - v0 > 2,
             'lateral_motion': np.abs(truthxy[..., 1]).max(-1) > 2}
    routes = np.array([Path(n).parts[-2] for n in names])
    flags, leads, boxes = annotations_and_collisions(names, methods, truth, masks['deceleration'])
    np.savez_compressed(OUT / 'collision-flags.npz', **flags)
    # Verify same proxy matches the previous assessment before interpreting new rules.
    if args.prediction_directory == 'artifacts/diagnostics/flow-v3-assessment':
        previous = np.load(ROOT / 'artifacts/diagnostics/flow-v3-assessment/overlap-flags.npz')
        assert np.array_equal(flags['first'].any(-1), previous['flow_v3'])
    assert not flags['expert'].any()
    stats = {k: geometry(v, state, truth) for k, v in methods.items()}
    for k in methods: stats[k]['bbox_overlap'] = flags[k].any(-1)
    cvade = stats['constant_velocity']['ADE_m'][0]
    poolade = stats['candidate_pool']['ADE_m'].reshape(3, 8, len(names)).transpose(0, 2, 1)
    poolhits = flags['candidate_pool'].any(-1).reshape(3, 8, len(names)).transpose(0, 2, 1)
    poolendv = p[..., -1, 4] * 30
    close_x = np.abs((p[..., -1, 0] - truth[None, :, None, -1, 0]) * 50) <= 2
    close_speed = np.abs(poolendv - speed[None, :, None, -1]) <= 1
    # Geometry gates are exploratory and intentionally reported also on expert.
    poolgeom = ((stats['candidate_pool']['max_accel_mps2'] <= 6) &
                (stats['candidate_pool']['max_jerk_mps3'] <= 10)).reshape(3, 8, len(names)).transpose(0, 2, 1)
    groups = {}
    for label, mask in masks.items():
        rr = np.unique(routes[mask]); group = {'windows': int(mask.sum()), 'routes': len(rr), 'route_names': rr.tolist(), 'methods': {}}
        if not mask.any(): groups[label] = group; continue
        for method, values in stats.items():
            row = {k: float(v[:, mask].mean()) for k, v in values.items()}
            row['ADE_route_macro_m'] = float(np.mean([values['ADE_m'][:, mask & (routes == route)].mean() for route in rr]))
            row['ADE_window_seed_p50_p90_p95_m'] = np.percentile(values['ADE_m'][:, mask], [50, 90, 95]).tolist()
            row['fraction_not_better_than_CV'] = float((values['ADE_m'][:, mask] >= cvade[None, mask]).mean())
            group['methods'][method] = row
        group['candidate_coverage'] = {
            'minADE8_m': float(poolade[:, mask].min(-1).mean()),
            'no_candidate_better_than_CV': float((poolade[:, mask].min(-1) >= cvade[None, mask]).mean()),
            'no_bbox_clear_candidate': float(poolhits[:, mask].all(-1).mean()),
            'any_candidate_matching_final_x_2m_and_speed_1mps': float((close_x[:, mask] & close_speed[:, mask]).any(-1).mean()),
            'any_candidate_matching_endpoint_and_bbox_clear': float((close_x[:, mask] & close_speed[:, mask] & ~poolhits[:, mask]).any(-1).mean()),
            'any_candidate_final_speed_drops_2mps': float((poolendv[:, mask] < v0[None, mask, None] - 2).any(-1).mean()),
            'any_candidate_with_diagnostic_accel_jerk_gates': float(poolgeom[:, mask].any(-1).mean()),
            'median_overlaps_when_all_original_candidates_clear': float((flags['median_tangent'].any(-1)[:, mask] & ~poolhits[:, mask].any(-1)).mean()),
        }
        groups[label] = group
    # Small, explicit failure examples; one example per route per behaviour.
    examples = []
    for label in ['strict_moving_cruise', 'stationary', 'deceleration']:
        for route in np.unique(routes[masks[label]]):
            ids = np.flatnonzero(masks[label] & (routes == route))
            n = ids[np.argmax(stats['median_tangent']['ADE_m'][:, ids].mean(0))]
            record = {'group': label, 'window': str(names[n]), 'v0_mps': float(v0[n]),
                      'expert_speed_mps': speed[n].tolist(), 'expert_xy_m': truthxy[n].tolist(),
                      'median_xy_m_seed0': median_xy[0, n].tolist(),
                      'candidate_final_x_m_seed0': xy[0, n, :, -1, 0].tolist(),
                      'candidate_final_speed_mps_seed0': poolendv[0, n].tolist(),
                      'candidate_ADE_m_seed0': poolade[0, n].tolist()}
            examples.append(record)
    report = {'checkpoint_sha256': cksha, 'manifest_sha256': sha(manifest_path),
              'test_opened': False, 'seeds': [1908, 2911, 3911], 'windows': len(names), 'routes': len(np.unique(routes)),
              'definitions': {'strict_moving_cruise': 'v0>=5km/h, every point |v-v0|<1m/s, |y|<1m, |yaw|<0.1rad',
                              'stationary': 'all speed magnitudes <1km/h and all displacement norms <0.5m',
                              'deceleration': 'final speed - current speed < -2m/s',
                              'median_tangent': 'Coordinatewise median XY, central-difference tangent heading/speed; stationary heading hold <0.3m/s',
                              'medoid': 'Original candidate minimising sum of pairwise trajectory ADE, no truth',
                              'dynamics': 'Unsmoothed position differences incl current velocity; >6m/s2 and >10m/s3 are diagnostic, not calibrated acceptance gates'},
              'groups': groups, 'actor_time_boxes': boxes,
              'lead_evidence': {'deceleration_windows': len(leads),
                                'corridor_lead': sum(v['lead_corridor_found'] for v in leads),
                                'lead_center_inside_original_xy': sum(v.get('lead_inside_original_xy', False) for v in leads),
                                'lead_region_contains_vehicle_class': sum(v.get('vehicle_class_voxels_in_axis_aligned_lead_region', 0) > 0 for v in leads)},
              'limitations': ['No closed-loop or counterfactual actor response; logged bbox proxy at 0.2s, not continuous swept collision',
                              'Static obstacles, legal lane boundaries and calibrated controller tracking NOT evaluated',
                              'Derived heading can differ from actual body heading; expert_tangent is a calibration comparison',
                              'Post-hoc behaviour groups and endpoint coverage use truth only for diagnosis, not online selection',
                              'Overlapping windows on 9 routes; 3 sampling seeds are not independent training runs',
                              'Lead annotation is privileged diagnostic data; nearby OCC class does not establish actor identity or complete observability',
                              'Reported acceleration/jerk include discretisation noise; no acceptance claim from chosen thresholds'],
              'seconds': time.perf_counter() - started}
    assert sha(ck) == cksha
    for name, obj in [('report.json', report), ('deceleration-leads.json', leads), ('examples.json', examples)]:
        (OUT / name).write_text(json.dumps(obj, indent=2), encoding='utf-8')
    np.savez_compressed(OUT / 'online-rule-trajectories.npz', names=names, median_tangent=methods['median_tangent'], medoid=medoid)
    with (OUT / 'window-metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['window', 'seed', 'method', 'ADE_m', 'FDE_m', 'bbox_overlap', 'max_accel_mps2', 'max_jerk_mps3'])
        for method in ['first', 'median_tangent', 'medoid', 'expert']:
            for s in range(len(methods[method])):
                for n in range(len(names)):
                    writer.writerow([names[n], s, method] + [stats[method][k][s, n] for k in ['ADE_m', 'FDE_m', 'bbox_overlap', 'max_accel_mps2', 'max_jerk_mps3']])
    print(json.dumps({'seconds': report['seconds'], 'lead_evidence': report['lead_evidence'],
                      'groups': {k: {'windows': v['windows'], 'routes': v['routes'],
                                    'median': v.get('methods', {}).get('median_tangent'),
                                    'coverage': v.get('candidate_coverage')} for k, v in groups.items()}}), flush=True)


if __name__ == '__main__':
    main()
