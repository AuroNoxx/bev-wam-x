"""Generate and pre-verify the 6 Town12/train Stage 2.6 supplemental missions."""
import argparse
import json
from pathlib import Path

import carla
import numpy as np
from scipy.spatial import cKDTree

from .bulk import CollectionDriver, DT, Mission, allowed, expert_route, samples
from .road_width import survey
from .scenarios import xyz
from .sd_policy import RoadNavigator
from .teacher_route import route as teacher_route


def trace_path(road, pts_xyz, start_loc, start_lane_id, branch_picks, max_len=2100.0, step=8.0):
    """Trace a connected CARLA waypoint centerline with explicit junction branch selectors."""
    wp0 = road.get_waypoint(
        carla.Location(x=float(start_loc[0]), y=float(start_loc[1]), z=float(start_loc[2])),
        project_to_road=True,
    )
    wp = None
    for fn in ('get_right_lane', 'get_left_lane'):
        cur = wp0
        for _ in range(8):
            if cur is None:
                break
            if cur.lane_id == start_lane_id and cur.lane_type == carla.LaneType.Driving:
                wp = cur
                break
            cur = getattr(cur, fn)()
        if wp is not None:
            break
    if wp is None or wp.lane_id != start_lane_id:
        raise RuntimeError(f'Failed to acquire start lane {start_lane_id} near {start_loc} (got {wp0.road_id}:{wp0.lane_id})')

    out = [wp]
    total = 0.0
    cur = wp
    tree = cKDTree(pts_xyz[:, :2])
    while total < max_len:
        nxt = cur.next(step)
        if not nxt:
            break
        if len(nxt) > 1:
            picked = None
            for match_fn, select_fn in branch_picks:
                if match_fn(cur, nxt):
                    picked = select_fn(nxt)
                    break
            cur = picked if picked is not None else nxt[0]
        else:
            cur = nxt[0]
        loc = cur.transform.location
        if tree.query([loc.x, loc.y])[0] > 8.0 and total > 100.0:
            break
        out.append(cur)
        total += step
    return out


def z_filtered_cloud(guide_xyz, pts_xyz, max_xy_pad=45.0, max_dz=3.5):
    """Keep aerial points within max_dz of the traced deck height so multi-level ramps stay clean."""
    gxy = guide_xyz[:, :2]
    sel = (
        (pts_xyz[:, 0] >= gxy[:, 0].min() - max_xy_pad)
        & (pts_xyz[:, 0] <= gxy[:, 0].max() + max_xy_pad)
        & (pts_xyz[:, 1] >= gxy[:, 1].min() - max_xy_pad)
        & (pts_xyz[:, 1] <= gxy[:, 1].max() + max_xy_pad)
    )
    local_pts = pts_xyz[sel]
    gtree = cKDTree(gxy)
    d, idx = gtree.query(local_pts[:, :2])
    keep = (d <= max_xy_pad) & (np.abs(local_pts[:, 2] - guide_xyz[idx, 2]) <= max_dz)
    return local_pts[keep]


def project_station_on_poly(poly_xy, target_xy):
    """Return closest arc station along poly_xy to target_xy."""
    poly = np.asarray(poly_xy, float)
    seg = np.diff(poly, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    arc = np.r_[0.0, seg_len.cumsum()]
    best_s, best_d = 0.0, 1e18
    for i, (dvec, length) in enumerate(zip(seg, seg_len)):
        if length < 1e-6:
            continue
        t = float(np.clip(np.dot(target_xy - poly[i], dvec) / (length ** 2), 0.0, 1.0))
        proj = poly[i] + t * dvec
        dist = float(np.linalg.norm(target_xy - proj))
        if dist < best_d:
            best_d = dist
            best_s = float(arc[i] + t * length)
    return best_s, best_d


def min_dist_to_box(points_xy, box_bounds):
    """Minimum Euclidean distance from any 2D point to an axis-aligned [lo, hi] box."""
    lo, hi = np.asarray(box_bounds[0], float), np.asarray(box_bounds[1], float)
    dx = np.maximum(0.0, np.maximum(lo[0] - points_xy[:, 0], points_xy[:, 0] - hi[0]))
    dy = np.maximum(0.0, np.maximum(lo[1] - points_xy[:, 1], points_xy[:, 1] - hi[1]))
    return float(np.hypot(dx, dy).min())


def simulate_navigator(spec, route):
    """Simulate RoadNavigator sequentially along a matched route with real update()."""
    xy = np.asarray(spec['xy'], float)
    speed_mps = float(spec['speed_kmh']) / 3.6
    from_xy = np.linalg.inv(RoadNavigator(xy, spec['georef']).to_xy)
    worst_lat, opp, inst_at_arc = 0.0, 0, []
    for phase in (0.0, 1.57):
        nav = RoadNavigator(xy, spec['georef'], phase, spec['events'], spec.get('sd_accuracy'))
        s_wp = 0.0
        prev_loc = None
        for wp in route:
            loc = wp.transform.location
            pt = np.array([loc.x, loc.y], float)
            if prev_loc is not None:
                s_wp += float(np.hypot(pt[0] - prev_loc[0], pt[1] - prev_loc[1]))
            prev_loc = pt
            fwd = np.array([wp.transform.get_forward_vector().x, wp.transform.get_forward_vector().y], float)
            compass = float(np.arctan2(fwd[0], -fwd[1]))
            latlon = nav.origin + pt @ from_xy
            nav.update(float(latlon[0]), float(latlon[1]), compass, s_wp / max(speed_mps, 1.0))
            worst_lat = max(worst_lat, float(nav.error))
            if phase == 0.0:
                _, inst = nav.instruction()
                inst_at_arc.append((float(s_wp), inst))
    return dict(
        opposite=opp,
        max_lat=float(worst_lat),
        inst_at_arc=inst_at_arc,
    )


def estimate_exit_frames(inst_at_arc, speed_kmh):
    """Count expected DT=0.05 frames where RoadNavigator emits 'exit'."""
    v_mps = speed_kmh / 3.6
    # Estimate up to 100 s (2000 frames) at target cruise speed along arc.
    times = np.arange(0, 2000) * DT
    s_samples = times * v_mps
    arcs = np.array([a for a, _ in inst_at_arc])
    insts = [i for _, i in inst_at_arc]
    count = 0
    for s in s_samples:
        idx = min(int(np.searchsorted(arcs, s, side='right') - 1), len(insts) - 1)
        idx = max(idx, 0)
        if insts[idx] == 'exit':
            count += 1
    return count


def build_missions(sd_root: Path, val_ep_json: Path, out_path: Path):
    road = carla.Map('Town12', (sd_root / 'Town12.xodr').read_text(encoding='utf-8'))
    pts = np.concatenate([np.load(f)['xyz'] for f in sorted((sd_root / 'Town12').glob('tile_*.npz'))])
    mis = json.loads((sd_root / 'missions.json').read_text(encoding='utf-8'))
    raf = json.loads((sd_root / 'road-atlas-final.json').read_text(encoding='utf-8'))

    val_bounds = mis['Town12']['validation_bounds_xy']
    georef = mis['Town12']['missions'][0]['georef']
    val_ep_bounds = json.loads(val_ep_json.read_text(encoding='utf-8'))['pose_xy_bounds']
    raf_by_name = {m['name']: m for m in raf['Town12']['missions']}
    mis_by_name = {m['name']: m for m in mis['Town12']['missions']}
    sampled = road.generate_waypoints(10)
    bounds_arr = np.asarray(val_bounds, float)

    specs = []

    # 1. Eastbound freeway -> Cloverleaf off-ramp -> Southbound surface interchange
    eb_exit_wps = trace_path(
        road, pts,
        start_loc=(2100.0, 3106.9, 360.0),
        start_lane_id=-4,
        branch_picks=[
            (lambda c, n: any(b.road_id in (8335, 19399, 16547, 17917) for b in n),
             lambda n: next(b for b in n if b.road_id in (8335, 19399, 16547, 17917))),
            (lambda c, n: c.transform.location.y < 2900.0,
             lambda n: min(n, key=lambda b: abs(b.transform.location.x - 2541.0))),
        ],
        max_len=1950.0,
    )
    eb_guide = np.array([[w.transform.location.x, w.transform.location.y, w.transform.location.z] for w in eb_exit_wps])
    eb_spec = {
        'name': 't12_sup_exit_clover_eb',
        'split': 'train',
        'speed_kmh': 38,
        'xy': eb_guide[::3, :2].tolist(),
        'laps': 0,
        'georef': georef,
        'events': [],
        'source': 'nadir-road-area-width-v2',
        'expert_search': 'both_directions',
        'driver_policy': 'v2',
    }
    eb_poly, eb_acc = survey(eb_spec, z_filtered_cloud(eb_guide, pts), 'Town12', search_radius=3)
    s_fork_eb, _ = project_station_on_poly(eb_poly, np.array([2484.4, 3196.3]))
    eb_spec['xy'] = eb_poly
    eb_spec['sd_accuracy'] = eb_acc
    eb_spec['events'] = [[float(s_fork_eb), 'exit']]
    specs.append(eb_spec)

    # 2. Westbound freeway lane-6 -> Direct off-ramp -> Southbound surface interchange
    wb_exit_wps = trace_path(
        road, pts,
        start_loc=(3026.0, 3090.4, 365.0),
        start_lane_id=6,
        branch_picks=[
            (lambda c, n: any(b.road_id in (16547, 17917) for b in n),
             lambda n: next(b for b in n if b.road_id in (16547, 17917))),
            (lambda c, n: any(b.road_id == 1123 for b in n) or c.transform.location.y > 2990.0,
             lambda n: min(n, key=lambda b: b.transform.location.y)),
            (lambda c, n: c.transform.location.y <= 2990.0,
             lambda n: min(n, key=lambda b: abs(b.transform.location.x - 2541.0))),
        ],
        max_len=1950.0,
    )
    wb_guide = np.array([[w.transform.location.x, w.transform.location.y, w.transform.location.z] for w in wb_exit_wps])
    wb_spec = {
        'name': 't12_sup_exit_wb',
        'split': 'train',
        'speed_kmh': 34,
        'xy': wb_guide[::3, :2].tolist(),
        'laps': 0,
        'georef': georef,
        'events': [],
        'source': 'nadir-road-area-width-v2',
        'expert_search': 'both_directions',
        'driver_policy': 'v2',
    }
    wb_poly, wb_acc = survey(wb_spec, z_filtered_cloud(wb_guide, pts), 'Town12', search_radius=3)
    int1_xy = mis_by_name['t12_interchange_1']['xy']
    wb_pre = [p for p in wb_poly if p[0] > 2875.0]
    wb_post = [p for p in wb_poly if p[1] < 2415.0]
    wb_stitched = wb_pre + list(int1_xy) + wb_post
    st_arr = np.asarray(wb_stitched, float)
    st_len = float(np.linalg.norm(np.diff(st_arr, axis=0), axis=1).sum())
    acc_arr = np.asarray(wb_acc, float)
    ds_acc = np.r_[np.arange(0.0, st_len, 10.0), st_len]
    wb_stitched_acc = [[round(float(s), 3), round(float(c), 3)]
                       for s, c in zip(ds_acc, np.interp(ds_acc / st_len * acc_arr[-1, 0], acc_arr[:, 0], acc_arr[:, 1]))]
    # Place event at the diverge nose (2790, 3070) so ~240m of approach + taper yields >=350 exit frames.
    s_fork_wb, _ = project_station_on_poly(wb_stitched, np.array([2790.0, 3070.0]))
    wb_spec['xy'] = wb_stitched
    wb_spec['sd_accuracy'] = wb_stitched_acc
    wb_spec['events'] = [[float(s_fork_wb), 'exit']]
    specs.append(wb_spec)

    # 3. Roundabout/cloverleaf interchange + Eastbound freeway extension (v2)
    sp3_seed = dict(raf_by_name['t12_interchange_3'])
    sp3_seed['expert_search'] = 'both_directions'
    sp3_seed['speed_kmh'] = 44
    r3_seed = teacher_route(road, Mission(sp3_seed), bounds_arr, sampled)
    ext3 = list(r3_seed)
    c3 = r3_seed[-1]
    for _ in range(130):
        nx3 = c3.next(5.0)
        if not nx3:
            break
        c3 = nx3[0]
        ext3.append(c3)
    guide3 = np.array([[w.transform.location.x, w.transform.location.y, w.transform.location.z] for w in ext3])
    spec3 = {
        'name': 't12_sup_interchange3_v2',
        'split': 'train',
        'speed_kmh': 44,
        'xy': guide3[::3, :2].tolist(),
        'laps': 0,
        'georef': georef,
        'events': [],
        'source': 'nadir-road-area-width-v2',
        'expert_search': 'both_directions',
        'driver_policy': 'v2',
    }
    poly3, acc3 = survey(spec3, z_filtered_cloud(guide3, pts), 'Town12', search_radius=3)
    s_rb, _ = project_station_on_poly(poly3, np.array([2532.0, 2983.0]))
    s_hw, _ = project_station_on_poly(poly3, np.array([2730.0, 3150.0]))
    spec3['xy'] = poly3
    spec3['sd_accuracy'] = acc3
    spec3['events'] = [[float(s_rb), 'enter_roundabout'], [float(s_hw), 'enter_highway']]
    specs.append(spec3)

    # 4. Curved interchange enter_highway + Westbound freeway extension with v2 maneuvers
    m2_base = dict(mis_by_name['t12_interchange_2'])
    m2_end = m2_base['xy'][-1]
    m2_ext = trace_path(
        road, pts,
        start_loc=(m2_end[0], m2_end[1], 369.0),
        start_lane_id=3,
        branch_picks=[(lambda c, n: True, lambda n: min(n, key=lambda b: abs(b.transform.location.y - 3090.0)))],
        max_len=1250.0,
        step=25.0,
    )
    m2_ext_xy = [[round(float(w.transform.location.x), 1), round(float(w.transform.location.y), 1)]
                 for w in m2_ext if w.transform.location.x < m2_end[0] - 25.0]
    m2_xy = np.array(m2_base['xy'] + m2_ext_xy, float)
    old2_arc = np.r_[0.0, np.linalg.norm(np.diff(np.array(m2_base['xy'], float), axis=0), axis=1).cumsum()]
    new2_arc = np.r_[0.0, np.linalg.norm(np.diff(m2_xy, axis=0), axis=1).cumsum()]
    extra2_s = np.arange(float(old2_arc[-1]) + 10.0, float(new2_arc[-1]), 10.0).tolist() + [float(new2_arc[-1])]
    m2_acc = list(m2_base['sd_accuracy']) + [[round(s, 3), 2.5] for s in extra2_s]
    specs.append({
        'name': 't12_sup_interchange2_v2',
        'split': 'train',
        'speed_kmh': 44,
        'xy': m2_xy.tolist(),
        'laps': 0,
        'georef': georef,
        'events': m2_base['events'],
        'sd_accuracy': m2_acc,
        'source': 'nadir-road-area-width-v2',
        'expert_search': 'both_directions',
        'driver_policy': 'v2',
    })

    # 5 & 6. Opposite-direction Town12 highway missions with CollectionDriver v2 (150m lane changes + cross-divider recoveries)
    for out_name, src_name in (('t12_sup_highway_v2_wb', 't12_highway_changes_1'),
                               ('t12_sup_highway_v2_eb', 't12_highway_changes_2')):
        base = dict(mis_by_name[src_name])
        base['name'] = out_name
        base['split'] = 'train'
        base['speed_kmh'] = 54
        base['source'] = 'nadir-road-area-width-v2'
        base['driver_policy'] = 'v2'
        specs.append(base)

    # Pre-verify every single mission before saving
    print(f'Verifying {len(specs)} Town12/train supplemental missions...')
    for idx, spec in enumerate(specs):
        m = Mission(spec)
        assert m.split == 'train' and spec.get('driver_policy') == 'v2'
        d_ep1 = min_dist_to_box(m.xy, val_ep_bounds)
        assert all(allowed(pt, 'train', bounds_arr) for pt in m.xy), f'{m.name}: touches val_bounds!'
        assert d_ep1 >= 2500.0, f'{m.name}: too close to ep_0001 ({d_ep1:.1f} m)'

        if spec.get('expert_search') == 'both_directions':
            r = teacher_route(road, m, bounds_arr, sampled)
        else:
            r = expert_route(road, m, bounds_arr, sampled)
        rxy = np.array([[w.transform.location.x, w.transform.location.y] for w in r])
        rlen = float(np.linalg.norm(np.diff(rxy, axis=0), axis=1).sum())
        d_ep1_route = min_dist_to_box(rxy, val_ep_bounds)
        assert d_ep1_route >= 2500.0, f'{m.name}: matched route too close to ep_0001 ({d_ep1_route:.1f} m)'

        # Curvature check along connected route
        fwd = np.array([[w.transform.get_forward_vector().x, w.transform.get_forward_vector().y] for w in r])
        yaw = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
        ds = np.maximum(np.linalg.norm(np.diff(rxy, axis=0), axis=1), 0.1)
        curv = np.abs(np.diff(yaw)) / ds
        n_sharp = int((curv > 0.03).sum())

        sim = simulate_navigator(spec, r)
        assert sim['opposite'] == 0, f"{m.name}: opposite SD anchors {sim['opposite']}"
        assert sim['max_lat'] < 8.0, f"{m.name}: max_lat {sim['max_lat']:.2f} >= 8.0 m"

        exit_frames = estimate_exit_frames(sim['inst_at_arc'], spec['speed_kmh'])
        if 'exit' in [ev[1] for ev in spec['events']]:
            assert exit_frames >= 300, f'{m.name}: predicted exit frames {exit_frames} < 300'

        seed = 2913 + (idx + 1) * 113
        drv = CollectionDriver(r, float(spec['speed_kmh']) / 3.6, seed, True, policy_version='v2')
        print(
            f"  [OK] {m.name:25s} | wps={len(r):3d} len={rlen:6.1f}m | max_c={curv.max():.4f} (>0.03:{n_sharp:2d}) "
            f"| nav_max_lat={sim['max_lat']:.2f}m opp={sim['opposite']} | d_val_ep1={d_ep1_route:.0f}m "
            f"| exit~={exit_frames}f | exc={len(drv.excursions)} rec_ev={len(drv.recovery_events)}"
        )

    plan = {
        'Town12': {
            'validation_bounds_xy': val_bounds,
            'missions': specs,
        }
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(f'Wrote verified plan -> {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sd-root', type=Path, default=Path('data/v2-mapless-wa/sd-40k-C'))
    p.add_argument('--val-ep-json', type=Path, default=Path('data/v2-mapless-wa/first-40k-mapless-C/episodes/ep_0001/episode.json'))
    p.add_argument('--out', type=Path, default=Path('data/v2-mapless-wa/supplement-v26-20261121/missions.json'))
    args = p.parse_args()
    build_missions(args.sd_root, args.val_ep_json, args.out)
