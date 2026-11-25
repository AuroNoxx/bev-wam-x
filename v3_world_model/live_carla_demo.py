"""Live zero-shot CARLA 1.0 + Car-OS 3D SR closed-loop infinite drive with random NPC traffic."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import queue
import faulthandler
import json
import math
import socket
import subprocess
from time import monotonic, perf_counter, sleep

import carla
import cv2
import numpy as np
import torch

from .collect_v3 import lane_bundle, raised_spawn
from .eval_stage32 import integrate_ego_delta, load_stage32_module
from .geometry import DT, H, NAMES, SIZES, W as BEV_W, ego_delta, lidar_bev, local, pixels
from .labels import neighbor, same_direction, xyz
from .observations import command_vector
from .perception_model import sdpa_context
from .sim import Rig, bgr, matrix, synchronous
from .sr_postprocess import SRPostProcessor
from .sr_viewer import prediction_scene


DISPLAY_STEPS = 5


def _find_carla_exe(carla_root):
    shipping = carla_root / "CarlaUnreal/Binaries/Win64/CarlaUnreal-Win64-Shipping.exe"
    if shipping.is_file():
        return shipping
    top = carla_root / "CarlaUnreal.exe"
    if top.is_file():
        return top
    raise FileNotFoundError(f"Missing CARLA 1.0 binary under {carla_root}")


@contextmanager
def launch_carla_server(carla_root, out_dir, port, quality):
    """Start a CARLA 1.0 UE5 server on `port` and terminate it cleanly on exit."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            pass
    except OSError:
        pass
    else:
        raise RuntimeError(f"CARLA RPC port {port} is already occupied")

    exe = _find_carla_exe(carla_root)
    flags = [
        "-RenderOffScreen",
        "-ResX=320",
        "-ResY=180",
        "-ExecCmds=r.ScreenPercentage 100,r.DynamicRes.OperationMode 0,r.MotionBlurQuality 0",
        "-dx12",
        "-nosound",
        f"-carla-rpc-port={port}",
        f"-quality-level={quality}",
        "-stdout",
        "-FullStdOutLogOutput",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "carla_server.log").open("wb") as log_file:
        proc = subprocess.Popen(
            [str(exe), *flags],
            cwd=carla_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            deadline = monotonic() + 140.0
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"CARLA server exited early with code {proc.returncode}")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                        break
                except OSError:
                    if monotonic() >= deadline:
                        raise TimeoutError(f"Timed out waiting for CARLA RPC on port {port}")
                    sleep(0.4)
            client = carla.Client("127.0.0.1", port)
            client.set_timeout(120.0)
            srv = client.get_server_version()
            cli = client.get_client_version()
            if cli != srv or srv.split("-", 1)[0] != "1.0":
                raise RuntimeError(f"Expected matching CARLA 1.0 versions, got client={cli} server={srv}")
            yield client, srv
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5.0)


def _freeze_lights_green(world):
    for tl in world.get_actors().filter("traffic.traffic_light*"):
        tl.set_state(carla.TrafficLightState.Green)
        tl.freeze(True)


def _branch_depth(wp, steps=26):
    cur = wp
    for i in range(steps):
        nxt = cur.next(3.0)
        if not nxt:
            return i
        fwd = cur.transform.get_forward_vector()
        cur = max(nxt, key=lambda w: fwd.dot(w.transform.get_forward_vector()))
    return steps


def _branch_bundle(wp, lookahead=8):
    cur = wp
    for _ in range(lookahead):
        nxt = cur.next(3.0)
        if not nxt:
            break
        fwd = cur.transform.get_forward_vector()
        cur = max(nxt, key=lambda w: fwd.dot(w.transform.get_forward_vector()))
    return lane_bundle(cur), int(cur.is_junction)


class InfiniteNavigator:
    """Self-extending multi-destination loop cycling highway <-> ramps <-> signal-free urban roads."""

    def __init__(self, road, start_wp, rng):
        self.road = road
        self.rng = rng
        self.want_zone = "urban" if lane_bundle(start_wp) >= 3 else "highway"
        self.leg_id = 1
        self.leg_dist_m = 0.0
        self.leg_target_m = float(self.rng.uniform(650.0, 950.0))
        self.turn_bias = float(self.rng.choice((-1.0, 0.0, 1.0)))
        self.total_dist = 0.0
        self.wps = []
        self.pts = np.empty((0, 3), dtype=np.float64)
        self.arc = np.zeros(1, dtype=np.float64)
        self.cursor = 0
        self.reset_from(start_wp)

    def reset_from(self, wp):
        self.wps = [wp]
        self.cursor = 0
        self._extend(target_ahead_m=280.0)
        self._rebuild_arrays()

    def _choose_branch(self, tail):
        options = tail.next(3.0)
        if not options:
            return None
        if len(options) == 1:
            return options[0]
        scored = []
        cur_bundle = lane_bundle(tail)
        fwd = tail.transform.get_forward_vector()
        right = tail.transform.get_right_vector()
        for w in options:
            depth = _branch_depth(w, steps=26)
            b_bundle, _ = _branch_bundle(w, lookahead=9)
            w_fwd = w.transform.get_forward_vector()
            align = float(fwd.dot(w_fwd))
            lat_dot = float(right.dot(w_fwd))
            if self.want_zone == "urban":
                zone_score = 3.0 if b_bundle < cur_bundle else (1.5 if b_bundle <= 2 else -1.0)
                if cur_bundle >= 3 and align < 0.96:
                    zone_score += 1.2
            else:
                zone_score = 3.0 if (b_bundle > cur_bundle or b_bundle >= 3) else -0.5
            score = (
                (depth >= 24) * 100.0
                + depth * 0.25
                + zone_score
                + self.turn_bias * lat_dot * 0.9
                + float(self.rng.uniform(0.0, 1.0))
            )
            scored.append((score, w))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _extend(self, target_ahead_m=280.0):
        step_est = 3.0
        cur_ahead = max(0, len(self.wps) - 1 - self.cursor) * step_est
        while cur_ahead < target_ahead_m:
            nxt = self._choose_branch(self.wps[-1])
            if nxt is None:
                break
            self.wps.append(nxt)
            cur_ahead += step_est

    def _rebuild_arrays(self):
        self.pts = np.stack([xyz(w.transform.location) for w in self.wps], axis=0)
        if len(self.pts) > 1:
            d = np.linalg.norm(np.diff(self.pts[:, :2], axis=0), axis=1)
            self.arc = np.r_[0.0, np.cumsum(d)]
        else:
            self.arc = np.zeros(1, dtype=np.float64)

    def update(self, pose, ego_loc, *, _reseeded=False):
        if len(self.pts) < 4:
            wp = self.road.get_waypoint(ego_loc, lane_type=carla.LaneType.Driving)
            if wp is not None:
                self.reset_from(wp)

        lo = max(0, self.cursor - 3)
        hi = min(len(self.pts), self.cursor + 24)
        dists = np.linalg.norm(self.pts[lo:hi, :2] - pose[:2, 3], axis=1)
        idx = lo + int(np.argmin(dists))
        min_dist = float(dists[idx - lo])
        if min_dist > 14.0 and not _reseeded:
            wp = self.road.get_waypoint(ego_loc, lane_type=carla.LaneType.Driving)
            if wp is not None:
                self.reset_from(wp)
                return self.update(pose, ego_loc, _reseeded=True)

        ds = max(0.0, float(self.arc[idx] - self.arc[self.cursor]))
        self.cursor = idx
        self.total_dist += ds
        self.leg_dist_m += ds

        cur_wp = self.wps[self.cursor]
        cur_bundle = lane_bundle(cur_wp)
        in_highway = cur_bundle >= 3
        zone_name = "HIGHWAY" if in_highway else ("JUNCTION" if cur_wp.is_junction else "URBAN/RAMP")
        if self.leg_dist_m >= self.leg_target_m:
            finished_leg = self.leg_id
            self.leg_id += 1
            self.leg_dist_m = 0.0
            self.leg_target_m = float(self.rng.uniform(650.0, 950.0))
            self.want_zone = "urban" if in_highway else "highway"
            self.turn_bias = float(self.rng.choice((-1.0, 0.0, 1.0)))
            print(
                f"[Nav] Reached leg #{finished_leg}; new destination #{self.leg_id}: "
                f"target={self.leg_target_m:.0f}m zone={self.want_zone} turn_bias={self.turn_bias:+.0f}",
                flush=True,
            )

        # Trim history and keep rolling future waypoints well ahead.
        if self.cursor > 30:
            trim = self.cursor - 15
            self.wps = self.wps[trim:]
            self.cursor -= trim
            self._extend(target_ahead_m=300.0)
            self._rebuild_arrays()
        elif (self.arc[-1] - self.arc[self.cursor]) < 180.0:
            self._extend(target_ahead_m=300.0)
            self._rebuild_arrays()

        progress = float(self.arc[self.cursor])
        sample_arcs = progress + np.array([12.0, 24.0, 36.0], dtype=np.float64)
        anchors_world = np.column_stack([np.interp(sample_arcs, self.arc, self.pts[:, c]) for c in range(3)])
        anchors_flu = local(anchors_world, pose).astype(np.float32)

        turn_lat = float(anchors_flu[-1, 1])
        turn_hint = "直行" if abs(turn_lat) < 2.2 else ("前方左转/匝道" if turn_lat > 0 else "前方右转/匝道")
        rem_m = max(0, int(round(self.leg_target_m - self.leg_dist_m)))
        goal_txt = "切城市/匝道" if self.want_zone == "urban" else "切高速主线"
        nav_hint = f"目的地#{self.leg_id} (剩{rem_m}m) · {zone_name} · {turn_hint} · 目标:{goal_txt}"
        return anchors_flu, in_highway, cur_wp, nav_hint


def _pick_ego_spawn(road, carla_mod):
    spawns = road.get_spawn_points()
    best = None
    for idx, tf in enumerate(spawns):
        wp = road.get_waypoint(tf.location, lane_type=carla_mod.LaneType.Driving)
        if wp is None:
            continue
        depth = _branch_depth(wp, steps=35)
        bundle = lane_bundle(wp)
        has_left = same_direction(wp, neighbor(wp, -1))
        has_right = same_direction(wp, neighbor(wp, 1))
        middle = has_left and has_right
        score = (
            (not wp.is_junction) * 200.0
            + middle * 160.0
            + (has_left or has_right) * 40.0
            + (bundle >= 3) * 80.0
            + min(depth, 35)
            + bundle * 2.0
        )
        if best is None or score > best[0]:
            best = (score, tf, wp, idx)
    if best is None:
        raise RuntimeError("No valid driving spawn point found on map")
    return best[1], best[2]


def _npc_blueprints(world):
    library = world.get_blueprint_library()
    candidates = []
    for bp in library.filter("vehicle.*"):
        if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels").as_int()) != 4:
            continue
        if any(bad in bp.id for bad in ("firetruck", "ambulance", "carlacola", "fusorosa")):
            continue
        candidates.append(bp)
    return candidates or list(library.filter("vehicle.*"))


def _configure_npc(npc, tm, tm_port, rng):
    npc.set_autopilot(True, tm_port)
    tm.auto_lane_change(npc, True)
    tm.ignore_lights_percentage(npc, 100.0)
    tm.vehicle_percentage_speed_difference(npc, float(rng.uniform(-20.0, 35.0)))
    tm.distance_to_leading_vehicle(npc, float(rng.uniform(2.5, 5.5)))


def _spawn_random_npcs(carla_mod, world, tm, ego_loc, count, rng):
    candidates = _npc_blueprints(world)
    spawns = list(world.get_map().get_spawn_points())
    near_spawns, far_spawns = [], []
    for tf in spawns:
        dist = tf.location.distance(ego_loc)
        if 22.0 <= dist <= 280.0:
            near_spawns.append(tf)
        elif dist > 280.0:
            far_spawns.append(tf)
    rng.shuffle(near_spawns)
    rng.shuffle(far_spawns)

    npcs = []
    tm_port = tm.get_port()
    for tf in near_spawns + far_spawns:
        if len(npcs) >= count:
            break
        bp = candidates[int(rng.integers(len(candidates)))]
        npc = world.try_spawn_actor(bp, raised_spawn(carla_mod, tf, 0.35))
        if npc is None:
            continue
        _configure_npc(npc, tm, tm_port, rng)
        npcs.append(npc)
    return npcs


def _recycle_distant_npcs(carla_mod, world, tm, npcs, ego_loc, nav, rng, target_count):
    """Keep dense NPC traffic around ego on huge maps by recycling distant actors."""
    kept = []
    for npc in npcs:
        if not getattr(npc, "is_alive", False):
            continue
        if npc.get_location().distance(ego_loc) > 360.0:
            try:
                npc.destroy()
            except Exception:
                pass
        else:
            kept.append(npc)

    needed = target_count - len(kept)
    if needed <= 0:
        npcs[:] = kept
        return

    candidates = _npc_blueprints(world)
    tm_port = tm.get_port()
    fwd = nav.wps[nav.cursor].transform.get_forward_vector() if nav.wps else None
    ahead_spawns, ring_spawns = [], []
    for tf in world.get_map().get_spawn_points():
        dist = tf.location.distance(ego_loc)
        if not (55.0 <= dist <= 240.0):
            continue
        if fwd is not None:
            dx = tf.location.x - ego_loc.x
            dy = tf.location.y - ego_loc.y
            dot = dx * fwd.x + dy * fwd.y
            (ahead_spawns if dot > 0.0 else ring_spawns).append(tf)
        else:
            ring_spawns.append(tf)
    rng.shuffle(ahead_spawns)
    rng.shuffle(ring_spawns)

    occupied = [n.get_location() for n in kept] + [ego_loc]
    for tf in ahead_spawns + ring_spawns:
        if len(kept) >= target_count:
            break
        if any(tf.location.distance(loc) < 12.0 for loc in occupied):
            continue
        bp = candidates[int(rng.integers(len(candidates)))]
        npc = world.try_spawn_actor(bp, raised_spawn(carla_mod, tf, 0.35))
        if npc is None:
            continue
        _configure_npc(npc, tm, tm_port, rng)
        kept.append(npc)
        occupied.append(tf.location)
    npcs[:] = kept


def _fillet_nav_curve(nav_anchors):
    """Build a monotone non-overshooting filleted corner from ego (0, 0) along (1, 0) to nav_anchors[0]."""
    pt0 = np.asarray(nav_anchors[0, :2], dtype=np.float64)
    pt1 = np.asarray(nav_anchors[1, :2], dtype=np.float64)
    pt2 = np.asarray(nav_anchors[2, :2], dtype=np.float64)
    chord = float(np.linalg.norm(pt0))
    if chord < 0.5 or pt0[0] <= 0.5:
        pts = np.vstack(([0.0, 0.0], pt0))
        arc = np.array([0.0, max(chord, 1e-3)], dtype=np.float64)
        return pts, arc, 0.0

    t0 = np.array([1.0, 0.0], dtype=np.float64)
    d1 = pt1 - pt0
    n1 = float(np.linalg.norm(d1))
    t1 = d1 / n1 if n1 > 0.5 else pt0 / chord
    turn_angle = math.acos(float(np.clip(np.dot(t0, t1), -1.0, 1.0)))

    # Recover corner apex along ego heading y=0 by intersecting with exit tangent through pt0.
    same_side = (pt0[1] * t1[1]) > 0.0
    if abs(t1[1]) > 0.15 and abs(pt0[1]) > 0.4 and same_side:
        x_corner = float(pt0[0] - pt0[1] * (t1[0] / t1[1]))
        x_corner = float(np.clip(x_corner, 0.5, max(0.5, pt0[0])))
        corner = np.array([x_corner, 0.0], dtype=np.float64)
        leg1 = x_corner
        leg2 = float(np.linalg.norm(pt0 - corner))
        fillet_r = min(3.5, 0.85 * leg1, 0.85 * leg2)
        if fillet_r > 0.4:
            p_in = np.array([x_corner - fillet_r, 0.0], dtype=np.float64)
            dir2 = (pt0 - corner) / max(leg2, 1e-6)
            p_out = corner + fillet_r * dir2
            u = np.linspace(0.0, 1.0, 13, dtype=np.float64)[:, None]
            arc_pts = ((1.0 - u) ** 2) * p_in + 2.0 * (1.0 - u) * u * corner + (u ** 2) * p_out
            pieces = []
            if p_in[0] > 0.2:
                pieces.append(np.array([[0.0, 0.0]], dtype=np.float64))
            pieces.append(arc_pts)
            if float(np.linalg.norm(pt0 - p_out)) > 0.2:
                pieces.append(pt0[None, :])
            pts = np.vstack(pieces)
        else:
            u = np.linspace(0.0, 1.0, 17, dtype=np.float64)[:, None]
            pts = 2.0 * (1.0 - u) * u * corner + (u ** 2) * pt0
        fillet_len = max(2.0 * fillet_r, 4.5)
    else:
        # Straight approach or gentle lateral recentering: monotone quadratic Bezier with zero overshoot.
        corner = np.array([0.5 * pt0[0], 0.0], dtype=np.float64)
        u = np.linspace(0.0, 1.0, 17, dtype=np.float64)[:, None]
        pts = 2.0 * (1.0 - u) * u * corner + (u ** 2) * pt0
        fillet_len = max(chord, 6.0)

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]

    # Estimate true corner curvature from both near (0->12m) and upcoming (12->24m) heading change.
    turn_near = turn_angle / max(fillet_len, 4.5)
    d2 = pt2 - pt1
    n2 = float(np.linalg.norm(d2))
    if n2 > 0.5:
        t2 = d2 / n2
        turn_mid = math.acos(float(np.clip(np.dot(t1, t2), -1.0, 1.0))) / max(min(n1, 8.0), 5.0)
    else:
        turn_mid = 0.0
    kappa = max(turn_near, 0.75 * turn_mid)
    return pts, arc, kappa


def _sample_polyline_at(pts_xy, ahead_m):
    """Interpolate a polyline starting from ego origin (0, 0) at arc distance ahead_m."""
    chain = np.vstack(([0.0, 0.0], np.asarray(pts_xy, dtype=np.float64)))
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(chain, axis=0), axis=1))]
    target_s = min(float(ahead_m), float(arc[-1]))
    tx = float(np.interp(target_s, arc, chain[:, 0]))
    ty = float(np.interp(target_s, arc, chain[:, 1]))
    return tx, ty, float(arc[-1])


def _nominal_action(trajectory_np, nav_anchors, speed_mps, zone_limit_mps, last_steer):
    """Derive nominal pure-pursuit steer and longitudinal throttle/brake from model traj + filleted SD nav."""
    traj_xy = trajectory_np[:, :2]
    fwd_ok = len(traj_xy) >= 12 and np.all(np.diff(traj_xy[:8, 0]) > 0.05)

    far_lat = float(abs(nav_anchors[0, 1]))
    mid_lat = float(abs(nav_anchors[1, 1]))
    end_lat = float(abs(nav_anchors[2, 1]))
    nav_pts, nav_arc, nav_kappa_geom = _fillet_nav_curve(nav_anchors)

    # Use shorter speed-adaptive lookahead in tight turns so pure pursuit stays inside the lane until apex.
    sharp_turn = (far_lat > 1.8) or (nav_kappa_geom > 0.06)
    if sharp_turn:
        ahead_m = float(np.clip(3.2 + 0.40 * speed_mps, 3.8, 9.5))
    else:
        ahead_m = float(np.clip(4.2 + 0.52 * speed_mps, 5.0, 14.0))

    use_nav = (not fwd_ok) or (far_lat > 1.8) or (mid_lat > 3.5) or (end_lat > 7.5)
    if use_nav:
        s_nav = min(ahead_m, float(nav_arc[-1]))
        nx = float(np.interp(s_nav, nav_arc, nav_pts[:, 0]))
        ny = float(np.interp(s_nav, nav_arc, nav_pts[:, 1]))
        if fwd_ok and np.linalg.norm(traj_xy[-1]) > 4.0:
            mx, my, m_len = _sample_polyline_at(traj_xy, ahead_m)
            if m_len >= 0.65 * ahead_m:
                # Near active turn: prioritize filleted nav; on straight approach ahead of far turn: favor model traj.
                nav_w = 0.78 if far_lat > 1.8 else 0.35
                tx = (1.0 - nav_w) * mx + nav_w * nx
                ty = (1.0 - nav_w) * my + nav_w * ny
            else:
                tx, ty = nx, ny
        else:
            tx, ty = nx, ny
        curve_y = float(max(abs(nav_anchors[1, 1] - 0.5 * nav_anchors[0, 1]),
                            abs(end_lat - 0.5 * abs(nav_anchors[1, 1]))))
    else:
        tx, ty, _ = _sample_polyline_at(traj_xy, ahead_m)
        curve_y = float(np.max(np.abs(traj_xy[:16, 1])))

    wheelbase, max_steer = 2.85, 0.72
    raw_steer = -math.atan2(2.0 * wheelbase * ty, max(1.0, tx * tx + ty * ty)) / max_steer
    steer = float(np.clip(last_steer + 0.62 * (raw_steer - last_steer), -0.85, 0.85))

    nav_kappa = min(0.24, max(nav_kappa_geom, curve_y / 180.0 + abs(steer) * 0.045))
    v_curve = math.sqrt(2.1 / max(nav_kappa, 1e-3))
    v_target = min(zone_limit_mps, v_curve)

    err = v_target - speed_mps
    if err >= 0.3:
        throttle, brake = float(np.clip(0.28 + 0.09 * err, 0.0, 0.78)), 0.0
    elif err <= -0.8:
        throttle, brake = 0.0, float(np.clip((-err - 0.5) * 0.28, 0.0, 0.75))
    else:
        throttle, brake = float(np.clip(0.18 + 0.05 * err, 0.0, 0.42)), 0.0
    return steer, throttle, brake, v_target


def _build_candidates(steer_nom, throttle_nom, brake_nom, horizon, num_candidates=16, speed_mps=0.0):
    """Build diverse constant-action 1.0 s candidates around nominal control."""
    launch_th = max(throttle_nom, 0.45) if brake_nom == 0.0 else throttle_nom
    specs = [
        (0.00, throttle_nom, brake_nom),
        (-0.04, throttle_nom, brake_nom),
        (+0.04, throttle_nom, brake_nom),
        (-0.08, throttle_nom, brake_nom),
        (+0.08, throttle_nom, brake_nom),
        (-0.14, throttle_nom * 0.8, brake_nom),
        (+0.14, throttle_nom * 0.8, brake_nom),
        (-0.22, min(throttle_nom * 0.65, 0.50), brake_nom),
        (+0.22, min(throttle_nom * 0.65, 0.50), brake_nom),
        (0.00, throttle_nom * 0.7, brake_nom),
        (0.00, min(0.80, throttle_nom + 0.15) if brake_nom == 0.0 else 0.0, brake_nom),
        (-0.05, launch_th * 0.9 if brake_nom == 0.0 else 0.0, brake_nom),
    ]
    if speed_mps > 2.0:
        specs.extend([
            (0.00, 0.0, 0.35),
            (0.00, 0.0, 0.75),
            (-0.14, 0.0, 0.45),
            (+0.14, 0.0, 0.45),
        ])
    else:
        specs.extend([
            (0.00, launch_th, brake_nom),
            (-0.05, launch_th, brake_nom),
            (+0.05, launch_th, brake_nom),
            (0.00, min(0.80, launch_th + 0.15) if brake_nom == 0.0 else 0.0, brake_nom),
        ])
    rows = [
        [float(np.clip(steer_nom + ds, -1.0, 1.0)), float(np.clip(th, 0.0, 1.0)), float(np.clip(br, 0.0, 1.0))]
        for ds, th, br in specs[:max(1, min(int(num_candidates), len(specs)))]
    ]
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(1).expand(-1, horizon, -1).contiguous()


def _sample_bev_prob(pts_xy, prob_map, default=0.0):
    """Sample `(4, H, W)` or `(H, W)` BEV probabilities at `(*, 2)` FLU metric points."""
    flat = pts_xy.reshape(-1, 2)
    ij = np.floor(pixels(flat) + 0.5).astype(np.int64)
    in_bounds = (ij[:, 0] >= 0) & (ij[:, 0] < BEV_W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
    out = np.full(len(flat), float(default), dtype=np.float32)
    valid_ij = ij[in_bounds]
    out[in_bounds] = prob_map[valid_ij[:, 1], valid_ij[:, 0]]
    return out.reshape(pts_xy.shape[:-1])


def _score_and_select_mpc(module, z_t, speed_mps, yaw_rate, cand_actions, traj_np, nav_anchors,
                          lane_probs, active_boxes, v_target, last_steer):
    """Run batched Stage 3.2 world-model rollout across candidates and pick lowest-cost safe action."""
    horizon = cand_actions.shape[1]
    bsz = cand_actions.shape[0]
    kin0 = torch.tensor([[speed_mps / 30.0, yaw_rate]], device=z_t.device, dtype=torch.float32).expand(bsz, 2)
    z0 = z_t.expand(bsz, -1, -1)

    t0 = perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False), sdpa_context("aotriton"):
        roll = module.world_model.rollout(z0, kin0, cand_actions)
    torch.cuda.synchronize()
    wm_ms = (perf_counter() - t0) * 1000.0

    cand_xy, cand_yaw_end = integrate_ego_delta(roll["next_delta"])
    cand_yaw = torch.cumsum(roll["next_delta"][:, :, 2].float(), dim=1).detach().cpu().numpy()
    cand_xy_np = cand_xy.detach().float().cpu().numpy()  # (B, H, 2)
    inv_thw_max = roll["inv_thw"].detach().float().cpu().numpy().max(axis=(1, 2))
    risk_max = roll["risk_logits"].detach().float().sigmoid().cpu().numpy().max(axis=(1, 2))

    # Physics longitudinal projection avoids observational throttle/speed confounding at v~0.
    th0 = cand_actions[:, 0, 1].cpu().numpy()
    br0 = cand_actions[:, 0, 2].cpu().numpy()
    a_lon = 4.2 * th0 - 8.5 * br0 - 0.04 * speed_mps
    v_end_phys = np.clip(speed_mps + a_lon * horizon * DT, 0.0, 30.0)

    # Reference path: aligned to current speed so startup matches real rollout distances.
    ref_xy = traj_np[:horizon, :2]
    if len(ref_xy) < horizon:
        pad = np.repeat(ref_xy[-1:], horizon - len(ref_xy), axis=0) if len(ref_xy) else np.zeros((horizon, 2))
        ref_xy = np.vstack((ref_xy, pad))
    anchor_dir = nav_anchors[0, :2] / max(float(np.linalg.norm(nav_anchors[0, :2])), 1.0)
    travel_m = max(speed_mps, 3.0) * horizon * DT
    arc_steps = np.linspace(max(0.4, 0.1 * travel_m), travel_m, horizon)
    nav_line = np.outer(arc_steps, anchor_dir)
    target_xy = 0.65 * ref_xy + 0.35 * nav_line

    costs = np.zeros(bsz, dtype=np.float64)
    path_err = np.linalg.norm(cand_xy_np - target_xy[None, :, :], axis=-1).mean(axis=-1)
    costs += 1.6 * path_err
    costs += 0.25 * (v_end_phys - v_target) ** 2
    s0 = cand_actions[:, 0, 0].cpu().numpy()
    steer_nom0 = s0[0]
    costs += 0.85 * np.abs(s0 - last_steer) + 0.35 * np.abs(s0 - steer_nom0)
    if speed_mps < 0.8:
        costs += 3.0 * np.abs(s0 - steer_nom0)

    # Evaluate road, solid yellow line, and curb edge across ego center and left/right edges (±0.92 m).
    lat_vec = np.stack((-np.sin(cand_yaw), np.cos(cand_yaw)), axis=-1)
    sample_pts = np.stack(
        (cand_xy_np, cand_xy_np + 0.92 * lat_vec, cand_xy_np - 0.92 * lat_vec),
        axis=2,
    )  # (B, H, 3, 2)
    driv = _sample_bev_prob(sample_pts, lane_probs[0], default=0.0)
    solid = _sample_bev_prob(sample_pts, lane_probs[2], default=0.0)
    edge = _sample_bev_prob(sample_pts, lane_probs[3], default=0.0)
    costs += 18.0 * (1.0 - driv).mean(axis=(1, 2))
    costs += 28.0 * solid.max(axis=2).mean(axis=1)
    costs += 18.0 * edge.max(axis=2).mean(axis=1)

    # World-model THW & risk + propagated detected vehicle clearance.
    costs += 4.5 * np.maximum(0.0, inv_thw_max - 0.55)
    costs += 3.0 * np.maximum(0.0, risk_max - 0.35)
    if len(active_boxes):
        ts = (np.arange(1, horizon + 1, dtype=np.float32) * DT)[None, :, None]
        bx0 = active_boxes[:, :2][None, None, :, :]
        bvel = active_boxes[:, 7:9][None, None, :, :]
        bsize = np.maximum(active_boxes[:, 3:5], 1.2)
        obj_xy = bx0 + ts[..., None] * bvel
        diff = np.abs(cand_xy_np[:, :, None, :] - obj_xy)
        clearance_x = np.maximum(0.0, diff[..., 0] - (0.5 * bsize[:, 0][None, None, :] + 2.4))
        clearance_y = np.maximum(0.0, diff[..., 1] - (0.5 * bsize[:, 1][None, None, :] + 1.05))
        dist = np.hypot(clearance_x, clearance_y)
        min_clear = dist.min(axis=(1, 2))
        costs += 24.0 * np.exp(-min_clear / 1.1)

    order = np.argsort(costs)
    best_idx = int(order[0])
    alt_paths = tuple(
        np.column_stack((cand_xy_np[idx, :, 0], cand_xy_np[idx, :, 1]))
        for idx in order[1:4]
    )
    return best_idx, cand_xy_np[best_idx], alt_paths, wm_ms


def _front_lidar_aeb(scan, ego_front_x):
    """Return min front gap (m) if solid LiDAR hits block the immediate corridor ahead of bumper."""
    if not len(scan):
        return math.inf
    pts = scan[:, :3]
    corridor = (
        (pts[:, 0] > ego_front_x + 0.2)
        & (pts[:, 0] < ego_front_x + 6.5)
        & (np.abs(pts[:, 1]) < 1.05)
        & (pts[:, 2] > 0.25)
        & (pts[:, 2] < 2.0)
    )
    if int(corridor.sum()) < 6:
        return math.inf
    return float(np.quantile(pts[corridor, 0], 0.05) - ego_front_x)


def _warmup_models(module, device, wm_horizon, wm_candidates):
    """Pre-compile ROCm/AOTriton kernels for both initial and recurrent perception + WM rollout."""
    rgb = tuple(torch.zeros((1, 3, h, w), dtype=torch.uint8, device=device) for w, h in SIZES)
    lbev = torch.zeros((1, 6, H, BEV_W), dtype=torch.float16, device=device)
    nav = torch.zeros((1, 3, 3), dtype=torch.float32, device=device)
    spd = torch.zeros((1, 2), dtype=torch.float32, device=device)
    cmd = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    delta = torch.zeros((1, 3), dtype=torch.float32, device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False), sdpa_context("aotriton"):
        p0 = module.perception(rgb, lidar_bev=lbev, prev_latent=None, ego_delta=None, cmd=cmd, nav_anchors=nav, ego_speed=spd)
        module.perception(rgb, lidar_bev=lbev, prev_latent=p0["z_t"], ego_delta=delta, cmd=cmd, nav_anchors=nav, ego_speed=spd)
        bsz = max(1, int(wm_candidates))
        act = torch.zeros((bsz, int(wm_horizon), 3), dtype=torch.float32, device=device)
        kin = torch.zeros((bsz, 2), dtype=torch.float32, device=device)
        module.world_model.rollout(p0["z_t"].expand(bsz, -1, -1), kin, act)
    torch.cuda.synchronize()


def drive(args, frames, camera, stopped, status):
    faulthandler.enable()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU runtime is required for live CARLA closed-loop demo")

    status("Loading checkpoint")
    module, _, meta = load_stage32_module(args.ckpt, device)
    t_w0 = perf_counter()
    status("Warming up GPU models")
    _warmup_models(module, device, args.wm_horizon, args.wm_candidates)
    print(
        f"Loaded & warmed Stage 3.2 step={meta['global_step']} in {perf_counter() - t_w0:.1f}s ({meta['checkpoint']})",
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    rng = np.random.default_rng(args.seed)
    if stopped.is_set():
        return
    status("Starting CARLA offscreen")
    with launch_carla_server(args.carla_root, args.out_dir, args.port, args.quality) as (client, server_ver):
        world = client.get_world()
        active_map = world.get_map().name.replace("\\", "/").rsplit("/", 1)[-1]
        if active_map != args.map:
            status(f"Loading {args.map}")
            world = client.load_world(args.map)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        _freeze_lights_green(world)

        if stopped.is_set():
            return
        status("Spawning vehicles")
        tm = client.get_trafficmanager(args.port + 5)
        road = world.get_map()
        ego_spawn, start_wp = _pick_ego_spawn(road, carla)
        ego_bp = world.get_blueprint_library().find("vehicle.lincoln.mkz")
        ego_bp.set_attribute("role_name", "hero")
        ego = world.try_spawn_actor(ego_bp, raised_spawn(carla, ego_spawn, 0.35))
        if ego is None:
            raise RuntimeError("Could not spawn ego vehicle")
        ego_front_x = float(ego.bounding_box.extent.x)
        npcs = _spawn_random_npcs(carla, world, tm, ego_spawn.location, args.npc, rng)
        print(f"Spawned ego on {args.map} + {len(npcs)} random NPC vehicles (server {server_ver})", flush=True)

        post = SRPostProcessor()
        nav = InfiniteNavigator(road, start_wp, rng)
        collisions = {"count": 0, "last_frame": -1}
        prev_pose = None
        prev_latent = None
        prev_scene = None
        last_steer = 0.0
        stuck_ticks = 0
        recoveries = 0
        latencies_perc, latencies_wm, latencies_ctrl = [], [], []

        try:
            with (synchronous(world, tm, args.seed, dt=DT / DISPLAY_STEPS),
                  Rig(carla, world, ego) as rig, ThreadPoolExecutor(max_workers=1) as clock):
                rig.cameras("rgb", sensor_tick=DT)
                rig.lidar(sensor_tick=DT)
                requested_frame = None
                chase_frames = queue.Queue(maxsize=1)

                def on_chase(image):
                    camera.publish(image)
                    if image.frame == requested_frame:
                        chase_frames.put_nowait(image)

                rig.attach("chase", "sensor.camera.rgb",
                           carla.Transform(carla.Location(x=-6.8, z=3.4), carla.Rotation(pitch=-14)),
                           dict(image_size_x=args.camera_size[0], image_size_y=args.camera_size[1],
                                fov=90, sensor_tick=0), on_chase)
                status("Synchronizing cameras and LiDAR")
                rig.attach(
                    "collision",
                    "sensor.other.collision",
                    carla.Transform(),
                    {},
                    lambda ev: collisions.update(count=collisions["count"] + 1, last_frame=ev.frame),
                )
                # Find the initial sensor phase before requesting the first model frame.
                phases = {}
                for _ in range(DISPLAY_STEPS * 5):
                    world.tick(45)
                for name, inbox in rig.queues.items():
                    samples = [inbox.get(timeout=10).frame for _ in range(3)]
                    while True:
                        try:
                            samples.append(inbox.get_nowait().frame)
                        except queue.Empty:
                            break
                    phases[name] = samples[-1] % DISPLAY_STEPS
                    if samples[-1] - samples[-2] != DISPLAY_STEPS:
                        raise RuntimeError(f"Unexpected sensor cadence for {name}: {samples}")
                if len(set(phases.values())) != 1:
                    raise RuntimeError(f"Unaligned sensor phases: {phases}")
                phase = phases["rgb/front"]
                while (world.get_snapshot().frame + 1) % DISPLAY_STEPS != phase:
                    world.tick(45)

                save_set = set(args.save_frames or ())
                shot_tick = 40 if args.max_ticks <= 0 else min(40, args.max_ticks - 1)

                def advance_camera():
                    for _ in range(DISPLAY_STEPS - 1):
                        world.tick(45)

                tick_idx = 0
                while args.max_ticks <= 0 or tick_idx < args.max_ticks:
                    if stopped.is_set():
                        break
                    t_ctrl0 = perf_counter()
                    need_shot = tick_idx == shot_tick or tick_idx in save_set
                    requested_frame = world.get_snapshot().frame + 1 if need_shot else None
                    fid = world.tick(45)
                    snap = world.get_snapshot()
                    if snap.frame != fid:
                        raise RuntimeError(f"Expected simulation frame {fid}, got {snap.frame}")
                    intermediate = clock.submit(advance_camera)
                    bundle = rig.read_sensors(snap)
                    sensor_ms = (perf_counter() - t_ctrl0) * 1000
                    ego_snap = snap.find(ego.id)
                    ego_tf = ego_snap.get_transform()
                    pose = matrix(ego_tf)

                    vel = xyz(ego_snap.get_velocity())
                    speed_mps = float(np.linalg.norm(vel[:2]))
                    yaw_rate = float(-np.deg2rad(ego_snap.get_angular_velocity().z))

                    nav_anchors, in_highway, cur_wp, nav_hint = nav.update(pose, ego_tf.location)
                    zone_limit_mps = (args.cruise_highway_kmh if in_highway else args.cruise_urban_kmh) / 3.6

                    scan_raw = np.frombuffer(bundle["lidar"].raw_data, dtype="<f4").reshape(-1, 4).copy()
                    if tick_idx == 0:
                        az = np.degrees(np.arctan2(scan_raw[:, 1], scan_raw[:, 0]))
                        print(f"LiDAR occupied {len(np.unique(np.floor((az + 180) / 5)))} of 72 sectors", flush=True)
                    lidar_pose = matrix(bundle["lidar"].transform)
                    scan_raw[:, :3] = local(scan_raw[:, :3] @ lidar_pose[:3, :3].T + lidar_pose[:3, 3], pose)
                    lbev = lidar_bev(scan_raw)

                    rgb_gpu = tuple(
                        torch.from_numpy(
                            np.ascontiguousarray(cv2.cvtColor(bgr(bundle[f"rgb/{name}"]), cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
                        ).unsqueeze(0).to(device, non_blocking=True)
                        for name in NAMES
                    )
                    lidar_gpu = torch.from_numpy(lbev).unsqueeze(0).to(device, non_blocking=True)
                    nav_gpu = torch.from_numpy(nav_anchors).unsqueeze(0).to(device, non_blocking=True)
                    speed_gpu = torch.tensor([[speed_mps, yaw_rate]], device=device, dtype=torch.float32)
                    cmd_str = "follow"
                    cmd_gpu = torch.from_numpy(command_vector(cmd_str)).unsqueeze(0).to(device, non_blocking=True)
                    delta_gpu = (
                        torch.from_numpy(ego_delta(prev_pose, pose)).unsqueeze(0).to(device, non_blocking=True)
                        if (prev_latent is not None and prev_pose is not None)
                        else None
                    )

                    t_p0 = perf_counter()
                    with torch.inference_mode(), torch.autocast(
                        "cuda", dtype=torch.bfloat16, cache_enabled=False
                    ), sdpa_context("aotriton"):
                        pred = module.perception(
                            rgb_gpu,
                            lidar_bev=lidar_gpu,
                            prev_latent=prev_latent,
                            ego_delta=delta_gpu,
                            cmd=cmd_gpu,
                            nav_anchors=nav_gpu,
                            ego_speed=speed_gpu,
                        )
                    torch.cuda.synchronize()
                    perc_ms = (perf_counter() - t_p0) * 1000.0
                    prev_latent = pred["z_t"].clone()
                    prev_pose = pose

                    traj_np = pred["trajectory"][0].detach().float().cpu().numpy()
                    intent_lbl = "Follow" if abs(nav_anchors[0, 1]) < 2.2 else ("Left Change" if nav_anchors[0, 1] > 0 else "Right Change")
                    scene = prediction_scene(
                        pred,
                        speed=speed_mps,
                        cruise=zone_limit_mps,
                        intent=intent_lbl,
                        previous_scene=prev_scene,
                        ego_motion=delta_gpu[0].cpu().numpy() if delta_gpu is not None else None,
                        lidar_bev=lbev,
                        postprocessor=post,
                        clean=True,
                        steer=last_steer,
                        throttle=0.0,
                        brake=0.0,
                        infer_ms=perc_ms,
                        limit=args.cruise_highway_kmh if in_highway else args.cruise_urban_kmh,
                        nav_hint=nav_hint,
                        mode="LIVE",
                    )
                    prev_scene = scene
                    active_boxes = scene["boxes"][scene["boxes"][:, -1] > 0.5]
                    ahead_boxes = active_boxes[(active_boxes[:, 0] > 0.0) & (active_boxes[:, 0] < 35.0) & (np.abs(active_boxes[:, 1]) < 1.9)]
                    eff_zone_limit = zone_limit_mps
                    if len(ahead_boxes):
                        gap_front = float(ahead_boxes[:, 0].min())
                        eff_zone_limit = min(zone_limit_mps, max(0.0, (gap_front - 8.0) / 1.8))

                    steer_nom, th_nom, br_nom, v_target = _nominal_action(
                        traj_np, nav_anchors, speed_mps, eff_zone_limit, last_steer
                    )
                    scene["cruise"] = v_target
                    cand_actions = _build_candidates(
                        steer_nom, th_nom, br_nom, args.wm_horizon, args.wm_candidates, speed_mps=speed_mps
                    ).to(device)
                    best_idx, _, alt_paths, wm_ms = _score_and_select_mpc(
                        module,
                        pred["z_t"],
                        speed_mps,
                        yaw_rate,
                        cand_actions,
                        traj_np,
                        nav_anchors,
                        scene["lane_probs"],
                        active_boxes,
                        v_target,
                        last_steer,
                    )
                    chosen_act = cand_actions[best_idx, 0].cpu().numpy()
                    steer_cmd = float(np.clip(chosen_act[0], -1.0, 1.0))
                    th_cmd = float(np.clip(chosen_act[1], 0.0, 1.0))
                    br_cmd = float(np.clip(chosen_act[2], 0.0, 1.0))

                    aeb_gap = _front_lidar_aeb(scan_raw, ego_front_x)
                    if math.isfinite(aeb_gap):
                        if aeb_gap < 3.0 or (speed_mps > 3.5 and aeb_gap < 5.2):
                            th_cmd, br_cmd = 0.0, 0.85

                    intermediate.result()
                    ego.apply_control(carla.VehicleControl(throttle=th_cmd, steer=steer_cmd, brake=br_cmd))
                    last_steer = steer_cmd

                    scene["steer"] = steer_cmd
                    scene["throttle"] = th_cmd
                    scene["brake"] = br_cmd
                    scene["alternatives"] = alt_paths
                    scene["infer_ms"] = perc_ms + wm_ms

                    paths = []
                    if tick_idx == shot_tick:
                        paths.append(args.out_dir / "live_sr_preview.png")
                    if tick_idx in save_set:
                        paths.append(args.out_dir / f"live_sr_tick{tick_idx:04d}.png")
                    packet = (scene, bgr(chase_frames.get(timeout=10)).copy() if need_shot else None, paths)
                    while not stopped.is_set():
                        try:
                            frames.put(packet, block=need_shot, timeout=.05 if need_shot else None)
                            break
                        except queue.Full:
                            if not need_shot:
                                break

                    ctrl_ms = (perf_counter() - t_ctrl0) * 1000.0
                    latencies_perc.append(perc_ms)
                    latencies_wm.append(wm_ms)
                    latencies_ctrl.append(ctrl_ms)
                    if tick_idx < 3:
                        print(
                            f"[diag tick {tick_idx}] perc={perc_ms:.1f}ms wm={wm_ms:.1f}ms "
                            f"ctrl={ctrl_ms:.1f}ms v={speed_mps * 3.6:.1f}km/h",
                            flush=True,
                        )

                    # Auto-recovery if stuck too long or collided badly at a halt.
                    stuck_ticks = stuck_ticks + 1 if speed_mps < 0.35 else 0
                    if stuck_ticks >= 80:
                        recoveries += 1
                        stuck_ticks = 0
                        target_idx = min(len(nav.wps) - 1, nav.cursor + 6)
                        rec_tf = raised_spawn(carla, nav.wps[target_idx].transform, 0.45)
                        kept_npcs = []
                        for npc in npcs:
                            if not npc.is_alive:
                                continue
                            if npc.get_location().distance(rec_tf.location) < 9.0:
                                npc.destroy()
                            else:
                                kept_npcs.append(npc)
                        npcs[:] = kept_npcs
                        ego.set_transform(rec_tf)
                        ego.set_target_velocity(carla.Vector3D(0, 0, 0))
                        ego.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                        prev_latent = None
                        prev_pose = None
                        prev_scene = None
                        post = SRPostProcessor()
                        print(f"[tick {tick_idx}] Auto-recovered forward 18m (recoveries={recoveries})", flush=True)

                    if tick_idx > 0 and tick_idx % 100 == 0:
                        _recycle_distant_npcs(carla, world, tm, npcs, ego_tf.location, nav, rng, args.npc)

                    log_every = max(1, int(args.log_every))
                    if tick_idx % log_every == 0:
                        win = latencies_ctrl[-log_every:]
                        row = {
                            "tick": tick_idx,
                            "sim_s": round(tick_idx * DT, 1),
                            "map": args.map,
                            "zone": "highway" if in_highway else ("junction" if cur_wp.is_junction else "urban"),
                            "total_dist_m": round(nav.total_dist, 1),
                            "speed_kmh": round(speed_mps * 3.6, 1),
                            "npcs": sum(1 for n in npcs if n.is_alive),
                            "collisions": collisions["count"],
                            "recoveries": recoveries,
                            "sensor_ms": round(sensor_ms, 2),
                            "perc_ms": round(float(np.median(latencies_perc[-log_every:])), 2),
                            "wm_ms": round(float(np.median(latencies_wm[-log_every:])), 2),
                            "ctrl_ms": round(float(np.median(win)), 2),
                            "vram_alloc_mb": round(float(torch.cuda.max_memory_allocated() / (1024 ** 2)), 1),
                            "vram_res_mb": round(float(torch.cuda.max_memory_reserved() / (1024 ** 2)), 1),
                        }
                        with metrics_path.open("a", encoding="utf-8") as mf:
                            mf.write(json.dumps(row, allow_nan=False) + "\n")
                        print(json.dumps(row, ensure_ascii=False), flush=True)

                    tick_idx += 1
        finally:
            post_npcs = [n for n in npcs if getattr(n, "is_alive", False)]
            for actor in post_npcs:
                try:
                    actor.destroy()
                except Exception:
                    pass
            try:
                if getattr(ego, "is_alive", False):
                    ego.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    from .stage32_demo_view import main
    main()
