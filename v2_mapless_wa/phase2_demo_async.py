"""Asynchronous Epic-quality dual-window viewer and live driver for Stage 2.6."""
import argparse
from contextlib import contextmanager
import ctypes
import json
import math
from pathlib import Path
import socket
import subprocess
import threading
import time

if hasattr(ctypes, 'windll'):
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import carla
import cv2
import numpy as np
from PIL import Image
import torch

from .collect import DT, matrix
from .geometry import CAMS, pixels, projection_tables
from .phase2_control import SpatialController
from .phase2_demo_lane import (
    AdjacentLaneRequest,
    LaneAssist,
    TargetLaneMemory,
    X as lane_stations,
    lane_curve,
    measure_route_lane_offset,
    shift_route,
)
from .phase2_infer import Inference, NativeCDriver, Stage26Driver, fold_for_inference, load_model
from .phase2_live import RoadSurvey, _active_maneuver, actor_observation, reference
from .phase2_model import NATIVE_C, configure_sdpa, mounts, route_inst_vector
from .render_probe import POSTPROCESS, bgr, locked_sizes
from .scenarios import same_direction
from .sd_policy import RoadNavigator, route_vertices


WINDOW = 'CARLA | Stage 2.6 Tactical BEV & Controls'
FIXED_DT = 0.025
PANEL_SIZE = (712, 1310)
PANEL_POS = (1840, 10)
UE_WIN = dict(width=1824, height=1310, x=8, y=10)

GAMMA_LUT = np.rint(np.sqrt(np.arange(256) / 255.0) * 255.0).astype(np.uint8)
INK = (234, 241, 247)
MUTED = (154, 170, 185)
BACKGROUND = (21, 16, 14)
PANEL = (39, 31, 27)
BORDER = (68, 56, 49)
CYAN = (244, 210, 73)
AMBER = (72, 180, 254)
RED = (121, 96, 253)
GREEN = (151, 226, 125)
COMMANDS = ('Keep lane', 'Lane left', 'Lane right')

BUTTONS = (
    ((22, 1092, 238, 1148), 'S   Speed -5', 'slower'),
    ((248, 1092, 464, 1148), 'W   Speed +5', 'faster'),
    ((474, 1092, 690, 1148), 'Space   Pause', 'pause'),
    ((22, 1162, 238, 1218), 'A   Lane left', 'left'),
    ((248, 1162, 464, 1218), 'D   Lane right', 'right'),
    ((474, 1162, 690, 1218), 'R   Replay', 'restart'),
)
KEYS = {
    ord('s'): 'slower',
    ord('w'): 'faster',
    ord('a'): 'left',
    ord('d'): 'right',
    ord('r'): 'restart',
    ord(' '): 'pause',
    27: 'close',
    ord('q'): 'close',
}


def label(canvas, value, x, y, scale=0.52, color=INK, weight=1):
    cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)


def mark(canvas, box, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)


def draw_curve(canvas, points, color, width):
    if len(points) > 1:
        cv2.polylines(canvas, [np.rint(points).astype(np.int32)], False, color, width, cv2.LINE_AA)


def rgb_view(camera):
    return cv2.LUT(bgr(camera), GAMMA_LUT)


class LatestSlot:
    def __init__(self):
        self._lock = threading.Lock()
        self._value = None
        self._seq = 0

    def put(self, value):
        with self._lock:
            self._value = value
            self._seq += 1

    def get(self):
        with self._lock:
            return self._value, self._seq


@contextmanager
def async_server(carla_root, out, port, quality='Epic'):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=0.3):
            pass
    except OSError:
        pass
    else:
        raise RuntimeError(f'CARLA RPC port {port} is already occupied')
    exe = carla_root / 'CarlaUnreal/Binaries/Win64/CarlaUnreal-Win64-Shipping.exe'
    flags = [
        '-windowed',
        f"-ResX={UE_WIN['width']}",
        f"-ResY={UE_WIN['height']}",
        f"-WinX={UE_WIN['x']}",
        f"-WinY={UE_WIN['y']}",
        '-dx12',
        '-nosound',
        f'-carla-rpc-port={port}',
        f'-quality-level={quality}',
        '-stdout',
        '-FullStdOutLogOutput',
    ]
    with (out / 'server.log').open('wb') as log:
        process = subprocess.Popen(
            [str(exe), *flags],
            cwd=carla_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            deadline = time.monotonic() + 140
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'CARLA exited {process.returncode}')
                try:
                    client = carla.Client('127.0.0.1', port)
                    client.set_timeout(120)
                    client.get_world()
                    break
                except RuntimeError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.5)
            yield client, process.pid
        finally:
            process.terminate()
            try:
                process.wait(15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(5)


def async_camera_bp(library, fov, size, sensor_tick=0.05):
    bp = library.find('sensor.camera.rgb')
    for k, v in (
        ('image_size_x', size[0]),
        ('image_size_y', size[1]),
        ('fov', fov),
        ('sensor_tick', sensor_tick),
        ('enable_postprocess_effects', POSTPROCESS),
    ):
        bp.set_attribute(k, str(v))
    if bp.has_attribute('motion_blur_intensity'):
        bp.set_attribute('motion_blur_intensity', '0.0')
    return bp


def async_lidar_bp(library, sensor_tick=0.05):
    bp = library.find('sensor.lidar.ray_cast')
    for k, v in dict(
        range=88,
        channels=64,
        rotation_frequency=20,
        points_per_second=640000,
        upper_fov=12,
        lower_fov=-30,
        sensor_tick=sensor_tick,
    ).items():
        bp.set_attribute(k, str(v))
    if bp.has_attribute('dropoff_general_rate'):
        bp.set_attribute('dropoff_general_rate', '0')
    return bp


def build_engine(model, sizes):
    configure_sdpa('aotriton')
    torch.cuda.tunable.tuning_enable(False)
    torch.set_num_threads(3)
    is_stage26 = bool(getattr(model, 'temporal', False) or getattr(model, 'nav_rich', False))
    if is_stage26:
        gpu_model = Stage26Driver(model).eval().cuda().to(memory_format=torch.channels_last)
        example = (
            *(torch.zeros((1, 3, h, w), dtype=torch.uint8) for w, h in sizes),
            torch.zeros((1, 6, 128, 128), dtype=torch.float16),
            torch.zeros((1, 3, 2), dtype=torch.float32),
            torch.tensor([[1, 0, 0]], dtype=torch.uint8),
            torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.float32),
            torch.tensor([-1.0], dtype=torch.float32),
            torch.zeros((1, 7, 256), dtype=torch.float32),
            torch.zeros((1, 3), dtype=torch.float32),
            torch.zeros((1,), dtype=torch.float32),
        )
    else:
        gpu_model = NativeCDriver(model).eval().cuda().to(memory_format=torch.channels_last)
        example = (
            *(torch.zeros((1, 3, h, w), dtype=torch.uint8) for w, h in sizes),
            torch.zeros((1, 6, 128, 128), dtype=torch.float16),
            torch.zeros((1, 3, 2)),
            torch.tensor([[1, 0, 0]], dtype=torch.uint8),
        )
    return gpu_model, example, is_stage26


class AsyncTacticalDashboard:
    def __init__(self, destination, spec, quality='Epic', title_tag='STAGE 2.6',
                 replay_minutes=0, exit_on_finish=False):
        self.destination = destination
        self.spec = spec
        self.quality = quality
        self.title_tag = title_tag
        self.replay_minutes = float(replay_minutes)
        self.replay_until = None
        self.exit_on_finish = bool(exit_on_finish)
        uv, seen = projection_tables(mounts(), (0.0,), sizes=locked_sizes())
        self.remaps = [np.ascontiguousarray(t) for t in uv[0]]
        self.camera_count = seen[0].sum(0).clip(1)
        self.coverage = seen[0].any(0)
        self.closed = False
        self.canvas = np.full((PANEL_SIZE[1], PANEL_SIZE[0], 3), BACKGROUND, dtype=np.uint8)
        self.start_route(spec)
        label(self.canvas, f"LOADING {spec['town'].upper()} / {spec['name'].upper()}", 22, 54, 0.68, INK, 2)
        label(self.canvas, f"{title_tag}   {quality.upper()}   ASYNC   HIP GRAPH", 22, 88, 0.46, MUTED)
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(WINDOW, PANEL_POS[0], PANEL_POS[1])
        cv2.setMouseCallback(WINDOW, self.click)
        cv2.imshow(WINDOW, self.canvas)
        cv2.waitKey(1)

    def start_route(self, spec):
        self.route = spec['name']
        self.target_kmh = float(spec.get('cruise_kmh', 55))
        self.tick = 0
        self.intent = 0
        self.intent_until = 0
        self.intent_tick = None
        self.request = AdjacentLaneRequest()
        self.lane_assist = LaneAssist()
        self.lane_status = 'Waiting for lane'
        self.lane_side = 0
        self.feedback_until = 0
        self.restart = False
        self.paused = False
        self.finished = False
        if self.replay_until is None:
            self.replay_until = time.monotonic() + self.replay_minutes * 60.0

    @property
    def command(self):
        return self.intent if self.tick < self.intent_until else 0

    def observe_command(self, output):
        if self.command and self.request.observe(output):
            self.intent = 0
            self.lane_status = 'New lane detected / Keep lane'
            return True
        elif self.intent and not self.command:
            self.intent = 0
            self.lane_status = 'Request expired / Keep lane'
        return False

    def action(self, name):
        if name == 'close':
            self.closed = True
        elif name == 'restart':
            self.restart = True
        elif not self.finished:
            if name == 'pause':
                self.paused = not self.paused
            elif name in ('faster', 'slower'):
                delta = 5 if name == 'faster' else -5
                self.target_kmh = min(90.0, max(5.0, self.target_kmh + delta))
            elif name in ('left', 'right') and not self.command and not self.lane_side:
                self.intent = 1 if name == 'left' else 2
                self.intent_tick = self.tick
                self.request.begin(self.intent)
                self.intent_until = self.tick + round(3.5 / DT)
                self.lane_status = 'Left requested' if self.intent == 1 else 'Right requested'

    def click(self, event, x, y, flags, state):
        if event == cv2.EVENT_LBUTTONUP:
            for (xmin, ymin, xmax, ymax), _, name in BUTTONS:
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    self.action(name)
                    break

    def poll(self, delay=1):
        key = cv2.waitKeyEx(delay)
        if ord('A') <= key <= ord('Z'):
            key += ord('a') - ord('A')
        if key in KEYS:
            self.action(KEYS[key])
        try:
            self.closed |= cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            self.closed = True

    def render_bev(self, views, density, mask, lines, exists, nav, planned, occupancy, route=None, target=None):
        warped = [cv2.remap(img, table, None, cv2.INTER_LINEAR) for img, table in zip(views, self.remaps)]
        fused = np.sum(warped, axis=0) / self.camera_count[..., None]
        canvas = np.full((128, 128, 3), (31, 24, 20), np.float32)
        canvas[self.coverage] = 0.50 * fused[self.coverage] + (26, 20, 16)
        returns = np.clip(density.astype(np.float32) / 4.0, 0.0, 0.7)[..., None]
        canvas = canvas * (1.0 - returns) + returns * np.array((151, 176, 190))
        for cells, color, mix in (
            (mask[0] > 0, GREEN, 0.20),
            (mask[3] > 0, CYAN, 0.17),
            (mask[1] > 0, (220, 222, 220), 0.55),
            (mask[2] > 0, AMBER, 0.70),
            (occupancy.max(0) > 0, RED, 0.65),
        ):
            canvas[cells] = canvas[cells] * (1.0 - mix) + np.array(color) * mix
        pane = cv2.resize(canvas[20:117, 3:125].astype(np.uint8), (440, 728), interpolation=cv2.INTER_NEAREST)
        draw = lambda points: (pixels(points) - [3, 20]) * [440.0 / 122.0, 728.0 / 97.0]
        for meter in (0, 20, 40, 60):
            py = round(draw(np.array([[meter, 0.0]]))[0, 1])
            cv2.line(pane, (0, py), (439, py), (73, 67, 63), 1)
            label(pane, f'{meter} m', 6, max(14, py - 5), 0.42, MUTED)
        for idx in (1, 2):
            if exists[idx] > 0:
                pts = lines[idx]
                vis = np.isfinite(pts).all(1) & (pts[:, 0] >= 0.0) & (pts[:, 0] < 60.0)
                if vis.sum() >= 3:
                    draw_curve(pane, draw(pts[vis]), INK, 2)
        draw_curve(pane, draw(nav if route is None else route), CYAN, 2)
        draw_curve(pane, draw(planned), AMBER, 3)
        if target is not None:
            for point in draw(target[2::2]):
                cv2.circle(pane, tuple(np.rint(point).astype(int)), 3, GREEN, -1, cv2.LINE_AA)
        for point in draw(nav[1:]):
            cv2.circle(pane, tuple(np.rint(point).astype(int)), 4, CYAN, -1, cv2.LINE_AA)
        car = np.array([[2.3, -0.95], [2.3, 0.95], [-2.3, 0.95], [-2.3, -0.95]])
        cv2.fillConvexPoly(pane, np.rint(draw(car)).astype(np.int32), INK, cv2.LINE_AA)
        return pane

    def paint_controls(self, status, collisions=0):
        mark(self.canvas, (16, 1084, 696, 1296), PANEL)
        cv2.rectangle(self.canvas, (16, 1084), (696, 1296), BORDER, 1)
        for box, text, name in BUTTONS:
            active = (name, self.command) in (('left', 1), ('right', 2))
            mark(self.canvas, box, (103, 70, 31) if active else (64, 49, 39))
            if name == 'pause':
                text = 'Space   Resume' if self.paused else 'Space   Pause'
            label(self.canvas, text, box[0] + 18, box[1] + 35, 0.56, CYAN if active else INK)
        status_col = AMBER if self.finished or self.paused else (RED if 'BRAKE' in status else GREEN)
        label(self.canvas, status[:28], 24, 1252, 0.50, status_col)
        label(self.canvas, f'{COMMANDS[self.command]}  |  Col {collisions}', 355, 1252, 0.48, MUTED)
        label(self.canvas, 'Async 20Hz model  |  Smooth spectator  |  Q / Esc exit', 24, 1280, 0.43, MUTED)

    def present(self, images, density, output, nav, path, occupancy, speed, throttle, brake,
                safety, progress, ue_fps, ctrl_hz, route=None, target=None, steer=0.0, collisions=0):
        views = [rgb_view(image) for image in images]
        nav_pts = np.vstack(([0.0, 0.0], nav))
        planned = np.vstack(([0.0, 0.0], path))
        mask = output['mask'][0].float().cpu().numpy()
        lines = output['lines'][0].float().cpu().numpy()
        exists = output['exists'][0].float().cpu().numpy()
        canvas = self.canvas
        canvas[:] = BACKGROUND

        label(canvas, f"{self.spec['town'].upper()} / {self.route.upper()}"[:36], 16, 36, 0.68, INK, 2)
        label(canvas, f"{self.title_tag}   {self.quality.upper()}   ASYNC   HIP GRAPH", 16, 62, 0.44, MUTED)
        gap = safety.get('lidar_gap_m', math.inf)
        gap_txt = f'{gap:.1f} m' if math.isfinite(gap) else '--'
        label(canvas, f"UE5 {ue_fps:4.1f} FPS   |   CTRL {ctrl_hz:4.1f} Hz   |   LIDAR {gap_txt}", 16, 86, 0.48, CYAN)

        label(canvas, 'COMPOSITE BEV (0-60 m)', 16, 108, 0.48, INK, 2)
        canvas[116:844, 16:456] = self.render_bev(
            views, density, mask, lines, exists, nav_pts, planned, occupancy, route=route, target=target
        )

        label(canvas, 'RGB   768 / 384 / 384', 470, 108, 0.46, INK, 2)
        for y, caption, view in zip((116, 288, 460), ('FRONT', 'FRONT LEFT', 'FRONT RIGHT'), views):
            canvas[y:y + 134, 468:700] = cv2.resize(view, (232, 134), interpolation=cv2.INTER_LINEAR)
            label(canvas, caption, 470, y + 154, 0.44, MUTED)

        mark(canvas, (468, 636, 700, 844), PANEL)
        cv2.rectangle(canvas, (468, 636), (700, 844), BORDER, 1)
        label(canvas, 'OVERLAY LEGEND', 480, 662, 0.46, INK, 2)
        legend = (
            (CYAN, 'Navigation (SD / 3A)'),
            (AMBER, 'Tracking path'),
            (INK, 'Current lane edges'),
            (GREEN, 'Selected target lane'),
            (RED, 'Future occupancy'),
        )
        for idx, (col, text) in enumerate(legend):
            yy = 694 + idx * 29
            mark(canvas, (480, yy - 7, 502, yy - 2), col)
            label(canvas, text, 512, yy, 0.44, col)

        mark(canvas, (16, 856, 696, 1072), PANEL)
        cv2.rectangle(canvas, (16, 856), (696, 1072), BORDER, 1)
        label(canvas, f'{speed:04.1f} km/h', 28, 908, 1.15, INK, 2)
        lim = safety.get('curve_kmh', self.target_kmh)
        label(canvas, f'Prog {progress:04.0f} m   |   Target {self.target_kmh:.0f}   |   Lim {lim:.0f} km/h',
              250, 898, 0.48, MUTED)
        label(canvas, f"Thr {throttle:3.0%}  Brk {brake:3.0%}  Str {'L' if steer < 0 else 'R'} {abs(steer):3.0%}",
              250, 926, 0.46, RED if safety.get('safety_brake') else INK)

        # Visual telemetry bars
        bx, bw = 96, 578
        for row_y, name, val, col in (
            (960, 'THR', float(np.clip(throttle, 0.0, 1.0)), GREEN),
            (994, 'BRK', float(np.clip(brake, 0.0, 1.0)), RED),
        ):
            label(canvas, name, 28, row_y + 13, 0.44, MUTED)
            mark(canvas, (bx, row_y, bx + bw, row_y + 15), BACKGROUND)
            if val > 0.01:
                mark(canvas, (bx, row_y, bx + max(2, int(bw * val)), row_y + 15), col)
        # Centered steering bar
        sy = 1028
        label(canvas, 'STR', 28, sy + 13, 0.44, MUTED)
        mark(canvas, (bx, sy, bx + bw, sy + 15), BACKGROUND)
        mid = bx + bw // 2
        cv2.line(canvas, (mid, sy - 2), (mid, sy + 17), MUTED, 1)
        sx = mid + int((bw * 0.5) * float(np.clip(steer, -1.0, 1.0)))
        if abs(sx - mid) >= 1:
            mark(canvas, (min(mid, sx), sy + 1, max(mid, sx), sy + 14), AMBER if steer >= 0 else CYAN)

        status = 'LIDAR BRAKE' if safety.get('safety_brake') else self.lane_status.upper()
        self.paint_controls(status, collisions=collisions)
        cv2.imshow(WINDOW, canvas)
        if self.tick in (1, 100):
            cv2.imwrite(str(self.destination / 'dashboard-async.jpg'), canvas)

    def wait_for_restart(self, result):
        self.finished = True
        if self.restart or self.closed or self.exit_on_finish:
            if self.exit_on_finish:
                self.closed = True
            return
        reason = (result.get('reason') or 'stopped').replace('_', ' ').upper()
        replay = self.replay_until is not None and time.monotonic() < self.replay_until
        status = f'{reason}   /   REPLAY IN 3 S' if replay else reason
        self.paint_controls(status, collisions=result.get('collision_count', 0))
        cv2.imshow(WINDOW, self.canvas)
        next_start = time.monotonic() + 3.0
        while not (self.closed or self.restart):
            self.poll(40)
            if replay and time.monotonic() >= next_start:
                self.restart = True


def _smooth_chase_transform(ego_tf, prev_yaw=None, dt_ue=0.0):
    target_yaw = float(ego_tf.rotation.yaw)
    if prev_yaw is None:
        cam_yaw = target_yaw
    else:
        alpha = (1.0 - math.exp(-dt_ue / 0.12)) if dt_ue > 0.0 else 0.0
        dy = ((target_yaw - prev_yaw + 180.0) % 360.0) - 180.0
        cam_yaw = ((prev_yaw + alpha * dy + 180.0) % 360.0) - 180.0
    fwd = carla.Rotation(yaw=cam_yaw).get_forward_vector()
    loc = carla.Location(
        ego_tf.location.x - 8.2 * fwd.x,
        ego_tf.location.y - 8.2 * fwd.y,
        ego_tf.location.z + 3.8,
    )
    return carla.Transform(loc, carla.Rotation(pitch=-11.0, yaw=cam_yaw)), cam_yaw


def _predict_chase_transform(ego_tf, ego_vel, prev_yaw=None, dt_ue=FIXED_DT):
    pred_loc = carla.Location(
        ego_tf.location.x + ego_vel.x * dt_ue,
        ego_tf.location.y + ego_vel.y * dt_ue,
        ego_tf.location.z + ego_vel.z * dt_ue,
    )
    return _smooth_chase_transform(carla.Transform(pred_loc, ego_tf.rotation), prev_yaw, dt_ue)


def spawn_corridor_npcs(world, road, library, ref, ego_tf, tm, spec):
    actors = []
    npc_target = int(spec.get('npc', 0))
    if npc_target <= 0:
        return actors
    rng = np.random.default_rng(714)
    if spec.get('npc_along_route'):
        exclude = ('truck', 'bus', 'firetruck', 'ambulance', 'carlacola', 'cybertruck', 'sprinter', 'fusorosa', 't2')
        car_bps = [
            b for b in library.filter('vehicle.*')
            if b.has_attribute('number_of_wheels') and int(b.get_attribute('number_of_wheels')) == 4
            and not any(tag in b.id.lower() for tag in exclude)
        ] or [library.find('vehicle.mini.cooper')]
        laps = max(1, int(spec.get('laps') or 1))
        lap_len = max(60, len(ref) // laps)
        step = max(14, lap_len // max(npc_target + 2, 1))
        candidates = []
        for idx in range(18, max(19, lap_len - 25), step):
            base_wp = ref[idx]
            lane_opts = [base_wp]
            for side in ('left', 'right'):
                adj = same_direction(base_wp, side)
                if adj is not None:
                    lane_opts.append(adj)
            rng.shuffle(lane_opts)
            candidates.extend(lane_opts[:2])
        for wp in candidates:
            if len(actors) >= npc_target:
                break
            if wp.transform.location.distance(ego_tf.location) < 42.0:
                continue
            if any(wp.transform.location.distance(a.get_transform().location) < 22.0 for a in actors):
                continue
            spawn_tf = carla.Transform(
                carla.Location(wp.transform.location.x, wp.transform.location.y, wp.transform.location.z + 0.45),
                wp.transform.rotation,
            )
            bp_npc = car_bps[int(rng.integers(0, len(car_bps)))]
            npc = world.try_spawn_actor(bp_npc, spawn_tf)
            if npc is not None:
                actors.append(npc)
                npc.set_autopilot(True, tm.get_port())
                tm.ignore_lights_percentage(npc, 100.0)
                tm.auto_lane_change(npc, True)
                tm.vehicle_percentage_speed_difference(npc, float(rng.uniform(-15.0, 12.0)))
    else:
        locations = road.get_spawn_points()
        rng.shuffle(locations)
        for t in locations:
            if len(actors) >= npc_target:
                break
            if not 36.0 < t.location.distance(ego_tf.location) < 180.0:
                continue
            npc = world.try_spawn_actor(library.find('vehicle.mini.cooper'), t)
            if npc is not None:
                actors.append(npc)
                npc.set_autopilot(True, tm.get_port())
    return actors


def _save_recording(record_cfg, recorded_frames, hero_bgr):
    if not record_cfg:
        return
    hero_path = record_cfg.get('hero')
    if hero_path is not None and hero_bgr is not None:
        hero_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(hero_path), hero_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f'HERO_SAVED {hero_path} bytes={hero_path.stat().st_size}', flush=True)
    webp_path = record_cfg.get('webp')
    if webp_path is not None and recorded_frames:
        webp_path.parent.mkdir(parents=True, exist_ok=True)
        sampled = []
        last_t = None
        for ts, rgb_frame in recorded_frames:
            if last_t is None or (ts - last_t) >= (1.0 / 15.0 - 1e-3):
                sampled.append(Image.fromarray(rgb_frame))
                last_t = ts
        if len(sampled) < 2 and len(recorded_frames) >= 2:
            sampled = [Image.fromarray(f) for _, f in recorded_frames]
        sampled[0].save(
            webp_path,
            format='WEBP',
            save_all=True,
            append_images=sampled[1:],
            duration=67,
            loop=0,
            quality=76,
            method=4,
        )
        if webp_path.stat().st_size > int(5.5 * 1024 * 1024):
            sampled[0].save(
                webp_path,
                format='WEBP',
                save_all=True,
                append_images=sampled[1:],
                duration=67,
                loop=0,
                quality=64,
                method=4,
            )
        if webp_path.stat().st_size > int(6.0 * 1024 * 1024):
            small = [im.resize((810, 415), Image.Resampling.LANCZOS) for im in sampled]
            small[0].save(
                webp_path,
                format='WEBP',
                save_all=True,
                append_images=small[1:],
                duration=67,
                loop=0,
                quality=68,
                method=4,
            )
            w, h = small[0].size
        else:
            w, h = sampled[0].size
        print(
            f'WEBP_SAVED {webp_path} frames={len(sampled)} size={w}x{h} bytes={webp_path.stat().st_size}',
            flush=True,
        )


def run_async_route(client, world, spec, engine, port, out_dir, dashboard, max_seconds=None, record_cfg=None):
    road = world.get_map()
    if road.name.rsplit('/', 1)[-1] != spec['town']:
        raise ValueError(f'Unexpected evaluation world {road.name}')
    vertices = route_vertices(spec)
    sd = RoadSurvey(vertices)
    ref = reference(road, sd, spec)
    tm = client.get_trafficmanager(port + 5)
    old = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DT
    settings.substepping = True
    settings.max_substep_delta_time = 0.0125
    settings.max_substeps = 4
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(421)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    library = world.get_blueprint_library()
    bp = library.find('vehicle.lincoln.mkz')
    bp.set_attribute('role_name', 'hero')
    tf = ref[0].transform
    tf.location.z += 0.38

    actors, sensors = [], []
    col_lock = threading.Lock()
    state_lock = threading.Lock()
    stop_pacer = threading.Event()
    pause_pacer = threading.Event()
    pacer_thread = None
    events = []
    cam_slots = [LatestSlot() for _ in range(len(CAMS))]
    lidar_slot = LatestSlot()
    gnss_slot = LatestSlot()
    imu_slot = LatestSlot()
    rec_slot = LatestSlot() if (record_cfg and (record_cfg.get('webp') or record_cfg.get('hero'))) else None

    def attach(blueprint, mount, callback):
        s = world.spawn_actor(blueprint, mount, attach_to=ego)
        sensors.append(s)
        s.listen(callback)

    try:
        ego = world.spawn_actor(bp, tf)
        actors.append(ego)
        box = ego.bounding_box
        half_width, half_length = box.extent.y, box.extent.x
        max_steer = max(w.max_steer_angle for w in ego.get_physics_control().wheels)
        controller = SpatialController(
            spec.get('cruise_kmh', 55),
            2.85,
            half_width,
            math.radians(max_steer),
            box.location.x + half_length,
            box.location.x - half_length,
            box.location.z + box.extent.z,
        )
        actors.extend(spawn_corridor_npcs(world, road, library, ref, tf, tm, spec))
        for idx, ((x, y, z, yaw, pitch, fov), size) in enumerate(zip(CAMS, locked_sizes())):
            mount = carla.Transform(carla.Location(x, y, z), carla.Rotation(yaw=yaw, pitch=pitch))
            attach(async_camera_bp(library, fov, size, sensor_tick=0.05), mount, cam_slots[idx].put)
        attach(async_lidar_bp(library, sensor_tick=0.05), carla.Transform(carla.Location(z=2.2)), lidar_slot.put)
        for sid, slot in (('sensor.other.gnss', gnss_slot), ('sensor.other.imu', imu_slot)):
            sbp = library.find(sid)
            if sbp.has_attribute('sensor_tick'):
                sbp.set_attribute('sensor_tick', '0.05')
            attach(sbp, carla.Transform(), slot.put)
        if rec_slot is not None:
            rec_bp = library.find('sensor.camera.rgb')
            for rk, rv in (
                ('image_size_x', 960),
                ('image_size_y', 688),
                ('fov', 90),
                ('sensor_tick', 0.05),
                ('enable_postprocess_effects', True),
            ):
                rec_bp.set_attribute(rk, str(rv))
            if rec_bp.has_attribute('motion_blur_intensity'):
                rec_bp.set_attribute('motion_blur_intensity', '0.0')
            rec_mount = carla.Transform(carla.Location(x=-8.2, z=3.8), carla.Rotation(pitch=-11.0))
            rec_cam = world.spawn_actor(
                rec_bp, rec_mount, attach_to=ego, attachment_type=carla.AttachmentType.SpringArmGhost
            )
            sensors.append(rec_cam)
            rec_cam.listen(rec_slot.put)

        def on_col(e):
            with col_lock:
                events.append(dict(frame=e.frame, actor=e.other_actor.id, kind=e.other_actor.type_id))

        attach(library.find('sensor.other.collision'), carla.Transform(), on_col)
        spectator = world.get_spectator()
        sp_tf, sp_yaw = _smooth_chase_transform(ego.get_transform(), None)
        spectator.set_transform(sp_tf)
        world.tick(10.0)
        init_snap = world.get_snapshot()
        init_ego = init_snap.find(ego.id) if init_snap is not None else None
        pacer_state = {
            'tf': init_ego.get_transform() if init_ego is not None else ego.get_transform(),
            'vel': init_ego.get_velocity() if init_ego is not None else carla.Vector3D(0.0, 0.0, 0.0),
            'ctrl': carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0),
            'fps': 0.0,
            'sim_t': float(init_snap.timestamp.elapsed_seconds) if init_snap is not None else 0.0,
        }

        def _ue_pacer():
            cam_yaw = sp_yaw
            next_tick_wall = time.perf_counter()
            last_tick_wall = None
            while not stop_pacer.is_set():
                if pause_pacer.is_set():
                    next_tick_wall = time.perf_counter()
                    last_tick_wall = None
                    time.sleep(0.02)
                    continue
                with state_lock:
                    ego_tf = pacer_state['tf']
                    ego_vel = pacer_state['vel']
                    ctrl_cmd = pacer_state['ctrl']
                tf_cam, cam_yaw = _predict_chase_transform(ego_tf, ego_vel, cam_yaw, FIXED_DT)
                try:
                    client.apply_batch([
                        carla.command.ApplyTransform(spectator.id, tf_cam),
                        carla.command.ApplyVehicleControl(ego.id, ctrl_cmd),
                    ])
                    world.tick(10.0)
                    snap = world.get_snapshot()
                except Exception:
                    if stop_pacer.is_set():
                        break
                    time.sleep(0.01)
                    continue
                now_wall = time.perf_counter()
                with state_lock:
                    if snap is not None:
                        actor_snap = snap.find(ego.id)
                        if actor_snap is not None:
                            pacer_state['tf'] = actor_snap.get_transform()
                            pacer_state['vel'] = actor_snap.get_velocity()
                        pacer_state['sim_t'] = float(snap.timestamp.elapsed_seconds)
                    if last_tick_wall is not None and now_wall > last_tick_wall:
                        inst_fps = 1.0 / (now_wall - last_tick_wall)
                        prev_fps = pacer_state['fps']
                        pacer_state['fps'] = inst_fps if prev_fps == 0.0 else 0.85 * prev_fps + 0.15 * inst_fps
                last_tick_wall = now_wall
                next_tick_wall += FIXED_DT
                sleep_s = next_tick_wall - time.perf_counter()
                if sleep_s > 0.001:
                    time.sleep(sleep_s)
                elif sleep_s < -2.0 * FIXED_DT:
                    next_tick_wall = time.perf_counter()

        pacer_thread = threading.Thread(target=_ue_pacer, name='ue-pacer', daemon=True)
        pacer_thread.start()

        navigator = RoadNavigator(vertices, spec['georef'], 0.72, spec.get('events', ()), spec.get('sd_accuracy'))
        lane_memory = TargetLaneMemory()
        is_stage26 = bool(getattr(engine, 'is_stage26', False))
        prev_map_tokens = torch.zeros((1, 7, 256), dtype=torch.float32)
        prev_compass = None
        prev_speed = None
        has_prev_flag = False
        cruise_sd_offset = None

        # Wait briefly for initial sensor frames
        warmup_deadline = time.monotonic() + 15.0
        need_slots = (*cam_slots, lidar_slot, gnss_slot, imu_slot) + ((rec_slot,) if rec_slot is not None else ())
        while time.monotonic() < warmup_deadline:
            dashboard.poll(10)
            if dashboard.closed or dashboard.restart:
                return None
            if all(s.get()[0] is not None for s in need_slots):
                break

        start_wall = time.monotonic()
        last_front_seq = -1
        prev_sim_t = None
        first_sim_t = None
        prev_infer_t = None
        ctrl_hz_ema = 0.0
        auto_lc_fired = False
        recorded_frames = []
        hero_bgr = None
        reason = 'tick_budget'
        max_ticks = int(spec.get('max_ticks', 12000))
        progress = 0.0
        lateral_errs = []
        last_progress = 0.0
        stagnant_since = time.monotonic()
        ctrl_ticks = 0

        while ctrl_ticks < max_ticks and not (dashboard.closed or dashboard.restart):
            if max_seconds is not None and (time.monotonic() - start_wall) >= max_seconds:
                reason = 'max_seconds'
                break

            dashboard.poll(1)
            if dashboard.paused:
                pause_pacer.set()
                with col_lock:
                    col_cnt = len(events)
                dashboard.paint_controls('PAUSED', collisions=col_cnt)
                cv2.imshow(WINDOW, dashboard.canvas)
                prev_sim_t = None
                prev_infer_t = None
                stagnant_since = time.monotonic()
                time.sleep(0.025)
                continue
            pause_pacer.clear()

            front_img, front_seq = cam_slots[0].get()
            if front_seq == last_front_seq or front_img is None:
                time.sleep(0.004)
                continue

            sync_deadline = time.perf_counter() + 0.012
            target_frame = front_img.frame
            while time.perf_counter() < sync_deadline:
                left_img = cam_slots[1].get()[0]
                right_img = cam_slots[2].get()[0]
                lidar_scan = lidar_slot.get()[0]
                rec_ok = rec_slot is None or ((r := rec_slot.get()[0]) is not None and r.frame >= target_frame)
                if (left_img is not None and left_img.frame >= target_frame
                        and right_img is not None and right_img.frame >= target_frame
                        and lidar_scan is not None and lidar_scan.frame >= target_frame and rec_ok):
                    break
                time.sleep(0.001)

            left_img = cam_slots[1].get()[0]
            right_img = cam_slots[2].get()[0]
            lidar_scan = lidar_slot.get()[0]
            gnss = gnss_slot.get()[0]
            imu = imu_slot.get()[0]
            if any(x is None for x in (left_img, right_img, lidar_scan, gnss, imu)):
                time.sleep(0.004)
                continue

            last_front_seq = front_seq
            sim_t = float(front_img.timestamp)
            if first_sim_t is None:
                first_sim_t = sim_t
            sim_elapsed = sim_t - first_sim_t
            dt = DT if prev_sim_t is None else float(np.clip(sim_t - prev_sim_t, 0.02, 0.25))
            prev_sim_t = sim_t
            now = time.perf_counter()
            if prev_infer_t is not None:
                hz_inst = 1.0 / max(1e-3, now - prev_infer_t)
                ctrl_hz_ema = hz_inst if ctrl_hz_ema == 0.0 else 0.85 * ctrl_hz_ema + 0.15 * hz_inst
            prev_infer_t = now
            ctrl_ticks += 1
            dashboard.tick = ctrl_ticks
            if record_cfg and record_cfg.get('auto_lc') and not auto_lc_fired and sim_elapsed >= float(record_cfg.get('auto_lc_sec', 7.5)):
                dashboard.action(record_cfg['auto_lc'])
                auto_lc_fired = True

            try:
                nav = navigator.update(gnss.latitude, gnss.longitude, imu.compass, time.monotonic() - start_wall)
            except ValueError:
                with state_lock:
                    pacer_state['ctrl'] = carla.VehicleControl(brake=1.0)
                reason = 'no_forward_sd_road'
                break

            progress = navigator.station
            user_command = dashboard.command
            command = user_command
            new_request = dashboard.intent_tick == ctrl_ticks
            coarse = navigator.points(np.arange(0.0, 61.0, 2.0))
            near_nav_maneuver = bool(
                navigator.station is not None
                and any(k != 'follow' and -110.0 <= (s - navigator.station) <= 130.0 for s, k in navigator.events)
            )
            route = None
            if lane_memory.world is not None:
                route = lane_memory.guidance(coarse, navigator)
                nav = route[[10, 20, 30]]
                cruise_sd_offset = None
            elif cruise_sd_offset is not None:
                if near_nav_maneuver:
                    cruise_sd_offset *= 0.85
                    if abs(cruise_sd_offset) < 0.05:
                        cruise_sd_offset = 0.0
                route = shift_route(coarse, cruise_sd_offset)
                nav = route[[10, 20, 30]]
            else:
                route = lane_memory.guidance(coarse, navigator)
                nav = route[[10, 20, 30]]

            if not user_command:
                command = lane_memory.command
            if lane_memory.rejected and not new_request:
                command = 0

            with state_lock:
                vel = pacer_state['vel']
            v = math.hypot(vel.x, vel.y)
            args, raw = actor_observation((front_img, left_img, right_img), lidar_scan, nav, command)
            stage26_extra = ()
            if is_stage26:
                inst_dist, inst_kind = _active_maneuver(navigator)
                route_inst_t = torch.from_numpy(route_inst_vector(inst_kind))[None]
                route_dist_t = torch.tensor([float(inst_dist)], dtype=torch.float32)
                if has_prev_flag and prev_compass is not None and prev_speed is not None:
                    dyaw = ((prev_compass - float(imu.compass) + math.pi) % (2.0 * math.pi)) - math.pi
                    ds = 0.5 * (prev_speed + v) * dt
                    dx = -ds * math.cos(0.5 * dyaw)
                    dy = -ds * math.sin(0.5 * dyaw)
                    ego_delta_t = torch.tensor([[dx, dy, dyaw]], dtype=torch.float32)
                    has_prev_t = torch.tensor([1.0], dtype=torch.float32)
                else:
                    ego_delta_t = torch.zeros((1, 3), dtype=torch.float32)
                    has_prev_t = torch.zeros((1,), dtype=torch.float32)
                stage26_extra = (route_inst_t, route_dist_t, prev_map_tokens, ego_delta_t, has_prev_t)

            output = engine.run_cpu(*args, *stage26_extra)
            path = output['path'][0].float().cpu().numpy().copy()
            occ = output['occ'][0].float().cpu().numpy().copy()
            confirmed = dashboard.observe_command(output)
            controller.cruise = dashboard.target_kmh / 3.6
            lane_memory.update(output, navigator, path, user_command, confirmed, command, raw, v, new_request)
            if new_request:
                selected = lane_memory.guidance(coarse, navigator)
                corrected = selected[[10, 20, 30]]
                applied_command = 0 if lane_memory.rejected else command
                if applied_command != command or not np.array_equal(corrected, nav):
                    command = applied_command
                    route, nav = selected, corrected
                    signal = torch.zeros_like(args[-1])
                    signal[0, command] = 1
                    output = engine.run_cpu(*args[:-2], torch.from_numpy(nav)[None], signal, *stage26_extra)
                    path = output['path'][0].float().cpu().numpy().copy()
                    occ = output['occ'][0].float().cpu().numpy().copy()
            if lane_memory.status:
                dashboard.lane_status = lane_memory.status

            if is_stage26:
                prev_map_tokens = output['map_tokens'].detach().float().cpu().clone()
                prev_compass = float(imu.compass)
                prev_speed = float(v)
                has_prev_flag = True

            target = lane_memory.local_target(navigator)
            control_lane = target
            if target is None and (not lane_memory.status or lane_memory.rejected or lane_memory.status == 'Target lost / Keep lane'):
                if output['exists'][0, 0].item() > 0:
                    sensed = lane_curve(output['lines'][0, 0].float().cpu().numpy())
                    if sensed is not None:
                        c_vec = coarse[12] - coarse[3]
                        if np.hypot(c_vec[0], c_vec[1]) > 1.0:
                            sensed_heading = math.atan2(float(sensed[16] - sensed[4]), 18.0)
                            route_heading = math.atan2(float(c_vec[1]), float(c_vec[0]))
                            heading_div = abs((route_heading - sensed_heading + math.pi) % (2 * math.pi) - math.pi)
                        else:
                            heading_div = 0.0
                        path_ref = np.interp(path[4:12, 0], lane_stations, sensed)
                        path_div = float(np.mean(np.abs(path[4:12, 1] - path_ref)))
                        if lane_memory.world is None and not near_nav_maneuver and heading_div <= math.radians(10.0):
                            meas_off = measure_route_lane_offset(coarse, np.column_stack((lane_stations[:17], sensed[:17])))
                            if meas_off is not None:
                                cruise_sd_offset = meas_off if cruise_sd_offset is None else 0.80 * cruise_sd_offset + 0.20 * meas_off
                        if not near_nav_maneuver and heading_div <= math.radians(8.0) and path_div <= 1.25:
                            control_lane = np.column_stack((lane_stations[:17], sensed[:17]))

            changing = target is not None and not lane_memory.acquired
            path, limit, offset = controller.lane_tracking(path, control_lane, changing, speed=v)
            if lane_memory.status and lane_memory.status != 'Target lost / Keep lane' and not lane_memory.rejected and (target is None or lane_memory.missing >= 4):
                horizon = 0.0 if target is None else max(0.0, target[-1, 0] - 4.0)
                limit = min(limit, math.sqrt(2.0 * 2.5 * horizon))
            if lane_memory.status == 'Selected lane blocked':
                limit = 0.0

            throttle, brake, steer, safety = controller.step(
                path, occ, raw, v, dt=dt, speed_limit=limit,
                lane_tracking=offset is not None, route_preview=coarse
            )
            with state_lock:
                pacer_state['ctrl'] = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
            if offset is not None and lane_memory.world is None:
                lateral_errs.append(abs(offset))

            with col_lock:
                col_cnt = len(events)
            with state_lock:
                ue_fps = pacer_state['fps']
            dashboard.present(
                (front_img, left_img, right_img),
                args[3][0, 5].numpy(),
                output,
                nav,
                path,
                occ,
                v * 3.6,
                throttle,
                brake,
                safety,
                progress,
                ue_fps,
                ctrl_hz_ema,
                route=route,
                target=target,
                steer=steer,
                collisions=col_cnt,
            )

            if rec_slot is not None:
                rec_img = rec_slot.get()[0]
                if rec_img is not None:
                    rec_bgr = bgr(rec_img)
                    r_start = float(record_cfg.get('start_sec', 5.5))
                    r_dur = float(record_cfg.get('duration_sec', 7.5))
                    if r_start <= sim_elapsed <= r_start + r_dur:
                        comp = np.empty((498, 972, 3), dtype=np.uint8)
                        comp[:, :696] = cv2.resize(rec_bgr, (696, 498), interpolation=cv2.INTER_AREA)
                        comp[:, 696:702] = BACKGROUND
                        comp[:, 702:] = cv2.resize(dashboard.canvas, (270, 498), interpolation=cv2.INTER_AREA)
                        recorded_frames.append((sim_elapsed, cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)))
                    if record_cfg.get('hero') is not None and hero_bgr is None:
                        hero_sec = (float(record_cfg['auto_lc_sec']) + 1.8) if record_cfg.get('auto_lc') else (r_start + 0.5 * r_dur)
                        if sim_elapsed >= hero_sec:
                            hero = np.empty((738, 1440, 3), dtype=np.uint8)
                            hero[:, :1032] = cv2.resize(rec_bgr, (1032, 738), interpolation=cv2.INTER_AREA)
                            hero[:, 1032:1040] = BACKGROUND
                            hero[:, 1040:] = cv2.resize(dashboard.canvas, (400, 738), interpolation=cv2.INTER_AREA)
                            hero_bgr = hero
                    if sim_elapsed >= r_start + r_dur + 0.5:
                        reason = 'record_complete'
                        break

            if ctrl_ticks % 100 == 0:
                print(
                    f"{spec['name']} ctrl={ctrl_ticks} progress={progress:.1f}m "
                    f"spd={v*3.6:.1f}km/h ue_fps={ue_fps:.1f} ctrl_hz={ctrl_hz_ema:.1f}",
                    flush=True,
                )
            if progress >= last_progress + 0.1 or v >= 0.55:
                last_progress = max(last_progress, progress)
                stagnant_since = time.monotonic()
            with col_lock:
                if events:
                    reason = 'collision'
                    break
            if time.monotonic() - stagnant_since > 30.0:
                reason = 'stalled_30s'
                break
            if navigator.error > 45.0:
                reason = 'deviated_from_sd_corridor'
                break
            if progress >= navigator.arc[-1] - 45.0:
                reason = 'reached_end'
                break

        if dashboard.restart:
            reason = 'demo_restart'
        elif dashboard.closed:
            reason = 'demo_closed'
        wall_s = time.monotonic() - start_wall
        _save_recording(record_cfg, recorded_frames, hero_bgr)
        with col_lock:
            col_cnt = len(events)
            col_list = list(events)
        with state_lock:
            ue_fps_final = pacer_state['fps']
        metrics = dict(
            route=spec['name'],
            town=spec['town'],
            quality=dashboard.quality,
            reason=reason,
            ctrl_ticks=ctrl_ticks,
            wall_seconds=wall_s,
            ue_fps_avg=ue_fps_final,
            ctrl_hz_avg=ctrl_hz_ema,
            progress_m=progress,
            route_length_m=float(navigator.arc[-1]),
            npc_count=max(0, len(actors) - 1),
            collision_count=col_cnt,
            collisions=col_list,
            lane_offset_p95_m=float(np.percentile(lateral_errs, 95)) if lateral_errs else None,
        )
        (out_dir / f"{spec['name']}.json").write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        print(json.dumps(metrics), flush=True)
        return metrics
    finally:
        stop_pacer.set()
        pause_pacer.clear()
        if pacer_thread is not None:
            pacer_thread.join(timeout=5.0)
        for s in reversed(sensors):
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass
        for a in reversed(actors):
            try:
                a.destroy()
            except Exception:
                pass
        try:
            tm.set_synchronous_mode(False)
            world.apply_settings(old)
        except Exception:
            pass


def main(args):
    route = json.loads(args.plan.read_text(encoding='utf-8'))
    if args.route:
        matches = [s for s in route if s['name'] == args.route]
    else:
        matches = [s for s in route if s['town'] == 'Town04_Opt' and s['name'] == 'town04_ring'] or route[:1]
    if len(matches) != 1:
        raise ValueError(f'Expected one demo route matching {args.route!r}, found {len(matches)}')
    selected = dict(matches[0])
    if args.npc is not None:
        selected['npc'] = args.npc
    if args.max_ticks is not None:
        selected['max_ticks'] = args.max_ticks
    if args.cruise_kmh is not None:
        selected['cruise_kmh'] = args.cruise_kmh
    if args.npc_along_route:
        selected['npc_along_route'] = True

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'demo-plan.json').write_text(json.dumps([selected], indent=2), encoding='utf-8')
    live_dir = args.out / 'live'
    live_dir.mkdir(parents=True, exist_ok=True)

    latency = json.loads(args.latency.read_text(encoding='utf-8'))
    model = load_model(args.weights)
    if latency['folded_batch_norm']:
        fold_for_inference(model)
    sizes = locked_sizes()
    if tuple(sizes) != NATIVE_C:
        raise ValueError(f'Unexpected camera sizes {sizes}')

    title_tag = 'STAGE 2.6' if 'stage26' in str(args.weights).lower() else 'PHASE 2.5'
    view = AsyncTacticalDashboard(
        args.out,
        selected,
        quality=args.quality,
        title_tag=title_tag,
        replay_minutes=args.replay_minutes,
        exit_on_finish=args.exit_on_finish,
    )
    results = []
    try:
        with async_server(args.carla_root, live_dir, args.port, quality=args.quality) as (client, _):
            world = client.load_world(selected['town'])
            for _ in range(4):
                world.wait_for_tick(10.0)
            time.sleep(1.0)
            gpu_model, example, is_stage26 = build_engine(model, sizes)
            engine = Inference(gpu_model, example, graph=latency['selected'] == 'hip_graph')
            engine.is_stage26 = is_stage26
            record_cfg = None
            if args.record_webp or args.record_hero or args.auto_lane_change:
                record_cfg = dict(
                    webp=args.record_webp,
                    hero=args.record_hero,
                    start_sec=args.record_start_sec,
                    duration_sec=args.record_duration_sec,
                    auto_lc=args.auto_lane_change,
                    auto_lc_sec=args.auto_lane_change_sec,
                )
            run_idx = 0
            while not view.closed:
                run_idx += 1
                run_dir = live_dir / f'run-{run_idx:02d}'
                run_dir.mkdir(exist_ok=True)
                view.start_route(selected)
                res = run_async_route(
                    client, world, selected, engine, args.port, run_dir, view,
                    max_seconds=args.max_seconds, record_cfg=record_cfg,
                )
                if res is not None:
                    results.append(res)
                else:
                    res = {'reason': 'aborted', 'collision_count': 0}
                view.wait_for_restart(res)
                if args.max_seconds is not None:
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('carla_root', 'out', 'weights', 'plan', 'latency'):
        p.add_argument('--' + field.replace('_', '-'), type=Path, required=True)
    p.add_argument('--port', type=int, default=2460)
    p.add_argument('--quality', type=str, default='Epic', choices=('Low', 'Medium', 'High', 'Epic'))
    p.add_argument('--replay-minutes', type=float, default=0)
    p.add_argument('--route', type=str, default=None)
    p.add_argument('--npc', type=int, default=None)
    p.add_argument('--max-ticks', type=int, default=None)
    p.add_argument('--max-seconds', type=float, default=None)
    p.add_argument('--cruise-kmh', type=float, default=None)
    p.add_argument('--npc-along-route', action='store_true')
    p.add_argument('--exit-on-finish', action='store_true')
    p.add_argument('--record-webp', type=Path, default=None)
    p.add_argument('--record-hero', type=Path, default=None)
    p.add_argument('--record-start-sec', type=float, default=5.5)
    p.add_argument('--record-duration-sec', type=float, default=7.5)
    p.add_argument('--auto-lane-change', type=str, default=None, choices=('left', 'right'))
    p.add_argument('--auto-lane-change-sec', type=float, default=7.5)
    args = p.parse_args()
    args.carla_root = args.carla_root.resolve()
    main(args)
