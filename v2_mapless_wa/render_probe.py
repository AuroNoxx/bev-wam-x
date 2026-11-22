"""Measure the CARLA 1.0 rendering tier and audit straight-road RGB alignment."""
import argparse
import json
import queue
import socket
import subprocess
import time
from pathlib import Path

import carla
import cv2
import numpy as np

from .collect import DT, SEM, check_versions, matrix, read
from .geometry import CAMS, NAMES, X, Y, Z, pixels, projection_tables, road_surface

TIERS = ('High', 'Medium')
WINDOW_FLAGS = ('-windowed', '-ResX=640', '-ResY=360')
POSTPROCESS = 'False'
FRAME_COUNT = 20
LOCKED_CONFIG = Path(__file__).resolve().with_name('config.yaml')


def locked_sizes(path=LOCKED_CONFIG):
    import yaml
    rgb = yaml.safe_load(path.read_text(encoding='utf-8'))['rgb']
    return tuple(tuple(rgb['sizes'][name]) for name in NAMES)


def locked_tier(path=LOCKED_CONFIG):
    import yaml

    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    rendering, rgb = cfg['rendering'], cfg['rgb']
    tier = rendering['quality_level']
    expected = dict(quality_level=tier, renderer='dx12', windowed=True, window_width=640, window_height=360)
    if tier not in TIERS or rendering != expected or cfg['simulator'] != 'CARLA 1.0' or cfg['engine'] != 'UE 5.8':
        raise ValueError(f'Unsupported CARLA rendering configuration in {path}')
    if (locked_sizes(path) != ((768,448),(384,224),(384,224)) or rgb['enable_postprocess_effects'] or
            rgb['preprocessing_gamma'] != .5 or rgb['stored_rgb'] != 'raw' or tuple(rgb['projection_heights_m']) != Z):
        raise ValueError(f'RGB size or postprocessing differs from the measured settings in {path}')
    if (tuple(tuple(rgb[name]) for name in NAMES) != CAMS or cfg['lidar'] != dict(channels=64, rotation_hz=20)
            or cfg['fixed_delta_seconds'] != DT):
        raise ValueError(f'Sensor rig or timing differs from the measured settings in {path}')
    return tier


def camera_blueprint(library, fov, size=(384,224)):
    bp = library.find('sensor.camera.rgb')
    for key, value in (('image_size_x', size[0]), ('image_size_y', size[1]), ('fov', fov),
                       ('sensor_tick', 0), ('enable_postprocess_effects', POSTPROCESS)):
        bp.set_attribute(key, str(value))
    return bp


def lidar_blueprint(library, semantic):
    name = 'sensor.lidar.ray_cast_semantic' if semantic else 'sensor.lidar.ray_cast'
    bp = library.find(name)
    for key, value in dict(range=88, channels=64, rotation_frequency=20, points_per_second=640000,
                           upper_fov=12, lower_fov=-30, sensor_tick=0).items():
        bp.set_attribute(key, str(value))
    if bp.has_attribute('dropoff_general_rate'):
        bp.set_attribute('dropoff_general_rate', '0')
    return bp


def bgr(image, size=None):
    if size is not None and (image.width, image.height) != size:
        raise RuntimeError(f'Unexpected RGB size on frame {image.frame}')
    return np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)[:, :, :3].copy()


def save_triptych(out, images, tier, fid):
    panels = []
    for i in (1, 0, 2):
        panel = cv2.copyMakeBorder(images[i], 0, 24, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))
        text = f'{tier}  {NAMES[i]}  frame {fid}'
        cv2.putText(panel, text, (6, 241), cv2.FONT_HERSHEY_SIMPLEX, .43, (235, 235, 235), 1, cv2.LINE_AA)
        panels.append(panel)
    if not cv2.imwrite(str(out), np.hstack(panels)):
        raise IOError(out)


def capture(world, calibration, alternate_straight):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    weather = carla.WeatherParameters.ClearNoon
    if alternate_straight:
        weather.sun_altitude_angle = 80
    world.set_weather(weather)
    road = world.get_map()
    if road.name.rsplit('/', 1)[-1] != 'Town10HD_Opt':
        raise RuntimeError(road.name)
    xy = (-17, 16.6) if alternate_straight else (20.5, 13.2)
    start = road.get_waypoint(carla.Location(*xy, 0.0))
    if alternate_straight:
        successor = start.next(8)
        if len(successor) != 1 or abs((successor[0].transform.rotation.yaw - start.transform.rotation.yaw + 180) % 360 - 180) > 2:
            raise RuntimeError('The alternate calibration path is no longer straight')
    lib = world.get_blueprint_library()
    tf = start.transform
    tf.location.z += .38
    ego = world.spawn_actor(lib.find('vehicle.lincoln.mkz'), tf)
    world.get_spectator().set_transform(carla.Transform(
        tf.location + carla.Location(x=5, z=3), carla.Rotation(yaw=180, pitch=-12)))
    sensors, queues, mounts = [], [], []

    def attach(bp, mount):
        sensor = world.spawn_actor(bp, mount, attach_to=ego)
        sensor_queue = queue.Queue()
        sensors.append(sensor)
        queues.append(sensor_queue)
        sensor.listen(sensor_queue.put)

    try:
        for x, y, z, yaw, pitch, fov in CAMS:
            mount = carla.Transform(carla.Location(x, y, z), carla.Rotation(pitch=pitch, yaw=yaw))
            attach(camera_blueprint(lib, fov), mount)
            mounts.append(matrix(mount))
        for semantic in (False, True) if calibration else (False,):
            attach(lidar_blueprint(lib, semantic), carla.Transform(carla.Location(z=2.2)))
        ego.apply_control(carla.VehicleControl(throttle=.28))
        warm_start = time.time()
        for _ in range(40):
            fid = world.tick(45)
            for sensor_queue in queues:
                read(sensor_queue, fid)
        frames = []
        capture_start = time.time()
        for _ in range(FRAME_COUNT):
            start_tick = time.perf_counter()
            fid = world.tick(45)
            snapshot = world.get_snapshot()
            bundle = [read(sensor_queue, fid) for sensor_queue in queues]
            if snapshot.frame != fid or any(x.frame != fid for x in bundle):
                raise RuntimeError(f'Sensor or world frame mismatch at {fid}')
            pose = matrix(snapshot.find(ego.id).get_transform())
            frames.append((fid, [bgr(x) for x in bundle[:3]], bundle[3:], pose,
                           time.perf_counter() - start_tick))
        capture_end = time.time()
        assert np.all(np.diff([f[0] for f in frames]) == 1)
        if alternate_straight:
            poses = np.stack([f[3] for f in frames])
            yaw = np.rad2deg(np.unwrap(np.arctan2(poses[:, 1, 0], poses[:, 0, 0])))
            if np.ptp(yaw) > 2 or np.any(np.linalg.norm(poses[:, :2, 3] - [tf.location.x, tf.location.y], axis=1) > 8):
                raise RuntimeError('Retained poses left the surveyed straight segment')
        return frames, mounts, warm_start, capture_start, capture_end
    finally:
        for sensor in reversed(sensors):
            sensor.stop()
            sensor.destroy()
        ego.destroy()


def signed_pairs(front, side):
    """Use side-minus-front orientation for both nearest-neighbor directions."""
    a, b = np.argwhere(front), np.argwhere(side)
    if not len(a) or not len(b):
        return np.empty((0, 2))
    delta = b[None, :, :] - a[:, None, :]
    d2 = (delta ** 2).sum(2)
    closest = np.concatenate((delta[np.arange(len(a)), d2.argmin(1)],
                              delta[d2.argmin(0), np.arange(len(b))]))
    return -closest  # Increasing grid row/column means decreasing forward/left in ego FLU.


def white_pixels(image):
    # Disabling postprocessing changes exposure. Measure neutral local highlights in native bytes.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    background = cv2.GaussianBlur(gray, (0, 0), 3)
    return (gray >= 8) & (gray - background >= np.maximum(3, .2 * background)) & (
        np.ptp(image, axis=2) <= np.maximum(4, .3 * gray))


def white_line_alignment(frames, mounts):
    roi = (X >= 3) & (X <= 25) & (abs(Y) < 8)
    matches = [[], []]
    contributing = [0, 0]
    for fid, images, scans, _, _ in frames:
        ordinary = np.frombuffer(scans[0].raw_data, np.float32).reshape(-1, 4).copy()
        ordinary[:, 1] *= -1
        ordinary[:, 2] += 2.2
        road_z, _, _ = road_surface(ordinary)
        uv, visible = projection_tables(mounts, (road_z,))
        sem = np.frombuffer(scans[1].raw_data, SEM)
        paint = sem['xyz'][sem['tag'] == int(carla.CityObjectLabel.RoadLines)].copy()
        paint[:, 1] *= -1
        paint[:, 2] += 2.2
        ii = np.rint(pixels(paint[:, :2])).astype(int)
        in_grid = ((ii >= 0) & (ii < 128)).all(axis=1)
        support = np.zeros((128, 128), np.uint8)
        on_plane = np.zeros(len(ii), bool)
        in_bounds = ii[in_grid]
        on_plane[in_grid] = abs(paint[in_grid, 2] - road_z[in_bounds[:, 1], in_bounds[:, 0]]) <= .12
        ii = ii[on_plane]
        support[ii[:, 1], ii[:, 0]] = 1
        support = cv2.dilate(support, np.ones((3, 3), np.uint8)).astype(bool)
        projected = []
        for image, grid in zip(images, uv[0]):
            white = white_pixels(image)
            projected.append(cv2.remap(white.astype(np.uint8), grid, None, cv2.INTER_NEAREST).astype(bool))
        for i in (1, 2):
            overlap = roi & support & visible[0, 0] & visible[0, i]
            offsets = signed_pairs(projected[0] & overlap, projected[i] & overlap)
            if len(offsets):
                matches[i-1].append(offsets)
                contributing[i-1] += 1

    def summary(offsets):
        if not len(offsets):
            return dict(n=0, p50_cells=None, mean_dx_cells=None, mean_dy_cells=None)
        dx, dy = offsets.mean(axis=0)
        return dict(n=len(offsets), p50_cells=float(np.median(np.linalg.norm(offsets, axis=1))),
                    mean_dx_cells=float(dx), mean_dy_cells=float(dy))

    pairs = [np.concatenate(samples) if samples else np.empty((0, 2)) for samples in matches]
    result = summary(np.concatenate(pairs))
    result['front_left'] = dict(summary(pairs[0]), frames=contributing[0])
    result['front_right'] = dict(summary(pairs[1]), frames=contributing[1])
    result['passed'] = bool(all(contributing) and result['p50_cells'] <= 1.0
                            and abs(result['mean_dx_cells']) <= .6 and abs(result['mean_dy_cells']) <= .6)
    return result


def run(carla_root, out, port, tier, calibration, alternate_straight):
    out.mkdir(parents=True, exist_ok=False)
    exe = carla_root / 'CarlaUnreal/Binaries/Win64/CarlaUnreal-Win64-Shipping.exe'
    args = [str(exe), *WINDOW_FLAGS, '-dx12', '-nosound', f'-carla-rpc-port={port}',
            f'-quality-level={tier}', '-stdout', '-FullStdOutLogOutput']
    with (out / 'server.log').open('wb') as log:
        server = subprocess.Popen(args, cwd=carla_root, stdout=log, stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        (out / 'server.pid').write_text(str(server.pid))
        try:
            deadline = time.monotonic() + 150
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'CARLA exited with {server.returncode}. See {out / "server.log"}')
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=1):
                        pass
                    client = carla.Client('127.0.0.1', port)
                    client.set_timeout(2)
                    client.get_server_version()
                    break
                except (OSError, RuntimeError):
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(.3)
            client.set_timeout(60)
            client_ver, server_ver = check_versions(client)
            print(f'{tier}: connected to CARLA {server_ver}, PID {server.pid}', flush=True)
            world = client.get_world()
            if world.get_map().name.rsplit('/',1)[-1] != 'Town10HD_Opt':
                world = client.load_world('Town10HD_Opt')
            frames, mounts, warm, begin, end = capture(world, calibration, alternate_straight)
            path = out / 'triptych.png'
            fid, images, *_ = frames[len(frames)//2]
            save_triptych(path, images, tier, fid)
            ticks = np.asarray([f[-1] for f in frames])
            metrics = dict(tier=tier, pid=server.pid, client=client_ver, server=server_ver,
                           map=world.get_map().name, flags=args[1:], postprocess=False,
                           first_frame=frames[0][0], last_frame=frames[-1][0], dropped_frames=0,
                           warm_start=warm, capture_start=begin, capture_end=end,
                           retained_fps=len(frames)/ticks.sum(), ticks_s=ticks.tolist(),
                           median_tick_ms=float(np.median(ticks)*1000), p95_tick_ms=float(np.percentile(ticks,95)*1000),
                           max_tick_ms=float(ticks.max()*1000))
            if calibration:
                metrics['world_positions'] = [frame[3][:3, 3].tolist() for frame in frames]
                metrics['alternate_straight'] = alternate_straight
                metrics['white_line_alignment'] = white_line_alignment(frames, mounts)
                for fid, images, scans, pose, _ in frames:
                    np.savez_compressed(out / f'{fid:07d}.npz', rgb=np.stack(images)[..., ::-1],
                                        lidar=np.frombuffer(scans[0].raw_data, np.float32).reshape(-1,4),
                                        semantic=np.frombuffer(scans[1].raw_data, SEM),
                                        world_from_ego=pose, extrinsics=mounts)
            (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
            print(json.dumps(metrics, indent=2), flush=True)
        finally:
            if server.poll() is None:
                server.terminate()
            server.wait(timeout=15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--carla-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--port', type=int, default=2460)
    quality = parser.add_mutually_exclusive_group(required=True)
    quality.add_argument('--tier', choices=TIERS)
    quality.add_argument('--config', type=Path)
    parser.add_argument('--calibration', action='store_true')
    parser.add_argument('--alternate-straight', action='store_true')
    args = parser.parse_args()
    tier = locked_tier(args.config) if args.config else args.tier
    run(args.carla_root.resolve(), args.out.resolve(), args.port, tier, args.calibration, args.alternate_straight)
