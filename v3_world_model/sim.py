"""Shared synchronous CARLA 1.0 sensor entry points."""
import queue
from contextlib import contextmanager
from time import monotonic

import numpy as np

from .geometry import CAMERAS, CFG, DT

COLLECTION_MAPS = tuple(CFG["collection_maps"])


def require_collection_map(value):
    town = value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if town.casefold() == "town04_opt" or town not in COLLECTION_MAPS:
        raise ValueError(f"{value!r} is prohibited for Stage 3 capture; allowed: {COLLECTION_MAPS}")
    return town


def connect(host, port, requested_map):
    require_collection_map(requested_map)  # Reject before importing/contacting CARLA.
    import carla
    client = carla.Client(host, port)
    client.set_timeout(45.)
    server = client.get_server_version()
    if client.get_client_version() != server or server.split("-", 1)[0] != "1.0":
        raise RuntimeError(f"Expected matching CARLA 1.0 client and server, got {client.get_client_version()} / {server}")
    world = client.get_world()
    actual = require_collection_map(world.get_map().name)
    if actual != requested_map:
        raise ValueError(f"Load {requested_map} before capture (active: {actual})")
    return carla, client, world, server


def matrix(transform):
    return np.asarray(transform.get_matrix(), dtype=np.float64)


@contextmanager
def synchronous(world, traffic=None, seed=0):
    original = world.get_settings()
    weather = world.get_weather()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.no_rendering_mode = False
    settings.fixed_delta_seconds = DT
    settings.substepping = True
    settings.max_substep_delta_time = CFG["physics"]["max_substep_delta_time"]
    settings.max_substeps = CFG["physics"]["max_substeps"]
    try:
        world.apply_settings(settings)
        if traffic is not None:
            traffic.set_synchronous_mode(True)
            traffic.set_random_device_seed(seed)
        yield
    finally:
        try:
            if traffic is not None:
                traffic.set_synchronous_mode(original.synchronous_mode)
        finally:
            world.apply_settings(original)
            world.set_weather(weather)


class Rig:
    def __init__(self, carla, world, ego):
        self.carla, self.world, self.ego = carla, world, ego
        self.library = world.get_blueprint_library()
        self.sensors, self.queues = [], {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            for actor in reversed(self.sensors):
                actor.stop()
        finally:
            for actor in reversed(self.sensors):
                actor.destroy()

    def attach(self, name, blueprint, transform, attributes, callback=None):
        bp = self.library.find(blueprint)
        for k, v in attributes.items():
            if not bp.has_attribute(k):
                raise RuntimeError(f"{blueprint} missing locked attribute {k}")
            bp.set_attribute(k, str(v).lower())
        actor = self.world.spawn_actor(bp, transform, attach_to=self.ego)
        self.sensors.append(actor)
        if callback is None:
            self.queues[name] = queue.Queue()
            callback = self.queues[name].put
        actor.listen(callback)

    def cameras(self, kind="rgb"):
        for name, cam in CAMERAS.items():
            x, y, z, yaw, pitch, fov = cam["pose"]
            width, height = cam["size"]
            tf = self.carla.Transform(self.carla.Location(x, y, z),
                                      self.carla.Rotation(yaw=yaw, pitch=pitch))
            attrs = dict(image_size_x=width, image_size_y=height, fov=fov, sensor_tick=0)
            if kind == "rgb":
                attrs["enable_postprocess_effects"] = CFG["rgb"]["enable_postprocess_effects"]
            self.attach(f"{kind}/{name}", f"sensor.camera.{kind}", tf, attrs)

    def lidar(self, semantic=False):
        cfg = CFG["lidar"]
        tf = self.carla.Transform(self.carla.Location(*cfg["pose_xyz"]))
        attrs = {k: v for k, v in cfg.items() if k != "pose_xyz"}
        attrs["sensor_tick"] = 0
        if not semantic:
            attrs["dropoff_general_rate"] = 0
        self.attach("semantic_lidar" if semantic else "lidar",
                    "sensor.lidar.ray_cast_semantic" if semantic else "sensor.lidar.ray_cast", tf, attrs)

    def tick(self):
        fid = self.world.tick(45)
        snapshot = self.world.get_snapshot()
        if snapshot.frame != fid:
            raise RuntimeError(f"World frame {snapshot.frame} != tick {fid}; use one clock owner")
        bundle = {}
        deadline = monotonic() + 30
        for name, inbox in self.queues.items():
            while True:
                frame = inbox.get(timeout=max(deadline - monotonic(), .001))
                if frame.frame < fid:
                    continue
                if frame.frame != fid or abs(frame.timestamp - snapshot.timestamp.elapsed_seconds) > .001:
                    raise RuntimeError(f"Sensor {name} skipped/misaligned at {fid}: {frame.frame}")
                bundle[name] = frame
                break
        return snapshot, bundle


def bgr(image):
    return np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)[..., :3]


def depth_cm(image):
    channels = bgr(image).astype(np.uint32)
    packed = (channels[..., 0] << 16) | (channels[..., 1] << 8) | channels[..., 2]
    return np.minimum(np.rint(packed * (100_000. / 16777215.)), 65535).astype(np.uint16)
