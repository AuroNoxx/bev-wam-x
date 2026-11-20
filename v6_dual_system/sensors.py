"""Shared sensor geometry and time alignment for collection and deployment."""
import hashlib
import io
import json
import queue
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/v6-dual-system"
LOG = ROOT / "logs/v6-dual-system"
DT = .05
CALIBRATION = {
    "rgb": ["sensor.camera.rgb", [1.3, 0., 2.0],
            {"image_size_x": "640", "image_size_y": "360", "fov": "80",
             "sensor_tick": "0.0", "enable_postprocess_effects": "True"}],
    "lidar": ["sensor.lidar.ray_cast", [0., 0., 2.4],
              {"channels": "64", "range": "120", "rotation_frequency": "20",
               "sensor_tick": "0.0", "points_per_second": "1280000",
               "upper_fov": "10", "lower_fov": "-30", "dropoff_general_rate": "0"}],
}
CAL_ID = hashlib.sha256(json.dumps(CALIBRATION, sort_keys=True).encode()).hexdigest()


def local_client(port):
    import carla
    client=carla.Client("127.0.0.1",port)
    cache=ROOT/".cache/v6-dual-system/carla-client"
    cache.mkdir(parents=True,exist_ok=True)
    client.set_files_base_folder(str(cache))
    client.set_timeout(90)
    return client


def write_json(path, value):
    tmp = path.with_suffix(".pending")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    # Windows may briefly deny replacement while a status reader holds the file.
    for attempt in range(11):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt==10:raise
            time.sleep(.05)


def encode_rgb(rgb):
    import cv2
    ok,jpeg=cv2.imencode(".jpg",rgb[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,92])
    if not ok:raise IOError("JPEG encode failed")
    return jpeg


def model_rgb(rgb):
    from PIL import Image
    with Image.open(io.BytesIO(encode_rgb(rgb))) as im:
        return np.asarray(im).copy()


class MissingFrame(ValueError):
    pass


class Sensors:
    def __init__(self, world, ego):
        import carla
        self.actors = []
        self.queues = {}
        self.buffered = {}
        self.pending = {"rgb": {}, "lidar": {}}
        try:
            for name, (kind, xyz, attrs) in CALIBRATION.items():
                bp = world.get_blueprint_library().find(kind)
                for key, value in attrs.items():
                    bp.set_attribute(key, value)
                actor = world.spawn_actor(bp, carla.Transform(carla.Location(*xyz)), attach_to=ego)
                self.actors.append(actor)
                stream = queue.Queue()
                self.queues[name] = stream
                actor.listen(stream.put)
        except BaseException:
            self.close()
            raise

    def read(self, frame):
        deadline = time.monotonic() + 5
        out = {}
        for name, stream in self.queues.items():
            while True:
                item = self.buffered.pop(name, None)
                if item is None:
                    item = stream.get(timeout=max(.001, deadline - time.monotonic()))
                if item.frame >= frame:
                    break
            if item.frame != frame:
                self.buffered[name] = item
                raise MissingFrame(f"{name}: expected {frame}, got {item.frame}")
            if name == "rgb":
                out[name] = np.frombuffer(item.raw_data, np.uint8).reshape(360, 640, 4)[:, :, 2::-1].copy()
            else:
                out[name] = np.frombuffer(item.raw_data, np.float32).reshape(-1, 4).copy()
        return out["rgb"], out["lidar"]

    def ready_pairs(self,first_frame):
        """Drain completed readbacks without blocking the 20 Hz physics thread."""
        for name,stream in self.queues.items():
            slots=self.pending[name]
            while True:
                try:item=stream.get_nowait()
                except queue.Empty:break
                if item.frame>=first_frame:slots[item.frame]=item
            for frame in list(slots):
                if frame<first_frame:del slots[frame]
        frames=sorted(self.pending["rgb"].keys() & self.pending["lidar"].keys())
        for frame in frames:
            image=self.pending["rgb"].pop(frame)
            cloud=self.pending["lidar"].pop(frame)
            rgb=np.frombuffer(image.raw_data,np.uint8).reshape(360,640,4)[:,:,2::-1].copy()
            scan=np.frombuffer(cloud.raw_data,np.float32).reshape(-1,4).copy()
            yield frame,rgb,scan

    def close(self):
        for actor in self.actors:
            actor.stop()
            actor.destroy()


def ego_points(scan):
    # Raw CARLA sensor coordinates are forward/right/up.
    return scan[:, :3] + np.asarray(CALIBRATION["lidar"][1], np.float32)


def grid(points):
    x, right, z = points.T
    ix = np.floor(x + 16).astype(np.int32)
    iy = np.floor((-right + 19.2) / .3).astype(np.int32)
    ok = (ix >= 0) & (ix < 128) & (iy >= 0) & (iy < 128)
    cell = (iy[ok] * 128 + ix[ok])
    height = np.zeros(128 * 128, np.float32)
    np.maximum.at(height, cell, np.clip((z[ok] + .5) / 4, 0, 1))
    density = np.log1p(np.bincount(cell, minlength=128 * 128)).clip(0, 5) / 5
    return height.reshape(128, 128), density.reshape(128, 128)


def bev(scan, old_scan, pose, old_pose):
    height, density = grid(ego_points(scan))
    align = np.linalg.solve(pose, old_pose).astype(np.float32)
    history = ego_points(old_scan) @ align[:3, :3].T + align[:3, 3]
    _, old_density = grid(history)
    return np.rint(255 * np.stack((height, density, .5 + .5 * (density-old_density)))).astype(np.uint8)


def state(ego):
    vel, acc = ego.get_velocity(), ego.get_acceleration()
    forward = ego.get_transform().get_forward_vector()
    return [vel.length(), acc.x*forward.x + acc.y*forward.y + acc.z*forward.z,
            -np.deg2rad(ego.get_angular_velocity().z)]
