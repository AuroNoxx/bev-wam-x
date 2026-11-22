"""CARLA world/rig are left-handed. Stored spatial observations and labels use FLU."""
import hashlib
import math
from pathlib import Path

import numpy as np
import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")
# Text mode normalizes checkout line endings before hashing. Windows and Colab share this identity.
_config_text = CONFIG_PATH.read_text(encoding="utf-8")
CONFIG_SHA256 = hashlib.sha256(_config_text.encode("utf-8")).hexdigest()
CFG = yaml.safe_load(_config_text)
CAMERAS = CFG["rgb"]["cameras"]
NAMES = tuple(CAMERAS)
SIZES = tuple(tuple(cam["size"]) for cam in CAMERAS.values())
HEIGHTS = tuple(CFG["rgb"]["projection_heights_m"])
DT = CFG["fixed_delta_seconds"]
H, W = CFG["bev"]["shape"]
X_MIN, X_MAX = CFG["bev"]["x_bounds_m"]
Y_MIN, Y_MAX = CFG["bev"]["y_bounds_m"]
DX, DY = (X_MAX - X_MIN) / H, (Y_MAX - Y_MIN) / W
Z_EDGES = np.asarray(CFG["bev"]["occupancy_z_edges_m"])
FUTURE_TICKS = tuple(round(t / DT) for t in CFG["bev"]["future_seconds"])
Z = len(Z_EDGES) - 1
FLU = np.array([1., -1., 1.])


def local(world_xyz, pose):
    return ((world_xyz - pose[:3, 3]) @ pose[:3, :3]) * FLU


def world_at(ego_xyz, pose):
    return (np.asarray(ego_xyz) * FLU) @ pose[:3, :3].T + pose[:3, 3]


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def pose_yaw(pose):
    return math.atan2(pose[1, 0], pose[0, 0])


def ego_delta(previous, current):
    """v2 convention: previous origin in current FLU and previous minus current CARLA yaw."""
    dx, dy, _ = local(previous[:3, 3], current)
    return np.asarray((dx, dy, wrap(pose_yaw(previous) - pose_yaw(current))), dtype=np.float32)


def recover_previous(delta, current, previous_local_z=0.0):
    """Return previous world origin and CARLA yaw, given its local height."""
    dx, dy, dyaw = delta
    return world_at((dx, dy, previous_local_z), current), wrap(pose_yaw(current) + dyaw)


def delta_error(delta, previous, current):
    z = local(previous[:3, 3], current)[2]
    origin, yaw = recover_previous(delta, current, z)
    return max(np.max(np.abs(origin - previous[:3, 3])), abs(wrap(yaw - pose_yaw(previous))))


def pixels(xy):
    """Metric FLU xy -> fractional column, row. Integers are cell centers."""
    x, y = np.moveaxis(np.asarray(xy), -1, 0)
    return np.stack(((Y_MAX - y) / DY - .5, (X_MAX - x) / DX - .5), axis=-1)


def metric(uv):
    col, row = np.moveaxis(np.asarray(uv), -1, 0)
    return np.stack((X_MAX - (row + .5) * DX, Y_MAX - (col + .5) * DY), axis=-1)


def grid(shape=(H, W)):
    rows, cols = shape
    u, v = np.meshgrid((np.arange(cols) + .5) * W / cols - .5,
                       (np.arange(rows) + .5) * H / rows - .5)
    return metric(np.stack((u, v), axis=-1))


def carla_pose(x=0., y=0., z=0., yaw=0., pitch=0., roll=0.):
    """CARLA Transform.get_matrix convention with angles in degrees."""
    sy, sp, sr = np.sin(np.deg2rad([yaw, pitch, roll]))
    cy, cp, cr = np.cos(np.deg2rad([yaw, pitch, roll]))
    return np.array([[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x],
                     [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
                     [sp, -cp * sr, cp * cr, z],
                     [0, 0, 0, 1]], dtype=np.float64)


def camera_mounts():
    return np.stack([carla_pose(*cam["pose"][:5]) for cam in CAMERAS.values()])


def project(xyz_flu, camera, mount):
    width, height = CAMERAS[camera]["size"]
    focal = width / (2 * np.tan(np.deg2rad(CAMERAS[camera]["pose"][-1] / 2)))
    native = (xyz_flu * FLU - mount[:3, 3]) @ mount[:3, :3]
    depth = native[..., 0]
    safe_depth = np.maximum(depth, 1e-6)
    uv = np.stack((width / 2 + focal * native[..., 1] / safe_depth,
                   height / 2 - focal * native[..., 2] / safe_depth), axis=-1)
    valid = ((depth > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < width - 1) &
             (uv[..., 1] >= 0) & (uv[..., 1] < height - 1))
    return uv, valid, depth


def projection_tables(mounts=None, shape=(H, W)):
    xy = grid(shape)
    planes = np.concatenate((np.broadcast_to(xy, (len(HEIGHTS), *xy.shape)),
                             np.broadcast_to(np.asarray(HEIGHTS)[:, None, None, None],
                                             (len(HEIGHTS), *shape, 1))), axis=-1)
    mounts = camera_mounts() if mounts is None else mounts
    projected = [project(planes, name, mount) for name, mount in zip(NAMES, mounts, strict=True)]
    uv = np.stack([np.where(valid[..., None], uv, -1) for uv, valid, _ in projected], axis=1)
    return uv.astype(np.float32), np.stack([p[1] for p in projected], axis=1)


def lidar_bev(scan):
    """Six v2 channels: lowest/highest Z, two height densities, mean intensity, log count."""
    ij = np.floor(pixels(scan[:, :2]) + .5).astype(np.int64)
    keep = ((ij >= 0) & (ij < (W, H))).all(axis=1)
    col, row = ij[keep].T
    index = row * W + col
    z, intensity = scan[keep, 2:].T
    n = np.bincount(index, minlength=H * W)
    lo, hi = np.full(H * W, np.inf), np.full(H * W, -np.inf)
    np.minimum.at(lo, index, z)
    np.maximum.at(hi, index, z)
    lo[n == 0] = hi[n == 0] = 0

    def density(select):
        return np.bincount(index[select], minlength=H * W) / (DX * DY)

    mean = np.bincount(index, weights=intensity, minlength=H * W) / np.maximum(n, 1)
    return np.stack((lo, hi, density((z >= .2) & (z < 1.2)),
                     density((z >= 1.2) & (z <= 2.5)), mean, np.log1p(n))
                    ).reshape(6, H, W).astype(np.float16)
