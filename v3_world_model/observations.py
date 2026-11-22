"""Stage 3.1 online inputs shared by the collector, pack reader and inference tools."""
import numpy as np

from .geometry import H, W, NAMES

AUDIT_SENSORS = tuple(f"rgb/{name}" for name in NAMES) + ("lidar",)
COMMANDS = ("follow", "left", "right")
OBS_FIELDS = {
    "lidar_bev": ((6, H, W), np.float16),
    "ego_delta": ((3,), np.float32),
    "nav_anchors": ((3, 3), np.float32),
    "ego_speed": ((2,), np.float32),  # speed (m/s), FLU yaw rate (rad/s)
    "cmd": ((3,), np.float32),
}


def command_vector(command):
    return np.eye(len(COMMANDS), dtype=np.float32)[COMMANDS.index(command)]


def validate_observation(obs):
    if obs.keys() != OBS_FIELDS.keys():
        raise ValueError("Invalid Stage 3.1 observation keys")
    for name, (shape, dtype) in OBS_FIELDS.items():
        value = obs[name]
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"Invalid Stage 3.1 observation field {name}")
    cmd = obs["cmd"]
    if obs["ego_speed"][0] < 0 or np.count_nonzero(cmd) != 1 or cmd.max() != 1:
        raise ValueError("Invalid speed or one-hot command")
