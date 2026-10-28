"""BEV WA-JEPA Unified Spatiotemporal Dataset.

Provides paired historical BEVs, causal observation context, multi-horizon future BEVs,
and trajectory labels with explicit qualification masks.
Directly accesses raw uncompressed/mmap episode data. No obsolete frozen latent cache.
"""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.navigation import eligible_navigation_anchors


def extract_causal_actions(applied_actions, t, num_lags=4):
    """
    Extract chassis actions strictly executed prior to decision step t:
    i.e., [a_{t-num_lags}, ..., a_{t-2}, a_{t-1}].
    Because applied_actions[k] was recorded during step k (the action executed at step k),
    at decision step t only actions up to t-1 have already been executed by the chassis.
    Actions at step >= t (including a_t) are not yet executed and strictly excluded to prevent
    causal lookahead leakage.

    Args:
        applied_actions: [N, 2] array or buffer of recorded/executed actions.
        t: int, current decision step index.
        num_lags: int, number of past steps (default 4).

    Returns:
        past_actions: [num_lags, 2] float32 array, zero-padded if t < lag.
        action_valid: [num_lags] bool array, True where t >= lag.
    """
    past_actions = np.zeros((num_lags, 2), dtype=np.float32)
    action_valid = np.zeros(num_lags, dtype=bool)

    for i, lag in enumerate(range(num_lags, 0, -1)):
        past_idx = t - lag
        if past_idx >= 0 and past_idx < len(applied_actions):
            past_actions[i] = applied_actions[past_idx]
            action_valid[i] = True

    return past_actions, action_valid


class BEWWAJointDataset(Dataset):
    """Unified multi-horizon BEV and Trajectory dataset for BEV WA-JEPA models."""

    def __init__(self, manifest_path, split="train", eligible_action_only=False, preload_ram=True):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.eligible_action_only = eligible_action_only
        self.preload_ram = preload_ram

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r") as f:
            self.manifest = json.load(f)

        if not self.manifest.get("control_verified", False):
            raise ValueError(f"Dataset {manifest_path} lacks verified control attestation")

        self.episodes = []
        self.anchors = []  # List of (episode_index, t) tuples
        self._raw_handles = {}

        for ep_idx, ep_meta in enumerate(self.manifest.get("episodes", [])):
            raw_path = Path(ep_meta["raw"])
            if not raw_path.exists():
                raise FileNotFoundError(f"Raw episode file missing: {raw_path}")

            if self.preload_ram:
                with np.load(raw_path) as raw:
                    data = {k: np.array(raw[k]) for k in (
                        "bev", "ego", "applied_action", "route_points", "route_valid",
                        "vehicle", "maneuver_allowed", "trajectory", "valid"
                    )}
                    n_bev = len(data["bev"])
                    n_states = len(raw["states"]) if "states" in raw else n_bev + 1
                    valid = data["valid"]
                    route_valid = data["route_valid"]
                    events = raw["events"]
                    noise_active = raw["noise_active"] if "noise_active" in raw else None

                    action_eligible = eligible_navigation_anchors(
                        valid, route_valid, events, noise_active=noise_active, horizon_steps=20
                    )

                    self.episodes.append({
                        "data": data,
                        "n_bev": n_bev,
                        "n_states": n_states,
                        "action_eligible": action_eligible,
                        "seed": ep_meta.get("seed", 0)
                    })
            else:
                with np.load(raw_path, mmap_mode="r") as raw:
                    n_bev = len(raw["bev"])
                    n_states = len(raw["states"]) if "states" in raw else n_bev

                    valid = raw["valid"]
                    route_valid = raw["route_valid"]
                    events = raw["events"]
                    noise_active = raw["noise_active"] if "noise_active" in raw else None

                    action_eligible = eligible_navigation_anchors(
                        valid, route_valid, events, noise_active=noise_active, horizon_steps=20
                    )

                    self.episodes.append({
                        "raw_path": raw_path,
                        "n_bev": n_bev,
                        "n_states": n_states,
                        "action_eligible": action_eligible,
                        "seed": ep_meta.get("seed", 0)
                    })

            for t in range(n_bev):
                if self.eligible_action_only:
                    if action_eligible[t]:
                        self.anchors.append((ep_idx, t))
                else:
                    self.anchors.append((ep_idx, t))

    def _get_raw(self, ep_idx):
        if self.preload_ram:
            return self.episodes[ep_idx]["data"]
        if ep_idx not in self._raw_handles:
            self._raw_handles[ep_idx] = np.load(self.episodes[ep_idx]["raw_path"], mmap_mode="r")
        return self._raw_handles[ep_idx]

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        ep_idx, t = self.anchors[index]
        ep = self.episodes[ep_idx]
        raw = self._get_raw(ep_idx)
        n_bev = ep["n_bev"]
        n_states = ep["n_states"]

        # 1. Historical BEVs (4 frames at t-3, t-2, t-1, t)
        history_indices = [max(0, t - lag) for lag in (3, 2, 1, 0)]
        history_bev = np.asarray(raw["bev"][history_indices])  # [4, 4, 128, 128] uint8
        history_valid = np.array([t >= lag for lag in (3, 2, 1, 0)], dtype=bool)

        # 2. Historical Motion & Applied Actions
        ego_indices = history_indices
        ego_motion = np.asarray(raw["ego"][ego_indices], dtype=np.float32)  # [4, 2] (vx/10., yaw_rate)
        applied_action, action_valid = extract_causal_actions(raw["applied_action"], t, num_lags=4)

        # 3. Observation Context
        route_points = np.asarray(raw["route_points"][t], dtype=np.float32)  # [8, 2]
        route_valid = np.asarray(raw["route_valid"][t], dtype=bool)  # [8]
        vehicle = np.asarray(raw["vehicle"][t], dtype=np.float32)  # [4]
        maneuver_allowed = np.asarray(raw["maneuver_allowed"][t], dtype=bool)  # [2]

        # 4. Multi-Horizon Future BEVs (+0.5s, +1.0s, +1.5s, +2.0s -> steps +5, +10, +15, +20)
        future_steps = (5, 10, 15, 20)
        future_bev = np.zeros((4, 4, 128, 128), dtype=np.uint8)
        future_bev_mask = np.zeros(4, dtype=bool)

        for slot, step_fwd in enumerate(future_steps):
            future_t = t + step_fwd
            if future_t < n_bev:
                future_bev[slot] = raw["bev"][future_t]
                future_bev_mask[slot] = True

        # 5. Future Trajectory (10 waypoints at +0.2s, ..., +2.0s -> steps +2, +4, ..., +20)
        trajectory = np.asarray(raw["trajectory"][t], dtype=np.float32)  # [10, 4]
        traj_valid = np.asarray(raw["valid"][t], dtype=bool)  # [10]

        future_traj_mask = np.zeros(10, dtype=bool)
        for j in range(10):
            step_traj = 2 * (j + 1)
            if (t + step_traj < n_states) and traj_valid[j]:
                future_traj_mask[j] = True

        # 6. Qualification Flags
        world_mask = True
        action_mask = bool(ep["action_eligible"][t])

        return {
            "history_bev": torch.from_numpy(history_bev),  # [4, 4, 128, 128] uint8
            "history_valid": torch.from_numpy(history_valid),  # [4] bool
            "ego_motion": torch.from_numpy(ego_motion),  # [4, 2] float32
            "applied_action": torch.from_numpy(applied_action),  # [4, 2] float32
            "action_valid": torch.from_numpy(action_valid),  # [4] bool
            "route_points": torch.from_numpy(route_points),  # [8, 2] float32
            "route_valid": torch.from_numpy(route_valid),  # [8] bool
            "vehicle": torch.from_numpy(vehicle),  # [4] float32
            "maneuver_allowed": torch.from_numpy(maneuver_allowed),  # [2] bool
            "future_bev": torch.from_numpy(future_bev),  # [4, 4, 128, 128] uint8
            "future_bev_mask": torch.from_numpy(future_bev_mask),  # [4] bool
            "trajectory": torch.from_numpy(trajectory),  # [10, 4] float32
            "future_traj_mask": torch.from_numpy(future_traj_mask),  # [10] bool
            "world_mask": torch.tensor(world_mask, dtype=torch.bool),
            "action_mask": torch.tensor(action_mask, dtype=torch.bool),
            "step": torch.tensor(t, dtype=torch.long),
            "episode_seed": torch.tensor(ep["seed"], dtype=torch.long)
        }

    def stats(self):
        raw_total_steps = sum(ep["n_bev"] for ep in self.episodes)
        eligible_action_count = sum(int(ep["action_eligible"].sum()) for ep in self.episodes)
        return {
            "episodes": len(self.episodes),
            "raw_total_steps": raw_total_steps,
            "dataset_anchors": len(self.anchors),
            "eligible_action_anchors": eligible_action_count,
            "action_qualification_rate": round(eligible_action_count / max(1, raw_total_steps), 4)
        }
