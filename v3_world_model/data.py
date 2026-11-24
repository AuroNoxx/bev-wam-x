"""Read verified extracted Stage 3.1 clips with disjoint observation and target namespaces."""
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import H, NAMES, SIZES, W
from .observations import OBS_FIELDS, validate_observation
from .pack_colab_v3 import read_manifest, unpack_targets
from .perception_model import CLIP_LEN
from .qc_v3 import load_npz, read_image

TRAIN_TARGETS = ("lanes", "vectors", "vector_valid", "vector_exists", "boxes", "occupancy",
                 "future_occupancy", "trajectory")


def color_jitter_table(*, strong=False):
    """One RGB transform shared by every camera and frame in a training clip."""
    if strong:
        brightness = np.random.uniform(.82, 1.18)
        contrast = np.random.uniform(.82, 1.18)
        gamma = np.random.uniform(.85, 1.15)
        color = np.random.uniform(.90, 1.10, size=3)
        base = (np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0
        levels = (base[:, None] - 127.5) * contrast + 127.5
        return np.rint((levels * brightness * color).clip(0, 255)).astype(np.uint8)[None]
    brightness = np.random.uniform(.92, 1.08)
    contrast = np.random.uniform(.88, 1.12)
    color = np.random.uniform(.94, 1.06, size=3)
    levels = (np.arange(256, dtype=np.float32)[:, None] - 127.5) * contrast + 127.5
    return np.rint((levels * brightness * color).clip(0, 255)).astype(np.uint8)[None]


def _shadow_mask_for_clip():
    """Sample clip-consistent normalized trapezoidal shadow specs on lower 55% road zone."""
    specs = []
    for cam_idx in range(len(NAMES)):
        if np.random.random() < 0.30:
            x0, x1 = sorted(np.random.uniform(0.05, 0.95, size=2).tolist())
            x2, x3 = sorted(np.random.uniform(0.05, 0.95, size=2).tolist())
            y_top = float(np.random.uniform(0.45, 0.72))
            y_bot = float(np.random.uniform(min(0.98, y_top + 0.14), 1.0))
            atten = float(np.random.uniform(0.55, 0.80))
            specs.append((np.array([[x0, y_top], [x1, y_top], [x3, y_bot], [x2, y_bot]], dtype=np.float32), atten))
        else:
            specs.append(None)
    # Independent 5% dropout per side camera (never front camera 0).
    cam_drop = [False] + [bool(np.random.random() < 0.05) for _ in range(1, len(NAMES))]
    return specs, cam_drop


def _apply_shadow_and_drop(image_rgb, spec, dropped):
    if dropped:
        return np.zeros_like(image_rgb)
    if spec is None:
        return image_rgb
    poly_norm, atten = spec
    h, w = image_rgb.shape[:2]
    poly_px = np.rint(poly_norm * np.array([w, h], dtype=np.float32)).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(mask, [poly_px], 1.0)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    scale = (1.0 - (1.0 - atten) * mask)[..., None]
    return np.rint((image_rgb.astype(np.float32) * scale).clip(0, 255)).astype(np.uint8)


class Clips(Dataset):
    def __init__(self, root, split, stride=CLIP_LEN, *, load_maptr=False, strong_aug=False):
        self.root = Path(root)
        self.training = split == "train"
        self.load_maptr = bool(load_maptr)
        self.strong_aug = bool(strong_aug)
        manifest = read_manifest(self.root / "manifest.json", check_shards=False)
        selected = set(manifest["splits"][split])
        self.entries = {e["id"]: e for e in manifest["episodes"] if e["id"] in selected}
        self.starts = [(episode, i) for episode, entry in self.entries.items()
                       for i in range(0, len(entry["frame_ids"]) - CLIP_LEN + 1, stride)]
        self.target_cache = OrderedDict()
        self.maptr_cache = OrderedDict()
        if not self.starts:
            raise ValueError(f"No complete clips in {split}")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        episode, start = self.starts[index]
        entry = self.entries[episode]
        if episode not in self.target_cache:
            self.target_cache[episode] = load_npz(self.root / episode / "targets.npz")
            if len(self.target_cache) > 8:
                self.target_cache.popitem(last=False)
        self.target_cache.move_to_end(episode)
        if self.load_maptr:
            if episode not in self.maptr_cache:
                mpath = self.root / episode / "maptr_targets.npz"
                if not mpath.is_file():
                    raise FileNotFoundError(f"Missing Stage 3.1c MapTR overlay file: {mpath}")
                self.maptr_cache[episode] = load_npz(mpath)
                if len(self.maptr_cache) > 8:
                    self.maptr_cache.popitem(last=False)
            self.maptr_cache.move_to_end(episode)

        rgb = [[] for _ in NAMES]
        inputs = {k: [] for k in OBS_FIELDS}
        targets = []
        jitter = color_jitter_table(strong=self.strong_aug) if self.training else None
        shadow_specs, cam_drop = _shadow_mask_for_clip() if (self.training and self.strong_aug) else ([None] * len(NAMES), [False] * len(NAMES))

        for i in range(start, start + CLIP_LEN):
            folder = self.root / episode / str(entry["frame_ids"][i])
            for cam_idx, (n, (w, h), bag) in enumerate(zip(NAMES, SIZES, rgb)):
                image = read_image(folder / f"{n}.webp")
                if image.shape != (h, w, 3) or image.dtype != np.uint8:
                    raise ValueError(f"Invalid packed image: {folder}/{n}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if jitter is not None:
                    image = cv2.LUT(image, jitter)
                if self.training and self.strong_aug:
                    image = _apply_shadow_and_drop(image, shadow_specs[cam_idx], cam_drop[cam_idx])
                bag.append(image.transpose(2, 0, 1))
            obs = load_npz(folder / "obs.npz")
            validate_observation(obs)
            for k in inputs:
                inputs[k].append(obs[k])
            tgt = unpack_targets(self.target_cache[episode], i)
            if self.load_maptr:
                ov = self.maptr_cache[episode]
                tgt["lanes_cont"] = np.unpackbits(ov["lanes_cont"][i], count=4 * H * W, bitorder="little").reshape(4, H, W)
                tgt["map_polylines"] = ov["map_polylines"][i].astype(np.float32)
                tgt["map_classes"] = ov["map_classes"][i].astype(np.int64)
                tgt["map_valid"] = ov["map_valid"][i].astype(bool)
            targets.append(tgt)
        batch = {k: torch.from_numpy(np.stack(values)) for k, values in inputs.items()}
        batch["rgb"] = tuple(torch.from_numpy(np.stack(im)) for im in rgb)
        target_keys = TRAIN_TARGETS + (("lanes_cont", "map_polylines", "map_classes", "map_valid") if self.load_maptr else ())
        return dict(inputs=batch, targets={k: torch.from_numpy(np.stack([t[k] for t in targets])) for k in target_keys})
