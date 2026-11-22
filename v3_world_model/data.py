"""Read verified extracted Stage 3.1 clips with disjoint observation and target namespaces."""
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import NAMES, SIZES
from .observations import OBS_FIELDS, validate_observation
from .pack_colab_v3 import read_manifest, unpack_targets
from .perception_model import CLIP_LEN
from .qc_v3 import load_npz, read_image

TRAIN_TARGETS = ("lanes", "vectors", "vector_valid", "vector_exists", "boxes", "occupancy",
                 "future_occupancy", "trajectory")


def color_jitter_table():
    """One RGB transform shared by every camera and frame in a training clip."""
    brightness = np.random.uniform(.92, 1.08)
    contrast = np.random.uniform(.88, 1.12)
    color = np.random.uniform(.94, 1.06, size=3)
    levels = (np.arange(256, dtype=np.float32)[:, None] - 127.5) * contrast + 127.5
    return np.rint((levels * brightness * color).clip(0, 255)).astype(np.uint8)[None]


class Clips(Dataset):
    def __init__(self, root, split, stride=CLIP_LEN):
        self.root = Path(root)
        self.training = split == "train"
        manifest = read_manifest(self.root / "manifest.json", check_shards=False)
        selected = set(manifest["splits"][split])
        self.entries = {e["id"]: e for e in manifest["episodes"] if e["id"] in selected}
        self.starts = [(episode, i) for episode, entry in self.entries.items()
                       for i in range(0, len(entry["frame_ids"]) - CLIP_LEN + 1, stride)]
        self.target_cache = OrderedDict()
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
        rgb = [[] for _ in NAMES]
        inputs = {k: [] for k in OBS_FIELDS}
        targets = []
        jitter = color_jitter_table() if self.training else None
        for i in range(start, start + CLIP_LEN):
            folder = self.root / episode / str(entry["frame_ids"][i])
            for n, (w, h), bag in zip(NAMES, SIZES, rgb):
                image = read_image(folder / f"{n}.webp")
                if image.shape != (h, w, 3) or image.dtype != np.uint8:
                    raise ValueError(f"Invalid packed image: {folder}/{n}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if jitter is not None:
                    image = cv2.LUT(image, jitter)
                bag.append(image.transpose(2, 0, 1))
            obs = load_npz(folder / "obs.npz")
            validate_observation(obs)
            for k in inputs:
                inputs[k].append(obs[k])
            targets.append(unpack_targets(self.target_cache[episode], i))
        batch = {k: torch.from_numpy(np.stack(values)) for k, values in inputs.items()}
        batch["rgb"] = tuple(torch.from_numpy(np.stack(im)) for im in rgb)
        return dict(inputs=batch, targets={k: torch.from_numpy(np.stack([t[k] for t in targets])) for k in TRAIN_TARGETS})
