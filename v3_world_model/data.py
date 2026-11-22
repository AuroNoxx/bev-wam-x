"""Read verified extracted Stage 3.1 clips with disjoint observation and target namespaces."""
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import NAMES, SIZES
from .pack_colab_v3 import read_manifest, unpack_targets
from .perception_model import CLIP_LEN
from .qc_v3 import load_npz, read_image

TRAIN_TARGETS = ("lanes", "vectors", "vector_valid", "vector_exists", "boxes", "occupancy",
                 "future_occupancy", "trajectory")


class Clips(Dataset):
    def __init__(self, root, split, stride=CLIP_LEN):
        self.root = Path(root)
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
        rgb, lidar, motion, targets = [[] for _ in NAMES], [], [], []
        for i in range(start, start + CLIP_LEN):
            folder = self.root / episode / str(entry["frame_ids"][i])
            for n, (w, h), bag in zip(NAMES, SIZES, rgb):
                image = read_image(folder / f"{n}.webp")
                if image.shape != (h, w, 3) or image.dtype != np.uint8:
                    raise ValueError(f"Invalid packed image: {folder}/{n}")
                bag.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
            obs = load_npz(folder / "obs.npz")
            if set(obs) != {"lidar_bev", "ego_delta"}:
                raise ValueError(f"Unexpected packed inputs at {folder}")
            lidar.append(obs["lidar_bev"])
            motion.append(obs["ego_delta"])
            targets.append(unpack_targets(self.target_cache[episode], i))
        # Scripted command is known before driving and is independent of future labels.
        intent = {"left_change": 1, "right_change": 2, "slow_overtake": 1}.get(entry["scenario"], 0)
        command = np.eye(3, dtype=np.float32)[[intent] * CLIP_LEN]
        return dict(inputs=dict(rgb=tuple(torch.from_numpy(np.stack(im)) for im in rgb),
                                lidar_bev=torch.from_numpy(np.stack(lidar)),
                                ego_delta=torch.from_numpy(np.stack(motion)),
                                cmd=torch.from_numpy(command)),
                    targets={k: torch.from_numpy(np.stack([t[k] for t in targets])) for k in TRAIN_TARGETS})
