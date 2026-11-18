"""Speed-balanced batches and causal metadata for the new RGB dataset only."""
import copy
import hashlib
import json
import math
import torch
from torch.utils.data import Sampler


def intent_labels(acceleration, trajectory_xy, dt=0.1):
    """Labels only: 4 longitudinal x 3 lateral classes over future 0..3 seconds."""
    end = round(3.0 / dt)
    if acceleration.shape[-2] < end or trajectory_xy.shape[-2] < end:
        raise ValueError("Intent labels require three seconds of future supervision")
    ax = acceleration[..., :end, 0].float().mean(-1)
    longitudinal = torch.where(ax < -3, 0, torch.where(ax < -.5, 1, torch.where(ax <= .5, 2, 3)))
    dy = trajectory_xy[..., end - 1, 1]
    lateral = torch.where(dy >= 1.2, 1, torch.where(dy <= -1.2, 2, 0))
    return longitudinal * 3 + lateral


class SpeedBinBalancedSampler(Sampler):
    """Stateful batch sampler: 70/30 source quotas, cycling occupied speed bins.

    Within each source, bins receive equal draw counts up to one draw. Missing
    bins are reported, never synthesized. The mixture is exact in expectation
    and differs from the running 30% target by less than one highway sample.
    TorchData owns loader state; this class only owns sampler RNG and progress.
    """
    def __init__(self, records, batch_size=8, num_batches=None, seed=71):
        self.batch_size = batch_size
        self.num_batches = num_batches or math.ceil(len(records) / batch_size)
        self.generator = torch.Generator().manual_seed(seed)
        self.groups = {"bench2drive": {}, "highway": {}}
        for index, row in enumerate(records):
            speed = float(row["speed_kmh"])
            if not 0 <= speed <= 120:
                raise ValueError(f"Invalid measured speed {speed}")
            source = row["source"]
            if source not in self.groups:
                raise ValueError(f"Unknown source {source}")
            bucket = min(5, int(speed // 20))
            self.groups[source].setdefault(bucket, []).append(index)
        if not all(self.groups.values()):
            raise ValueError("Both Bench2Drive and actual highway data are required for 70/30 sampling")
        self.bin_order = {s: sorted(g) for s, g in self.groups.items()}
        self.bin_cursor = {s: 0 for s in self.groups}
        self.position = 0
        self.total_drawn = 0
        self.highway_drawn = 0
        self.identity = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()

    def __len__(self):
        return self.num_batches

    def _draw(self, source):
        bins = self.bin_order[source]
        cursor = self.bin_cursor[source]
        bucket = bins[cursor % len(bins)]
        self.bin_cursor[source] = cursor + 1
        members = self.groups[source][bucket]
        return members[torch.randint(len(members), (), generator=self.generator).item()]

    def __iter__(self):
        # Lightning can stop at len(loader) without requesting StopIteration.
        # Reset on the next iterator, not after the final yield (which might
        # never be resumed). Keep mid-epoch resume positions untouched.
        if self.position >= self.num_batches:
            self.position = 0
        while self.position < self.num_batches:
            target = math.floor((self.total_drawn + self.batch_size) * .3 + .5)
            n_highway = target - self.highway_drawn
            indices = [self._draw("highway") for _ in range(n_highway)]
            indices += [self._draw("bench2drive") for _ in range(self.batch_size - n_highway)]
            order = torch.randperm(self.batch_size, generator=self.generator).tolist()
            self.position += 1
            self.total_drawn += self.batch_size
            self.highway_drawn += n_highway
            yield [indices[i] for i in order]

    def state_dict(self):
        return {"identity": self.identity, "batch_size": self.batch_size, "num_batches": self.num_batches,
                "rng": self.generator.get_state(), "position": self.position,
                "total_drawn": self.total_drawn, "highway_drawn": self.highway_drawn,
                "bin_cursor": copy.deepcopy(self.bin_cursor)}

    def load_state_dict(self, state):
        if (state["identity"], state["batch_size"], state["num_batches"]) != (self.identity, self.batch_size, self.num_batches):
            raise ValueError("Sampler data/config identity changed")
        self.generator.set_state(state["rng"])
        self.position = state["position"]
        self.total_drawn = state["total_drawn"]
        self.highway_drawn = state["highway_drawn"]
        self.bin_cursor = copy.deepcopy(state["bin_cursor"])

    def coverage(self):
        return {s: {str(b): len(g.get(b, [])) for b in range(6)} for s, g in self.groups.items()}


def audit_manifest(manifest):
    """Require real timestamps, route-family splits, and separate actor/labels."""
    families = {}
    for split in ("train", "validation", "test"):
        for row in manifest["splits"][split]:
            family = row["initial_condition_family"]
            if family in families and families[family] != split:
                raise ValueError("Route/initial-condition family leaks across splits")
            families[family] = split
            now = float(row["current_timestamp"])
            if not row["history_timestamps"] or max(row["history_timestamps"]) > now:
                raise ValueError("Future observation in actor history")
            if min(row["future_timestamps"]) <= now:
                raise ValueError("Future targets must be strictly later")
            if row["actor_path"] == row["label_path"]:
                raise ValueError("Actor and future labels must be separate files")
    return {"families": len(families), "split_counts": {s: len(manifest["splits"][s]) for s in manifest["splits"]}}
