"""Evaluate SR predictions before rendering, using the held-out packed split.

Reports lane IoU and one-to-one vehicle center matching, not detection AP.
Threshold sweeps are diagnostic; they never change the deployed threshold.
Example: python -m v3_world_model.evaluate_sr --dataset .cache/stage31-r2/dataset-full
  --checkpoint .cache/stage31-r2/run-stage31-a100/best.ckpt --clips 24 --out report.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from v1_dual_system.checkpoint_io import VerifiedCheckpointIO
from .data import Clips
from .geometry import CONFIG_SHA256, grid
from .pack_colab_v3 import sha256
from .perception_model import BEVWAMXPerception, CLIP_LEN, sdpa_context


def match_centers(predicted, truth, radius=3.):
    """Maximize matches within radius, then minimize their center distances."""
    if not len(predicted) or not len(truth):
        return 0
    distances = np.linalg.norm(predicted[:, None, :2] - truth[None, :, :2], axis=-1)
    penalty = (min(distances.shape) + 1) * radius
    costs = np.where(distances < radius, distances, penalty)
    rows, columns = linear_sum_assignment(costs)
    return int((distances[rows, columns] < radius).sum())


@torch.inference_mode()
def evaluate(dataset, checkpoint, clips=24, backend="aotriton"):
    if clips is not None and clips <= 0:
        raise ValueError("clips must be positive")
    torch.set_num_threads(2)
    state = VerifiedCheckpointIO().load_checkpoint(checkpoint, map_location="cpu")
    if state["task_state"]["config_sha256"] != CONFIG_SHA256:
        raise ValueError("Checkpoint sensor contract mismatch")
    fingerprint = sha256(Path(dataset) / "manifest.json")
    if state["task_state"]["pack_sha256"] != fingerprint:
        raise ValueError("Checkpoint and validation manifest fingerprints differ")
    step = state["global_step"]
    model = BEVWAMXPerception(lidar_dropout=0., legacy=state["task_state"].get("architecture") != "stage31b").cuda().eval()
    model.load_state_dict({k.removeprefix("model."): v for k, v in state["state_dict"].items()})
    del state
    data = Clips(dataset, "val")
    tail_skip = {}
    if clips is None:
        for episode, entry in data.entries.items():
            length = len(entry["frame_ids"])
            remainder = length % CLIP_LEN
            if remainder:
                tail_skip[len(data.starts)] = CLIP_LEN - remainder
                data.starts.append((episode, length - CLIP_LEN))
        indices = range(len(data))
    else:
        indices = np.linspace(0, len(data) - 1, min(clips, len(data))).astype(int)
    thresholds = (.05, .1, .2, .3, .5)
    roi = (grid()[..., 0] >= 0) & (grid()[..., 0] <= 40) & (abs(grid()[..., 1]) <= 18)
    records = []
    for clip_number, index in enumerate(indices):
        batch, latent = data[int(index)], None
        episode, offset = data.starts[index]
        for t in range(CLIP_LEN):
            inputs = {k: v[t:t+1].cuda() for k, v in batch["inputs"].items() if k != "rgb"}
            with sdpa_context(backend), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                pred = model(tuple(x[t:t+1].cuda() for x in batch["inputs"]["rgb"]),
                             prev_latent=latent, **inputs)
            latent = pred["z_t"]
            if t < tail_skip.get(index, 0):
                continue
            masks = pred["lane_logits"][0].float().cpu().numpy() > 0
            truth_masks = batch["targets"]["lanes"][t].numpy().astype(bool)
            boxes = pred["boxes"][0].float().cpu().numpy()
            probabilities = pred["obj_logits"][0].float().sigmoid().cpu().numpy()
            truth = batch["targets"]["boxes"][t].numpy()
            truth = truth[truth[:, -1] > .5]
            counts = {}
            for threshold in thresholds:
                selected = boxes[probabilities > threshold]
                counts[str(threshold)] = dict(tp_3m=match_centers(selected, truth),
                                              predicted=len(selected), truth=len(truth))
            records.append(dict(episode=episode, map=data.entries[episode]["map"], offset=offset+t,
                                intersection=(masks & truth_masks & roi).sum(axis=(1, 2)).tolist(),
                                union=((masks | truth_masks) & roi).sum(axis=(1, 2)).tolist(),
                                objects=counts, max_object_probability=float(probabilities.max())))
        print(f"SR evaluation {clip_number + 1}/{len(indices)} clips", flush=True)

    def summarize(rows):
        intersections = np.sum([r["intersection"] for r in rows], axis=0)
        unions = np.sum([r["union"] for r in rows], axis=0)
        result = dict(frames=len(rows), lane_iou=(intersections / np.maximum(1, unions)).tolist())
        result["objects"] = {}
        for threshold in map(str, thresholds):
            counts = {key: sum(r["objects"][threshold][key] for r in rows)
                      for key in ("tp_3m", "predicted", "truth")}
            counts.update(precision_3m=counts["tp_3m"] / max(1, counts["predicted"]),
                          recall_3m=counts["tp_3m"] / max(1, counts["truth"]))
            result["objects"][threshold] = counts
        return result

    return dict(checkpoint=str(checkpoint), checkpoint_step=step, pack_sha256=fingerprint, split="val", clips=len(indices),
                sampling=("every validation frame scored once" if clips is None else "evenly spaced clips")
                         + "; recurrent state reset per clip",
                lane_roi="forward 0..40 m, lateral -18..18 m",
                lane_channels=["drivable", "dashed", "solid", "edge"],
                object_metric="one-to-one XY center distance < 3m, class-agnostic, full label ROI; not AP",
                summary=summarize(records),
                by_map={town: summarize([r for r in records if r["map"] == town])
                        for town in sorted({r["map"] for r in records})}, records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--clips", type=int, default=24)
    scope.add_argument("--all", action="store_true", help="Score every validation frame exactly once")
    parser.add_argument("--sdpa", choices=("aotriton", "math", "cuda_flash"), default="aotriton")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.dataset, args.checkpoint, None if args.all else args.clips, args.sdpa)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
