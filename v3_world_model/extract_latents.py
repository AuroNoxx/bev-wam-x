"""Offline extraction of frozen Stage 3.1b recurrent perception latents for Stage 3.2 training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from v1_dual_system.checkpoint_io import VerifiedCheckpointIO

from .geometry import CONFIG_SHA256, NAMES, SIZES
from .observations import OBS_FIELDS, validate_observation
from .pack_colab_v3 import read_manifest, sha256
from .perception_model import BEVWAMXPerception, sdpa_context
from .qc_v3 import load_npz, read_image


def _atomic_write_json(path, payload):
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _atomic_torch_save(path, payload):
    tmp = path.with_suffix(".pt.part")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_frame_obs(args):
    root, ep_id, frame_id = args
    folder = root / ep_id / str(frame_id)
    cams = []
    for name, (w, h) in zip(NAMES, SIZES):
        img = read_image(folder / f"{name}.webp")
        if img.shape != (h, w, 3) or img.dtype != np.uint8:
            raise ValueError(f"Invalid packed image {folder}/{name}: {img.shape} {img.dtype}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        cams.append(img)
    obs = load_npz(folder / "obs.npz")
    validate_observation(obs)
    return cams, {k: obs[k] for k in OBS_FIELDS}


def _collate_step(pool, dataset, entries, frame_index):
    tasks = [(dataset, entry["id"], entry["frame_ids"][frame_index]) for entry in entries]
    loaded = list(pool.map(_load_frame_obs, tasks))
    rgb = [
        torch.from_numpy(np.ascontiguousarray(np.stack([item[0][c] for item in loaded])))
        for c in range(len(NAMES))
    ]
    obs = {
        k: torch.from_numpy(np.ascontiguousarray(np.stack([item[1][k] for item in loaded])))
        for k in OBS_FIELDS
    }
    return rgb, obs


def extract_all(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU runtime is required for Stage 3.2 latent extraction")

    manifest = read_manifest(args.dataset / "manifest.json", check_shards=False)
    pack_sha = sha256(args.dataset / "manifest.json")
    ck = VerifiedCheckpointIO().load_checkpoint(args.checkpoint, map_location="cpu")
    task_state = ck.get("task_state", {})
    if task_state.get("config_sha256") != CONFIG_SHA256:
        raise ValueError("Checkpoint config_sha256 does not match geometry.CONFIG_SHA256")
    if task_state.get("pack_sha256") != pack_sha:
        raise ValueError("Checkpoint pack_sha256 does not match dataset manifest")
    ckpt_sha = sha256(args.checkpoint)
    ckpt_step = int(ck.get("global_step", 0))
    arch = task_state.get("architecture", "stage31b")
    if arch not in ("stage31", "stage31b", "stage31c"):
        arch = "stage31"

    model = BEVWAMXPerception(lidar_dropout=0.0, arch=arch, pretrained=False).to(device).eval()
    model.load_state_dict({k.removeprefix("model."): v for k, v in ck["state_dict"].items()})
    del ck

    args.out.mkdir(parents=True, exist_ok=True)
    episodes = list(manifest["episodes"])
    if args.limit_episodes > 0:
        episodes = episodes[:args.limit_episodes]

    grouped = defaultdict(list)
    for entry in episodes:
        grouped[len(entry["frame_ids"])].append(entry)

    batches = []
    for _, group in grouped.items():
        for i in range(0, len(group), args.batch_episodes):
            batches.append(group[i:i + args.batch_episodes])

    total_frames = sum(len(e["frame_ids"]) for e in episodes)
    done_frames = 0
    started = time.perf_counter()
    print(f"Extracting Stage 3.1b latents: {len(episodes)} episodes ({total_frames} frames) -> {args.out}", flush=True)

    with ThreadPoolExecutor(max_workers=max(2, args.workers)) as pool:
        for batch_idx, batch_entries in enumerate(batches):
            if args.skip_existing and all((args.out / f"{e['id']}.pt").is_file() for e in batch_entries):
                done_frames += sum(len(e["frame_ids"]) for e in batch_entries)
                continue

            seq_len = len(batch_entries[0]["frame_ids"])
            bsz = len(batch_entries)
            prev_z = None
            z_steps = []
            obs_accum = {k: [] for k in ("ego_delta", "cmd", "nav_anchors", "ego_speed")}

            next_fut = pool.submit(_collate_step, pool, args.dataset, batch_entries, 0)
            for t in range(seq_len):
                rgb, obs = next_fut.result()
                if t + 1 < seq_len:
                    next_fut = pool.submit(_collate_step, pool, args.dataset, batch_entries, t + 1)

                rgb_gpu = tuple(x.to(device, non_blocking=True) for x in rgb)
                lidar_gpu = obs["lidar_bev"].to(device, non_blocking=True)
                delta_gpu = obs["ego_delta"].to(device, non_blocking=True)
                cmd_gpu = obs["cmd"].to(device, non_blocking=True)
                nav_gpu = obs["nav_anchors"].to(device, non_blocking=True)
                speed_gpu = obs["ego_speed"].to(device, non_blocking=True)

                with torch.inference_mode(), sdpa_context(args.sdpa), torch.autocast(
                    "cuda", dtype=torch.bfloat16, cache_enabled=False
                ):
                    pred = model(
                        rgb_gpu,
                        lidar_bev=lidar_gpu,
                        prev_latent=prev_z,
                        ego_delta=delta_gpu,
                        cmd=cmd_gpu,
                        nav_anchors=nav_gpu,
                        ego_speed=speed_gpu,
                    )
                prev_z = pred["z_t"]
                z_steps.append(prev_z.detach().to(torch.bfloat16).cpu())
                for k in obs_accum:
                    obs_accum[k].append(obs[k])

            z_batch = torch.stack(z_steps, dim=1)  # (B, T, 128, 256) bfloat16
            obs_batch = {k: torch.stack(v, dim=1).to(torch.float32) for k, v in obs_accum.items()}

            for b, entry in enumerate(batch_entries):
                ep_id = entry["id"]
                targets = load_npz(args.dataset / ep_id / "targets.npz")
                payload = {
                    "episode_id": ep_id,
                    "map": entry["map"],
                    "dt": float(entry["dt"]),
                    "z": z_batch[b].clone(),
                    "ego_delta": obs_batch["ego_delta"][b].clone(),
                    "cmd": obs_batch["cmd"][b].clone(),
                    "nav_anchors": obs_batch["nav_anchors"][b].clone(),
                    "ego_speed": obs_batch["ego_speed"][b].clone(),
                    "ego_state": torch.from_numpy(np.asarray(targets["ego_state"], np.float32)).contiguous(),
                    "flags": torch.from_numpy(np.asarray(targets["flags"], bool)).contiguous(),
                    "thw": torch.from_numpy(np.asarray(targets["thw"], np.float32)).contiguous(),
                    "lane_state": torch.from_numpy(np.asarray(targets["lane_state"], np.int64)).contiguous(),
                    "nav_state": torch.from_numpy(np.asarray(targets["nav_state"], np.float32)).contiguous(),
                    "trajectory": torch.from_numpy(np.asarray(targets["trajectory"], np.float32)).contiguous(),
                }
                _atomic_torch_save(args.out / f"{ep_id}.pt", payload)

            done_frames += bsz * seq_len
            if (batch_idx + 1) % 8 == 0 or done_frames == total_frames:
                elapsed = time.perf_counter() - started
                fps = done_frames / max(elapsed, 1e-6)
                print(
                    f"Extracted {done_frames}/{total_frames} frames ({fps:.1f} fps, elapsed {elapsed:.1f}s)",
                    flush=True,
                )

    kept_ids = {e["id"] for e in episodes}
    out_manifest = {
        "architecture": "stage32-latents-v1",
        "perception_architecture": arch,
        "config_sha256": CONFIG_SHA256,
        "pack_sha256": pack_sha,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_step": ckpt_step,
        "splits": {
            split: [ep for ep in ids if ep in kept_ids]
            for split, ids in manifest["splits"].items()
        },
        "episodes": [
            {
                "id": e["id"],
                "map": e["map"],
                "dt": float(e["dt"]),
                "num_frames": len(e["frame_ids"]),
                "file": f"{e['id']}.pt",
            }
            for e in episodes
        ],
        "total_frames": total_frames,
    }
    _atomic_write_json(args.out / "manifest.json", out_manifest)
    print(f"Finished Stage 3.2 latent extraction in {time.perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sdpa", choices=("auto", "math", "aotriton", "cuda_flash"), default="aotriton")
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    extract_all(args)


if __name__ == "__main__":
    main()
