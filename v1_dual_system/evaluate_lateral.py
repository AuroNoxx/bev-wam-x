"""Evaluate original validation windows and their reflected counterparts, without CARLA."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import FastData
from .deployment import load_driver
from .sensors import write_json


def metrics(values):
    return {"windows": len(values), "near_m": float(values[:, :15].mean()),
            "far_m": float(values[:, 15:].mean()), "all_m": float(values.mean())}


def evaluate(checkpoint):
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    model, step, identity = load_driver(checkpoint)
    model.layout = torch.contiguous_format
    model.to("cuda").eval()
    dataset = FastData("validation", True)
    if identity != dataset.identity:
        raise ValueError("Checkpoint and validation manifest differ")
    errors, mirror_errors, intents, speeds = [], [], [], []
    loader = DataLoader(dataset, batch_size=32, num_workers=2)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            rgb, bev, state, cmd, truth = [v.cuda() for v in batch[:5]]
            xy, _ = model(rgb, bev, state, cmd)
            mirror_state = state.clone()
            mirror_state[:, 2].neg_()
            reflected, _ = model(rgb.flip((-1,)), bev.flip((-2,)), mirror_state, cmd[:, [0, 2, 1, 3]])
            reflected[:, :, 1].neg_()
            mirror_errors.append((xy-reflected).abs().float().cpu().numpy())
            errors.append((xy[:, :, 1]-truth[:, :, 1]).abs().float().cpu().numpy())
            intents.append(cmd[:, :3].argmax(1).cpu().numpy())
            speeds.append((state[:, 0]*3.6).cpu().numpy())
    values, mirror, labels, kmh = map(np.concatenate, (errors, mirror_errors, intents, speeds))
    turns = mirror[labels != 0]
    result = dict(step=step, manifest=identity, all=metrics(values),
                  by_intent={k: metrics(values[labels == i]) for i,k in enumerate(("KEEP","LEFT","RIGHT"))},
                  high_speed=metrics(values[(kmh>=118)&(kmh<=122)]),
                  mirror_turns=dict(windows=len(turns), y_mean_m=float(turns[:,:,1].mean()),
                                    y_max_m=float(turns[:,:,1].max()), max_xy_m=float(turns.max())),
                  near_pass=bool(values[:, :15].mean()<.15),
                  mirror_equal=bool(np.all(turns==0)))
    write_json(checkpoint.parent/"lateral-audit.json",result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    evaluate(parser.parse_args().checkpoint)
