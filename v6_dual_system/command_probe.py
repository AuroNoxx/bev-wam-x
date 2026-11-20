"""Compare KEEP/LEFT/RIGHT on one recorded observation in the sensor-frame FLU system."""
import argparse
import json
from pathlib import Path

import numpy as np

from .sensors import write_json


NAMES = ("keep", "left", "right")


def save_observation(folder, key, frame, sensor_frame, map_, pose, rgb, bev, state, target, reply):
    import carla
    from .collect_fast_data import neighbor, path_from

    source = map_.get_waypoint(carla.Location(*pose[:3, 3].tolist()))
    references = {}
    lane_ids = {}
    for name, start in zip(NAMES, (source, neighbor(source, True), neighbor(source, False))):
        if start is None:
            continue
        path = path_from(start, distance=160)
        points = np.array([[p.transform.location.x, p.transform.location.y, p.transform.location.z]
                           for p in path])
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        references[name] = local[:, :2] * [1, -1]
        lane_ids[name] = [start.road_id, start.section_id, start.lane_id]
    np.savez(folder / f"key-{frame}-{key}.npz", rgb=rgb, bev=bev, state=state, target=target,
             key=key, frame=frame, sensor_frame=sensor_frame, pose=pose, live=reply["xy"],
             executed=reply["intent"], veto=reply["veto"], lanes=json.dumps(lane_ids), **references)


def signed_distance(xy, path):
    # Signed shortest distance to the polyline. FLU positive is left of its tangent.
    delta = np.diff(path, axis=0)
    lengths = np.sum(delta * delta, axis=1)
    rel = xy[:, None, :] - path[None, :-1, :]
    t = np.clip(np.sum(rel * delta, axis=-1) / np.maximum(lengths, 1e-12), 0., 1.)
    disp = rel - t[:, :, None] * delta
    index = np.argmin(np.sum(disp * disp, axis=-1), axis=1)
    nearest = disp[np.arange(len(xy)), index]
    direction = delta[index]
    return (direction[:, 0] * nearest[:, 1] - direction[:, 1] * nearest[:, 0]) / np.sqrt(lengths[index])


def compare(folder, ckpt, output=None):
    import torch
    from .deployment import load_driver

    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.set_per_process_memory_fraction(.15)
    model, step, identity = load_driver(ckpt)
    model.to(device="cuda", memory_format=torch.channels_last).eval()
    result = dict(checkpoint=str(ckpt.resolve()), step=step, manifest=identity, observations=[])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for source in sorted(folder.glob("key-*.npz")):
            with np.load(source) as obs:
                args = [torch.as_tensor(obs[name], device="cuda").unsqueeze(0) for name in ("rgb", "bev")]
                args.append(torch.as_tensor(obs["state"], device="cuda", dtype=torch.float32)[None])
                observation = dict(file=source.name, key=str(obs["key"]), frame=int(obs["frame"]),
                                   sensor_frame=int(obs["sensor_frame"]), speed_kmh=float(obs["state"][0]*3.6),
                                   target_kmh=float(obs["target"]), lanes=json.loads(str(obs["lanes"])),
                                   executed=int(obs["executed"]), veto=bool(obs["veto"]), predictions={})
                for index, command in enumerate(NAMES):
                    cond = torch.tensor([[index == 0, index == 1, index == 2, float(obs["target"])]],
                                        device="cuda", dtype=torch.float32)
                    xy = model(*args, cond)[0][0].float().cpu().numpy()
                    errors = {lane: signed_distance(xy, obs[lane]) for lane in NAMES if lane in obs}
                    observation["predictions"][command] = dict(
                        xy_flu_m=xy.tolist(),
                        signed_error_at_1_2_3_4s_m={lane: e[9::10].tolist() for lane, e in errors.items()})
                    if command=="right":
                        mirror_state=args[2].clone()
                        mirror_state[:,2].neg_()
                        mirror_xy=model(args[0].flip((-1,)),args[1].flip((-2,)),mirror_state,
                                        cond[:,[0,2,1,3]])[0][0].float().cpu().numpy()
                        observation["reflected_left_xy_flu_m"]=mirror_xy.tolist()
                    if index == observation["executed"]:
                        observation["live_replay_max_xy_delta_m"] = float(np.max(np.abs(xy-obs["live"])))
                result["observations"].append(observation)
    if {o["key"] for o in result["observations"]} != {"a", "d"}:
        raise ValueError("The probe requires both A and D capture files")
    write_json(output or folder / "command-comparison.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path,help="Separate result file when comparing another checkpoint")
    args = parser.parse_args()
    compare(args.folder, args.checkpoint,args.output)
