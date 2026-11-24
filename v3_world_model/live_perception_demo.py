"""Live Stage 3.1 pure-perception audit in CARLA 1.0 using Traffic Manager autopilot + GT comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import faulthandler
import json
import multiprocessing as mp
from pathlib import Path
import queue
from time import perf_counter

import carla
import cv2
import numpy as np
import torch

from v1_dual_system.checkpoint_io import VerifiedCheckpointIO

from .audit_stage31 import DIST_BUCKETS, LANE_NAMES, _hungarian_match, _is_zero_lidar_ghost
from .collect_v3 import NPCS, raised_spawn
from .geometry import CONFIG_SHA256, DT, H, NAMES, SIZES, W as BEV_W, ego_delta, grid, lidar_bev, local
from .labels import OBJECT_CLASSES, RoadSurfaces, local_objects, xyz
from .live_carla_demo import (
    DISPLAY_STEPS,
    InfiniteNavigator,
    _freeze_lights_green,
    _pick_ego_spawn,
    _recycle_distant_npcs,
    _spawn_random_npcs,
    launch_carla_server,
)
from .observations import command_vector
from .perception_model import BEVWAMXPerception, sdpa_context
from .sim import Rig, bgr, matrix, synchronous
from .sr_postprocess import SRPostProcessor
from .sr_viewer import SRViewer, prediction_scene
from .stage32_demo_view import LiveCamera, inset_size


def load_perception_checkpoint(ckpt_path, device):
    """Load Stage 3.1 (or Stage 3.2 perception branch) weights with sensor-contract check."""
    ck = VerifiedCheckpointIO().load_checkpoint(Path(ckpt_path), map_location="cpu")
    task_state = dict(ck.get("task_state", {}))
    if task_state.get("config_sha256") != CONFIG_SHA256:
        raise ValueError("Checkpoint config_sha256 does not match geometry.CONFIG_SHA256")
    arch = task_state.get("architecture", "stage31b")
    state_dict = ck["state_dict"]
    if any(k.startswith("perception.") for k in state_dict):
        weights = {k.removeprefix("perception."): v for k, v in state_dict.items() if k.startswith("perception.")}
    else:
        weights = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model = BEVWAMXPerception(lidar_dropout=0.0, arch=arch, pretrained=False)
    model.load_state_dict(weights, strict=True)
    model = model.to(device).eval()
    return model, task_state, {
        "checkpoint": str(ckpt_path),
        "architecture": arch,
        "global_step": int(ck.get("global_step", -1)),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def _warmup_perception(model, device):
    """Pre-compile ROCm/AOTriton kernels for initial and recurrent perception passes."""
    rgb = tuple(torch.zeros((1, 3, h, w), dtype=torch.uint8, device=device) for w, h in SIZES)
    lbev = torch.zeros((1, 6, H, BEV_W), dtype=torch.float16, device=device)
    nav = torch.zeros((1, 3, 3), dtype=torch.float32, device=device)
    spd = torch.zeros((1, 2), dtype=torch.float32, device=device)
    cmd = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    delta = torch.zeros((1, 3), dtype=torch.float32, device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False), sdpa_context("aotriton"):
        p0 = model(rgb, lidar_bev=lbev, prev_latent=None, ego_delta=None, cmd=cmd, nav_anchors=nav, ego_speed=spd)
        model(rgb, lidar_bev=lbev, prev_latent=p0["z_t"], ego_delta=delta, cmd=cmd, nav_anchors=nav, ego_speed=spd)
    torch.cuda.synchronize()


def _actor_cls_id(actor):
    tid = actor.type_id
    if tid in NPCS:
        return NPCS[tid]
    if tid.startswith("walker."):
        return 4
    attrs = getattr(actor, "attributes", {}) or {}
    if "number_of_wheels" in attrs and int(attrs["number_of_wheels"]) == 2:
        return 3
    if any(k in tid for k in ("harley", "kawasaki", "yamaha", "vespa", "diamondback", "gazelle", "crossbike")):
        return 3
    ext = actor.bounding_box.extent
    if ext.x * 2.0 > 5.5 or ext.z * 2.0 > 2.2 or any(k in tid for k in ("carlacola", "sprinter", "ambulance", "firetruck", "t2", "hgv", "bus")):
        return 2
    if ext.z * 2.0 > 1.72 or any(k in tid for k in ("patrol", "wrangler", "etron")):
        return 1
    return 0


def _collect_gt_boxes(world, snap, ego_id, ego_loc, ego_pose):
    """Build evaluation-only GT vehicle/walker boxes in current ego FLU coordinates."""
    actors = []
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == ego_id:
            continue
        a_snap = snap.find(actor.id)
        if a_snap is None:
            continue
        tf = a_snap.get_transform()
        if tf.location.distance(ego_loc) > 95.0:
            continue
        bbox = actor.bounding_box
        bbox_pose = matrix(tf) @ matrix(carla.Transform(bbox.location, bbox.rotation))
        actors.append((bbox_pose[:3, 3], bbox_pose[:3, 0], xyz(a_snap.get_velocity()), 2.0 * xyz(bbox.extent), _actor_cls_id(actor)))

    if not actors:
        return np.empty((0, 11), dtype=np.float32)

    from .labels import ActorState

    states = [ActorState(c, h, v, s, cls_i, i) for i, (c, h, v, s, cls_i) in enumerate(actors)]
    boxes, _ = local_objects(states, ego_pose)
    return boxes


def _score_live_frame(pred, lbev, gt_boxes, gt_lanes, roi_mask):
    """Score raw model predictions (and clean_frame+LiDAR) against CARLA ground truth."""
    geom = pred["boxes"][0].detach().float().cpu().numpy()
    cls_ids = pred["cls_logits"][0].detach().float().cpu().numpy().argmax(axis=-1)
    scores = pred["obj_logits"][0].detach().float().sigmoid().cpu().numpy()
    yaw = np.arctan2(geom[:, 6], geom[:, 7])
    pred_valid = scores > 0.5
    raw_boxes = np.column_stack((geom[:, :6], yaw, geom[:, 8:10], cls_ids, pred_valid)).astype(np.float32)
    sel = raw_boxes[pred_valid]

    m_p, m_g, _ = _hungarian_match(sel[:, :2] if len(sel) else sel, gt_boxes[:, :2] if len(gt_boxes) else gt_boxes, radius=3.0)
    matched_mask = np.zeros(len(sel), dtype=bool)
    matched_mask[m_p] = True
    unmatched = sel[~matched_mask]
    dup_fp, ghost_fp = 0, 0
    if len(unmatched):
        if len(gt_boxes):
            d_any = np.linalg.norm(unmatched[:, None, :2] - gt_boxes[None, :, :2], axis=-1).min(axis=1)
        else:
            d_any = np.full(len(unmatched), np.inf, dtype=np.float32)
        is_dup = d_any < 3.0
        dup_fp = int(is_dup.sum())
        non_gt = unmatched[~is_dup]
        if len(non_gt):
            ghost_fp = int(_is_zero_lidar_ghost(non_gt[:, :2], lbev).sum())

    # Near (<40m) GT recall for clear urban/highway close-range tracking.
    near_gt = (np.hypot(gt_boxes[:, 0], gt_boxes[:, 1]) < 40.0) if len(gt_boxes) else np.empty((0,), dtype=bool)
    near_tp = int(near_gt[m_g].sum()) if len(m_g) else 0

    clean_boxes, _ = SRPostProcessor.clean_frame(raw_boxes, scores, lidar_bev=lbev)
    c_p, _, _ = _hungarian_match(clean_boxes[:, :2] if len(clean_boxes) else clean_boxes,
                                 gt_boxes[:, :2] if len(gt_boxes) else gt_boxes, radius=3.0)

    pred_lanes = pred["lane_logits"][0].detach().float().cpu().numpy() > 0.0
    gt_m = gt_lanes.astype(bool)
    inter = (pred_lanes & gt_m & roi_mask[None]).sum(axis=(1, 2))
    union = ((pred_lanes | gt_m) & roi_mask[None]).sum(axis=(1, 2))
    return dict(
        raw_tp=len(m_p), raw_pred=len(sel), raw_gt=len(gt_boxes),
        near_tp=near_tp, near_gt=int(near_gt.sum()),
        dup_fp=dup_fp, ghost_fp=ghost_fp, miss_fn=max(0, len(gt_boxes) - len(m_p)),
        clean_tp=len(c_p), clean_pred=len(clean_boxes),
        lane_inter=inter, lane_union=union,
    )


def drive_perception_audit(args, frames, camera, stopped, status):
    faulthandler.enable()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU runtime is required for Stage 3.1 perception live audit")

    status("Loading Stage 3.1 checkpoint")
    t_w0 = perf_counter()
    model, _, meta = load_perception_checkpoint(args.ckpt, device)
    status("Warming up perception model")
    _warmup_perception(model, device)
    print(
        f"Loaded & warmed {meta['architecture']} step={meta['global_step']} "
        f"({meta['parameters']:,} params) in {perf_counter() - t_w0:.1f}s ({meta['checkpoint']})",
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    rng = np.random.default_rng(args.seed)
    if stopped.is_set():
        return
    status("Starting CARLA offscreen")
    with launch_carla_server(args.carla_root, args.out_dir, args.port, args.quality) as (client, server_ver):
        world = client.get_world()
        active_map = world.get_map().name.replace("\\", "/").rsplit("/", 1)[-1]
        if active_map != args.map:
            status(f"Loading {args.map}")
            world = client.load_world(args.map)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        _freeze_lights_green(world)

        if stopped.is_set():
            return
        status("Building evaluation road GT")
        t_r0 = perf_counter()
        road = world.get_map()
        surfaces = RoadSurfaces(road)
        print(f"Built evaluation RoadSurfaces for {args.map} in {perf_counter() - t_r0:.1f}s", flush=True)

        status("Spawning vehicles (TM Autopilot)")
        tm = client.get_trafficmanager(args.port + 5)
        ego_spawn, start_wp = _pick_ego_spawn(road, carla)
        ego_bp = world.get_blueprint_library().find("vehicle.lincoln.mkz")
        ego_bp.set_attribute("role_name", "hero")
        ego = world.try_spawn_actor(ego_bp, raised_spawn(carla, ego_spawn, 0.35))
        if ego is None:
            raise RuntimeError("Could not spawn ego vehicle")

        # Drive ego cleanly with CARLA Traffic Manager autopilot to isolate Stage 3.1 perception.
        tm_port = tm.get_port()
        ego.set_autopilot(True, tm_port)
        tm.auto_lane_change(ego, True)
        tm.ignore_lights_percentage(ego, 100.0)
        tm.vehicle_percentage_speed_difference(ego, -10.0)
        tm.distance_to_leading_vehicle(ego, 3.5)

        npcs = _spawn_random_npcs(carla, world, tm, ego_spawn.location, args.npc, rng)
        print(
            f"Spawned TM-autopilot ego on {args.map} + {len(npcs)} random NPCs "
            f"(server {server_ver}, sr_mode={args.sr_mode})",
            flush=True,
        )

        post = SRPostProcessor() if args.sr_mode == "clean" else None
        nav = InfiniteNavigator(road, start_wp, rng)
        collisions = {"count": 0, "last_frame": -1}
        prev_pose = None
        prev_latent = None
        prev_scene = None
        stuck_ticks = 0
        recoveries = 0
        driven_dist_m = 0.0

        g = grid()
        roi_mask = (g[..., 0] >= 0.0) & (g[..., 0] <= 40.0) & (np.abs(g[..., 1]) <= 18.0)
        cum = dict(raw_tp=0, raw_pred=0, raw_gt=0, near_tp=0, near_gt=0,
                   dup_fp=0, ghost_fp=0, miss_fn=0, clean_tp=0, clean_pred=0,
                   lane_inter=np.zeros(4, dtype=np.int64), lane_union=np.zeros(4, dtype=np.int64))
        latencies_perc, latencies_ctrl = [], []

        try:
            with (synchronous(world, tm, args.seed, dt=DT / DISPLAY_STEPS),
                  Rig(carla, world, ego) as rig, ThreadPoolExecutor(max_workers=1) as clock):
                rig.cameras("rgb", sensor_tick=DT)
                rig.lidar(sensor_tick=DT)
                requested_frame = None
                chase_frames = queue.Queue(maxsize=1)

                def on_chase(image):
                    camera.publish(image)
                    if image.frame == requested_frame:
                        chase_frames.put_nowait(image)

                rig.attach("chase", "sensor.camera.rgb",
                           carla.Transform(carla.Location(x=-6.8, z=3.4), carla.Rotation(pitch=-14)),
                           dict(image_size_x=args.camera_size[0], image_size_y=args.camera_size[1],
                                fov=90, sensor_tick=0), on_chase)
                rig.attach(
                    "collision", "sensor.other.collision", carla.Transform(), {},
                    lambda ev: collisions.update(count=collisions["count"] + 1, last_frame=ev.frame),
                )

                phases = {}
                for _ in range(DISPLAY_STEPS * 5):
                    world.tick(45)
                for name, inbox in rig.queues.items():
                    samples = [inbox.get(timeout=10).frame for _ in range(3)]
                    while True:
                        try:
                            samples.append(inbox.get_nowait().frame)
                        except queue.Empty:
                            break
                    phases[name] = samples[-1] % DISPLAY_STEPS
                    if samples[-1] - samples[-2] != DISPLAY_STEPS:
                        raise RuntimeError(f"Unexpected sensor cadence for {name}: {samples}")
                if len(set(phases.values())) != 1:
                    raise RuntimeError(f"Unaligned sensor phases: {phases}")
                phase = phases["rgb/front"]
                while (world.get_snapshot().frame + 1) % DISPLAY_STEPS != phase:
                    world.tick(45)

                save_set = set(args.save_frames or ())
                shot_tick = 40 if args.max_ticks <= 0 else min(40, args.max_ticks - 1)

                def advance_camera():
                    for _ in range(DISPLAY_STEPS - 1):
                        world.tick(45)

                tick_idx = 0
                while args.max_ticks <= 0 or tick_idx < args.max_ticks:
                    if stopped.is_set():
                        break
                    t_ctrl0 = perf_counter()
                    need_shot = tick_idx == shot_tick or tick_idx in save_set
                    requested_frame = world.get_snapshot().frame + 1 if need_shot else None
                    fid = world.tick(45)
                    snap = world.get_snapshot()
                    if snap.frame != fid:
                        raise RuntimeError(f"Expected simulation frame {fid}, got {snap.frame}")
                    intermediate = clock.submit(advance_camera)
                    bundle = rig.read_sensors(snap)
                    sensor_ms = (perf_counter() - t_ctrl0) * 1000.0

                    ego_snap = snap.find(ego.id)
                    ego_tf = ego_snap.get_transform()
                    pose = matrix(ego_tf)
                    if prev_pose is not None:
                        driven_dist_m += float(np.linalg.norm(pose[:2, 3] - prev_pose[:2, 3]))
                    ctrl = ego.get_control()
                    speed_mps = float(np.linalg.norm(xyz(ego_snap.get_velocity())[:2]))
                    yaw_rate = float(-np.deg2rad(ego_snap.get_angular_velocity().z))

                    nav_anchors, in_highway, cur_wp, _ = nav.update(pose, ego_tf.location)

                    scan_raw = np.frombuffer(bundle["lidar"].raw_data, dtype="<f4").reshape(-1, 4).copy()
                    if tick_idx == 0:
                        az = np.degrees(np.arctan2(scan_raw[:, 1], scan_raw[:, 0]))
                        print(f"LiDAR occupied {len(np.unique(np.floor((az + 180) / 5)))} of 72 sectors", flush=True)
                    lidar_pose = matrix(bundle["lidar"].transform)
                    scan_raw[:, :3] = local(scan_raw[:, :3] @ lidar_pose[:3, :3].T + lidar_pose[:3, 3], pose)
                    lbev = lidar_bev(scan_raw)

                    rgb_gpu = tuple(
                        torch.from_numpy(
                            np.ascontiguousarray(cv2.cvtColor(bgr(bundle[f"rgb/{name}"]), cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
                        ).unsqueeze(0).to(device, non_blocking=True)
                        for name in NAMES
                    )
                    lidar_gpu = torch.from_numpy(lbev).unsqueeze(0).to(device, non_blocking=True)
                    nav_gpu = torch.from_numpy(nav_anchors).unsqueeze(0).to(device, non_blocking=True)
                    speed_gpu = torch.tensor([[speed_mps, yaw_rate]], device=device, dtype=torch.float32)
                    cmd_gpu = torch.from_numpy(command_vector("follow")).unsqueeze(0).to(device, non_blocking=True)
                    delta_gpu = (
                        torch.from_numpy(ego_delta(prev_pose, pose)).unsqueeze(0).to(device, non_blocking=True)
                        if (prev_latent is not None and prev_pose is not None)
                        else None
                    )

                    t_p0 = perf_counter()
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False), sdpa_context("aotriton"):
                        pred = model(
                            rgb_gpu, lidar_bev=lidar_gpu, prev_latent=prev_latent,
                            ego_delta=delta_gpu, cmd=cmd_gpu, nav_anchors=nav_gpu, ego_speed=speed_gpu,
                        )
                    torch.cuda.synchronize()
                    perc_ms = (perf_counter() - t_p0) * 1000.0
                    prev_latent = pred["z_t"].clone()
                    prev_pose = pose

                    # Evaluation-only live GT (never fed into model or SR).
                    gt_lanes = surfaces.masks(pose)
                    gt_boxes = _collect_gt_boxes(world, snap, ego.id, ego_tf.location, pose)
                    fs = _score_live_frame(pred, lbev, gt_boxes, gt_lanes, roi_mask)
                    for k in ("raw_tp", "raw_pred", "raw_gt", "near_tp", "near_gt",
                              "dup_fp", "ghost_fp", "miss_fn", "clean_tp", "clean_pred"):
                        cum[k] += fs[k]
                    cum["lane_inter"] += fs["lane_inter"]
                    cum["lane_union"] += fs["lane_union"]

                    cum_p = cum["raw_tp"] / max(1, cum["raw_pred"])
                    cum_r = cum["raw_tp"] / max(1, cum["raw_gt"])
                    cum_iou = cum["lane_inter"] / np.maximum(1, cum["lane_union"])
                    nav_hint = (
                        f"TM控车·3.1感知 | P:{cum_p:.0%} R:{cum_r:.0%} "
                        f"幽灵:{fs['ghost_fp']} 漏检:{fs['miss_fn']} | 虚线IoU:{cum_iou[1]:.2f}"
                    )

                    intermediate.result()
                    delta_np = delta_gpu[0].cpu().numpy() if delta_gpu is not None else None
                    is_clean = args.sr_mode == "clean"
                    scene = prediction_scene(
                        pred,
                        speed=speed_mps,
                        cruise=(80.0 if in_highway else 50.0) / 3.6,
                        intent="Follow",
                        previous_scene=prev_scene if is_clean else None,
                        ego_motion=delta_np if is_clean else None,
                        lidar_bev=lbev if is_clean else None,
                        postprocessor=post if is_clean else None,
                        clean=is_clean,
                        steer=float(ctrl.steer),
                        throttle=float(ctrl.throttle),
                        brake=float(ctrl.brake),
                        infer_ms=perc_ms,
                        limit=80.0 if in_highway else 50.0,
                        nav_hint=nav_hint,
                        mode="LIVE" if is_clean else "RAW",
                    )
                    # User-facing SR shows only current detections + road/path, never MPC branches.
                    scene["alternatives"] = ()
                    prev_scene = scene

                    paths = []
                    if tick_idx == shot_tick:
                        paths.append(args.out_dir / "live_sr_preview.png")
                    if tick_idx in save_set:
                        paths.append(args.out_dir / f"live_sr_tick{tick_idx:04d}.png")
                    packet = (scene, bgr(chase_frames.get(timeout=10)).copy() if need_shot else None, paths)
                    while not stopped.is_set():
                        try:
                            frames.put(packet, block=need_shot, timeout=0.05 if need_shot else None)
                            break
                        except queue.Full:
                            if not need_shot:
                                break

                    ctrl_ms = (perf_counter() - t_ctrl0) * 1000.0
                    latencies_perc.append(perc_ms)
                    latencies_ctrl.append(ctrl_ms)
                    if tick_idx < 3:
                        print(
                            f"[diag tick {tick_idx}] perc={perc_ms:.1f}ms ctrl={ctrl_ms:.1f}ms "
                            f"v={speed_mps * 3.6:.1f}km/h raw={fs['raw_tp']}/{fs['raw_pred']}/{fs['raw_gt']}",
                            flush=True,
                        )

                    stuck_ticks = stuck_ticks + 1 if speed_mps < 0.25 else 0
                    if stuck_ticks >= 80:
                        recoveries += 1
                        stuck_ticks = 0
                        target_idx = min(len(nav.wps) - 1, nav.cursor + 6)
                        rec_tf = raised_spawn(carla, nav.wps[target_idx].transform, 0.45)
                        kept_npcs = []
                        for npc in npcs:
                            if not npc.is_alive:
                                continue
                            if npc.get_location().distance(rec_tf.location) < 9.0:
                                npc.destroy()
                            else:
                                kept_npcs.append(npc)
                        npcs[:] = kept_npcs
                        ego.set_transform(rec_tf)
                        ego.set_target_velocity(carla.Vector3D(0, 0, 0))
                        ego.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                        ego.set_autopilot(True, tm_port)
                        prev_latent = None
                        prev_pose = None
                        prev_scene = None
                        if post is not None:
                            post.reset()

                    if tick_idx > 0 and tick_idx % 100 == 0:
                        _recycle_distant_npcs(carla, world, tm, npcs, ego_tf.location, nav, rng, args.npc)

                    log_every = max(1, int(args.log_every))
                    if tick_idx % log_every == 0:
                        row = {
                            "tick": tick_idx,
                            "sim_s": round(tick_idx * DT, 1),
                            "map": args.map,
                            "zone": "highway" if in_highway else ("junction" if cur_wp.is_junction else "urban"),
                            "total_dist_m": round(driven_dist_m, 1),
                            "speed_kmh": round(speed_mps * 3.6, 1),
                            "raw_p3m": round(cum_p, 4),
                            "raw_r3m": round(cum_r, 4),
                            "near_r40m": round(cum["near_tp"] / max(1, cum["near_gt"]), 4),
                            "clean_p3m": round(cum["clean_tp"] / max(1, cum["clean_pred"]), 4),
                            "clean_r3m": round(cum["clean_tp"] / max(1, cum["raw_gt"]), 4),
                            "dup_fp_cum": int(cum["dup_fp"]),
                            "ghost_fp_cum": int(cum["ghost_fp"]),
                            "lane_iou_cum": [round(float(x), 4) for x in cum_iou],
                            "sensor_ms": round(sensor_ms, 2),
                            "perc_ms": round(float(np.median(latencies_perc[-log_every:])), 2),
                            "ctrl_ms": round(float(np.median(latencies_ctrl[-log_every:])), 2),
                            "vram_alloc_mb": round(float(torch.cuda.max_memory_allocated() / (1024 ** 2)), 1),
                            "vram_res_mb": round(float(torch.cuda.max_memory_reserved() / (1024 ** 2)), 1),
                        }
                        with metrics_path.open("a", encoding="utf-8") as mf:
                            mf.write(json.dumps(row, allow_nan=False) + "\n")
                        print(json.dumps(row, ensure_ascii=False), flush=True)
                    tick_idx += 1

            cum_p = cum["raw_tp"] / max(1, cum["raw_pred"])
            cum_r = cum["raw_tp"] / max(1, cum["raw_gt"])
            cum_iou = (cum["lane_inter"] / np.maximum(1, cum["lane_union"])).tolist()
            summary = dict(
                driver="CARLA Traffic Manager Autopilot (perception audit only, not model closed-loop)",
                simulator=server_ver,
                map=args.map,
                sr_mode=args.sr_mode,
                checkpoint=meta,
                ticks=tick_idx,
                total_dist_m=round(driven_dist_m, 1),
                collisions=collisions["count"],
                recoveries=recoveries,
                raw_detection=dict(
                    tp=int(cum["raw_tp"]), predicted=int(cum["raw_pred"]), truth=int(cum["raw_gt"]),
                    precision_3m=float(cum_p), recall_3m=float(cum_r),
                    near_40m_recall=float(cum["near_tp"] / max(1, cum["near_gt"])),
                    duplicate_fp=int(cum["dup_fp"]), zero_lidar_ghost_fp=int(cum["ghost_fp"]),
                    missed_fn=int(cum["miss_fn"]),
                ),
                clean_lidar_detection=dict(
                    tp=int(cum["clean_tp"]), predicted=int(cum["clean_pred"]), truth=int(cum["raw_gt"]),
                    precision_3m=float(cum["clean_tp"] / max(1, cum["clean_pred"])),
                    recall_3m=float(cum["clean_tp"] / max(1, cum["raw_gt"])),
                ),
                lane_iou_roi={name: float(val) for name, val in zip(LANE_NAMES, cum_iou)},
                latency_ms=dict(
                    perc_median=round(float(np.median(latencies_perc)), 2) if latencies_perc else 0.0,
                    perc_p95=round(float(np.percentile(latencies_perc, 95)), 2) if latencies_perc else 0.0,
                    ctrl_median=round(float(np.median(latencies_ctrl)), 2) if latencies_ctrl else 0.0,
                    ctrl_p95=round(float(np.percentile(latencies_ctrl, 95)), 2) if latencies_ctrl else 0.0,
                    control_period_ms=round(DT * 1000.0, 1),
                ),
                vram=dict(
                    peak_alloc_mb=round(float(torch.cuda.max_memory_allocated() / (1024 ** 2)), 1),
                    peak_reserved_mb=round(float(torch.cuda.max_memory_reserved() / (1024 ** 2)), 1),
                ),
            )
            tmp = args.out_dir / "perception_audit_summary.json.part"
            tmp.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
            tmp.replace(args.out_dir / "perception_audit_summary.json")
        finally:
            for actor in [n for n in npcs if getattr(n, "is_alive", False)]:
                try:
                    actor.destroy()
                except Exception:
                    pass
            try:
                if getattr(ego, "is_alive", False):
                    ego.destroy()
            except Exception:
                pass


def _simulate_worker(args, frames, camera, stopped, messages):
    drive_perception_audit(args, frames, camera, stopped, messages.put)


def run_perception_demo(args):
    size = (960, 540)
    title = f"BEV-WAM-X | Stage 3.1 Perception Audit (TM Autopilot, {args.sr_mode.upper()})"
    viewer = SRViewer(size=size, backend="auto" if args.headless else "gpu",
                      window=not args.headless, maximized=not args.headless)
    viewer.show_occupancy = False
    ctx = mp.get_context("spawn")
    frames = ctx.Queue(maxsize=2)
    messages = ctx.Queue()
    stopped = ctx.Event()
    args.camera_size = (1920, 1080) if args.headless else viewer.glfw.get_framebuffer_size(viewer.window)
    camera = LiveCamera(ctx, args.camera_size)
    worker = ctx.Process(target=_simulate_worker, args=(args, frames, camera, stopped, messages))
    try:
        if not args.headless:
            glfw = viewer.glfw
            camera_texture = viewer.ctx.texture(args.camera_size, 4)
            camera_texture.swizzle = "BGRA"
            sr_texture = viewer.ctx.texture(size, 4)
            viewer.resolved.release()
            viewer.resolved = viewer.ctx.framebuffer(sr_texture)

        def draw(texture, viewport):
            viewer.ctx.viewport = viewport
            texture.use()
            viewer.hud_vao.render(viewer.gl.TRIANGLE_STRIP, vertices=4)

        def present(show_sr=False):
            viewer.ctx.screen.use()
            viewer.ctx.screen.clear(0.04, 0.04, 0.04)
            viewer.ctx.disable(viewer.gl.BLEND | viewer.gl.DEPTH_TEST)
            fw, fh = glfw.get_framebuffer_size(viewer.window)
            cw, ch = args.camera_size
            scale = min(fw / cw, fh / ch)
            w, h = int(cw * scale), int(ch * scale)
            x, y = (fw - w) // 2, (fh - h) // 2
            draw(camera_texture, (x, y, w, h))
            if show_sr:
                viewer.hud_program["flip_y"].value = True
                sw, sh, margin = inset_size(w, h)
                draw(sr_texture, (x + margin, y + margin, sw, sh))
                viewer.hud_program["flip_y"].value = False
            glfw.swap_buffers(viewer.window)

        def status(message):
            print(message, flush=True)
            if not args.headless:
                w, h = args.camera_size
                image = np.full((h, w, 3), 24, dtype=np.uint8)
                cv2.putText(image, message, (36, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (235, 235, 235), 2, cv2.LINE_AA)
                camera_texture.write(cv2.cvtColor(image, cv2.COLOR_BGR2BGRA))
                present()

        if not args.headless:
            glfw.set_window_title(viewer.window, title)
        status("Starting Stage 3.1 Perception Audit")
        worker.start()
        previous_frame = -1
        have_sr = False
        while True:
            if not args.headless:
                glfw.poll_events()
                if glfw.window_should_close(viewer.window):
                    break
            try:
                status(messages.get_nowait())
            except queue.Empty:
                pass
            try:
                scene, chase, paths = frames.get(timeout=0.002)
            except queue.Empty:
                if worker.exitcode is not None:
                    break
                scene = None
            if scene is not None and (not args.headless or paths):
                sr = viewer.render(scene, readback=bool(paths), present=False)
                have_sr = True
                if paths:
                    h, w = chase.shape[:2]
                    sw, sh, margin = inset_size(w, h)
                    chase[h - margin - sh:h - margin, margin:margin + sw] = cv2.resize(sr, (sw, sh))
                    encoded = cv2.imencode(".png", chase)[1].tobytes()
                    for path in paths:
                        path.write_bytes(encoded)
            if not args.headless:
                current = camera.upload(camera_texture, previous_frame)
                if current != previous_frame:
                    previous_frame = current
                    present(have_sr)
    finally:
        stopped.set()
        viewer.close()
        if worker.pid is not None:
            while worker.is_alive():
                try:
                    frames.get(timeout=0.05)
                except queue.Empty:
                    pass
            worker.join()
        frames.close()
        messages.close()
    if worker.exitcode:
        raise RuntimeError(f"Live perception audit exited with code {worker.exitcode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("models/v3_stage31b_step7968/best.ckpt"))
    parser.add_argument("--carla-root", type=Path, default=Path("runtime/Carla-1.0-Win64-Shipping"))
    parser.add_argument("--map", default="Town12")
    parser.add_argument("--port", type=int, default=2460)
    parser.add_argument("--quality", choices=("Low", "Medium", "High", "Epic"), default="Medium")
    parser.add_argument("--npc", type=int, default=35)
    parser.add_argument("--sr-mode", choices=("raw", "clean"), default="raw")
    parser.add_argument("--max-ticks", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-frames", type=int, nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, default=Path("logs/stage31-perception-demo"))
    parser.add_argument("--seed", type=int, default=404)
    args = parser.parse_args()
    run_perception_demo(args)


if __name__ == "__main__":
    main()
