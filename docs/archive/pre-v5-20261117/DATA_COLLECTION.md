> 历史记录：已由 ARCHITECTURE.md 和 DATA_TRAINING_PLAN.md 取代当前方案。保留证据，不代表当前入口或新版架构结果。

# Continuous local highway collection

## Start / stop / inspect (PowerShell in this project)

```powershell
.\scripts\Start-Collection.ps1
Get-Content .\logs\collector-status.json
.\scripts\Stop-Collection.ps1
```

The launcher stops the preview client before starting collection: only the collector may tick the simulator. The dashboard is named `Highway OCC Flow - LIVE COLLECTION - BEV / OCC / RGB`. Closing it or pressing Escape requests a clean stop. A stop file is also supported; the next start archives that marker. Data stays under `data/raw/<timestamp>/`. No cloud connection is used.

Default collection is continuous, not a fixed sample budget. Episodes last at most 45 simulation seconds and end earlier at the highway segment boundary or a collision. Seeds increment between episodes. Collection stops if free project-disk space drops below 30 GiB, on explicit stop, or on a sensor/error condition. Sensor errors are not silently skipped or retried. The failed episode remains incomplete and its error is retained. Restart creates a new run; it does not append to an incomplete episode.

## What is actually collected

- CARLA 0.9.15 Town04, road 45, dry daytime, four-lane carriageway. This first baseline does **not** implement the proposed reduced 2–3-lane road or unrestricted highway routing.
- CARLA Traffic Manager demonstrations: cruise, slow lead vehicle, mixed traffic. Mixed traffic permits automatic lane changes; this is not proof of successful overtaking or safe-return coverage. Every episode remains `imitation_eligible: false` until quality review.
- Physics and sensor capture at 20 simulation Hz; every second aligned frame is saved, giving 10 simulation Hz. CPU label generation and compression can make wall-clock collection slower than real time; no timestamps are altered to disguise this.
- Four paired RGB cameras at 448×256, 110° horizontal FOV, yaw 0/-100/+100/180 degrees. These are inputs for a future visual student, not the source of the oracle BEV. Coverage and occlusion are not yet formally certified.
- Semantic LiDAR: 64 channels, 150 m range, 200k points/s, 20 Hz rotation. Raw structured points, semantic tags and object IDs are preserved.
- Exact frame IDs and timestamps, camera matrices/intrinsics, LiDAR/world and ego/world matrices, ego/neighbor transforms, velocities, accelerations, bounding boxes, lane/road IDs and station, route target, current controls and collisions.
- OpenDRIVE map and sampled road geometry accompany each run.

## BEV and OCC are different label products

`bev_actor_id` and `bev_velocity` rasterize privileged vehicle bounding boxes. This is a full-box oracle approximation, including occluded vehicles, not a vision prediction and not precise surface geometry. `bev_lane_id` uses sampled map lanes; samples do not replace the full OpenDRIVE geometry or exact drivable polygons.

`occ` is **observed surface occupancy**, not complete dense mesh truth. Semantic LiDAR hit points are occupied; measured ray segments before hits are sampled at 0.25 m to mark observed free cells; untouched cells remain unknown. Sparse rays and finite marching leave missing cells unknown. It does not fill unobserved vehicle interiors, infer occluded free space, or synthesize missing returns. `occ_semantic` labels observed hit cells only (255 elsewhere); overlapping surface classes in one cell use the last processed hit.

Grid order is `(height, left, forward) = (4,64,360)`, 0.5 m XY cells, forward range [-60,120) m, left range [-16,16) m, height edges [-1,0,1,2,3] m relative to ego origin. Values: 0 unknown, 1 observed free, 2 occupied surface. Road/ground surfaces can be occupied in the lowest height layer; occupancy alone is not a drivable-space label. LiDAR observes only part of this volume; the recorded observed fraction makes that limitation visible.

Raw CARLA coordinates use x-forward/y-right/z-up. Derived BEV/OCC and trajectories use x-forward/y-left/z-up; derived yaw is radians, left-positive. Raw matrices and angular velocity retain CARLA conventions. BEV velocity is world velocity rotated into the ego axes, not relative velocity with ego velocity subtracted.

## Episode files and future targets

Each `.npz` contains paired RGB, observed OCC/semantics, oracle BEV IDs/velocity/lanes, raw LiDAR, matrices and sensor frame/timestamp arrays. `frames.jsonl` contains per-frame metadata and controls. `events.json` records collisions. `episode.json` records the seed, scenario, completion, termination, counts and review status.

`future_trajectories.npz` contains 25 waypoints at 0.2-second intervals, a 5-second horizon. Fields are `(dt,x,y,yaw,speed)` in the source ego frame. A validity mask marks episode-end targets that do not exist; they are not extrapolated. Targets never cross episode boundaries. These are recorded TM trajectories, not counterfactual rollouts or human-certified expert trajectories. The sampled `control_at_frame` is not a claim of complete 20 Hz control history; raw planning records are at 10 Hz.

All splits remain **unassigned**. Do not randomly split adjacent frames or count the same route/seed as independent held-out evidence. Define episode/configuration groups and held-out traffic/road conditions before training. This single-road collection does not establish cross-road generalization.

## Verification

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_collection.py .\data\raw\<run>
```

The verifier reads every sample in completed episodes and checks frame/timestamp agreement, 0.1-second cadence, array shapes, unknown masks, highway location and future target masks. Incomplete episodes are excluded; inspect `ERROR.txt` and `episode.json` rather than treating file existence as success.

Initial pipeline check: 75 synchronized samples, 25 complete five-second targets, zero collisions, full archive verification passed. Around 2.55 samples/s wall-clock and 45 MB for that short run; this is an initial measurement, not a promised steady-state throughput. No model training has started.

Live run `20261114-192621` (seed 20261115): first highway segment completed automatically with 292 samples, 242 full five-second targets and zero collision events; all completed archives passed validation. The collector automatically entered the slow-lead episode. A live sample verified one neighbor, 40 occupied BEV box cells with nonzero velocity, and identical sensor frame IDs. At that check the dashboard showed 375 total saved samples and approximately 2.63 samples/s wall-clock. Counts continue increasing while the process runs.

An initial failed probe (`20261114-192214`) is retained with `ERROR.txt`: a sensor did not deliver the expected startup frame. The corrected collector uses `sensor_tick=0` (capture every physics tick), a ten-tick startup warmup, and strict frame matching thereafter. Warmup frames are deliberately outside the recorded episode; post-warmup missing frames cause failure rather than silent skipping. The successful pilot is `20261114-192311` and is separate from the live run. Code provenance for the live run is saved in `source_snapshot.json`.

## Remaining before a full training dataset

Quality filtering, held-out splits, verified lane-change/overtake/abort/emergency-braking coverage, action-diverse counterfactual rollouts, exact drivable polygons, denser occupancy supervision and explicit camera coverage tests remain. The current corpus is a reproducible oracle/observed-OCC collection baseline, not a finished autonomous-driving dataset.
