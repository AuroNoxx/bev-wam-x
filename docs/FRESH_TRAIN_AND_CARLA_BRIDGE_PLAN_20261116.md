# Fresh-start budget and learned-policy CARLA bridge plan

Status: planning/preflight only; NO long training or CARLA launch.

## Fresh comparison initial conditions
Two predeclared independent seeds, each creates ONE fresh MaskedHistoryJoint(768,15,12) state and clones identical shared tensors into fusion and history-only arms. Fresh optimizer moments/data iterator/RNG; no prior fused checkpoint loaded. Initial EMA encoder equals online encoder and is frozen for backprop. The8step smoke verifies identical full initial state, EMA copy and first8batch hashes. First implementation attempted the model-only state at the wrapper namespace and failed before any update; corrected model.load_state_dict in isolatedv2 run, not a training instability.

Same normalized4-frame/history validity/time/geometry contract,20-state,25x5trajectory output,5296train windows, batch8. Fusion adds existing world objectives and predicted context; no-JEPA removes both and freezes legacy inactive heads. Same base FM/physical objectives, AdamW and paired per-step RNG. Effective active parameters155.22M/152.85M, not identical active FLOPs. No test access. Existing validation already influenced development; identify untouched route-level validation BEFORE making final selection claims, do not casually relabel reused data as heldout.

## Measured startup and estimate
Disposable8updates/arm, no checkpoint saved: warmed steps including batch reads0.2272s JEPA,0.1353s no-JEPA. First JEPA CUDA/Adam startup9.02s; allocated3.56/3.08GiB, reserved4.23/3.86GiB in this eager smoke (not promised Lightning peak).

Direct1e-4 without warmup produced second-step loss1166/679 and gradient norms18518/7211; all finite, but clearly NOT evidence of stable training. Shared linear128step warmup to1e-4 is a proposed correction under bounded startup testing, not a proven convergence fix. Clip1 retained; per-component weighted gradient/loss logs must reveal severe clipping instead of hiding it.

Proposed ONE experiment with milestones, not endless new short variants:
-2seeds x2arms, total maximum4096updates per arm (~6.19epochs;662steps/epoch).
-common128step linear warmup; cosine thereafter toward1e-5 at4096.
-milestone1024per arm (~1.55epochs) for finite/resume/data isolation and multi-metric review; do not silently extend or tune losses at this point.
-full4096per-arm raw update estimate:JEPA15.5min, no-JEPA9.2min; bothseeds total~49.5min raw. Allow checkpoint/validation/framework/IO overhead: provisional60-80min overall.
-first1024per-arm milestone:raw12.4min across4arms, provisional15-20min including overhead. Calibrate against actual first128/256logged steps rather than presenting estimates as promises.
-save recoverable rolling state and immutable agreed milestones; around1.9GB/checkpoint. Budget up to45GBnew storage for4arms, initial states, best/latest/milestones/logs; existing E:free~240GB at planning. Do not remove old weights to make room.
-resource conflict guard before each arm; no concurrent CARLA/heavyothertraining. Explicit failure stop, no blind restart. Numerical nonfinite, checkpoint verify failure, input leakage, or resource conflicts stop immediately. User-visible progress at milestones, no automatic run beyond4096.

World validation must compare prediction vs persistence in the SAME per-checkpoint EMA space. A fixed old/fresh encoder embedding is not automatically a common metric space for newly learned predictors. Pair this with fixed physical semantic labels on verified occupied-support subsets, dynamic proxies and target/mask drift; no free or entering-actor claims outside support. No-JEPA untrained legacy future head is N/A, not a competing world predictor. Driving acceptance remains multi-dimensional and ultimately closed-loop, not ADE/loss alone.

## Independent bridge engineering completed
New pure module scripts/learned_policy_bridge_contract.py and11passing tests reject future fields, wrong XYZ/TZYX shapes, invalid/missing current observations, reordered/future timestamps, NaN/negative-speed/malformed-heading candidates. Unknown0/255 maps to255; no free certification. Candidate decode preserves xy50/10,speed30,FLU and0.2s25point times. Outputs explicitly execution_authorized=false and collision/TTC/road/driving verdict=None. This is boundary validation only, NOT an operational policy bridge, regridder or safety supervisor.

## Learned candidates -> CARLA gap checklist
1. Existing run_occ_closed_loop.py is CLASSICAL sensor-only control, explicitly learned_models_loaded=false. Preserve it as baseline; do not imply its successes validate learned candidates.
2. Existing sensor grid is0.5m, origin(-20,-25,-1), XYZ(200,100,12), sparse hits/ray support. Learned input is TZYX(4,20,64,160), origin(-25.6,-25.6,-5), voxelXYZ(0.8,0.8,0.4). Shape reshape is not conversion. Need tested rasterization, axis sign, semantic mapping/version, ego mask, extent/unknown handling. Sparse ray support is not whole-volume free truth. Domain gap versus offline accumulated OCC persists.
3. Build strictly past timestamped0/.2/.4/.6s history; warp each past grid into CURRENT ego FLU using causal ego odometry. Missing/stale packets get explicit masks, not future interpolation. Validate dynamics objects remain past observations, not privileged NPC states. Source offline OCC provenance still not certified past-only.
4. Supply exactly normalized current20-state (speed,accel,yaw-rate,near/far navigation and commands) with validated frame/sign and permitted navigation source. Missing fields cannot silently become valid zeros. Actual state/IMU frame and latency need calibration.
5. Bridge Python environments/versions deliberately: Torch training venv and CARLA runtime may differ. Define timestamped bounded localIPC if needed; no API keys/external services. Deadline/missing-output policy must not stall synchronous simulation or reuse stale proposals indefinitely.
6. Learned25x5candidate is NOT throttle/brake/steer. Need explicit candidate selection with no ground-truth future, t0/full dynamics gate, vehicle footprint/observed hazards/unknown treatment, trajectory tracking to actuator limits, steering FLU->CARLA sign, latency compensation. Existing classical controller tracks a route/target speed, not a certified learned-trajectory tracking interface.
7. Preserve hard safety authority independent of model: invalid/late/unknown observation or failed candidate gates -> explicit conservative baseline/stop handoff, logging intervention reason. Do not equate clipping to a safe model. No universal acceleration/jerk thresholds silently adopted as safety guarantees.
8. Matched scenario fingerprint/seed, initial states, current sensorframe, modelcheckpoint,allcandidates/selected/executed controls, takeover source, collision/lane callback coverage, trajectory actuals and safety/efficiency/comfort logs. Truth participants may enter evaluation only, never actor. No-event without coverage remainsunknown.
9. Before learned control: offline adapter unit tests, recorded causal sensor replay, non-actuating shadow inference/timing, then separately authorized short low-speed closed loop on unused CARLA with only owned objects cleaned. No existing high-speed0-120claim. No learned CARLA run performed here.

NoAstra API/automaticRL wiring assumed; authentic evidence can still be forwarded by parent to the existing coach conversation.

## Subsequent bounded warmup evidence
Shared128step warmup tested through32updates per arm only, same freshseed/data/RNG; see FRESH_SHARED_WARMUP_CHECK_20261116.md. It avoids the prior second-step giant loss spike, but all raw gradients still clip and random-start candidate dynamics remain unusable. No full-warmup or convergence certification; no long training launched.
