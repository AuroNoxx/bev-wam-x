# Reactive history dynamics paired 256-step diagnostic

## Authorization and scope
Latest user-authorized minimal-loop investigation, coordinated by root. New isolated cache and checkpoints only; no test, default overwrite, closed-loop or automatic long continuation.

## Causal source / information gap
Source hold_v3: one 60s train episode road388 seed3101 and one 60s dev road713 seed4101. Train cruise8m/s, dev8.5m/s: route/cruise/weather/event timing are confounded. 514 windows each after first3.6s removal and full5s target availability. Reused dev, highly overlapping windows, not514 independent driving trials.

Teacher observes oracle bounding-box gap and lead velocity at frame f, applies control, row/sensors/ego describe f+1. Actor uses only current/past semantic LiDAR+ego pose truth, current speed, current HD-map near navigation, last-issued cruise and permission (age0.1s). Those commands are taken from the recorded f observation, never from the next row. Teacher target_speed, hold flag, relative lead fields, lead target/schedule and episode phase are excluded. Future trajectory and future OCC are labels only. Future scheduled movement permission is not known: horizons crossing such command transitions remain inherently uncertain.

Actor is not real-perception validated: semantic rays and ego pose are simulator oracle. Lead is implicit in sparse ray-hit OCC, not explicitly tracked. Teacher has privileged front-car information that actor may fail to recover. Unknown mean99.247%, min99.185%, max99.315%; unseen255 never free. No asserted collision-free label from absent callback.

## Filtering and tests
Any history/supervision-span invalid frame, guard or safety takeover removes a training window with recorded reason, without editing raw data. Current source has none, so all514/514 survive. Filtering on future validity is dataset supervision censorship, not an actor feature or proof failed driving is solved.

Checked raw/sensor/ego frame equality, pose equality, f-to-f+1 teacher age, contiguous0.1s history, navigation FLU/50, class1..7/unknown255. Poison teacher-only state/decision/phase leaves actor exactly unchanged. FP32 and BF16 model predictions exactly invariant to future-label poisoning. Zero acceleration first-layer column gives exact paired initial predictions in both precisions.

## Independent variable and budget
Existing small NavigationScaledFlow: width128,2layers,4heads, two-dimensional velocity residual [10,.5]m/s; internal navigation gain50 compensated at initialization. No previously trained checkpoint loaded. Shared all inputs and losses; baseline state3=0, history state3=(speed_t-speed_t-.6)/elapsed/5; valid flag shared1. No static gate in this experiment.

Seeds2070/2071, each base/history256steps, batch12, same3072 sequence perseed with all514unique train windows covered. Order reversed in secondseed. AdamWlr1e-4,wd.01,clip1; original64warmup+512stepcosine retained but stop256. Same FM/noise/time RNG, first3 trace checked. Existing .2position and .1each world terms unchanged. Current-support world limitation remains.

Lightning BF16, StatefulDataLoader, VerifiedCheckpointIO, latest every64, best at validation128/256. TensorBoard logging. Validation position is only checkpoint fit diagnostic, not driving score. Final full dev K4/16Euler same noise pergroup; report command-hold/static-permitted/deceleration/acceleration/steady, speed errors, motion and progress proxies, world vs persistence, collision/TTC unknown. Safety and control feasibility require later separate validation.

## Completed results
Four arms completed and process exited, total308.64s including evaluation; train66.86-68.48s/arm,966,939 trainable parameters,peak0.936GiB. Paired sequence byte-equal, first3 FM RNG hashes matched. All best/latest at256 with optimizer/scheduler/loops/callback state present; no actual resume performed. 514 unique train windows seen in3072draws perarm.

Speed MAE at0.6s all514dev: seed2070 base2.140/history1.729m/s; seed2071 base1.638/history1.654. Five-second deceleration169windows:6.020->5.275 for2070 but5.178->5.314 for2071. Thus proposed feature gain does not reproduce across seeds. Seed2070 static-permitted20window5s error4.155->4.584, acceleration128window2.256->2.576 worsened. Prediction jerk peak means139-163m/s^3 remain poor. World prediction vs own contemporaneous EMA persistence MSE:2070base .391/.105, history .404/.108;2071base .416/.094, history .419/.099. Target spaces differ across arms; ratios are within-arm diagnostics, not shared-reference world quality comparison.

No continuation or closed loop. Only two sourceepisodes and256 updates do not establish convergence or generalization. Missing teacher-observer information, sparse OCC and pure-noise generation learning remain distinct candidates, not proven causal explanations of all error.
