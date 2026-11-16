# Independent low-speed reactive pilot: frozen pre-collection contract

Date: 2026-11-16. Scope: design and subsequent data integrity audit only. No training or closed-loop authorization is implied by this document. Collection is separately coordinated by carla_build under root authorization. Preserve all previous data, checkpoints, code defaults and failed runs.

## Single intended change
Replace the previous one-train/one-dev scripted-command dataset with independent reactive episodes. Do not simultaneously change attention, representation, loss, teacher policy, control guards or model scale. This is a limited data pilot, not a claim of statistical sufficiency or real-road safety.

Collection design: 6 x 60 recorded seconds, 4 train (roads 388/392, 2 each), 2 dev (713); seeds 6101..6104 and 7101..7102. Both splits use the same predetermined uniform parameter ranges: cruise 7.5..9 m/s, initial bumper gap 14..22 m; lead start 8..10 s, deceleration 22..25 s, restart 35..38 s, final deceleration 47..50 s. Lead schedules are privileged simulation configuration, NEVER actor fields. Record the actual sampled values and source hashes before collection. Do not replace seeds after observing performance. Road remains split-confounded: same parameter range does not remove road confounding, and 2 dev episodes are not generalization certification.

Seven seconds of brake settling precede the recorded episode. During every recorded episode movement_permission=True and cruise is constant. Ego stops in reaction to the lead, not to a hidden scheduled ego command. This dataset does not validate obeying permission=False, command changes, reverse, occlusion or high-speed operation.

## Actor information boundary
Keep current implemented actor input: four OCC frames at [-.6,-.4,-.2,0] seconds, their validity/age, current ego horizontal speed, causal .6 s speed-derived acceleration and validity, current cruise/permission, current near navigation. No new explicit lead-state branch in this pilot. Oracle semantic ray labels, ego pose and HD-map navigation remain simulation/oracle limitations, not production perception.

OCC: transform historical observed points into current ego frame, use existing voxel/class mapping and self-return filtering; 255 means unknown, never free. No future pose alignment, future sweeps or object-ID-selected points in actor. No lead truth, bbox, target speed, phase, schedule, elapsed episode time, future command mask or future valid length in actor.

Separate current command record (value/frame/timestamp) from previous observation used to issue control. Preserve observation f -> applied control -> post-control raw/pose f+1. Constant command must make the two command values agree; do not silently treat prior teacher gap/speed as current observation. Navigation is recomputed from the current pose and fixed intended road geometry, not global-nearest overlapping-road matching. World-frame/CARLA y-right to ego-FLU y-left conversion stays unchanged.

GT lead pose/ID/bbox may be saved under an evaluation-only namespace for coverage/error attribution. Their removal, permutation or extreme-value poisoning must leave actor tensors exactly unchanged.

## Frozen model/label contract for any later decision
Retain the current 25 x 2 velocity-residual FM representation, scales [10,.5], 25 samples at .2 s over 5 s and existing cumulative trapezoid decoder. Do not combine this data change with the 12-coefficient rear-axle prototype. Keep the last bidirectional reference topology and FP32 + MATH numerical setting if a later comparison is approved; causal attention and incomplete supervision are not part of this data pilot.

Existing loss is FM MSE + .2 gated position SmoothL1(error/5 m, flow_t >= .5) + .1 each JEPA latent/variance/semantic. No added kinematic loss just because a config contains dormant fields. These are a frozen comparison reference, NOT an endorsement of the currently poor dynamics/world quality. New training budget, initialization, control arm and learning decision require review after data audit; no automatic warm-start or from-scratch training.

Future trajectories/world fields are labels only. Build windows strictly inside one episode, with timestamp continuity and no train/dev overlap. A supervised window spanning missing frames, invalid/collision/takeover states is separately marked invalid; retain its raw records and include the failure in episode-level outcomes. Do not censor poor but valid behavior merely for high loss, slow progress or ambiguity. Report selection counts and bias.

## Predefined measurements (no dev tuning)
Use episode-level outcomes first; report window counts second. All unknown/unsupported metrics stay null with reasons and event coverage, never zero.

Data: actual sampled parameter distributions by split, raw sync/pose/navigation errors, 4-frame history availability, unknown fraction, lead visibility via evaluation-only ID, association/low-motion uncertainty, valid windows and exclusion reasons. Report actual stop and restart events rather than intended schedule coverage.

Event definitions for diagnostics: settled stop is horizontal speed < .05 m/s continuously for >= 1 s; sustained movement is > .5 m/s for >= 1 s. Report lead/ego event timestamps and latency, with no-event/insufficient-followup explicitly missing. Also report continuous speed distributions so threshold-defined summaries do not hide creeping.

Offline later: per-episode and per-phase position/speed fit (diagnostic, not safety), complete t0-inclusive acceleration/jerk, signed position progress versus predicted speed, backwards displacement/negative longitudinal speed, stopped-state drift, recovery lateral/yaw response only if present, fixed-noise candidate diversity; world same-reference persistence and valid/moving occupancy separately. No invented collision or TTC from offline predicted points without the required scene geometry/time model.

Closed-loop later, only separately approved: route-aligned progress, lateral displacement/heading, actual speed, stop/restart response, projected bumper gap and overlap, positive-closing TTC plus null reason, collision callback coverage, lane departure, control/guard intervention counts and first-event time. Compare identical episode initial conditions and lead realization, do not select best candidate seed after seeing outcomes. Paused synchronous simulation timing is not a realtime claim; cold/warm p50/p95/p99 wall latency separately.

Existing safeguards remain unchanged: collision stop; speed >14 m/s stop; fixed road/lane departure stop; lateral-overlap projected gap <2 m or valid TTC <1 s takeover; candidate lookahead x<-.5 m, max |y|>15 m or max predicted speed>20 m invalid; worker error/timeout stop. Verify exact running guard source before any later run. Collection teacher safeguards and model guards are distinct and must be separately enumerated, not declared identical by name.

## Gates and stop conditions
1. Collection manifest complete or explicitly failed; verify all saved frame/timestamp/pose packets and sources. Failure is evidence, not permission to retry new seeds.
2. Audit actor whitelist, future/GT poison invariance, navigation/units and command constancy; reject affected windows with logged reasons if violated. No silent repairs to raw data.
3. Count independent stop/restart events and route/parameter coverage. Missing coverage is reported and blocks that category's claim, not filled with zero metrics.
4. Return audit and actual resource estimate to root. Decide whether any controlled training is justified. No training or CARLA model run follows automatically.
5. Do not call a run successful merely because guards did not fire. A minimal learned-loop milestone needs actual start, meaningful progress, reactive slowing, stop and restart coverage with no guard intervention. It remains a low-speed oracle simulation milestone, not a safety certification.
