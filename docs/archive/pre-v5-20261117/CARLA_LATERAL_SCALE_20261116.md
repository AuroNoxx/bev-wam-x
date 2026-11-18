# CARLA lateral coordinate scale diagnostic, 2026-11-16

User-authorized bounded single-variable test, not deployment acceptance. No official test, old weights or guards changed. Artifacts: artifacts/carla-lateral-scale512.

## Preflight
1570 unique training windows only: lateral velocity RMS 0.08047146 m/s, absolute P95 0.22392562, maximum 0.39738891. Choose lateral coordinate scale 0.5 m/s, longitudinal remains 10 m/s. This is normalization, not clipping or a physical limit. q roundtrip error 2.38e-7; physical equivalence error 7.45e-9; positive FLU y, no clipping and future-poison actor isolation passed.

## Protocol
Same seed 2060, initial parameter values, exact 6144 draws (original 50% real recovery mixture, not rejected 20% corrective reweight), navigation, model width128/layers2/heads4, batch12, 512 steps, objective weights, 64-step warmup and AdamW schedule. Fresh training, not checkpoint continuation. Baseline is on_recovery in carla-nav-recovery-factorial512. First3 forward RNG traces identical. Extra subclass construction changes ambient CPU RNG but loss RNG explicitly forked/seeded per update; loader fixed order/no shuffle. Each dev sample gets same four seeds and 16 Euler steps. Constructor parameters identical; only lateral q scaling changes target/physical noise and effective physical objective weighting. This does not isolate these latter mechanisms from one another.

Lightning + StatefulDataLoader + VerifiedCheckpointIO; latest/best separate. Actual 512 steps in144.56s; peak allocated0.9354GiB. Completed process exit0, no continuation.

## Results old -> scaled
Recovery628 windows: mean0.6s abs lateral error0.12001 ->0.01847m; offcenter114:0.12222 ->0.02601m. Peak lateral acceleration per candidate mean8.888 ->0.667m/s2; terminal lateral candidate std0.4509 ->0.1387m. Position diagnostic5.221 ->5.059m, peak vector jerk178.49 ->163.08m/s3.
Recovery deceleration162: predicted5s speed6.131 ->6.245m/s vs label1.836, position diagnostic5.222 ->5.384m (worse).
Recovery start68: reverse endpoint fraction0.6029 ->0.5441, average forward endpoint -2.396 ->-1.677m (still unusable); predicted endpoint speed1.972 ->1.752 vs label7.084m/s. Stopped120: endpoint speed1.814 ->1.636 vs label0, still drifting.
Original/startstop dev lateral error also drops to~0.015m but deceleration speed error worsens slightly. Longitudinal candidate endpoint std remains~7.9m. Label metrics are diagnostic only, not driving quality ground truth.

## Interpretation and next minimum
Evidence supports a substantial physical lateral noise-scale contribution. Lower noise is not evidence of conditional corrective learning: inspect sign-specific moving-recovery response versus straight/zero-nav baselines before calling it recovery. Do not adopt default or launch more sampling sweeps. Longitudinal noise/intent ambiguity remains: static wait and start share nearly identical observations while script waits seven seconds with no explicit go intent; deceleration follows scripted timing rather than a reactive expert. Next useful read-only check is conditional target ambiguity and longitudinal noise-to-target scale with fixed temporal/semantic groups, plus sign-specific recovery sensitivity. Any new scale experiment should remain a separately approved single variable, not coupled loss changes.

No closed loop for this checkpoint. Existing nav worker creates CompareModel with old scale10: it MUST load ScaledVelocityFlow from checkpoint settings and pass an independent parity test before any future closed loop. Collision/TTC/lane safety unknown offline; no claim of safety. Sparse positive semantic LiDAR/pose are simulator oracle; unhit cells unknown, not free. One seed and repeatedly inspected dev only.

## Follow-up frozen navigation counterfactual
artifacts/carla-lateral-scale-nav-frozen/report.json: 12 distinct dev states (two recovery signs, indices50/80/90/100/110/330), fixed four Gaussian candidates, paired actual/flip-y/zero-navigation. No updates. Scaled0.6s mean-y response to flip-y is approximately0 to0.00006m. At moving index80 true0.6s y is+0.0616/-0.0615m but predicted+0.0129/-0.0016m. At one index100 true-0.0371m but predicted+0.0208m. Thus reduced output noise is established, learned robust corrective response is not. Root notified, CARLA agent told NOT to run pending decision, since conditional no-obvious-anomaly gate is not satisfied. scripts/carla_scaled_policy_worker.py prepared to load checkpoint velocity_scales_mps, but not yet independently validated and not used live.
