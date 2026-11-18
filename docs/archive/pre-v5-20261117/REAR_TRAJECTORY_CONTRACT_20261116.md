# Fixed rear-reference prototype contract, 2026-11-16

Implementation is isolated and NOT connected to model training, inference worker or CARLA control. `scripts/rear_trajectory_contract.py`, `scripts/test_rear_trajectory_contract.py`; result `artifacts/rear-trajectory-contract-v1/tests.json`. CPU forward/backward only, no optimizer or checkpoint updates.

## Causal and representation contract
Decoder accepts only current signed longitudinal velocity and generated B x 6 x 2 continuous acceleration/curvature coefficients. Six fixed knots, five seconds; integrate rear speed/body yaw/rear position, transform to actor-origin displacement. Body yaw is not travel direction during reverse. No future trajectory, teacher lead pose/velocity or schedule enters forward. True current signed velocity must come from actor-observable odometry, not future labels.

Frozen historical geometry is used for equivalence with previous offline reconstruction. New independent same-blueprint physical snapshot from carla_build: artifacts/carla-model3-physics-contract-20261116/report.json, wheelbase2.874914811m, rear x about-1.43744m, front max steer35deg/rear0. This strongly corroborates historical geometry but does NOT retroactively certify each collection episode. Steering-speed curve/control-to-wheel dynamics remain uncalibrated; do not map normalized steering linearly as a proven actuator model.

## Labels and masks
- Trajectory, signed speed and body yaw losses are computed in measured valid target domain.
- Interval yaw rate is finite yaw difference including t0, not an assumed pre-t0 derivative.
- Curvature identifiability uses fixed ground-truth observation mask |midpoint signed speed|>=0.5 m/s plus interval validity. Never mask based on predicted speed: that would permit evading loss by predicting stop.
- At low speed, measured position/speed/body-yaw evidence remains usable; only unsupported curvature-related rate supervision is omitted. At exact rest finite curvature gives exactly zero motion/yaw and zero curvature gradient, as intended.
- No curvature coefficient target/loss exists. Least-norm fitted coefficients in underdetermined windows are NOT ground truth. Masked samples have explicit counts; count0 means no supervision, not perfect performance.
- Loss outputs retain SI units and counts. No mixed-unit total has been adopted for training. Existing scales/weights must be declared before a subsequent paired study, using train only.

## Passed tests
Torch float64 decoder equals previously audited numpy label decoder within1e-11; reconstruction target loss zero; low-speed target poisoning leaves loss exactly unchanged; gradients finite; standstill cannot move/yaw with curvature alone and curvature gradient is zero. No optimizer step. Existing full2912-window oracle-label reconstruction remains the expressivity evidence, not learned quality.

## Honest comparison design
Cannot claim that replacing the old 25-point velocity FM with 12 acceleration/curvature coefficients and new decoded losses is a single-variable FM experiment: head dimension, target distribution and loss all change. In particular, direct FM coefficient supervision would falsely teach the low-speed nullspace.

A defensible first single-variable diagnostic is a common deterministic12-output head, same fixed encoder/features/data/RNG/optimizer/budget and decoded position/signed-speed/body-yaw/rate losses, comparing the same fixed curvature decoder with actor-origin approximation versus rear-reference correction. This tests reference correction only, not FM vs regression or an entire new policy. Alternatively freeze generated coefficients and compare the two decoders without training (already audited on labels). Any later FM training method must separately address unidentifiable latent targets, train/inference distribution, stochastic candidates and loss normalization; that is a distinct design decision, not silently bundled here.

## Remaining
No action bounds, curvature-rate/steering-speed calibration or learned closed-loop claims. Real data have only small corrections, no substantive turning/reverse. Do not auto-train: first decide the minimal comparison and confirm physical reference/validity contract. Preserve old weights/test/guards.
