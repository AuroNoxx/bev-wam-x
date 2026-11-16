# Rear-reference sensitivity, 2026-11-16

CPU-only audit, no simulator connection, training or weight changes. Script scripts/audit_rear_reference_sensitivity.py; artifacts/rear-reference-sensitivity-v1/report.json. 2912 windows in 3.75 s.

## Provenance and limits
Verified historical logs/occ-closed-loop/20261115-220619-smoke/report.json.geometry: wheelbase 2.874995708465576 m, rear_axle_x -1.4375089406967163 m, max steering .6108652381980153 rad. Same tesla.model3 blueprint historical measurement; current hold_v3/recovery episodes did not save their wheel physics. These are NOT episode-certified parameters. Missing full wheel xyz, center of mass/reference details, steering-speed curve, tire/slip and actuator dynamics.

## Correction
With d = 1.4375089406967163 m, relative actor displacement = integrated rear displacement + d*[cos(yaw)-1, sin(yaw)]. Current yaw frame is FLU, body yaw is independent of travel direction. Rear lateral velocity diagnostic = actor lateral velocity - d*yaw_rate. Pose central differences used for that diagnostic are offline labels only, NOT actor input. Pure longitudinal offset leaves longitudinal velocity unchanged under planar rigid-body assumptions. Pitch/roll/uneven ground are not modeled fully.

Analytic forward/reverse/rest rear-to-actor-to-rear round trips and numerical trajectory tests pass (<0.0001 m synthetic error).

## Results (nominal historical geometry)
Recovery dev start position error: 0.04970 -> 0.02788 m; endpoint 0.07112 -> 0.04104 m.
Recovery train start: 0.04963 -> 0.02737 m.
Recovery dev other: 0.01404 -> 0.00845 m; stop: 0.02174 -> 0.02131 m; static unchanged.
Mean apparent lateral speed recovery dev: 0.00491 -> 0.000714 m/s; train: 0.00396 -> 0.000523 m/s.
Reactive dev other position: 0.01275 -> 0.00963 m; reactive train almost unchanged, and its tiny lateral residual increases slightly. This is a reference-point correction supported by trends, not universal perfect equivalence.

## Preset sensitivity, not parameter fitting
Offset +/-0.10 m: recovery dev start errors 0.02725 / 0.02788 / 0.02895 m. Do not pick the smaller value on dev; nominal historical geometry stays nominal. This range is a diagnostic perturbation, not a measured uncertainty interval.
Wheelbase +/-5% changes the inferred steering conversion at fixed curvature, not the curvature-based trajectory itself. Largest inferred knot steer is 0.16864 rad, below historical 0.61087 rad. This only checks geometric amplitude on these labels; does not certify steering rate, lateral acceleration, tires, controller tracking or other speeds.

## Low-speed supervision
Fixed |midpoint signed speed| >=0.5 m/s mask retained. Rank-deficient windows remain 982/2912; 370 windows have no observable curvature interval at all (reactive train 2, recovery train 246, recovery dev 122). Zero coefficients there are arbitrary reconstruction defaults, not known targets. Future coefficient supervision must not regress arbitrary least-norm nullspace values. Prefer supported yaw-rate/trajectory-domain losses with explicit validity; preserve standstill yaw and body heading separately. No turn/reverse expansion of real data occurred.

## Conclusion
Rear-reference correction resolves much of the prior apparent lateral mismatch and improves label reconstruction, but this is still historical-geometry, narrow-coverage offline evidence. No learned policy improvement established. Do not automatically train or run CARLA; first retain a decoder/validity contract distinguishing observed low-speed dynamics, unsupported curvature and physical constraints.
