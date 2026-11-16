# Signed-speed and curvature representation audit, 2026-11-16

Scope: offline CPU only. No training, checkpoint edits, simulator connection or driving. No official test. Script `scripts/audit_signed_curvature_expressivity.py`; evidence `artifacts/signed-curvature-expressivity-v1/report.json`. 2912 label windows, 4.07 seconds.

## Fixed contract
Six acceleration knots and six curvature knots over 5 seconds. Integrate longitudinal signed speed, then body yaw rate = signed speed * curvature, and planar position derivative = signed speed * [cos(body yaw), sin(body yaw)]. 0.01-second numerical integration; no threshold tuning on dev.

Current signed speed is current world velocity projected on current body forward axis. Coordinates: CARLA x forward/y right, FLU y sign reversed; relative body yaw sign reversed. Body heading is distinct from travel heading in reverse. Decoder takes only current speed and predicted coefficients; offline fitting of coefficients uses future labels, never an online actor input. No lead ground-truth data enters decoder.

Curvature fit uses interval yaw change and midpoint signed speed, fitting yaw_rate = v * kappa only for |v| >= 0.5 m/s. It does not divide by near-zero speed. Under-observed curvature is marked by design-matrix rank; entirely static windows get zero only as an arbitrary reconstruction convention, not supervision truth. A future training loss must mask such unsupported coefficients/observations.

## Results
Reactive train/dev 514 each; recovery train 1256, dev 628. No real reverse window (< -0.05 m/s), and actual turn coverage is only small correction, <= about 3.8 degrees over 5 seconds from previous audit.

Mean position reconstruction error:
- Reactive dev start 54 windows: 0.0225 m (p95 0.0472); stop 20: 0.00933 m.
- Recovery dev start 68: 0.0497 m (p95 0.0754), endpoint mean 0.0711 m; stop 92: 0.0217 m; stationary 120: 0.0000273 m.
- Recovery dev mean body lateral velocity magnitude 0.00491 m/s, maximum 0.0706 m/s. A zero-slip actor-origin approximation cannot represent this exactly; recovery start position error is larger than the previous free planar-vector representation (~0.0269 m).

Curvature design rank deficient: reactive train 129/514, dev 89/514; recovery train 510/1256, dev 254/628, total 982/2912. This is an identifiability limitation, not a fitting or training success.

Synthetic forward/reverse circles and rest passed analytic reconstruction tolerance <0.0001 m. Finite curvature cannot rotate/translate a stationary vehicle; reverse uses negative speed and independently integrated body yaw. These tests verify the idealized equations only, not CARLA steering, slip or controller feasibility. Acceleration and steering-rate bounds are not implemented.

## Vehicle geometry still unverified
Collector source explicitly spawns `vehicle.tesla.model3`; saved manifests/frames inspected here do not provide wheel physics, wheelbase, max steer or actor-origin-to-rear-axle offset. Do not substitute public real-car dimensions. Curvature can be integrated mathematically without wheelbase, but mapping steering angle = atan(wheelbase * curvature), rear-axle reference, steering saturation and steering-rate limits require actual simulator vehicle metadata. Current positions are the actor origin, NOT verified rear axle, so the no-slip equations at that point remain approximate.

## Decision
Promising label-compression candidate, not a verified vehicle action decoder and not yet authorization to train or drive. Smallest next step is static geometry/physics metadata verification, then reference-point-corrected reconstruction and masks for low-speed curvature. If unavailable, explicitly retain an idealized signed-speed/curvature trajectory model without claiming real steering equivalence. No invented forward/reverse intent, no future start countdown, and no zero-filled unsupported curvature targets.
