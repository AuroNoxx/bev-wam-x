# CPU longitudinal observability probe

## Scope
Authorized frozen-policy diagnostic only. No neural training, checkpoint modification, test access or CARLA run. Script: `scripts/probe_longitudinal_acceleration.py`; outputs: `artifacts/carla-longitudinal-acceleration-probe/`.

Train: 1,570 unique previously used train windows, roads 388/392. Dev: road 713, original 214, startstop 314, recovery 628. These are reused development episodes, not fresh generalization tests. Strong overlap and scripted dynamics mean window counts are not independent sample counts.

## Pre-fixed protocol
Numpy float64 closed-form ridge, mean MSE + 0.01 standardized coefficient L2; unpenalized intercept. Train-only standardization. Baseline current horizontal speed and near navigation x/y. Sole extra feature: horizontal-speed difference from t-0.6 to current divided by actual elapsed time. Seven past/current rows only. Targets future-minus-current horizontal speed at 0.2/0.6/1 seconds. No phase, frame index, target speed, future control or countdown features. No nonnegative clipping or static gate, so this is not a deployable speed predictor. Synthetic constant-acceleration and future-poison invariance checks passed. CPU runtime 1.95 seconds.

Groups use current speed <0.2 m/s for static; otherwise future 0.6-second delta below -0.2 for decelerating, absolute delta <=0.2 for steady, otherwise accelerating. Future-dependent groups are reporting strata only, never predictor inputs. Different from previous five-second trajectory groups.

## Results (MAE m/s, horizons 0.2 / 0.6 / 1 s)
Recovery dev:
- Decelerating n122: baseline .2618/.7728/1.2519; +acceleration .0399/.1402/.2742.
- Steady n244: .0334/.1212/.2387 -> .0170/.0735/.1627.
- Accelerating n74: .4327/1.2313/1.9370 -> .0768/.2715/.4960.
- Static n188: .0204/.0678/.2135 -> .0503/.2029/.4118 (worse).

Original dev decelerating n61: 0.6 s .7776 -> .1442. Startstop dev decelerating n61: .7777 -> .1443.

Simple constant-acceleration extrapolation is stronger than fitted ridge for moving/decelerating cases here: recovery deceleration .0288/.1057/.2309. But it also worsens static compared with constant-speed. This is evidence the scripted dynamics are locally simple and history carries useful information, not evidence a large model is needed or that general driving is solved.

## Interpretation
The missing ego dynamic feature hypothesis survives for short-horizon moving/decelerating behavior. It does not resolve unobservable scripted future start times or teach front-car-reactive braking. Static deterioration rules out adopting this global linear predictor as a fix. Unbounded linear/extrapolation behavior near zero and hidden go/hold modes both need distinct treatment; no post-hoc gate was introduced or tuned. Policy integration, five-second trajectory behavior and closed-loop quality remain untested.
