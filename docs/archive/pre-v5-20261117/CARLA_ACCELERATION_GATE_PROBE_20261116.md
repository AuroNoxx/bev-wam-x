# Static acceleration gating CPU diagnostic

Script: scripts/probe_longitudinal_acceleration_gate.py. Report: artifacts/carla-longitudinal-acceleration-gate/report.json.

Same 1,570 unique train windows, roads388/392; dev713 with original214/startstop314/recovery628. Fixed ridge alpha0.01, train-only standardization. Additional acceleration feature multiplied by clip((current speed-.2)/.3,0,1), only current/past information. Predictor output is NOT gated or clamped. No main model training/checkpoint update/CARLA/test use. Runtime1.95s. Synthetic acceleration, future-poison invariance and gate-boundary tests passed.

0.6-second delta-speed MAE m/s on recovery dev, columns base / acceleration / gated acceleration:
- Static188: .0678 / .2029 / .1220.
- Steady244: .1212 / .0735 / .0698.
- Deceleration122: .7728 / .1402 / .1129.
- Acceleration74: 1.2313 / .2715 / .2757.

Static 1-second error .2135 / .4118 / .2985. Therefore gate mitigates but does not remove static regression. Very sparse gate-transition coverage: train7, dev original0/startstop1/recovery2; recovery transition0.6s errors1.5288 / 1.1773 / 1.4152. Cannot certify threshold or transition behavior. All readouts refit all coefficients/intercepts, so static feature zero is not a promise of unchanged base output. Existing stop/go ambiguity persists. Do not adopt as policy fix or automatically start a new experiment.
