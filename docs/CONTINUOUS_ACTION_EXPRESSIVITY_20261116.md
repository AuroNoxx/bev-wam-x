# Continuous action representation: offline feasibility, 2026-11-16

No training, no checkpoint modification, no CARLA rerun. CPU label reconstruction took 3.41 s. Preset 6 and 11 knots across 5 s; neither selected by dev tuning. Script: scripts/audit_continuous_action_expressivity.py. Results: artifacts/continuous-action-expressivity-v1/report.json.

## Representation and information boundary
Piecewise-linear continuous planar acceleration, integrated analytically into velocity and position. 12 or 22 coefficients replace 50 independent velocity coordinates. Initial condition is the actual current signed ego-frame planar velocity, not future velocity or scalar speed assumed forward. Offline least-squares encoder uses future true velocity labels; this encoder is NOT an actor and MUST NOT be used online. Decoder takes only current velocity and generated coefficients. Future positions are evaluation only, not fitted.

Coefficient encode/decode reconstruction, t=0 integrals, rest, constant reverse, constant acceleration and zero crossing passed. Additional analytic circular-motion checks gave 6-knot mean position errors 0.0000614 m forward, 0.00000293 m reverse. These synthetic checks are not real turning validation.

## Label reconstruction
Read all 514 reactive train + 514 reactive dev, 1256 recovery train + 628 recovery dev. No official test. The actual routes only turn by <=0.066 rad (~3.8 degrees) across 5 seconds, so broad turns and reverse remain unrepresented in real data.

6 knots, reactive dev:
- Start 54 windows: velocity vector MAE 0.0430 m/s, mean position error 0.0226 m, endpoint 0.0293 m.
- Stop 20: velocity MAE 0.0258 m/s, mean position error 0.0100 m, endpoint 0.0202 m; endpoint velocity error 0.0399 m/s.
6 knots, recovery dev:
- Start 68: velocity MAE 0.0510 m/s, mean position error 0.0269 m.
- Stop 92: velocity MAE 0.0388 m/s, mean position error 0.0232 m.
- Entirely static 120: velocity MAE 0.0000093 m/s, mean position error 0.0000278 m.
11 knots reduce velocity reconstruction errors but often increase jerk. E.g. recovery dev start peak jerk mean 2.61 -> 5.18 m/s^3. No comfort threshold tuned against these results. Smoothness does not guarantee physical limits.

Old heading-times-horizontal-speed label proxy differs from true planar velocity by mean 0.00045/0.00203 m/s reactive train/dev and 0.00454/0.00489 m/s recovery train/dev. Old trapezoid position error means 0.0088--0.0147 m. These tiny discrepancies cannot explain the learned model's extreme jerk by themselves.

## Explicit failure / limitation
This is an expressive smooth planar representation, NOT yet a feasible car action representation. Counterexample: from rest and forward body heading, acceleration [0,1] produces pure lateral departure (0.02 m in 0.2 s). A no-slip car cannot do that. There are also no acceleration/curvature/steering bounds. Signed vector motion supports reversal but body heading is not travel direction during reverse; existing norm-speed plus velocity-direction output loses that distinction. At rest travel heading is undefined.

Therefore do NOT train or deploy this as the completed physical repair. Next smallest check is signed longitudinal speed + continuous bounded curvature/steering with integrated body yaw/position, using a verified wheelbase/steering contract. First reconstruct labels and test standstill/reverse/low-speed singularities, keeping body yaw independent of velocity direction. Do not infer gear or future start intent from future labels. Existing dataset cannot certify large turns or reverse. No learned quality, closed-loop safety or sampling improvement has been established.
