# Navigation coordinate-scale paired diagnostic 2026-11-16

Completed two512-update arms; no closed loop, no test set, no guard/default/old weight changes. Artifacts carla-navscale-paired512 and carla-navscale-response-frozen.

## Contract and preflight
Canonical cached/online state nearFLU/50 remains unchanged. New model internally multiplies only state slot5 by50; first state Linear column5 divided50 at initialization. Longitudinal navigation and other19 slots unchanged; physical lateral velocity output scale remains0.5m/s. This preserves initial FP32 physical function but changes optimizer conditioning (AdamW/clip), intentionally part of the parameterization experiment.
FP32 context max discrepancy7.15e-7, physical output mixed-channel max6.81e-6; BF16 context0.01716, mixed-channel max0.03282, below predeclared0.02/0.2 tolerances, not bitwise equivalent. Future poisoning actor exactly invariant. Three actual miniloop raw/cache navigation encodings exactly equal. First attempt preflight mistakenly looked for separate startstop-cache entries, found none and failed before any training; retained error log, corrected to actual miniloop source then passed.
Actual exact6144 train draws:3072 original miniloop,3072 recovery, not separate startstop-cache. Recovery itself contains stopped/start/motion/decel. Train roads388/392; dev713. Same seed2060, parameters initial physical equivalence, fixed batch12/order, 64warmup, AdamW1e-4,512schedule,clip1; identical first3 stochastic traces. Gain1 reproduced previous scaled run diagnostics exactly.
Training144.28/134.67s respectively (first cold), peaks0.9354/0.9357GiB. Both exit512, latest/best isolated, evaluator complete. No continuation.

## Results gain1 -> gain50
Recovery628: position diagnostic5.059->5.166m;0.6s lateral error0.01847->0.01881m; vector jerk163.08->167.39m/s3. Offcenter114 lateral0.02601->0.02582m (negligible), position7.563->7.647m.
Recovery decel162 endpoint speed6.245->6.266 vs label1.836m/s. Stopped endpoint1.636->1.680 vs zero; reverseendpoint49.17%->50.63%. Start endpointreverse54.41%->54.78%; mean forward -1.677->-1.745m. No meaningful overall improvement.
Frozen12dev states x4 same-noise candidates: flipping navigation y changes mean0.6s predicted y by mean0.0000261->0.0004671m, max0.0000584->0.001795m. Sensitivity increases but remains tiny and not reliable corrective behavior. At index80 target +/-0.0615m, gain50 predicts+0.0093/-0.0042m; index100 negative correction target-0.0371m still predicts+0.0183m. Thus gain alone insufficient under this budget. Do not adopt or continue gain sweep.

Future work should first test conditional target identifiability/ambiguous scripted wait-go behavior and isolate whether conditional encoder can learn signed correction on known motion data, rather than launching another unconstrained sampler adjustment. Labels remain one scripted behavior, not unique correct driving. Collision/TTC/lane safety unknown offline, semantic/pose oracle limitation, one trainingseed and reuseddev. Existing workers do not yet load nav_y_gain and must not silently execute this checkpoint.
