# Explicit causal state interaction paired trial 2026-11-16
Artifacts carla-interaction-paired512 and carla-interaction-response-frozen. Completed all2seeds x2arms x512 updates and fixed dev evaluation. No closed loop, default/guard changes, officialtest or overwritten oldweights. Train wall144.28+134.76+134.93+135.05=549.02s, peaks<=0.93571GiB, process exited0.

## Single change
Keep outputlateral scale0.5, internal nav-y gain50, existing objective and architecture. Baseline existing state slot19 is zero on all1570unique training windows. Candidate internally computes slot19=(current speed/10m/s)*(current near-goal FLU y/1m); canonical loader/online state stays unchanged. Set firststateLinear column19 tozero at initialization so candidate andbaseline physical functions exactly equal (FP32/BF16 context and same-noise16Euler outputs maxdifference0). Same initial weights elsewhere, seeds2060/2061, same6144sequence, batch12, optimizer/schedule512,64warmup,clip1. Candidate extra-feature column can then learn. Parameter count unchanged; zero initialization means this column has different optimizer history only as part of intended featurelearning. No auxiliaryprobe output/loss inserted. Future poison actor unchanged. First3 train RNG traces match perseed.

## Result base -> interaction
Early corrective dev82 scalar0.6s y prediction from FM fourcandidate mean:
2060 MAE0.035918->0.035890m, direction53.85%->55.13%;2061 0.036534->0.036474m,46.15%->50%. Prior ridge97.4% and0.00939m remains far better on this diagnostic, not a policy comparison.
Frozen navigationflip average y06 response2060 .000467->.000472m;2061 .000406->.000414m, negligible.
Recovery628 position diagnostics2060 5.166->5.077,2061 4.355->4.387m. Vector jerk167.39->162.56 and143.79->142.95m/s3. Deceleration5s speed6.266->6.181 and6.355->6.278m/s vslabels1.836. Startreverseendpoint54.78%->55.88%,50.74%->51.84%; stationaryreverse50.63%->51.25%,45.63%->48.54%. No overall usable improvement.

Do not adopt feature or run closedloop. Existing workers cannot silently load thismodel feature contract. Evidence says a learnable low-dimensional condition exists, but injecting its product alone does not make FM exploit it under currentbudget. Next proposed only: frozen weighted-gradient/conditioning signal diagnostic to locate objective or pathway bottleneck, no automatic longertraining or feature stacking. Statistical2seed, repeatedlyinspecteddev, limitedscriptedroads/oracle semanticpose remain. ADE/yerrors are fittingdiagnostics, not driving acceptance.
