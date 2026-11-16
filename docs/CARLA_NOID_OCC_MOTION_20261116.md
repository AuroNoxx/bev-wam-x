# No-ID final-OCC past-motion diagnostic

Script `scripts/diagnose_occ_noid_motion.py`, artifact `artifacts/occ-noid-motion-diagnostic/`. Completed CPU14.93s; no model training, test-set use or closed-loop.

## Strict input boundary
Readout accepts only the four actor OCC grids and current near navigation. Grids use existing permitted ego-frame registration, not lead pose. No raw continuous point positions are fed to the readout; no object_idx, lead ID, lead pose, true gap/velocity, schedule, teacher decision, future observation or label. On independent audit, these fields are accessed only after prediction for identity/accuracy auditing.

Preset geometry before dev/audit: class4 BEV8-connected components with at least2cells in navigation-forward3..60m and +/-1.75m corridor. Select nearest current candidate; ambiguity margin0.8m. Backwards nearest-centroid association across .2s steps capped4.6m, ambiguity margin0.8m. Ambiguity yields unknown, not zero. Estimate velocity by least squares centers across[-.6,-.4,-.2,0]. Train stationary-history p95 absolute apparent .6s displacement fixed movement decision threshold at0.698678m (147windows); never tuned on dev/audit. This high threshold reflects this particular quantized centroid estimator, not a universal sensing limit.

## Results
- Train502/514valid,dev512/514,audit412/414. All unknowns were ambiguous history association (12/2/2), not no-hit.
- Audit all412valid anchor predictions had all selected cells in all4frames match true lead instance in posthoc audit. This is single-lead data, not multi-object tracking validation.
- Audit overall past .6s mean longitudinal speed MAE0.270m/s; zero-motion MAE4.745.
- Moving>=2m/s:269valid/271,MAE0.291,p95abs error0.865m/s.
- Static(-.05..05m/s):116valid,MAE0.207,p95abs error1.116;4/116motion false alarms3.45% undertrain threshold.
- Slow(.05..5m/s):8valid,MAE0.347 vszero baseline0.225;only1/8passesmovement threshold. Smallcount.
- Medium(.5..2m/s):19valid,MAE0.317.
- Two actual current-speed onsets>.2m/s: indices88/355; first causal detections94/358, delays0.6/0.3s. Temporalwindow necessarily averagespast motion. Few events do not validate delaybounds.
- Dev overallMAE0.257, movingMAE0.268,staticfalsealarm4/115. TrainMAE0.308.

Train/dev GT pastspeed reference integrates recorded post lead-forward velocity, approximate because heading can rotate; audit uses actual lead-pose pastdisplacement in the same currentego/navigationframe. Therefore preferaudit errors for exactgeometry comparison. Audit is a separate recording on the same road/seed4101 asdev, not an independently randomizedscenario distribution. Overlappingwindows are notindependenttrials.

Synthetic4m/s cellmotion, exactstaticzero and unknown-not-zero tests passed. No confidence or safety certification.

## Interpretation
Conventional leadmotion is information-theoretically present in finalOCC without ID. This does not show the trainedencoder reads it or FM uses it. Slow onset remains weak for this simple quantized-centroid estimator; point-surface/centroid changes, quantization andassociation canintroduceapparentmotion. Continuous raw-ID centroid accuracy cannotbe substituted forfinalOCC accuracy. No directly executable control recommended; futurediagnosis should separate encoderrepresentation fromaction-generation, not anotherblindlosschange.
