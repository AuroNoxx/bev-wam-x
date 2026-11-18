# JEPA/no-JEPA paired continuation ablation (2026-11-16)

## Scope and fairness
Authorized isolated experiment.2seeds1930/1931 x2arms x64updates=256updates; all finished normally. Same original2048 FUSION-trained checkpoint core weights; new AdamW1e-5->1e-6cosine64,weight_decay0.01,batch8,BF16,clip1. Original FM and physical losses identical; NO previous rollout regularizer. Training42routes x3windows=126, four shuffled epochs per arm. Evaluation9validation routes x3=27DEVELOPMENT windows,4fixed-noise16Euler candidates/window. This is not an untouched test set. No test access/old-weight writes/default changes. Old narrow16window results use a different selection/noise and are not directly comparable to this table.

JEPA arm uses history h plus two predicted future latent grids (480context tokens) and original latent/variance/semantic losses0.1each. No-JEPA actor uses only exactly matching history h (160tokens), removes auxiliary losses and freezes inactive predictor/future_time/semantic head. Copied history-only computation is asserted exactly equal to the shared forecast h at initialization. Inactive EMA/legacy heads are retained for artifact compatibility/diagnostics, not future actor inputs; this is not a stripped export benchmark. Total stored parameters thus differ from active trainable counts.

Data generator seeds and per-step FM noise/time seed are paired; first3measured FM RNG hashes agree. Arm order counterbalanced betweenseeds. Before/after metrics retained. Removing JEPA context changes initial behavior: before training first residual6.163(with) vs9.103(without), full2.034vs3.057m/s. Therefore instantaneous removal is not equivalent to a trained no-JEPA model. The result answers REMOVAL DURING CONTINUATION, not from-scratch architecture superiority.

## Development results, JEPA -> no-JEPA
seed1930:
- first residual5.897->5.618m/s (lower); full1.833->1.949 (higher).
- mean per-trajectory peak accel16.382->16.620m/s^2; jerk111.626->112.268m/s^3 (slightlyhigher).
- forward proxy35.212->36.549m; backward segments0.463->0.435; nearzero0.01074->0.00815; spread3.640->3.444m.
- global first accel41.177->43.521; global jerk253.714->249.159.
seed1931:
- first residual6.300->6.344 (slightlyhigher); full1.929->1.878 (lower).
- mean per-trajectory peak accel19.358->17.370; jerk124.577->111.052 (lower).
- forward proxy40.844->37.135m; backward segments0.176->0.306; nearzero0.00444->0.00889; spread3.584->3.960m.
- global first accel48.363->46.450; global jerk292.015->261.061.

No arm dominates every metric across both seeds. Forward displacement is not route completion or desired-progress ground truth; lower forward movement can reflect either appropriate braking or undesirable slowing. Collision/TTC/route progress remain explicitly null. Global peaks remain large. ADE is not overall acceptance. All64updates in every arm triggered global clipping.

## Independent world diagnostic
Across all27development windows, using one frozen original2048 EMA target/reference (held identical for all arms):initialJEPA MSE0.248089; afterJEPA1930/1931=0.246836/0.246987, persistence0.193760 unchanged. Small local improvement does not beat persistence. Own-EMA results also retained separately to expose target drift. The no-JEPA legacy predictor is inactive and receives no world loss; its recorded~0.250MSE is diagnostic only, not a trained alternative world model.

Actor contexts are bitwise unchanged by future/trajectory poisoning during evaluations. Source OCC generation causality remains uncertified, and current-occupied-support target has no verified free-space truth or complete entering-object coverage.

## Resource result
Active trainable parameters:JEPA155,221,278; noJEPA152,851,223. Same core transformer width768/layers15/heads12.
Training+checkpoint wall times:JEPA28.73/25.82s; noJEPA20.36/19.89s. Counterbalanced small-run timings suggest~23-29%less time for this no-JEPA continuation, not a deployment speed guarantee.
Peak allocated:JEPA4.25-4.26GiB; noJEPA3.36GiB. Reserved4.58-4.63vs3.82GiB. After-evaluation~2.78vs1.62s includes4candidate sampling and diagnostics, not single-frame deploymentlatency or Orin measurement.
Total experiment loop146.80s including initialization/evaluations/save/cleanup, excluding initial imports/source setup. Four new checkpoints preserved. No training remains running beyond preexistingTensorBoard.

## Decision
This establishes a viable, cheaper no-JEPA continuation control, not a universally better driving model. Do not automatically remove JEPA or claim its context useless solely from latent MSE. Neither arm passes a real driving/safety gate. Keep both results, no default promotion or long-run extension. A from-scratch comparison would require matched fresh shared initialization and enough training, which has NOT been done.

Artifacts: artifacts/jepa-nojepa-continuation-paired/{summary,selection,config,clipping-summary}.json plus per-arm full reports/checkpoints andTensorBoard logs. Script:scripts/train_jepa_nojepa_paired.py.
