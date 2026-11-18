# Exogenous-command supervision prefix audit, 2026-11-16

Read-only frozen-prediction analysis. No training/model modification/closedloop. Script scripts/audit_permission_prefix.py; artifacts/permission-prefix-audit-v1/report.json and label-only manifests.

## Exact temporal contract
At anchor i, actor currently receives last-issued permission/cruise from row i (0.1s old), not a future command. Row j stores observation/control at frame j-1 and its resulting post-control sensor frame j. If the issued command first differs at row j, target endpoints BEFORE j are valid under the anchor command; endpoint j itself is already affected and excluded. Do not shift this cutoff by a tick or treat the changed post-frame as old-command supervision. A later command returning to its original value does not revalidate the suffix.

Future switch index/time is label-only metadata. It must never enter context/state, online masks, predicted horizon, noise sampling condition or actor branch. At inference actor still outputs its normal horizon under the current known command. Missing command metadata is unknown, not certified unchanged.

## Counts
Each split has514windows,34affected and480unchanged. These34overlappingwindows correspond to just ONE switch per split, not34independent events. No cruise change found.
Each affected group retains272/850target points. Entire split retains12272/12850points: removes578 (4.498%). Two windows have zero valid0.2s target points. Valid horizon counts at0.2/.6/1/5s:512/508/504/480.

Current permission=false prevalence falls from6.615% of original point weight to2.216% of retained points under point-normalized training. Thus censoring alone reduces hold training weight, even though hold remains important. Per-window-normalizing nonempty prefixes instead gives very short windows disproportionate per-point influence. Neither is neutral; report both point- and window-weighted evaluation. Do not silently rebalance or claim same effective supervision.

## Proposed loss rule (not executed)
For deterministic decoded supervision use original data draws/horizon, preserve original denominator per25points to keep each retained point's coefficient unchanged; multiply each applicable target loss by its valid-prefix mask. Explicitly log active points and all-masked samples (zero contribution, not a successful prediction). Evaluation means divide by actual valid count and separately show original unmasked descriptive metrics and event strata. Any later reweighting for lost hold coverage is a separate change.
Position/signed speed/body yaw require valid endpoint. Interval yaw-rate requires both interval boundaries before switch, plus original sensor-valid/low-speed-identifiability masks. Curvature coefficients have no targets. No curvature nullspace becomes zero-supervision truth. Do not erase arbitrary future-world targets: separately specify whether JEPA is conditional on held intent or predicts marginal observed futures.

IMPORTANT FM caveat: masking only output loss is not enough if the full training bridge xt still contains future-command-dependent ground-truth suffix and attention mixes it into prefix predictions. A future FM experiment must explicitly handle unobserved/censored suffixes in its training path; no padded future truth may remain a hidden teacher condition. This audit does not implement that design.

## Frozen scoring shows why average change is not training improvement
On affected retained prefix, true speed is0 and constant-velocity baseline exactly reproduces stationary truth. Existing model speed MAE is0.731/0.934m/s and position error0.615/0.788m for seeds2070/2071. Thus model anticipates motion while current command remains hold; clipping the scoring horizon does NOT solve it.
Overall prefix position scores2070/2071 are2.005/2.685m versus baseline2.575m. Unchanged480windows remain2.036/2.728m versus baseline2.633m. Seed2071 weakness remains outside command-changing windows; this issue cannot explain all failure.

## Do not filter genuine uncertainty
Only a specifically recorded exogenous ego-command change defines this prefix. Do not censor because lead vehicle brakes/restarts, OCC is sparse, a trajectory is unusual, label error is high, or multiple safe actions exist. Those are precisely interaction/partial-observation/multimodal problems to retain and stratify. Factual expert trajectories are not the only safe actions. Additional no-change hold episodes may be useful later, but no collection or training was started here.
