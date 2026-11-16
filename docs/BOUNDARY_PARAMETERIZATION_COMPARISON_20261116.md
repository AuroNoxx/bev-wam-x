# Boundary-aware parameterization feasibility, isolated output experiment

No training/checkpoint/test access. Teacher2048 artifacts remain available and unchanged; current no compute conflict. This compares eight existing predicted candidates, not a newly learned policy.

Variants: original; integrate original speed/heading from observed t0; bounded acceleration[-4,+2]m/s² and yaw-rate0.3rad/s; same plus scalar jerk5m/s³. Limits are illustrative, not calibrated vehicle safety thresholds. Jerk demonstration assumes initial acceleration0, not the actual measured acceleration. All variants use same candidate desired speeds/headings; no future truth supplied.

Five synthetic tests passed. Mean kinematic residual across8 candidates: original1.72667m/s; all integration variants~3e-15 by construction. Largest absolute scalar acceleration:29.4518 unchanged for pure integration,4.0 for both bounded variants. Largest scalar jerk:160.394 original/pureintegration,30.0 acceleration/yaw-only,5.0 withjerk bound.

Crucial tradeoff: mean position change vs original2.016/3.238/3.266m; largest final displacement change10.467/12.456/13.022m respectively. Mean final forward displacement32.474m original vs29.826/27.916/28.004m. These are spatial proxies, not route progress. Large path changes could enter obstacles; no collision result exists. Do not install as an automatic postprocessor or claim success from constructed consistency.

Braking counterexample: start20m/s, all desired future speeds0. Pure integration reaches0 in0.2s and travels2m (unphysical acceleration100m/s²). Bound−4m/s² takes5s and50m to stop; adding jerk5 reaches55.72m in5s still1.2m/s. Thus imposing comfort limits can be incompatible with a40m stopping requirement. This is a synthetic kinematic example, not CARLA collision evidence. Yaw-rate limiting alone does not certify lateral acceleration/curvature.

Recommendation: retain2048 teacher as offline initializer, not validated driving teacher. No extra training yet from this experiment. Next useful bounded training comparison should use same train/validation budget and corrected first-segment peak metrics: original baseline versus explicit first-speed boundary/first-displacement penalties versus acceleration/heading-rate integrated decoder. The decoder route changes model target semantics and cannot be treated as checkpoint-compatible without an explicit adapter. Require progress/geometry-deviation checks and candidate rejection when dynamics/geometry constraints conflict; do not optimize a single residual. Source causal provenance and world-model persistence failure remain separate blockers to driving readiness, not reasons to erase weights.

Artifacts: artifacts/teacher-boundary-parameterization-comparison/{report,summary}.json. All25points/segment metrics retained for each variant. Script:scripts/compare_boundary_parameterizations.py.
