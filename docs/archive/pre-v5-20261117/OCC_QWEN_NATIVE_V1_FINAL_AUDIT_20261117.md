# Native OCC-Qwen v1: completed training and final audit

Completed 2026-11-17. All 20 distinct validation epochs are present; final global step 13240. The old trainer exited normally at max_epochs=20. status.json reports epoch21 only because its completion callback adds one after Lightning has advanced the epoch counter; checkpoints and validation records confirm 20 actual epochs.

## Preserved weights and reproducible evaluation

Both `artifacts/occ-qwen-native-training/best.ckpt` and `latest.ckpt` are epoch20/step13240. Model and optimizer tensor contents are equivalent; file hashes differ due to serialization/metadata. Both retain full frozen backbone, EMA, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG and exact loader generator/cache identity. All 354 model tensors and93 optimizer tensors per checkpoint are finite. Neither weights nor source data were changed by audit.

CPU integrity evidence: `artifacts/occ-qwen-native-final-audit/checkpoint-audit.json`.
GPU evaluation evidence: `artifacts/occ-qwen-native-final-audit/best-evaluation.json`.
Evaluator: `scripts/audit_occ_qwen_native_final.py`.

Validation protocol: all1124 Town13 validation windows, batch8, one candidate,8 Euler steps, seed1907, same loss-then-sample RNG consumption as production validation, dt0.2s,25 points. Test split untouched. Re-evaluated ADE4.57515893m reproduces logged4.57515955m to reduction precision.

| Metric | Final best |
|---|---:|
| ADE |4.575159m|
| Endpoint1/2/3/5s |1.137768 /3.082183 /5.296320 /10.000568m|
| Speed MAE |2.238077m/s|
| Validation loss |0.15952892|
| Position-derived vector jerk mean /P95 /maximum |10.366884 /22.450689 /236.330963m/s3|
| Vector-velocity-derived jerk mean |10.366883m/s3|
| Scalar-speed-derived jerk mean /P95 |6.936797 /18.453526m/s3|

42.82% of vector-jerk timepoints exceed10m/s3; every evaluated trajectory has at least one exceedance. Scalar-speed jerk is a different metric:22.46% of timepoints and98.04% of trajectories exceed10. No claim of satisfying the CARLA jerk guard, real-time operation, or reliable closed-loop driving follows from offline ADE.

## World prediction against matched persistence

Same checkpoint EMA encoder, current-ego coordinates, native source labels, original latent scale, pooled4x4 occupied-support mask, mean over64 latent features. Persistence repeats the encoded current frame at both horizons. Future labels never enter the actor.

| Horizon | Generated World latent MSE | Static persistence MSE |
|---|---:|---:|
|+1s|0.260432|0.507493|
|+2s|0.281341|0.762381|

This run beats persistence under that protocol. Historical claims that World is worse than persistence must NOT be repeated as facts about this completed run. This result still does not prove counterfactual physical fidelity or that World improves Action.

## Execution and next stage

The run began unoptimized, then safely resumed the same full step1150 checkpoint with native aligned cache and channels_last. Optimizer/scheduler/EMA/data sequence preserved; exact shared RNG loader recovery was tested across epochs. A few unsaved steps at switching could have been recomputed; no zero-repetition claim. Fused AdamW failed the predeclared parameter-difference gate and stayed disabled; workers0-to2 migration was excluded. New optimized checkpoint state audited atstep1550. Performance evidence remains in the training artifact folder.

The next isolated v2 follows `NEXT_STAGE_ARCHITECTURE_PLAN.md`. Its proposed explanations remain hypotheses. Preserve v1best/latest and native cache for matched evaluation. If the v2 evaluation needs a model-independent action-noise stream, evaluate BOTH models again under that exact common protocol and report it separately from this reproduced training-validation result. Do not compare changed RNG draws as if they were identical.

## Verified equivalent test-checkpoint archive and cleanup
Three obsolete pretrained resume-test checkpoints were compacted into one shared frozen backbone plus three complete remainder checkpoints. Reloaded reconstruction matched every tensor byte, NumPy array and metadata field exactly; archive/original SHA values were checked again immediately before removing only the three originals. The archive and restoration script remain. Net space recovered ~5.638GB (5,637,786,797 bytes before small text-file overhead). Evidence: artifacts/occ-qwen-native-final-audit/cleanup-completed.json and artifacts/occ-qwen-native-pretrained-resume-test/lossless-archive-v1/manifest.json. Currentbest/latest, oldstep1150rollback, all source data and native cache were not removed. No other irrecoverable checkpoint was deleted.
