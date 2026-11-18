# Three-arm short training comparison, 2026-11-16

Completed three96-update Lightning runs, each initialized from unchanged teacher2048/best weights, with fresh AdamW lr1e-5, BF16,b8,seed1801. Same128 evenly spaced train windows and16 previously used validation windows; no test. All64 validation candidates per arm use4 candidates/16Eulersteps and fixed evaluation seeds. Full checkpoints saved only in new artifact directories. No automatic further training or simulator execution.

Arms: baseline unchanged loss; first_penalty adds coefficient.1 times late-bridge(t=.8) first-segment residual SmoothL1 plus .5*relu((|first acceleration|−6)/6)^2; integrated retains5-channel FM targets/checkpoint compatibility but physical xy is derived by velocity/heading trapezoidal integration from t0. Integration therefore achieves low residual by construction and is not an optimized new3-channel model.

Results after96updates (first residual and whole residual are means; peak first acceleration is global over64candidates; forward is a proxy not route progress):
- baseline:34.01s,4.547GiB reserved; first residual5.021m/s,whole1.514; peakfirst|a|36.721m/s²; forward22.341m; mean backward segments/trajectory.922.
- first_penalty:40.78s,5.537GiB; first3.419,whole1.569; peakfirst|a|22.255;forward25.363m;backwardsegments.234.
- integrated:32.61s,4.547GiB; first1.37e-7,whole1.49e-6 by construction; peakfirst|a|41.293;forward18.368m;backwardsegments.109.

Shared original model before fine-tuning: firstresidual5.465,whole1.821,peakfirst|a|40.273,forward27.253m. Integrated before training alreadyhaswhole1.91e-6 andforward23.913m due changed mapping. It must not be credited as learned residual improvement.

Mean per-trajectory max|scalar accel| after:13.820/12.412/14.364m/s²; mean per-trajectory max|scalar jerk|94.411/94.494/95.316m/s³ (baseline/first/integrated). These are not means across timesteps and not global maxima. First-penalty does not solve dynamics or comfort.

Mean point movement relative to shared original sampled trajectories:2.560/2.949/5.480m; largest final-point changes8.306/12.979/16.345m. This is not label error and not proof of collision, but shows semantic/path tradeoff, particularly integrated. No collision/TTC/route progress/traffic rule metrics available; remain null. Forwarddistance reduction may be valid braking or unwanted stalling; scene evidence required.

Important limits: one seed, tiny subset, warm-started teacher, short runs. First-penalty uses extra model forward and extra noise draws, so equalupdates != equalFLOPs and training noise stream diverges despite same initial seed. Cannot make strict causal claim; next replication should isolate random generators and use multiple seeds. Shared labels are supervision, not unique correct driving. Source causal certification/world persistence limitations persist. No default/model promotion.

Artifacts: artifacts/teacher-boundary-short-training/summary.json, paired-path-change.json and per-arm report.json/latest.ckpt/run-identity.json. Reconstruction requires explicit mode documented in each sidecar. Script:scripts/train_boundary_short_comparison.py.

Recommendation: explicit boundary penalty is worth a controlled multi-seed replication, not adoption yet. Integrated candidate does not solve first-speed spikes and changes path substantially; do not choose it solely for zero residual. Original2048teacher stays available for future controlled training. No weights deleted or overwritten outside these isolated outputs.
