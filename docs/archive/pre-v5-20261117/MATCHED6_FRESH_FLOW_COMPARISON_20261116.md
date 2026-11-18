# Fresh matched6 Flow / deterministic comparison (2026-11-16)

Authorized bounded end-to-end comparison, not an extension of the fixed-context unit test. See frozen design.json and strata_label_only.json in artifacts/matched6-fresh-flow-regression2048, written before the first update.

Two seeds 7301/7302, paired identical initialization and full data sequence, FM and regression, 2048 steps each. Existing model width128 / 2 layers / 4 heads, 966939 trainable parameters; no teacher scaling experiment. All main parameters trainable, EMA teacher maintained. Batch12, 24576 draws of2176 unique train windows,4train episodes;1088 held development windows from2episodes. Complete source data already synchronized, constant current commands and actor future-poison preflight verified. Development data previously inspected, not independent final validation.

Optimizer AdamW1e-4 wd.01,128step warmup then cosine to1e-5 at2048, clip1,FP32 MATH. Shared25point two-dimensional velocity representation and decoder, original gatedposition/world objectives preserved. FM vector-field and deterministic qMSE differ by design; combined loss is not a comparable driving-quality score. Same update budget is not equal inference FLOPs.

Hard caps:2048updates and15min perarm,60min combined. Nonfinite/contract failure stops; no unapproved extension or blind restart. StatefulDataLoader,Lightning,VerifiedCheckpointIO,rollinglatest plus milestone/best. Best label-fit checkpoint is bookkeeping only; pre-registered primary comparison uses final2048, fixedpure-noise16Euler4candidates, candidate0 primary without oracle best-of-K.

Milestones512/1024/2048 include full held metrics, candidate spread, own-EMA world versus same-EMA persistence. Own-EMA targets differ across methods, so absolute cross-model latent MSE is not itself a common quality scale. Teacher GTlead speed/gap only generates supervision and posthoc labels; actor receives actualhistoricalOCC/state/nav, no futuretiming/GTIDs. SemanticLiDAR,ego pose,mapnavigation remain simulator-oracle inputs. 99%unknown OCC is not free space. Actual train/dev cruise differs despite frozen same sampling distribution.

Predefined observability strata: future start before lead actual motion; early actual motion below frozen no-ID threshold; detectable motion; static first1second; deceleration. Prior GT only defines evaluation groups. Detector is simple no-ID diagnostic, not a certified observability oracle. Sparse detectable-start subgroup may not cover both held episodes, in which case admission gate is unknown, not pass.

Admission gates recorded in design.json are conservative engineering pilot conditions, not safety certification: bothseeds/heldepisodes, requiredgroupcoverage; detectable-start and decel endpoint-speed errors improve relevant simple baseline by20%; hold1smean drift<=.30m; meanpeakaccel<=6m/s2,meanpeakjerk<=15m/s3,negativevx<=1%; no substantial regression-relative hold/jerk tradeoff. All conditions required. These thresholds will not be changed to admit observed results. No automaticCARLA launch: successfulofflinegate still requires report and unchangedguards. Collision,TTC,andactualrouteprogress remainunknownoffline.

## Final result (all four arms completed)
Training wall time2267.00s (37.78min), no extension. Each2048updates,24576draws,cursor/optimizer/scheduler verified. Paired initial weights and first3 random-stream traces exact. Allocated GPU peak0.990GiB. Only TensorBoard Python processes remain after final frozen evaluation.

|seed/method|dev position diagnostic m|endpoint speed MAE m/s|mean peak acceleration m/s2|mean peak jerk m/s3|negative vx timestep fraction|
|---|---:|---:|---:|---:|---:|
|7301 Flow|2.534|2.637|6.396|51.177|9.49%|
|7301 regression|2.109|1.911|6.846|34.088|5.30%|
|7302 Flow|2.206|2.409|4.475|35.135|8.31%|
|7302 regression|1.809|2.001|5.778|28.623|1.94%|

All admission gates failed; no CARLA launched. Important tradeoffs: Flow improves current-hold first1s drift vs regression (Flow episode4/5 .219/.431m and .236/.353m; regression .939/.799m and1.135/2.188m), but fails episode5 hold threshold and is noisier overall. Deceleration endpoint error improvements >=20% overconstant occur on both episodes for regression, only seed7302 episode4 forFlow. Detectable-start dev has just1window episode5 and noneepisode4, so fullcoverage isunknown. This is distinct from observed dynamic failures.

All4candidate distributions: fraction containing anyvx<-.05m/s:Flow20.54%/19.16%,regression6.62%/2.30%. Peak jerk95thpercentile:Flow140.14/94.55m/s3,regression80.45/64.15. No best-of-K selection or rerun noise cherry-picking.

Frozen train stride8(272windows) position errors Flow2.690/2.407m,regression1.755/1.486m; Flow train errors/jerk are also poor, so held-route shift alone does not explainfailure. This sample is not full train coverage nor proof of an architectural impossibility.

Same-own-EMA worldprediction/persistence error ratio:Flow1.771/1.941,regression1.747/1.908. Worldforecast did not beat persistence in this metric; absolute latent errors are not comparable across different learned EMA spaces, nofree truth/safety conclusion.

Fixed12train teacher-forced Flow bridge audit:longitudinal channel dominates; at t=.95 normalizedvectorfieldMSE .1546/.1264 vs t=.05 .0271/.0239. This is teacher-forced bridge diagnostic, NOT pure-noise inference quality, and too small to identify cause or justify automatic loss/scheduler changes.

Conclusion: samebudget regression is generally better for label-fitting/deceleration/smoothness but fails static/dynamic requirements too; Flow is not validated as a closed-looppolicy. Fixed-context learnability did not transfer into a passing end-to-end result. Do not automatically enlarge model, prolongtraining, or relaxguards. Decide the next intervention only after reviewing these failures and incompleteobservability/teacher information gap.

Artifacts: summary.json,eacharm eval512/1024/2048.json, predictions*.npy, train_fixed_stride8_eval.json, FM_bridge_diagnostic.json, latest/best/milestone checkpoints. Main train code train_matched6_fresh_methods2048.py; frozen evaluator evaluate_matched6_fresh2048_final.py.
