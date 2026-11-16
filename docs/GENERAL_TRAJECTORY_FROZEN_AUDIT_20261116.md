# General trajectory generation frozen audit

No training. Source checkpoints: both scene-balanced512 seeds and the completed single-seed eight-window fine-tune. The second tiny-fit seed was not started after the user's correction to stop focusing on parking. First tiny-fit seed completed512 updates; its post-fit CPU/GPU evaluator failure was recovered from the savedcheckpoint without retraining. Its pure-noise fit gate did not pass (position1.78m, velocityRMSE1.95m/s, stop6.7%, normal finalspeedMAE2.67m/s).

## 1. Analytic numerical control
Six synthetic physical trajectories: constantvelocity, braking, acceleration, lanechange lateral pulse, turning and merging. Known conditional endpoint used only in the synthetic analytic field v=(q_target-x)/(1-flowtime). Test existing Euler-loop formulation at4/16/32steps and the velocity2 physical decoder at0.2s spacing. All18 tests passed: maxvelocityerror4.77e-7m/s, maxpositionerror1.91e-6m. This rejects a gross deterministic endpoint/unit/axis defect in the tested sampler+decoder path. It does NOT certify the learned field, all possible numerical cases, vehicle dynamics, perception causality or safety.

## 2. Six-scenario frozen study
Previously fixed42 windows, seven per directory scenario label: HardBreakRoute,HighwayCutIn,HighwayExit,MergerIntoSlowTraffic,MergerIntoSlowTrafficV2,SequentialLaneChange. These labels do not confirm that the named event was triggered or successfully handled in each window. Reused development data, with some training-route overlap, not independent validation.

Four identical initial Gaussian candidates per window,16Euler steps. No future-label actor access; poisoning future/trajectory leavescontext exactly unchanged. Independently shuffle historicalOCC, or19 non-speed statefields across scenes while retaining initialspeed. These are mismatched-condition sensitivity tests, not realistic counterfactual driving.

For each deterministic recorded trajectory, squared error decomposes exactly into candidate mean bias squared plus candidate spread variance. This is label-relative error, not optimal driving error or irreducible uncertainty. Values below average window-level RMSEs, not a pooled squared decomposition.

- Balanced2020, across six scenes: mean-biasRMSE1.50-3.14m/s, candidate-spreadRMS1.81-2.72m/s, mean peakvectorjerk266-335m/s3.
- Balanced2021: bias1.63-2.96, spread1.82-2.52, jerk293-346. Problems are not specific to parking.
- HistoricalOCC swap changes sampled velocityRMS0.17-0.41m/s; non-speed-state swap0.58-1.08. The model responds to conditions, but limited sensitivity in this selected set does not establish adequate perception use or prove complete ignorance.
- The eight-window fine-tune cuts spread in ALLsix scene groups to0.67-1.28m/s and reducesjerk to151-294, but conditional mean bias worsens in FIVEofSIX scene groups relative to its source model. Lower position-error or spread cannot alone be called better driving; these changes suggest denoising/concentration and correct conditional mean are distinct unresolved issues.
- t=0 one-step label-endpoint estimates versus16-step rollout: balanced2020 meanbias2.10->2.33m/s, spread1.47->2.18; balanced2021 2.14->2.32 and1.52->2.12. A learned trajectory-field quality issue remains plausible, not established as an integrator implementation bug.

Three checkpointSHA256s unchanged. Audit wall25.21s, no train/evaluation processes left. Collision/TTC/free-space/actualeventoutcome allunknown.

## Next proposed decision, not executed
A fixed mixed-scene tiny-set learnability gate, balanced across allsix scene categories instead of stop-heavy selection. First use a deterministic two-dimensional velocity regression reference with the same causal condition encoder, and compare against the currentFM pure-noise generator under explicit matched update/data budgets. Treat this as a diagnostic of conditional representation versus generation difficulty, not a fair claim that one production architecture is superior. Keep multi-dimensional motion,normalprogress and recorded-target fitting separate; no automatedlongtrain or defaultswitch.
