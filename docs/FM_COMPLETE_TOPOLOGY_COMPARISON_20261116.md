# Complete-window FM topology comparison, 2026-11-16

Completed and stopped: 2seeds x2arms x256updates,5.37min (322.17s) total. Noclosedloop,partial-windowtraining,old-weight/test/default changes. scripts/train_fm_complete_topology256.py, scripts/evaluate_jepa_fp32.py, scripts/summarize_fm_complete_topology.py; artifacts/fm-complete-topology-paired256.

## Fixed comparison
FP32 + explicitMATH attention in training ANDfinalevaluation. Same480complete train windows, same3072draws and480unique perarm, batch12, pairedseed/noise/time schedule, identical matchinghistory256 weights. Only trajectoryselfattention bidirectional vs fixed lowertriangular changes. Same25x2velocity residual representation/cumulative trapezoiddecoder, FM MSE +.2 gatedpositionSmoothL1(5m,t>=.5) +.1eachJEPA latent/variance/semantic. Originalconfig containsotherphysicalweight fields but CompareTask does not use those terms; not silently added here. AdamW1e-4,wd.01,clip1,warmup64/original512cosine truncated256. Differentseed execution order counterbalanced.

Warm-start caveat: sourcehistory256 weights previously sawall514windows including34command-change examples. Currenttraining usesonly480; this is a paired topology-switch fine-tune, NOT from-scratch complete-case learning. Architectural switch changes initialfunction despite identicalweights. Final256used for comparison, not dev-selectedbest. Validationcheckpoint monitor remains label-fit only, not drivingquality.

## Integrity / runtime
Bothseed first3noise/timeRNG traces exact betweenarms. Sourcecheckpointhashes unchanged. Allpreflight/evaluationfuture-label poison checks exact. Four latestcheckpoints step256/loadercursor3072, optimizer/scheduler/taskstate present,latest/best retained. Perarmwall75.66/67.76/67.74/67.51s; firstarm includesinitialization/compile overhead. Peakallocated~.991GiB. No numericalexceptions, allbudgetstops normal.

## Primary480complete held windows, K4 Euler16
Bidirectional -> causal:
- seed2070 positiondiagnostic4.272 ->5.051m; decel5.728 ->7.635m;5sdecel speedMAE4.839 ->6.064m/s.
- seed2071 position5.150 ->5.725m; decel7.021 ->8.695m;5sdecel speedMAE5.524 ->6.533m/s.
- meanpeakvectorjerk INCLUDING realt0-derived boundary acceleration:119.24 ->139.62 and121.24 ->143.67m/s3.
- candidateRMSspread3.286 ->2.498m and3.767 ->2.132m; lowerdiversity alone is notbetter.
- static-permitted20windows andaccelerating128windows getbetter longhorizonlabel fits, but those gains coexist withworseearlyspeed/deceleration/steady/dynamics; noselectivepass.

World same-initial-reference space:
2070pred.52581 ->.52550 vspersistence.11330;2071pred.46153 ->.51938 vspersistence.10303. Bothfarworse thanpersistence. OwnEMA targetspacealsoreported separately; no crossarmown-spaceequivalence claim.

## Secondary34command-change held windows
Onlyvalidpre-switch prefixscored for position/speed. Position1.854 ->3.399m and2.218 ->3.370m;0.6sspeedMAE1.820 ->3.186 and2.135 ->3.187m/s. No5sscorebecauseallfuture5ssuffixesinvalid. Fullhorizondynamics inJSONexplicitly marked descriptive, notprefix-only. Currentholdcoverageabsentfrom480training, socannotclaimholdlearningvalidated.

## Decision
This boundedcomparison does NOT support replacingthecurrentmodel with fixedcausalattention forquality. Numericalcausalisolation andlearneddrivingqualityaredifferent requirements. Bothmodelsremainphysicallypoor; causal isworseoverall/deceleration/jerk inbothseeds. DonotprogressautomaticallytoCARLA,partial-windowtrainingorlongertraining. Doesnotprovefrom-scratchcausalFMimpossible givenwarm-start/distribution-switch andsmallone-episode limitations. Collision/TTC/routeprogressremainunknownoffline. Keepresultsandwaitfornextdecision.
