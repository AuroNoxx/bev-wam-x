# Frozen full-route / Flow-path diagnosis
Completed 72.62seconds, four latest1024 checkpoints unchanged, no optimizer/update/test. Artifacts:artifacts/route-flow-frozen-full.

All828 held-route development windows evaluated, six route IDs, overlap heavily; report scenario-macro as well as micro, do not treat828 as independent scenarios. Whole-source causality remains unverified. Same six route IDs as before, not broader road/domain validation.

FM->regression full828 micro position-label fit: seed2050 5.796->4.200m, seed2051 6.210->4.379m. Macro six scenarios5.636->4.206 and6.137->4.586m. Micro vector velocityRMSE2.571->1.937 and2.699->2.022m/s. Lower regression dispersion/jerk still not safety; regression2051 nearzero fraction0.102 vsFM0.0116 and lower forward33.21 vs35.60m, so investigate stagnation rather than count it as pure gain.

17 nonoverlapping-support windows, same4Gaussian noises per window, frozenFM:
2050 roll1/4/16/32 position3.929/4.786/5.335/5.444m; velocity candidate spread1.093/1.145/1.574/1.659m/s; vector peak jerk206.3/168.3/228.7/241.1.
2051 position5.360/6.473/7.078/7.201; spread1.107/1.164/1.571/1.648; jerk189.4/163.5/225.3/236.9.
Thus increasing integration steps does not recover quality and32isclose-but-worse than16. This points toward learned field/path behavior rather than merely too fewEulersteps; not a complete proof. One-step is not universally best(jerk4lower) and no default change.

Bridge t=.9 uses90%true target by construction, ONLYoffline diagnostic, neveractorcondition. Its position~1m vs~4-5m at0 must not be called successfuldenoising: endpoint error = (1-t)*field-error, so shrinking1-t alone improves endpoint. Need normalized field-error or gradients to infer time-specific competence.

Codeaudit: q=(vectorvelocity-[v0,0])/10; physicaltime.2s independent ofFlowtime[0,1]. IIDunitGaussian means10m/s velocity-residual scale. 18analytic oracle tests again pass; not proof learnedfield. Shared architecture/EMA losses andpairedinitialization unchanged. FM unitweight targets(target-noise); regression unitweight targets q, hence unequal target scale/gradient despite matching scalarcoefficients. Position smoothL1 scale5m weighted.2, gatedt>=.5 withfactor2. This means lowFlowtime gets no positionaux; actual data-conditioned gradient balance unmeasured. Configlistskin/unit/negativeweights inherited but CompareTask does notapply them (twoDintegral scheme); no pretending they were active.

Next minimal single-variable candidate: shift only physical position auxiliarytime coverage from gatedlate-half toalltime(constantgate1), keeping expectedgate weight1, allotherloss/data/noise fixed; BUT first frozen time-binned field/weightedgradient audit should establish lowtime mismatch and checkdominance. This is a hypothesis, not yet authorized experiment or proven fix. Alternative merely increasingsteps unsupported. Preserve currentdefault; no newtrainingstarted.
