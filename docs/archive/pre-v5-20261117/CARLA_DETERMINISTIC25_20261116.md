# Deterministic25 velocity head paired diagnostic 2026-11-16
Artifacts carla-deterministic25-paired512. Both seeds2060/2061 completed512 and fixeddev evaluation,144.35/134.98s; checkpoints latest/best isolated, no automatic continuation. Matching FM baselines are carla-interaction-paired512/{2060,2061}-interaction, not gain1baseline. No officialtest, guards, closedloop, or oldweights altered.

Same conditional encoder/Transformer, dimensionality, initialparameters, internalnav50/currentvelocityinteraction, lateraloutputscale0.5, decoder,6144data sequence,512updates,batch12, optimizer64warmup etc. Change action objective to direct normalized25x2velocity residual MSE withzero trajectory/t=0; position/worldaux stay. Noise/t draws still consumed to keep mainlossRNG identical, though noise unused by regression. First3RNG traces matchFM. No total loss numericalcomparison acrossobjectives. FM takes16samplingforwards, regression1; equalupdates not equalinferencecost. K4 regression copies are identical and never counted as independent successful trials/multimodality.

Preflight parameters identical bothseeds; output finite, duplicates exact, futurepoisonactorinvariant. Labeltrajectory vsvelocitytrapezoidal integration normalizedXY discrepancy9.65e-5 on2samples, notforcechanged. Integratedvelocity positionconsistency preserved (devfirstresidual~2e-7m/s).

## Recovery628 FM -> regression
2060 positiondiagnostic5.077->2.633m, vectorpeakjerkmean162.56->5.03m/s3;2061 4.387->3.178m,142.95->25.14. Smoother but not gooddriving.
Earlycorrective82 y06 FMmean4 vsregression:2060 .03589->.03502m,2061 .03647->.03549; regressionsign50% both vsridge97.4%/.00939m. Conditioning failure persists evenwithoutflow-noise objective.
2060start mean5sendx-1.008m,stopped-0.774m; bothreverseendpoint100% (deterministiccopies not independent candidates).2061start+3.662m,stopped+4.968m (large drift). Decelendv6.160/6.912m/s vslabel1.836. Staticendspeed.517/.749 vszero. Currentgo/hold futuretiming ambiguity remains.

## Interpretation
Noise-free head greatly improves smoothness but does not repair conditioned lateral behavior or longitudinal stop/start/decel. Previous hypothesis 'FM-specific bottleneck' is insufficient: sharedconditionencoding/actionhead/optimization also implicated. Lowdim currentstateinteraction ridge can learn y06; fullTransformer regressionandFM cannot underthisbudget. Do not immediatelybuild residualFM on an inadequate meanpolicy. Proposednext only: inspect condition influence/pathway and optimization onsharedhead before furthertraining. No safetyclaim; routeandcoverageoracle limits unchanged.
