# Frozen backbone / lateral-only residual isolation 2026-11-16
artifacts/carla-frozen-lateral512. Two baselines are prior unnormalizeddirectstate512 seed2060/2061; retainall90stateentries includingEMA andold200parameterbranch. Add100zero-init lateralheadparameters4->25, interactionRMS calibratedtrainonly. Only q_y changes; q_x andintegratedx mustremainexact. Derivedscalar speedcanchange becausevychanges.

Initialparity firstfailed withtinyFP32difference(7.08e-7 normalizedoutput,1.19e-7x) comparingtrainableflagbaseline to frozenmodel. Settingreference equallyfrozen (no weightchange) restoredexactFP32/BF16bothseeds. Didnotrelax tolerances. Failedpreflightlogsretained; prematuretrainingentryexitedbecausepreflightfileabsentbeforetraining. Successfulpreflightvalidatednonzerolateralhead doesn'tchangex, futurepoisoninvariance andstaticinteractionzero. EMAupdate disabledexplicitly. Same6144draws,512updates,lr/schedule; freezingchangesgradientcompetition intentionally. RNGfirst3match.

Completed47.48/38.67seconds; all90originalentries elementwiseequalaftertraining. Independent postcheck across1156devwindows perseed verifiedq_x andintegratedx exactlyequal toproperlyfrozenbaseline inBF16. No test, closedloop, guard/default or oldweightchanges.

Earlycorrective82 MAE .034696->.024938m and.034971->.025371; significantsign50%->97.44%/96.15%. Predabsamplitude .01065/.01033m vslabel~.03510m, undercorrects. Whole628y06error .00896->.00495,.01163->.00730. Staticy06error .00195->.00029 and.00575->.00410. Noclaimfullcorrection ordrivingsafety.

Longitudinalknownbadbehavior preserved exactly under matchedexecution: staticxapproximately-.706/+4.600m; decelspeed6.10/6.75m/s vslabel1.836. Endscalar speedtinychanges canbevyderived. Freezingbadbaselineisnotacceptance. Localcausalconditioncanlearnsignedcorrectionwithoutsharedbackboneinterference; two-seed/smallroads/reuseddev limitation remains. No automaticextension.
