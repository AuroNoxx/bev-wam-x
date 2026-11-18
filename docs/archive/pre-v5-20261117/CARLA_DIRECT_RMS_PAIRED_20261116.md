# Direct interaction RMS normalization paired trial 2026-11-16
artifacts/carla-direct-rms-paired512 and carla-direct-rms-perturb-frozen. Completed2seeds x512;144.34/135.03s, first3RNGmatch respective unnormalized directbaselines. No closedloop/test/default/guard changes.

Only directfeature4=(v/10)*near-y divided bytrain-drawRMS0.010698423131561092.6144draws/1570unique, calibrationselectionSHA saved. No centering, staticv0 givesexactzero; first3featuresunchanged. ZeroheadFP32/BF16 initialoutputs exactlyequal bothseeds, futurepoisonactorunchanged, slotunitstested. Sameoriginalbudget/data/loss/optimizer with200newheadparams inherited.

Recoverydev earlycorrective82:2060MAE.034696->.027577m, significantdirection50%->85.90%;2061 .034971->.029013m,50%->57.69%. Navflip y06response .000969->.010877 and.001055->.011837m, roughly11x. Physicalconditioning scale is a real optimizationfactor, not a completefix.

Whole recovery628:2060position2.611->2.662m,jerk4.786->4.652;2061position3.102->3.720,jerk24.020->33.853m/s3. Staticendx2060-.706->-.840m,2061+4.600->+5.724m. Decelendv2060 6.099->6.007m/s,2061 6.748->7.099m/s vslabel1.836. Unacceptable overall. No adoption/closedloop.

UnknownOCC intervention trajectoryXYRMSdelta .663/1.591m; oppositeposeOCCswap .0158/.0859. Sensitivity alone doesn'tcertifycorrectobstacleresponse. Reuseddev,2seeds, limitedroadsandoracleinputs remain.

Nextproposedonly: frozen decomposition of directbranch vsbackbone contributions and gradients to explain changedlongitudinalbehavior/seedinstability despite modifyinginteractionfeature. Do notcontinue gain sweeps orblindlongtraining.
