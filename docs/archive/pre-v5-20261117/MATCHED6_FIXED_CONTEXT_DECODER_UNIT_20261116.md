# Fixed-context original decoder learnability unit test

Date2026-11-16. Completed and stopped. scripts/test_fixed_context_decoder_fit.py; artifacts/matched6-fixed-context-decoder256. This is NOT a driving model or main FM training, and no CARLA/test/new data was used.

## Preset scope
Five exact-actor empirical mean velocity targets plus five preselected train deceleration labels. Contexts are actual480x128 outputs of each original1024checkpoint encoder, computed without future labels and then frozen. Future-label poisoning leaves context exactly unchanged. Original Transformer decoder/25timepositions/trajectory+time projections/shared2D output retained; no architecture replacement. Only549762decoder-side parameters trainable; encoder/predictor/world branch frozen and excluded from computation duringfit. No auxiliaryworld/positionloss inthisisolation: actionqMSE only. This changeisforlearnabilitydiagnosis, notafairmaintrainingqualityablation.

Twoinitializations fromexistingseeds7201/7202,256updatesfullbatch10, AdamW1e-4 wd.01 constantLR,clip1,FP32+MATH. Hardlimit2minutesperarm, noextension. Lightning+StatefulDataLoader+VerifiedCheckpointIO. Predeclared unitpass: everytarget vector-componentvelocityRMSE<=.1m/s ANDcorrectstart/decelendpointtrend; notadrivingsafetycriterion.

## Outcome
Botharmsfinished256 normally. Successfulworktotal8.59s; train+eval3.04/2.58s, peakallocated.199/.199GiB. No nonfinitevalues. Sourcecheckpointshashesandallfrozenparametersunchanged. Mainmodelsuntouched. Unitlatestcheckpointsretainoptimizer/scheduler/taskstate. best.ckptselectsfinal256overinitialbythetwoexplicitlyevaluatedactionMSEvalues; NOTbestofallintermediateupdates, andnotbestdrivingquality.

ActionMSE7201 .030284->.008700,7202 .026586->.007698 (about71%lower). But0/10targets passeachseed.
- Seed7201 physicalcomponentRMSE.717..1.159m/s.
- Seed7202 .544..1.087m/s.
- Fivestarttargets requirevxrise4.57..6.48m/s; predictionsrise1.67..1.70 /1.90..1.98.
- Fivedeceltargets requirevxchange−1.04..−2.76m/s; predictionsinsteadRISE1.60..1.66 /1.63..1.81.
Thusdecoderlearnedalargertimecurvebutsharedasimilarrisingcurveacrossconditions, includingoppositerequirements. Averages/lossreductionalonewouldhidemajorfailure.

## Gradients
Timepositionandoutputgradientsnonzerothroughtraining. Selectedsteps0,1,15,63,127,255saved. Seed7201globalpreclipnorm1.60,6.03,.749,.0396,.434,.648;seed7202 .233,5.17,.407,.181,.157,.312. Finalpositiontemporal-contrastgradientpostclip .000930/.001205. Thisisnotanall-zero-gradientorall-steps-clipped explanation.

## What this does / does not establish
Withthisfixedcontextandshortbudget,theoriginaldecoderdoesnotlearnoppositeconditionaltemporalshapes,evenafterexplicitlabelambiguityisreplacedbyonefixedtargetperinputandJEPA/positiongradientcouplingisremoved. Itcanlearnanoveralltimepattern, soweakphysical-time/conditioninteractionisamorefocusedhypothesisthantimeaxisabsence.

ThisdoesNOTproveTransformerexpressiveincapacity, sufficientoptimizationbudget, orfullyinformativefrozencontexts. Contextinformationloss/near-degeneracyremainspossible; thetestcannotassignblametodecoderalone. Itdoesnotshowheldgeneralizationbecauseall10itemsarefititems. Anewarchitecture, longerbudgetorclosedlooprequiresaseparatedecision. Nothingfurtherstarted.
