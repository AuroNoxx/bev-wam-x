# Frozen branch/backbone decomposition 2026-11-16
artifacts/carla-direct-decompose-frozen/report.json. Fourcheckpoints(2seeds, raw/RMS branch)SHA unchanged;628recoverydev contributionanalysis,8actualtrainstates gradientanalysis. No updates/test/closedloop/defaultchanges.

Important scope: 'lateral-related interactionfeature normalization' is NOT lateral-output-only. Feature4 feedsall50 linearoutputs, including25longitudinal. Actions/position/world jointlyupdatebackbone andglobalclip. Staticv0 yieldsinteraction0, but otherbranchfeatures andlearnedbackbone remain.

Seed2061staticendx raw4.601->RMS5.718m inthisdecomposition:backbone4.597->5.707, directdelta+.00465->+.01089m. Thus~1.11m extra drift almostentirelycomesfrom changedbackbone, not instantstatic interaction.2060static-.707->-.849:backbone-.703->-.842,direct-.00398->-.00698m. Branchcomponents computedBF16 thenpromotedFP32 foraddition; originalfullforwardBF16 roundingdiffersfewmm/cm. These diagnosticnumbersdon'treplaceoriginaleval.

8trainstates frozenFP32 weightedgradients:2061 actiony backbone norm.406->1.123, branchy .01984->.05454; actionx/ybackbonecos+.0269->-.0472.2060y .150->.335,cos+.133->+.139. Expectedpositionweight.2(not actualrandomtgate) andworld includedseparately. Localconflict is evidenceofcoupling, not definitivecause ofallseedinstability. Sharedclipping furthercouplesupdates; no claimclipalonecausedit.

Identicalobservations withdifferent scriptedfuturestart times remain; twoinitializationslearn different longitudinalfunctions. Changingconditionalgradientthroughsharednetwork can move thisunstable solution. DirectbranchRMSactivation works,butglobalbehaviornotpreserved.

Proposednextminimumonly: freezechosenbackbone+longitudinaloutputs, fit lateral-onlynormalized directresidual withsamefixeddata andverifylongitudinalq/XYx unchangedexactly. This isolates learninginterference; cannot repairknownlongitudinalfailure orprove driving. Noautomaticexperiment.
