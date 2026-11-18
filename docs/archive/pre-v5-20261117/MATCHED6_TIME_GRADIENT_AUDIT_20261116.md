# Frozen physical-time and weighted-gradient audit, 2026-11-16

Scope: no optimizer/EMA update, no training, no CARLA, no test. scripts/audit_matched6_time_gradients.py; artifacts/matched6-time-gradient-audit/report.json. Two frozen matched6 step512 checkpoints, two fixed train-only batches each: 3earlystart,3observedmotiondecel,3hold,3steady. Available train counts32/592/131/205. These are stratified correlated windows, not independent trials. Position gates are fixed representative Bernoulli gates with the actual factor2; not a replay of the original final minibatch. FP32+MATH.

## Time axis integrity
Regression decoder inputxt=0 anddiffusiont=0 is intentional. Physicalfuturetime identity is providedby25distinct learnedpositionembeddings. Reversing theseembeddings whilekeepingcontextfixed reverses the25q outputs to maxerror4.47e-7..5.07e-7. Settingpositionszero makes alltimeoutputsidentical to2.25e-7..2.98e-7. Thus no evidenceofaccidentalbroadcast,missingtimeaxis orlabeltimeindexerror inthispath. Permutation is a controllednumericaltest, not an actualdrivinginput.

Initialpositionstd .020213/.020221;step512std .020120/.020135. ParameterRMSchange .001908/.001855;notfrozen/notdetached. All25timeembeddingsreceivenonzerogradients. Existingactionvelocityencodingroundtrippasses;representationdoesnotrequireconstantvelocity. Temporallyflatlearnedpredictionsaredistinctfromanunrepresentabletarget.

## Actual weighted gradient terms (4batch/checkpoint combinations)
ActionMSEglobalnorm .940..3.222;alreadyweightedpositionauxiliary2.930..6.877;alreadyweightedworld .279.. .431. Action-versusposition globalcosine .9914.. .9979, stronglyalignedinthese batches, not an observeddestructiveconflict. Worldversusaction −.011..−.080, worldversusposition−.006..−.063: smallnegativealignment, insufficient evidenceofJEPA dominatingthefailure. Worlddoesnotdirectlydifferentiatetrajectoryposition/outputparameters here, although itdoesaffectsharedcontext.

Positionparametergradientnorm action .01015.. .02320 andweightedposition .02015.. .04657;temporal-contrastcomponent (subtracttime-mean gradient) .00837.. .01004 and.00538.. .01195 respectively. Therefore thereis a nonzero gradient encouragingtime-dependentlearning, not complete cancellation.

Combinedglobalnorm4.640..9.660, hypotheticalexistingclip1multipliers .1035.. .2155. Theseare4frozenrepresentativebatches, notthewholetrainingdistribution. Gradientnorms acrossdifferent-sizedparameters arenotdirect learningrate comparisons. Adam adaptivepreconditioning preventsinferringeffectiveparameterprogress fromrawnorm orclippingalone; doNOTincreaseclipbasedonthisaudit.

## Interpretation
Togetherwithpriorfrozenresponseaudit: models learnlargecondition-dependentvelocityoffsets butlittlefuture-timevariation, inbothfitandheldsamples. The actionandpositionobjectives providea temporalgradient, andpositionembeddingschanged,butnotmuch. Possiblefactorsremain limited2.82passbudget, weakphysical-timeconditioning/sharedheadoptimization, andfuture-leadmultimodal averaging. Theseobservations cannot distinguish theircausalcontribution orproveextraepochsfixit. Currentgradient evidence doesnotjustifydroppingworldloss, retuningauxiliaryweights, orreplacingattention topology.

A minimalnextdecision, ifrootauthorizescontinuation: keepdata/representation/loss/topology/seedunchanged andallowonlyoneboundedresumedmilestone, preserving512checkpoints andexplicitlyrecordingtheextensionofitscompletedLRschedule. Predeclareevaluationofoppositefuturevelocityslopesforstartvsdecel, immediatehold, dynamics, progressandseedagreement, notjustaggregateADE. Stopratherthanautomaticallyextendagainiftemporalresponsefails. Alternativephysical-timeencodingisaseparatesingle-variablehypothesis, notsomethingtoaddsimultaneously. No suchcontinuation wasstartedbythisaudit.

Checkpointfilesandallmodeltensors, includingbuffers, verifiedunchangedaftercontrolledembeddingrestoration. No training process remains.
