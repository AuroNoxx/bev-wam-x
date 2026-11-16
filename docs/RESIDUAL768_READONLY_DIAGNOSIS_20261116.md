# Read-only768 dynamics horizons and world baseline

Scripts analyze_residual768_readonly.py,results artifacts/residual768-readonly-dynamics-world/report.json.42same-route/internaldiagnosticwindows x4savedcandidates, noGPUtraining,testorcheckpointchanges.

clip1first1s /1–3s /3–5s meanvectorconsistencyresidual(m/s):1980 2.946/5.473/11.112;1981 7.895/8.503/13.022.1980errorlargelylongitudinal (latealong10.065,cross3.002);1981latelong8.088,cross8.502soheading/geometryalso matters.1980late25.24%positionalsegmentsbackwardwithallheadingsforward.1981earlyheadspeed20.671 versusposition-chordspeed12.144. Notjustfirstpointfailure.

WorldownEMAforecast beatsownpersistence1/42windows forbothclip1seeds;clip2only1/42and2/42. Everyv0speedbucketmeanworse;under5bucketonly2windowsso notrobustslow-speedfinding. Medianforecast/persistenceMSEratiosclip1 1.315/1.228. Gap–trajectorycorrelation.300/.113,descriptiveonly,cannotclaimworlderrorcausestrajectoryerror. Worldstillunvalidatedeveniftraininglossfalls.

## Nextsingle-variable candidate, not implemented
Hypothesis: independentper-timeXYnoise undergrowingresidualscales makeslatepoint-to-pointderivatives unnecessarilyhigh-variance. ReplaceONLYXYbase-noisecovariance withstationaryAR1rho=.9, retainingeachtimevariance1 andallotherchannelnoiseunchanged. IdenticalcorrelationtransformationmustapplytoFMtrainingnoiseandpure-noiseinferencesampler; changinginferencealone wouldbeOOD. Existingweights cannot beclaimed compatiblewithouttest.

CPUalgebracheck20000syntheticnoisesequences:physicalnoisedifferenceRMS x/y121.342/21.906→38.916/7.022m/s;finalpositionstd32.042/5.574→31.838/5.524m. Marginalvariancesremain~1. Thisisbase-noiseproperty, NOT modelkinematicimprovement andnotproofcurrentnoise iscausalbottleneck. Fixedrho isapredeclaredchoice,notoptimizedvalue. Candidate diversity/turnresponsecouldworsen.

If authorized test, firstvalidatecovariance/inversionandpairedRNG, thenonebounded2seedcomparisonholdingrepresentation/loss/clip1/data/LR/updatebudgetfixed. Evaluatefullhorizonphysics,heading,speed-versus-position,progress,diversity,worldbaselines. Do notsimultaneouslyadjustworldlossorclip. Worldpersistencefailure needs itsownseparatehypothesis;noisechange mustnotbepresentedasfixingworldmodel. No newtrainingorcode-defaultchange performed.
