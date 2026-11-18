# Bounded native continuation128→512: original versus residual

Script scripts/continue_residual_pair512_v2.py; output artifacts/residual-representation-resume512-v2. Bothseeds1980/1981,bothrepresentations,384additionalupdates each,hardstop512. Total455.75s including evaluations/saves. No subsequenttraining; onlypreexistingTensorBoard remains.

## Recovery evidence
All4 source128 checkpoints verified: config/settings exact,Adamstep128,schedulerlast_epoch128,LR1e-4,savedtaskRNG andstatefulloader. Independent restoredloader producednext3batchstatehashes exactly matchingTraineractual batches;pairedarmsmatch. PendingRNGconsumed atfirsttrainingstep. Originaloptimizer/moments/schedule preserved,notweights-only restart. Full CUDA uninterrupted bitwiseequivalence notcertified. Firstscript failed beforeupdates becauseLightningdefaultweights_only rejectednumpyRNGpayload. v2 explicitlyloads trustedlocallycreatedfullcheckpoint(weights_only=False);failedscript/log retained,no duplicatedupdates. New256/512checkpoints perarm,8total,noneoverwriteoriginal128orotherweights.

Schedule unchanged128warmupthen4096horizoncosine,onlyauthorizedto512;LRstill~9.79e-5 at512.84optimizationtrainwindows,42same-route/internaldiagnosticwindows,notindependentvalidation;testuntouched. Theeffectiveoptimizationrevisitsfewwindows frequently;cannotcomparethisexperiment'sabsolutequality to5296-window1024training as equal exposure/generalization.

## Results at512
|Seed|Arm|First residual m/s|Full residual m/s|Mean trajectory peak speed-head jerk m/s3|Forward proxy m|Mean negative segments/25|
|---|---|---:|---:|---:|---:|---:|
|1980|original|98.880|35.263|415.645|37.659|10.964|
|1980|residual|3.952|12.975|322.359|6.821|9.262|
|1981|original|69.202|20.744|302.172|30.880|10.429|
|1981|residual|4.214|12.416|222.066|55.928|2.071|

No hard dynamics pass. Meanpeak acceleration still34.8–49.4m/s2,globalfirstacceleration59.4–93.6m/s2. Residualseed1981nearzerospeed fraction33.26% despiteforwardproxy55.93m andspeed-disagreement11.535m/s,showing unresolvedhead/positionconsistency. Lowerforwardproxy alone is NOT proofbad driving versusexpert;scene safety/requiredprogress unavailable. Do notrewardmoreprogresswithoutcollision/roadcontrols.

World ownEMA/persistence:1980original.43769/.25890,residual.42783/.26000;1981original.47344/.29864,residual.47092/.29821. Neitherbeatsbaseline. Latentspacesdifferacrossmodels so within-checkpointcomparison is intended.

## Warmup hypothesis
1981residualnegativecosfraction100%at128→0%at512,cosmean−.97975→+.92536 withoutchanginglabels,losses,headorrepresentation. All4cosmeanspositive(.925–.961)andallnegativefractions0%. This supports an immature/optimization-dependent128-step failure,not an immutablecoordinateerror. Doesnotprovefullconvergence or excludeotherlossinteractions.

Lasttraininglossesoriginal/residual:1980.5113/.9590,1981.9005/.9519;notdirectcrossrepresentationqualitycomparison. Continuinggradientnormmedians94–102 (downfrominitial~386–439),all384updatesstillclipped,peaks459–848. No nonfinite/OOM. SourceOCCcausality,free-space,collision,TTC,closedloopunknown.

## Decision
No defaultswitch or automaticlongrun. Residualhasconsistentinitial/fullconsistency benefit versus equallyimmatureoriginalhere,nowalso lowerjerk,butneitherisusableandseed-dependentprogress remains. If continuing, useonepredeclaredbudget withrepresentationcorrectlyversioned andphysicalmetrics throughout;do notreacttoeveryearlyheadingflip withanotherloss. Needactualcausalperception/controller/closedloopvalidation before anydrivingclaim. Currenttaskcompleteandstopped.
