# Spatially indexed frozen-context probe

Script scripts/probe_reactive_spatial_motion.py; artifact artifacts/reactive-spatial-motion-probe/report.json. Completed50.62s, no mainmodel update/CARLA/test. CheckpointSHA unchanged and future/teacher-extrafield contamination leaves context unchanged.

Pre-fixed token-order flatten480x128 -> fixedGaussian random projection384dimensions, seed2160. No labels used inprojection. Compareoriginalthree-blockspatialmean384dimensions, samealpha1 andtrain-onlystandardization; alsoeachplusstate. Fixedprojection was not retried orselectedbydev. Preserves tokenposition distinctions beforecompression but notallinformation; matchedoutputdimensiondoesnotguaranteeequaleffectivecapacity orregularizationbias.

Auditcommon412frames MAE m/s:
| checkpoint | meanpool384 | spatial384 | spatial+state |
|---|---:|---:|---:|
|2070base|4.004|1.705|1.702|
|2070history|3.077|1.773|1.743|
|2071base|2.924|3.165|3.003|
|2071history|1.061|3.213|3.095|

No-ID finalOCC geometricbaseline0.270 onthese sameframes. SpatialtrainMAE0.349..0.496 vsheld1.705..3.213. Independentnewrecording auditsharesroad/seed withdev; trainisoneepisode withdifferentroad/cruise, overlappingwindowsnotindependenttrials.

Conclusion: meanpooling canhide someinformation (seed2070) but isnotthesoleexplanation; spatialcontrol worsensboth2071 models. Neitherlinearprobe establishes fullencoderinformationabsence ordownstreamcausalcontrol utility. Projectioncompression,regularization,limitedtraining anddomainshift remainconfounded. Do notselectwinningarm,alterdefault orlaunchmoretraining fromthese resultsalone.
