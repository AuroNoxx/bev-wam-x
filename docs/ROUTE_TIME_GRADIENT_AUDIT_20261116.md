# Frozen flow-time actual weighted-gradient audit
Completed19.16s, no training/optimizer, all model parameters andbuffers andsourcecheckpointSHAsunchanged. TwoFMseeds, two actualtrainbatches12 each(seedstreamoffset0/6144), fixedtime sweepsandactualmixedtime replay. BF16trainingmode, alltrainableparameters inclsharedworldbackbone. Frozenfinal1024weights, not historicaltrainstepweights; sample-limitedmeans not populationcausality.

Exacttime RNG reconstruction: seed2050positionactive6109/12288(49.72%),20516128/12288(49.87%); perdecile1158–1279, no grossfrequencyimbalance. Lowt positiongradientexactzero fromgate, notmissinglowtFMtraining. Eulerinference visits0..15/16 buttrainingrandomuniformt withtrue-target bridge; inferenceenterslearnedstate, nottruebridge.

Meanacrossfourbatches:
- t0 FMgrad1.536, actualweightedposition0, weightedworld(combined0.1each).085. Counterfactualalltimeweightedposition2.168; cosineFM/position.683; clipcoefficient.710->.312.
- t.5 FMgrad1.300, actualposition1.440, alltime.720; currentclip.502->.602.
- t.9 FMgrad2.604, actualposition.180; t.99FMgrad3.860,actualposition.0026. PositionendpointJacobian shrinksnear1.
- Actualmixedtimereplay FM1.806,position.627,world.085; counterfactualalltimeposition.555,clip.465->.491,combineddirectioncos.935. Thus alltimeisnotuniformstrengthboost; itredistributesgradient andchangesclipping.

FieldRMSE(q/time) vsendpointqRMSE: t0 .241/.241; t.5 .353/.176; t.9 .779/.078; t.99 1.067/.0107. Near-targetendpointimprovementisstronglyconfoundedby(1-t),notproofgoodlearnedfield. Highernear1fieldRMSE mayreflectirreduciblenoiserecovery asbridgehidesnoise; these numbersalone cannot identifydatasetvsoptimizationcause.

Decision: do not changeauxiliarygateortrain. Userprioritizesdatasetcoverage/causality/labelsnow; no newtrainingorlargeparameterscans. Findingsshowa realcoveragechoicebut do not establishit asdominantfailurecause. Next independentworkshoulddataauditandbroaderrepresentativeevaluation; anylaterobjectiveexperimentmustkeepmatcheddataandbudgetandexplicitlyaccountgradient/clippingchange.
