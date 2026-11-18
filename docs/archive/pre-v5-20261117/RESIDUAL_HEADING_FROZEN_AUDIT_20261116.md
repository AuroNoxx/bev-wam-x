# Residual heading: initial /128-step frozen audit

Script scripts/audit_residual_heading_frozen.py. Artifact artifacts/residual-heading-frozen-audit/report.json.8fixedTRAINwindows,one shared rollout noise,2seeds; freshinitial and trained128residual plus trained128original. No parameter updates,test or external calls.

Labels cos min0.995680,mean0.999711;sin−.092856..+.079052. No evidence of backward labels in thisbatch. Same randomnoiseinitialcos+.038. Finalrawcos:
-1980residualinitial−.620,trained128+1.631;
-1981residualinitial−.909,trained128−3.959;
-1981originaltrained128+1.389.
Failed1981residual rollout crosses to99.5%negativecos byflowt=.25 and100%negative by.5. Negative drift happens BEFORE finalnormalization; normalization cannot create the sign flip.

Bothfreshinitializations produce negativeaveragefinalcos, yet one learnspositive. Thus initialization alone does not determine failure. Originalrepresentation onseed1981 endspositive, so outcome depends on optimization/representation; not proven intrinsicresidualdefect.

## Weighted output-head gradients (not full-model attribution)
Fixed8trainbatch,post128 frozenmodel,seed-specificreplayedtime/noise. Failed1981residual:
FM weightedloss17.3504,headgradnorm752.667,cosbiasgradient−2.96875.
Position.0675,headgrad11.323,cosbias0.
Kinematics.1762,headgrad12.702,cosbias−.003967.
Unitheading.2415,headgrad39.559,cosbias−.180664.
Negative-speed.0156,headgrad14.680,cosbias0.
Combinedcosbiasgradient−3.15315. Negativegradient means descent increasescos, so currentobjective supplies a corrective signal, not a universallystable backwards optimum. JEPAauxdoesnotdirectlyconnecttrajectoryoutputhead, so its headgradientzero; sharedbackbonegradients NOT assessedhere. Adamactualupdate/globalclippingcannot be inferred from these localnumbers.

Actual recorded training total-loss values (steps96→128):1980original2.488→2.267,residual2.051→2.428;1981original4.604→3.360,residual2.624→10.406. All128stepsclipped in allarms. Gradnormmedian entiretraining approximately434/386/434/439;failedresidualmax5701,final32median266.6. These are actualrecordedglobalnorms,not the frozenbatchheadnorm. Frozenlosscomponents are NOT claimed to be originallastbatchcomponents.

## Conclusion and narrow next candidate
Evidence currently favors immature/unstable128-step optimization rather than an unnoticedlabelcoordinateflip or zero corrective headinggradient. Warmup reachesmaximumLR exactly at stopping; no subsequentconvergenceinformation. Do not directlypatchlabels,flipcos,replacehead,or claim safety.

If another trainingtest is authorized, a singlevariable peakLR reduction (same128warmup, samefresh2seeds/data/representation,128or256boundedupdates) is a more direct falsification of optimizationinstability than stackingnewlosses. Record loss spikes and rawheading throughout, comparefulltrajectorypositionderived ANDspeedhead dynamics,progress andworldpersistence. A smallerLR might merelydelaylearning; must not judge bylowerlossorcorrectcosalone. No suchrun hasbeen started. Existingoriginal1024modelsandallweightsremainunchanged.
