# Pure-noise rollout dynamics paired trial

Astra-reviewed protections implemented: explicit scalar-forward t0 assumption, unavailable prehistory acceleration/jerk not fabricated; primary16Eulerstep development evaluation,4vs16 separate; position-derived speed comparison, nearzero-speed proportion, diversity, forward/lateral proxies; gradient norms/clipping; RNG and buffer assertions. Same128train/16reused DEVELOPMENT windows,notfinalheldout. No safety claim.

Completed seeds1810/1811×baseline/rollout,64updates each,teacher2048 weights,newAdamW1e-5,BF16,b8. Extra pure-noise4Eulerstep rollout onfirst2batchsamples every4updates,coefficient.01. No future labels in extraactorpath; ordinary original supervisedloss stilluseslabels as intended. Equalupdates/data not equalFLOPs. Paired first3FM RNGhashes equal, everyextra call restoresouterRNG and doesnotmutatemodelbuffers. No training beyondbudget,allprocesses exited normally;originalweights/testuntouched.

16-step main development metrics baseline -> rollout:
Seed1810:firstresidual6.618->5.627m/s;whole1.730->1.700;globalpeakaccel41.293 unchanged;globalpeakjerk206.463 unchanged. Nearzero12.50%->14.1875%;candidateRMSspread4.032->4.041m;forward17.596->17.610m.
Seed1811:firstresidual5.708->6.685 (worse);whole2.178->2.021;globalpeakaccel39.101->39.504 (worse);globalpeakjerk200.720->200.674. Nearzero6.875%->7.1875%;spread3.642->3.787m;forward19.445->19.676m.
Mean per-trajectory maxaccel/jerk for1811 also worsen13.256->13.441m/s²,81.224->85.952m/s³. Progress proxy not true routeprogress and no collision/TTC evidence. Allnearzero/spread exploratoryguardrails pass but requiredboth-seed initial/full-dynamics nonregression gateFAILS. No modelpromotion.

Worst firstsegment is samewindow13/candidate2:observed8.258523m/s;1810botharms firstpredspeed0 ->−41.292617m/s².1811baselinefirst.438334 vsrollout.357767 ->−39.10095 vs−39.50378. Zero-output clamp can produce saturation; no rawpreclampvalues were saved,so cannot claim everyzero was negativebeforeclamp.

All64updates inall4arms clippedatnorm1. Medianpreclipnorm17–20,max69–80. Additionalfirstprobe unweightedgradient norms(kin,acc,jerk):1810(106.08,22.66,220.79),1811(133.84,81.35,241.10). Extraweight.01 changes scale, and totalclipping rescales combinedgradient; no assumptionthat nominalweights equal effective optimization influence.

4vs16 same-noise2window meanpositiondifference:1810baseline1.233m/rollout1.275m;1811baseline1.282m/rollout1.340m. Discretizationmismatch persists/slightlygrows. Same2windowworldlatent~.232 vspersistence~.163 forbotharms,notimprovedworldreadiness.

Trainwall1810baseline34.41s/rollout27.91s,1811baseline25.55s/rollout27.93s. Firstcoldrun/IOconfounds mean no speedup claim. Peakreservedbaseline4.55–4.60GiB,rollout5.86GiB. Componentgradients measuredfirstrolloutonly,notentiredatasetgradientdistribution.

Conclusion:smallwhole-horizon residual benefit,notrobustfirstboundary/dynamicbenefit;predefineddevelopmentgatefailed. Do not adopt or scale thisregularizer onthisevidence. Nextdecisionneeds actualboundaryparameterization and4/16discretization/stability investigationrather than indefinitelyraisingpenalties. No automatic nextrun.

Artifacts:artifacts/teacher-rollout-dynamics-paired/{summary,gradient-and-global-summary,worst-first-cases}.json andper-armfullcandidate/checkpointreports. Script:train_rollout_dynamics_paired.py. Configstillinherits legacyshort_probe fields frompriortrial; authoritativeactualbudget64steps,2seeds,androllout_regularizer block,asthisreportandprocessoutputsdocument.
