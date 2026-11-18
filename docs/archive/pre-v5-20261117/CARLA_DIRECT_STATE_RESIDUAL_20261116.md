# Zero-initialized direct state action residual trial 2026-11-16
artifacts/carla-directstate-paired512 and carla-directstate-perturb-frozen. Two seeds2060/2061 x512,144.36/134.78seconds; latest/best isolated, RNGfirst3 matched originalregression, exit0. No test/closedloop/default/guard/oldweightchanges.

Compared to deterministic25 head, add exactly200parameters in zero-initialized bias-free Linear4->50, outputadded to normalized25x2velocityresidual. Inputs current speed/10, near-x/10,near-y/1, speed/10*near-y/1. Loadercanonicalnav unchanged; no newobservation/label/teacherfeature. KeepOCCbranch, originalloss/data6144/optimizer/512budget. FP32/BF16 initialoutputs exactlyequal bothseeds. Causalpoison andslotunits/irrelevantfields tests passed.

Earlycorrective dev82 MAE baseline->direct:2060 .035017->.034696m,2061 .035492->.034971; significantsignaccuracy remains50% each. Whole628 lateralerror .01236->.00896,.01397->.01163; smallnumericalgain notcorrectivelarning. Static5sendx-0.706/+4.600m; decelendv6.099/6.748m/s vslabel1.836, bad. No deploymentadoption.

OCCcounterfactual10devstates: allunknown trajectoryXY RMSdelta .578/1.619m (baselines.720/3.279); oppositeposeOCCswap .0192/.0716m. Navflipmean|y06change| .000547->.000969 and.000683->.001055. OODsensitivitydoesnotcertifyobstacleattention orcorrectavoidance; guards remain.

Read-only weights: newbranchmaxabs .02010/.01951; first3lateralinteractioncoeff .008–.015. Nonzerolearning but tinyphysicalcontribution. Shortbudget/lowsharedlr could constrainzerohead learning; notproven impossible architecture. Nextproposedonly: comparelinearbranch gradient/reachableamplitude withreferenceleast-squares answer before additionalarchitecturalchanges; no automaticlongtrain.
