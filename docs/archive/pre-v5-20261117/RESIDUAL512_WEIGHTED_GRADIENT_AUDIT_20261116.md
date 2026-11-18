# Frozen weighted full-model gradient audit at residual512

Reproduction scripts/audit_residual512_gradients.py; output artifacts/residual512-weighted-gradient-audit/report.json. Bothresidual512seeds,identical4TRAINwindows andtime/noise seed1990. No parameterupdate,optimizerstep,test orweightoverwrite. Gradients are alltrainablemodelparameters,alreadyweighted by actual objective. Threegroups:FM,all existingphysicalterms,JEPAaux. Not rawunweightedlossgradientcomparison.

|Seed|FM norm|Physical norm|JEPA norm|FM–physical cosine|Combined norm|clip1 coefficient|
|---|---:|---:|---:|---:|---:|---:|
|1980|54.353|21.310|.34224|+.68024|70.600|.014164|
|1981|50.585|11.622|.27067|+.42229|56.486|.017704|
JEPAwithFM/physicalcosinesnear0(.00038–.00267). PhysicalandFM arealigned onthisbatch,not strongmutualconflict. JEPAgradientmuchsmallerinaggregateparameterEuclidean norm,which doesNOT alone establish ineffectivelearning:parameterlocations,Adamnormalization andtaskscale matter.

## Frozen hypothetical Adam update
Saved moments/step/lr,fullgradient computedabove,NOapplication. Excludes decoupledweightdecay(equalacrossvariants). ActualLR9.79363e-5. clip1update norm.11094/.13675; hypotheticalclip5.36794/.45056. Ratio3.3166/3.2949; directioncosine.66770/.82883. clip1update/parameternorm~.000345/.000425. Currentglobalclipping doesnotmakeupdatezero; relaxingclip isnotmerely a scalarlearning-ratechange withhistoricalAdamstate.

## Minimum falsifiable next plan (NOT STARTED)
If authorized, compareclip1vsclip2only fromsame512savedstates,2seeds×2arms×64updates,unchangedLR/objective/representation/data/noisestream. Choose2ratherthan5asconservativeperturbation,notbecause2provedoptimal. CheckactualAdamupdates/gradientstatistics,headandposition-deriveddynamics,progress/backwards/nearzero/diversity,worldvscontemporaneouspersistence. Definefailureonnonfinite/OOMorclearpairedmulti-metricdeterioration;do notpromotebasedonlossoroneconsistencymetric. Keeporiginal512andallpriorartifacts.

Alternative ofsimultaneouslyraisingphysical/JEPAweights isnotjustifiedbythissinglebatch. Fairtestmust acknowledge2seedsand42route-correlateddevelopmentwindows,notdriverqualityorfinalacceptance. Currentprocessfinished,nolongtrainingrunning.
