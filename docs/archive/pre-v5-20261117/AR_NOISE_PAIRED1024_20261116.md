# Matched XY AR1 noise single-variable continuation768→1024

Scripts experimental_ar_noise.py and continue_ar_pair1024.py. Output artifacts/ar-noise-paired1024. Bothseedsbranchfromsameiidclip1-residual768checkpoint;256updatesperarm,originaloptimizer/scheduler/RNG/loader,clip1unchanged. ARarmusesrho.9 forXY only in BOTHFMbridge andinferencesampler. Other3noisechannelsunchanged. Thisisdistribution-switchadaptation,NOTfairfromscratcharchitecturecomparison. Hardstop1024,noautomaticextension.

Preflight:XYstationaryvariance.993582,lag1correlation.899227;rho0originalnoiseexact;actualfullmodelrho0lossandalltermsidenticaltooriginalcode;trainingandsamplingnoisedrawsexact;futurelabelpoisonactorinvarianceexact;finiteARsample. Pairedfirst3actualdatahashesandbaseFM RNGtracesmatch(includingepochboundary). Inferencesamplerrhoisrecordedinsideeachcheckpointnoise_runtime;loadingARweightswithoriginaliid-samplerisNOTtheevaluatedconfiguration.

|Seed|Noise|First residual m/s|Full residual m/s|Mean trajectory peak speed-head jerk m/s3|Forward proxy m|Nearzero speed fraction|Candidate RMSspread m|
|---|---|---:|---:|---:|---:|---:|---:|
|1980|iid|2.025|6.365|163.801|37.647|.0957|3.231|
|1980|AR.9|1.725|4.906|210.370|29.702|.0786|3.603|
|1981|iid|2.725|6.797|227.768|53.391|.0250|3.449|
|1981|AR.9|2.656|5.610|236.067|23.605|.6590|3.726|

Bothseedsimprovevectorconsistency,withoutthisspreadmetriccollapsing. Bothspeed-headpeakjerkmeansworsen.1981ARparticularlyhas65.9%nearzerospeedpointswhilepositionsstilladvance23.6m:clearunresolvedcoupling,notautomaticallysaferbraking. Backwardsegmentsrise1980 1.780→2.274,1981 1.161→3.964. Progressproxyalonecannotjudgeexpertdifferenceortrafficcorrectness,butcombinedconsistency/nearzerotradeoffsdisallowclaimingoverallimprovement.

WorldownEMA/persistence1980iid.393342/.319659,AR.418856/.318141;1981iid.455581/.402414,AR.488774/.401158. Neitherwinsbaseline;ARratiosworse. CrossmodelEMAspacesdiffer,do notinterpretabsolutemseascommonmetric.

Total324.75s,eightnewverified896/1024checkpoints,oldweights/testunchanged.NoOOM/nonfinite.Noactive training remains.84train/42same-routecorrelateddiagnostic,2seeds,4candidates16Euler,notindependentfinalvalidationorclosedloop. NoexternalLLMtraining.

Decision:doNOTadoptARdefault;retainiidandclip1. Noise-covariancehypothesisyieldedlimitedpositiveconsistencyevidencebutfailedmultidimensionaltradeoff. Do notcontinueblindlyorstacknewpenalties. Nextproposalneedsaccountforindependentposition/speed/headingoutputsandworldbaselinefailure,ratherthanreportinglowerresidualasdrivingsuccess.
