Four compatible old Bench2Drive checkpoints: route-regression-flow1024 seeds2050/2051 FM/regression latest1024. CPU strict_load succeeded,966939 trainable parameters. Trained on3683 unique Town12 windows/29recordings, not full42; zero overlap with fixed1124Town13val. Keep original DiskSamples20-dimensional state,4history frames,25x2velocity residual scaled10m/s both axes and trapezoid decoder. CARLAmatched6 checkpoints are NOT compatible semantically despite similar weight shapes. Original checkpoint manifest_sha is placeholder, not cryptographic provenance; savedstreams/design and source contract support traceability. Frozen FP32MATH inference differs from originalBF16 execution. K1 fixedfirstcandidate primary,K4distribution secondary,neverbestofK; evaluate exactbaseline1/2/3/5shorizons and nine-routebootstrap descriptive intervals. No training/install/test/closedloop. Metadata and SHA256:artifacts/bench2drive-existing-checkpoint-contract-plan/report.json. Initial resourceguard blocked on own idleCARLA server; root coordinating release before inference.


## Executed frozen comparison
Root authorized inference after own idle CARLA server was stopped by carla_build. Original resource guard passed. No model/data/guard changes. Four frozen models completed in42.02s, all source checkpointSHA256 unchanged. No training/Carla processes remain, only TensorBoard.

Input/normalization preflight: three fixed val anchors per checkpoint, strictstate load; 20state slots,4history frames andnominal ages intact; poisoning future/trajectory labels leaves actor output exactlyunchanged. Velocity encode/decode roundtrip max1.91e-6m/s. Integrating labelvelocity differs from pose-label positions by .062/.097/.064m mean onthe3anchors; not silently replaced. K1 independently computed equalsK4designatedfirst within recorded tolerance. Source OCC causality stillunverified.

Primary fixedcandidate0 endpointposition errors[m],1/2/3/5s:
|model|1s|2s|3s|5s|
|---|---:|---:|---:|---:|
|constantvelocity|.901|2.503|4.270|8.631|
|2050Flow|1.695|3.752|5.946|10.411|
|2051Flow|1.852|4.154|6.599|11.644|
|2050regression|1.151|2.647|4.286|7.855|
|2051regression|.977|2.482|4.093|7.773|

5s speedmagnitudeMAE: CV2.513m/s; Flow2.666/2.867; regression2.155/2.276. Flowdoesnot beat CV at anyofthesehorizons. Regression partially improves longerhorizons,not universally.

Completehold72windows,5spositionerrors: stationary.028m,CV.088m;Flow10.866/8.670m;regression4.896/2.389m. Decel169windows5s:CV11.768m;Flow14.310/15.771m;regression10.423/11.843m. These are labelfit metrics,not closedloopactualdrift or safety acceptance.

Paired9-routebootstrap descriptive95% intervals for5spositionerror difference(model-minusCV):2050Flow+1.780m[+.171,+2.783];2051Flow+3.013m[+1.629,+3.881];2050regression-.777m[-2.413,+.384];2051regression-.858m[-1.814,-.166]. Only9routes and repeateddevelopmenthistory; do not treat overlapping1124windows as independenttrials or intervals as certification.

K4spatialspreadRMS:Flow4.374/4.909m,deterministic0. Nooraclebest-of-K. Runtimepermodel includingI/O:Flow13.61/13.74s,regression6.10/6.23s. Batch4forwardP95(Flow4candidates) .0353/.0372s;regression.00782/.00782s. GPUpeakallocated.0973GiB. These are warmed local offlinebatch timings,not car-end-end latency.

Authoritative artifacts: artifacts/bench2drive-frozen-oldweights-v1/summary.json, metrics.csv, permodelreport/rows/predictions/preflight, fixed_noise.pt. Source scripts include evaluate_bench2drive_oldweights_frozen.py. The initial blockedrunlog was retained; no guardwasbypassed.

Conclusion: compatible input and properbaseline comparison now exist, but these are our oldsmallmodels trainedon3683windows, not matureofficialweights or fullofficialtraining. Observed failurecannot indict Bench2Drive or FlowMatching universally. No automatictraining,installation,thresholdadjustment or deployment follows.
