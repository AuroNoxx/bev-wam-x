# Bounded CARLA synchronous and recording health check

Executed existing project CARLA1.0-a1770de89 executable, offscreen low resolution, after verifying no other CARLA/Python training task or2100listener. No driver/package/security changes. Officialtest andweights untouched. NoLLMcalls ortraining.

Added pure episode-contract helpers and7tests (all passed), alongside15 existingmetric tests: contiguousframes,0.05s delta, emptyrecording/missingframe/transporterror notcoverage, completeaccounting notsensor/safetycertification.

First connection attempt did not produce a usableworld withinbounded startup attempt; failed report retained at artifacts/carla-metric-health-20261116/report.json. OnceRPC wasobserved listening, retry on same server succeeded: Town10HD_Opt,10contiguoussynchronousticks,0.05s each, snapshot frameexactmatch, originalsettingsrestored. Complete scenariofingerprint captured. Noactors/sensors inemptyworldtest, so all drivingsafety metrics stayedunavailable.

Stationaryinstrumentation first attempt hit bindingcompatibility: is_listening iscallable inthisversion, andserializingthemethodfailed. Failure note retained. Corrected callable/property handling and reran tonewoutput; no previousresultoverwritten.

Successful stationaryretry: TeslaModel3 withconstantbrake1/handbrake,20settlingticks then40recordedticks (2simulationseconds). EachrecordedframehadmatchingIMUpacket andsnapshot;40/40accounted,nomissingframes. Collisioncallbackregistered/listeningatbothboundaries,rawcallbacks0. No deliberatecollisioncalibrationperformed: collisioncoverage remainsunknown and nozero-collisionpassisshown. No drivingplanner,traffic,routecompletion,worldprediction orLLM. Walltime0.111s forrecordedloop isheadlessstationaryhealth, notend-to-endplannerperformance.

All3ownedactors/sensorsdestroyed successfully, originalsettingsrestored. Independentreadbackverified0vehicles,0sensors,synchronous=false. Then onlytheownednewserverPIDs9008/20536 wereverifiedbyexecutablepath/portargument andclosed tofreeGPU/CPU; follow-upcheckshowednoCARLAprocessor2100listener. ExistingTensorBoard unaffected.

Artifacts:
- artifacts/carla-metric-health-20261116-retry/report.json
- artifacts/carla-stationary-metric-20261116/serialization-failure.json
- artifacts/carla-stationary-metric-20261116-retry/report.json

Remaining: nonstationaryrulebaseline withcalibratedcollision/laneboundaryevents,knownleadTTCsemantics,routeprogress/tracking,scenarioreset/repeatability,thenlearnedplanner;closedloopdrivingqualitystillnotmeasured. Successfulhealthdoesnotmakeformaltrainingreadyorcertifysafety.
