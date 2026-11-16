# Explicit observed-motion actor paired512

Isolated experiment: artifacts/carla-observed-motion-paired512. Two seeds warm-start their history256 checkpoint. Both arms reset AdamW and the same 64-step warmup / 512-step cosine, batch12 and 512 updates. This is not strict optimizer resume. No old weights overwritten.

Both arms add the same 4416-parameter zero-initialized 5-to-32-to-128 action-memory residual. Baseline receives zeros; observed arm receives causal OCC-only geometry and motion. Existing world supervision stays unchanged. FP32/BF16 paired/predecessor equality and future-label poisoning tests must pass before training.

Features: candidate front-surface distance from ego origin /50 (not true bumper gap), past0.6s lead motion along navigation /15, ego-minus-lead closing /15, association valid, low-motion uncertainty. Missing=[0,0,0,0,1]. Uncertainty threshold0.698678m is train-only. Ego scalar speed projected along navigation is a documented approximation. No lead ID, pose, true gap/velocity, future or teacher decision enters actor.

Pre-registered gates in design.json: both seeds deceleration5s MAE improve>=20%; other groups0.6s/5s MAE worsen<=10%; at least95% candidates satisfy peak acceleration<=3m/s2 and jerk<=5m/s3. Include t0 velocity; do not invent pre-t0 acceleration. Failure means no closed loop. These are diagnostic engineering gates, not safety certification; existing guards unchanged.

One train episode road388/cruise8 vs dev713/cruise8.5; highly overlapping windows, no independent driving trial count, no occlusion or highspeed coverage. Preserve unknown. Teacher privilege and geometry low-speed noise remain limitations. No automatic extension or automatic Astra RL.

## Completed result
All4 arms512 updates completed. Total training+evaluation581.08s, train132.74-133.87s/arm,971355trainable parameters,peak0.936GiB. Same sequence andfirst3 FM RNG hashes perseed; oldcheckpointSHA unchanged. Newbest/latest includeoptimizer,scheduler,loop state; noactualresumeexecuted.
Failure against pre-registered gates: seed2070deceleration5sMAE5.278->5.319(-0.79%improvement);seed20714.996->4.977(+0.38%), below20%requirement. Everyarm0/2056candidatespassfullvectoracceleration<=3andjerk<=5. Meanvectorpeakjerk100.1..118.4m/s3. NoCARLAclosedloop,nochangedguards,noautomaticextension.
State ofknowledge: explicitobservedmotion didnotdelivermeaningfulactionquality underthissmallwarmstartbudget. Thisdoesnotprovegeometryunusable: itwasreadableinprioraudit. Do notattributefailureuniquelytoFlow,representation,dataquantityoroptimization. Onlyoneepisodepertrain/dev remains. Newbranchpluscondition was the intervention, not a completeperception/planningfix.
