# Frozen sampling-path audit after fresh1024

Reproduction: scripts/audit_fresh1024_sampling_path.py. Results artifacts/fresh1024-frozen-path-audit/report.json. Four1024 checkpoints read-only;8 common reused development windows; one fixed noise seed1970; same16/32Euler paths; no optimizer or parameter updates. Compute14.83s.16 unit tests for physical decode/encode roundtrip, constant velocity/acceleration,first interval,FLU left axis and learned-bridge contracts pass.

## Observations
Target labels on these windows: first/full residual0.07752/0.13757m/s. Position-derived average speed7.7585 versus recorded7.7806m/s. Unit heading norm1.0. Thus no evidence here of a gross scale or axis error shared by all labels. Small negative label speeds exist(2%points); magnitude/provenance not yet characterized. t0 assumes scalar speed along ego-forward, lateral velocityunknown. Source OCC past-only generation still uncertified.

|seed/mode|16step first residual|16step full|32step full|bridge t=.95 full|
|---|---:|---:|---:|---:|
|1960 JEPA|29.734|11.485|12.705|6.487|
|1960 noJEPA|71.456|14.277|15.411|5.587|
|1961 JEPA|41.040|7.364|8.455|3.510|
|1961 noJEPA|17.974|6.426|8.107|5.079|
All residuals m/s. This8window/onecandidate table differs from earlier27window/fourcandidate report and is not a replacement or improvement claim.

All four raw trajectories in this audit have no negative predicted speed. Normalizing heading and clamping speed therefore leave these kinematic metrics unchanged; heading is already normalized within residual calculation. So clamp is not the proximate cause in this batch, despite previously observed bad negative-speed boundary cases. Increasing Euler16→32 worsens full residual in all4; more integration steps alone is not a fix.

Label mean first displacement1.244m. Model16step mean first displacement6.421,13.076,9.917,3.647m respectively. Large initial spatial offset and step-to-step position variation are real predicted defects, not just a mislabeled acceleration diagnostic. Mean position-derived versus predicted speed:13.14/6.88,14.63/6.24,11.18/12.81,9.49/7.51m/s.

The target-informed bridge is explicitly a diagnostic, NOT actor input or an allowable online sampler. Its t=.95 full residual3.51–6.49 remains far above labels, so trajectory quality is already poor close to targets. Pure-noise sampling further amplifies errors in most cases. This supports a combination of insufficient learned local consistency and path/distribution mismatch; it does NOT isolate a causal training fix.

## Mechanism and priorities
Source code confirms fixed output units[x/50,y/10,sin,cos,v/30] and.2s. Derivatives amplify normalized point error:0.01 x means0.5m and2.5m/s overone interval; differences of independent point errors can be worse. FM averages normalized channels, whereas physically important position differences have much larger unit gains. Existing kinematics loss does include t0, but is averaged across25segments and applied only to target-informed endpoint estimates at t>=.5. No hard position/velocity continuity is enforced. These are design facts, not proof of which change will work.

Priority before additional training: decompose common FM errors by channel/time, compare physically weighted endpoint residuals on SAMEnoise/t, inspect target negative speeds and timing/frame metadata on bad windows; retain current labels as diagnostic baseline, not expert-optimal driving truth. Then choose ONE predeclared change or bounded continuation based on evidence rather than stack penalties. Do not infer that JEPA removal,clamping,moreEulersteps or ADE improvement solves safety. No4096 continuation started. Collision,TTC,free-space and real dynamics remain unknown.
