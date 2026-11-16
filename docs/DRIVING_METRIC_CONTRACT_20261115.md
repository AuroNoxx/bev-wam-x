# Multidimensional driving metric contract and offline baseline harness

Implemented new isolated files:
- scripts/driving_metric_contract.py
- scripts/test_driving_metric_contract.py
- scripts/evaluate_rule_baseline_logs.py
- scripts/run_rule_metric_fixture.py

15 unit tests passed. These cover unknown vs instrumented zero; stable event-ID deduplication/conflict rejection; exact derivative units; no derivative acrossmissing/gap>0.25s; strictly increasing simulation timestamps; valid in-path lead conditions; nonclosing/standstillTTC; planning event dedup; matching fingerprints; required scenariofields andseedchanges. Event IDs must come from instrumented sensor/frame/contact semantics; do not invent broad time-window merging of distinct impacts. Callback event count is not automatically a collision-episode count.

Every metric carries value/unit/available/reason/n. Missing input returnsnull rather than0. No overall safety/driving pass or ADE-based winner. Safety, feasibility/comfort, efficiency and latency are separate. World prediction and cloudLLM currently unavailable, explicitly null. Scalar-speed acceleration/jerk are not lateral/vector comfort; lateral acceleration/curvature/tracking error require actual measurements. Simulator time is separate from planning wallmilliseconds. With only one planning observation a p95 is merely that sample, not credible tail-latency characterization.

TTC requires validated in-path lead, bumper gap>=0 and closing speed>0.1m/s. Headway additionally requires ego speed>0.5m/s. No qualifying lead is unavailable/not-applicable, not zero risk. A observedzeroTTC is allowed atcontact. Fixed thresholds here define numerical eligibility only, not safety thresholds. Instrumented empty event list can yield0 observed events, but never implies the episode passed. A failed/truncated episode remains failed.

Full scenario fingerprint helper requires simulator version,map,weather,trafficseed,route,initialactors,fixed timestep,sensor configuration andcontroller version. Keep controller consistent when comparing planners. Caller must supply verified complete values; equality of externally supplied opaque strings is not proof of scenario provenance. Legacy adapter has no fingerprint and cannot be used for a matched policy comparison.

## Executed offline evidence
Toy1D metric plumbing fixture (not CARLA): same scenario andstep schedule, rulefollowing produced0contactonsets andprogress102.95m; deliberately unsafeacceleration produced1contactonset andprogress264m. This demonstrates why progress alone must not rank driving and why no aggregate winner is returned. The toy dynamics continue aftercontact and are intentionally not a realistic collisionresponse. Artifact: artifacts/driving-metric-contract/toy-follow/.

Read-only existing CARLA rulebaseline adapter: logs/occ-closed-loop/20261115-220918-live had2telemetryrows over0.1simseconds and statusfailed, simulator RPC timeout90000ms. It reports legacyemptycollisioncallbacks but no coveragecertification; safetycounts/TTC/routecompletion therefore remainnull, notzero/pass. Planning latency has only1unique event78.06ms, insufficient for performance qualification. Old telemetry speed is sampled beforetick while timestamp is assigned aftertick, so its scalar derivative is approximate and not a calibrated comfort result. Canonical result: artifacts/driving-metric-contract/legacy-live-v2.json (v1 retained as earlier output).

## CARLA and remaining work
Process/port check: noCARLAprocess andno2100listener; onlyexistingTensorBoard. No new CARLA episode launched, noLLMcredentials/calls, notraining. Existing live runner writes globals/current status and lacks fullcanonicaleventcoverage/timealignment/fingerprint, so it was not blindly relaunched as a metricvalidation. Next integration should be a new bounded runner/adapter that captures same-frame snapshot speed/pose,time andevents, complete coverage start/end, consistentrouteprogress, planningeventIDs and scenario fingerprint; first verify synchronous server/tickhealth, then short fixed rulebaseline. Unknownleadvalidity must remainunknown, and onlyownedactors may becleanedup. Closed-loop learned-policy/world/LLM results remain unmeasured.

No officialtest orweights touched. This is evaluation infrastructure, not prooftraining is ready or driving is safe.
