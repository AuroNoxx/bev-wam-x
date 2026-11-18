# Route-balanced TRAIN-only frozen JEPA audit (2026-11-16)

## Scope
Zero updates, fixed original2048 checkpoint and its EMA, no test enumeration or old-weight writes.42Town12 training routes,3deterministically spaced windows per route=126windows;6scenario families,7routes/21windows each. This is already-seen TRAIN data, not generalization or held-out acceptance. Compute/report phase10.35s excluding imports/setup, within5minute cap.

Scenarios:HardBreakRoute,HighwayCutIn,HighwayExit,MergerIntoSlowTraffic,MergerIntoSlowTrafficV2,SequentialLaneChange. Speed buckets0-2m/s:50windows;2-5:2;5-10:48;>=10:26. Speed/scenario names do not establish0-120km/h coverage.

## Fixed-space results
Same EMA future target, same current persistence encoder and valid masks for every within-window comparison. Window macro means (equal3windows/route and21/scenario):
-1s all:JEPA0.266076 vs persistence0.178379; JEPA wins0/126windows.
-2s all:0.330785 vs0.263058; wins2/126.
-1s dynamic-class proxy:0.209798 vs0.116752; wins1/126.
-2s dynamic-class proxy:0.247225 vs0.173925; wins3/126.
-1s changed-known-class:0.268096 vs0.166510; wins4/126.
-2s changed-known-class:0.308819 vs0.227647; wins4/126.

All6scenario-family means and all4speed-bin means favor persistence.2-5m/s has only2windows; avoid strong subgroup claims. Group counts and raw route/window metrics are retained. Dynamic-class presence is not measured actor motion; same-class motion can be absent from change proxy.

## Multiple target permutations: important qualification
Five seeds1921-1925 generate bijective, no-self-route permutations WITHIN each scenario family. Every comparison intersects original/subgroup validity with shuffled target validity and compares true and shuffled targets on EXACTLY this same mask. This is a conditional sensitivity diagnostic, not a physically possible alternative future or causal oracle.

1s true macro range0.2382-0.2428 vs shuffled0.8386-0.8769.
2s true0.2576-0.2715 vs shuffled0.8367-0.8845.
True beats shuffled in115-122/126all-group windows depending on horizon/permutation. Dynamic/change proxies also consistently favor true at macro level. These masked values must not be directly compared to the different-mask unshuffled full-set means above.

Thus the prior narrow two-route shuffle failure does NOT support a blanket collapse diagnosis. The model retains scene-specific information on the broader TRAIN sample, while it still fails to improve upon current-state persistence as a future predictor under this objective.

## Input and mask audit
Every batch recomputes actor forecasts after removing future/trajectory fields and after poisoning future255/trajectory999; both h and predicted latents are bitwise identical. All126cached history pose IDs<=current and future pose IDs>current. Model parameter versions unchanged; no optimizer/EMA update. This verifies the code-level dependency and stored index boundary, NOT source generation causality, physical timestamps, or hidden future information in upstream OCC. No future target or validity mask is supplied to actor context.

Valid future-token fractions only37.11%/30.97% at1s/2s. Current-occupied support excludes14.13%/28.72% of all future known-occupied voxels and41.32%/52.08% of future vehicle/rider-class occupied voxels (voxel-weighted). Unknown0/255 is not certified free. The objective cannot currently certify prediction of new occupied regions outside current occupied support; mask-defined absent regions are unknown, not successful predictions. No collisions/TTC/free-space truth are inferred.

## Link to a fair Flow without JEPA comparison (not executed)
Question should be explicit: does the CURRENT predicted latent context help Flow, rather than whether JEPA can ever help driving?
1. Optional frozen sensitivity-only probe: exact same history/state/noise, full h+pred context vs history-only h (or repeated h to match token count). Never replace pred with true future EMA features. Changed token count and out-of-distribution ablation mean this cannot establish a trained no-JEPA baseline or safety superiority.
2. For a bounded trained comparison, keep identical core history encoder/state/Flow decoder initialization, masks, data order/noise, FM/physical objective, optimizer and update budget. Fusion arm retains auxiliary losses/predicted context; no-JEPA arm removes predicted future context AND auxiliary training losses. Report active parameters, FLOPs/latency/memory as well as equal updates; unused heads must not be counted as active capacity. Preserve per-arm initial evaluations and separate outputs.
3. Starting both from the old fusion-trained checkpoint tests removal during CONTINUATION, not a from-scratch architectural benefit. A genuine from-scratch claim requires matched shared initialization and sufficient training, not this audit or64steps.
4. Primary development comparisons remain initial/full kinematics, acceleration/jerk, forward/reverse proxies, candidate spread/near-zero and later closed-loop safety/efficiency, never ADE alone. Use route-balanced windows with declared reused development status and reserve an untouched validation split before model selection. No test access.

No Flow ablation/training was launched in this run. Current evidence justifies a controlled no-JEPA comparison; it does not justify silently removing JEPA from default or concluding its context is useless solely from latent MSE.

Artifacts: artifacts/jepa-route-balanced-train-audit/{report,selection,coverage-permutation-summary}.json. Script: scripts/audit_jepa_route_balanced_train.py.
