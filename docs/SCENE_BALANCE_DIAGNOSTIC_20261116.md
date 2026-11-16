# Route-isolated behavior sampling diagnostic

Current authorization: bounded Lightning comparison of representative versus behavior-balanced sampling, not full training or policy acceptance. No official test used.

## Split and limits
- All24 existing supplemental diagnostic windows remain excluded from training. Their14 scenario/Town/Route identifiers, including repeated weather recordings, are wholly excluded.
- Train pool:2555 windows on27 remaining route groups. Old42 diagnostic windows also excluded.
- Held diagnostic:24 existing stop/slowing windows plus14 normal-moving examples, one per held route group,38 total. Previously inspected development data, not final test.
- Strict isolation removes all HardBreak routes and all initially-and-finally-stopped training examples. This is a known scenario/behavior coverage gap, not silently repaired by leaking held routes. Moving-to-stop31 unique windows and slowing186 remain.

## Single intervention
Both arms use identical fresh velocity2/current-support-world model, same seed initialization, optimizer/loss/BF16/batch8/clip1/512 updates/64warmup/512 cosine schedule. No combined world-mask change. Both use4096 samples with replacement, generated from shared uniform variates and different distribution CDFs; per-step FM noise RNG matched. Representative baseline balances scenario then route, with natural behavior frequency within these weights. Balanced arm only reweights behavior to20%moving-stop,20%slowing,60%other. Stationary-stop has zero training support.

Reweighting changes route/scenario marginals and repeat counts; these are reported, not hidden. It does not create independent data. Seed2020 actual moving-stop draws31 versus783; seed2021 37 versus807. This is a sampling-distribution ablation, not architecture or total-loss competition.

## Budget and outputs
Two seeds2020/2021, counterbalanced two-arm order, each512 updates. About0.967M active parameters. Per-arm4min and overall15min guard; finite-loss/gradient checks. Latest/best only, VerifiedCheckpointIO and StatefulDataLoader, localTensorBoard. Best is a label-fit diagnostic, not best driving.

Review all38-window strata, including normal progress, vector acceleration/jerk and stopping frequency. Missing collision/free/TTC/closed-loop metrics remainunknown. Historical labels are a recorded action, not uniquely correct driving. Do not adopt the new sampling default solely on stop fit, especially with sparse31-example repetition and noHardBreak training.

## Completed results
All four512-update arms completed normally in323.08s total. Latest/best verified; peak allocated<=0.651GiB. No training/evaluation process remains. First-three FM RNG traces matched. The sole intervention was sampling distribution; no action/world loss or architecture change.

Representative -> balanced, seeds2020/2021:
- Moving-to-stop group mean final speed7.152->6.228 and7.712->5.949m/s; none of32 candidates per arm actually ends below0.5m/s. This is a recorded-outcome diagnostic, not safety acceptance.
- That group's mean peak vector acceleration29.18->37.56 and27.94->37.97m/s2; jerk248.55->315.98 and238.07->320.30m/s3. Both seeds worsen dynamics.
- Normal-moving group forward proxy44.76->43.52 and48.73->41.83m (recorded mean41.19m), final speed9.38->8.43 and10.12->8.18m/s. No end-stop collapse, but lower progress is not automatically better. Label-fit is mixed4.64->5.50 and6.00->5.01m.
- Normal-moving vector jerk242.32->304.21 and242.93->316.78m/s3, both worse.
- Candidate spread rises4.51->6.00 and4.42->5.19m. More spread is not proven useful diversity and may reflect retained noise.
- Gradient clip hit98.8-99.4% across all arms, mean norms4.3-4.8. This does not isolate clipping as the cause and does not justify changing it.

Conclusion: aggressive20/20/60 reweighting changes sampled action speed but does not solve stopping and harms dynamics on both seeds. Do not adopt as default or extend this run automatically. Sparse31 unique moving-stop windows, scenario shift, repeated developmental evaluation and zero stationary-stop train coverage limit inference. Next useful diagnosis is train-versus-held stop behavior and pure-noise contraction at fixed weights, not another simultaneous architectural/loss change. World predictions should separately receive shared semantic-target evaluation rather than only EMA-latent scores.
