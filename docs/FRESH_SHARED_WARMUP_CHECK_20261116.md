# Shared warmup startup check (2026-11-16)

One fresh matched seed1950, same full5296train data loader and per-step FM RNG as the earlier8step no-warmup smoke. JEPA/no-JEPA each32updates, DISCARDED without checkpoint. Common128step linear warmup toward1e-4, observed ONLY through step32(lr2.5e-5); no claim about reaching the eventual peak LR. Initial shared tensors and EMA copy identical, paired first32batch-state hashes match. First-step losses also exactly match prior smoke.

Direct1e-4 vs warmup second-step losses:JEPA1166.14->5.496, noJEPA679.28->6.468. First8warm steps avoid the prior giant startup spike. All32steps remain finite, but all gradients still require clip1; warmup does not remove large raw gradients. No full-convergence or driving claim.

Independent8fixed route-spaced TRAIN diagnostic windows,4fixed-noise16step candidates each:
-JEPA full kinematic residual370.57->44.25m/s, first291.49->111.62; mean per-trajectory peak jerk3317.54->1693.13m/s^3; forward proxy28.87->8.35m, backward segments12.156->12.094.
-noJEPA full345.62->40.83, first369.19->153.84; mean peak jerk4003.48->1311.60; forward proxy-37.21->-29.56m, backward segments12.50->12.59.

These are very immature random-start models and both remain physically unusable. Lower loss/residual does not establish good driving, and the noJEPA negative forward displacement is not hidden by averaging loss. Candidate spread shrinks strongly but cannot be labeled desirable or collapsed from this smoke alone. Collision/TTC and closed-loop scores unknown. ADE not used as overall acceptance.

Warmed data+update timing remains~0.228s JEPA,~0.134s noJEPA. Allocated peak3.57/3.10GiB (includes diagnostic setup); not a promised Lightning peak. Both short test copies ended, no long training or CARLA run started. No test/old checkpoint writes.

Recommendation: the evidence supports shared warmup as a more controlled STARTUP than abrupt1e-4 at this seed, not another automatic expansion. Use the already proposed single formal bounded run with checkpoints and milestones rather than repeatedly changing losses. The formal run must monitor the remaining warmup/peak-LR transition and enforce finite/resume/multi-metric gates. Plan remains unlaunched pending parent/user direction.

Artifacts: artifacts/fresh-jepa-warmup-preflight/{report,stability-summary}.json. Script: scripts/preflight_fresh_jepa_warmup.py.
