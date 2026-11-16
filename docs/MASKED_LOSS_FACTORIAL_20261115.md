# Physical vs world auxiliary factorial diagnostic

Four identical train anchors, short4-frame masked actor,967710 non-EMA parameters,width128/layers2/heads4,batch4,512steps,lr5e-4. Same initialization seed1701 and bridge RNG seed444. BF16 eager/default attention; deterministic algorithms were not enforced, so bitwise repeatability is NOT claimed. Three inference seeds8181/8182/8183 are not three independent training seeds. All models retain the same fusion architecture; world denotes training loss presence, not removal of the prediction branch.

Physical package: position, kinematics, unit heading, nonnegative speed, existing configured weights. World package: latent JEPA, cross-example variance and semantic losses, weight0.1 each. Package-level ablation does not isolate each component or gradient conflict.

|Loss|Mean ADE across sampling seeds (m)|ADE range (m)|Mean kinematic residual (m/s)|Loop incl periodic eval (s)|
|---|---|---|---|---|
|fm|2.962|2.867–3.115|14.940|16.33|
|fm_physics|5.827|5.725–5.991|5.402|16.08|
|fm_world|2.526|2.331–2.867|6.294|32.14|
|fm_physics_world|5.800|5.725–5.888|5.761|32.09|

## Interpretation
In this tiny run physical-package addition increases ADE but reduces kinematic residual relative to pureFM. Removing it is therefore not a free quality improvement. FM+world improves both metrics relative to pureFM on this seed, so these results do not support blaming JEPA for the earlier6m trajectory error. Full joint is close to physics-only in ADE. Next diagnosis should reduce/ramp physical weights, separate position from velocity consistency, and check multiple training seeds before a default change. A derivative-based physical term evaluated on noisy one-step bridge endpoints may create an optimization tradeoff; this remains a hypothesis, not a proven root cause.

All residuals remain much higher than these targets'0.005–0.109m/s. No configuration meets a declared driving-quality or safety gate. ADE only measures resemblance to recorded trajectories, not optimality. No collision/closed-loop tests or held-out evaluation occurred.64 sampler steps worsened ADE in the tested seed for all4 configurations, so simply increasing integration steps is not supported.

Runtime96.6s total measured optimizer+periodic-eval sections, excluding data loading/model initialization/final evaluation. Peak reserved below0.5GiB on RX7900XT; not Orin timing. Four runs exited, no weights saved;32 existing checkpoint hashes unchanged. Source causal provenance still unverified; see BENCH2DRIVE_OCC_PROVENANCE_LIMITS_20261115.md. No formal/long training, no test set use.
