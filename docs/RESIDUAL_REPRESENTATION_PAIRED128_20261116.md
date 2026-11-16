# Experimental time-scaled residual representation: bounded paired128

Files: scripts/experimental_residual_representation.py and scripts/probe_residual_representation.py. Output artifacts/residual-representation-paired128. Original code/defaults/checkpoints unchanged. Four fresh128-update arms complete,190.36s total. No continuation,test,CARLA or external LLM.

## Transform
Only x/y change: flow_xy=(physical_xy-[observed_v0*t,0])/scale(t). Decoder is exact inverse, then existing heading normalization/speed clamp. v0 is current state speed, not future speed; forward ego FLU. sin/cos/speed channels unchanged. This is numerical conditioning and an observed-state prior, not a hard dynamics controller or guaranteed t0 acceleration bound.

Scale fitted only from84 optimization train windows(two slots per42routes),not42other-slot diagnostics. q90 absolute residual at each future time,positivefloor[0.5,0.25]m,monotonic nondecreasing. Scales x/y at.2s=[.5,.25],1s=[2.989,.692],5s=[31.872,5.571]m. These floors are explicit design choices, not optimized safety limits. Longer-horizon input labels never enter online prior; fixed fitting statistics are training artifacts.

Tests: encode/decode maxabs5.96e-8 normalized units, future-label poisoning exact invariance with supplied target heldfixed, zero residual decodes current-v0 constant velocity prior, positive y is left,first timestamp.2s. Physical boundary at t0 itself is a known prior, not output point. Independent frame/source causality not certified.

## Experiment
DenseJEPA155M only,not a secondJEPA/noJEPA comparison. Seeds1980/1981,sharedexactfreshinitialweights,paireddata/FM RNG first3 checks pass,armordercounterbalanced,84train,batch8,AdamW,originalFM+physics+JEPAaux,128warmup to1e-4. Hardstop128 means no post-warmup stability claim. All4arms clip every step. Four independent full128 checkpoints saved. Same normalized random draws acrossarms imply different physicalnoise distributions by design; equal updates and data, not identical physicalperturbation.42diagnosticwindows are different filenames but same routes and can overlap in time,not independentvalidation.

|Seed|Representation|First residual m/s|Full residual m/s|Mean trajectory peak acceleration m/s2|Mean trajectory peak jerk m/s3|Forward proxy m|
|---|---|---:|---:|---:|---:|---:|
|1980|original|178.244|43.101|78.083|666.202|36.935|
|1980|residual|4.929|16.894|99.049|847.499|25.004|
|1981|original|446.421|102.259|54.591|428.450|−88.132|
|1981|residual|9.098|30.819|90.724|785.110|−11.923|

Residual improves spatial/velocity consistency relative to very immature original128 models but worsens peak acceleration andjerk in both seeds. Negative progress remains inseed1981. Thus no overall improvement claim,no default switch,no driving readiness. All worldMSE~1.78–1.83 vs contemporaneouspersistence~.25,also immature and poor. Absolute ownEMAerrors across independently learned bases are not directly equivalent.

## Interpretation / next decision
Evidence supports the hypothesis that large absolute-position/noise scale causes severe initial-position errors, and representation conditioning materially affects them. It does NOT prove this representation solves dynamics or that learning has converged. Velocity/heading still independently generated,so better position boundaries do not enforce speed/acceleration/jerk. Do not fix this by arbitrary clamp/smooth unless consequences are evaluated.

Before more training, inspect whether acceleration/jerk degradation is dominated by the unchanged speed head versus position-derived motion and whether coupled velocity integration would remove genuine inconsistency or merely hide it by construction. Existing integrated-decoder negative results must be considered. Any extension should be a single bounded plan with the same multidimensional gates,not endless tiny unpaired probes. Original1024teacher weights remain available; no automatic4096 continuation.
