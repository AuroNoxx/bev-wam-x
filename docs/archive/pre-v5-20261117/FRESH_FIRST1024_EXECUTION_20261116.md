# Fresh JEPA versus no-JEPA: first 1024-step milestone

Execution: scripts/train_fresh_jepa_first1024.py. Output: artifacts/fresh-jepa-paired-first1024. No previous checkpoint loaded. Seeds 1960 and 1961, shared exact fresh state per seed; mode order counterbalanced. Batch 8, 5296 train windows, original FM and physical losses. JEPA adds original 0.1 latent/variance/semantic losses. No-JEPA removes future context and freezes future heads. This is equal update/data budget, not equal FLOPs or active parameter count.

Both arms use 128 linear warmup to 1e-4, cosine toward 1e-5 at hypothetical step4096. CURRENT AUTHORIZATION hard stops each arm at 1024; this script never continues to4096. AdamW weight decay0.01, gradient clip1, BF16. Save independent verified full checkpoints every256 inside fit, preserving loader/optimizer/scheduler/RNG and all older artifacts. Per-arm10-minute guard, overall40-minute guard checked between arms. No automatic retry on failure.

27 reused development windows across9routes, four candidates each,16Euler steps, identical sampling stream. Report initial-boundary and full kinematics, acceleration/jerk, forward proxy/backward segments, predicted/position-derived speed disagreement, near-zero ratio and diversity. None is a closed-loop driving verdict. Collision,TTC,route progress remain unknown. No-JEPA future metrics are N/A. Fresh JEPA world prediction compares with persistence in the SAME contemporaneous EMA space; absolute latent MSE across runs is not a common quality scale.

Actor future-poison independence is checked in evaluation. Source OCC generation causality remains unverified, unknown/unfilled cells are not free-space truth, nominal history timing and support mask limitations remain. No official test data, CARLA, external LLM or RL used. First3 data/flow RNG checks assert paired equality; initial state exact equality is asserted over all tensors.

Initial actual seed1960 JEPA: step1 loss23.076 after8.90s startup; step32 loss4.914 at16.05s, approximately0.2305s/update over steps2–32. Peak allocated4.233/reserved4.549GiB. These are startup observations, not final quality. Read per-arm progress/report and final summary for completion.

## Completed milestone
All four arms reached1024 and stopped. Final artifact: final_review.json. Total training wall838.0s; launch-to-final including evaluation device-fix recovery1086.9s. Four full verified checkpoints per arm (steps256/512/768/1024),16 total. No updates beyond1024, no test/CARLA/LLM execution.

|Seed|Mode|Train s|Last training loss|First residual m/s|Full residual m/s|Mean trajectory peak jerk m/s3|Forward proxy m|
|---|---|---:|---:|---:|---:|---:|---:|
|1960|JEPA|258.74|0.35989|25.705|14.154|222.374|39.956|
|1960|No JEPA|162.91|0.43755|71.717|15.447|251.807|12.955|
|1961|No JEPA|162.26|0.25930|14.771|7.505|202.636|37.058|
|1961|JEPA|254.09|0.35524|40.931|7.447|222.558|53.414|

Losses have different terms and are not a cross-arm ranking. Forward proxy is not route progress, safety or reward. These are predictions not actual vehicle motion. Global first scalar acceleration maxima JEPA60.22/90.85m/s2 and no-JEPA53.16/68.20, confirming no executability claim. JEPA own-EMA world MSE0.328929 versus persistence0.305882 (1960),0.446365 versus0.443680 (1961): neither beats persistence. No-JEPA world head N/A.

One software exception occurred AFTER first arm completed/saved: evaluation reference remained CPU after Lightning teardown while inputs were GPU. Preserved original log/script, corrected in v2, loaded completed1024 only for evaluation, and trained only the other3arms. First failed process lingered during shutdown; resource guard rejected recovery twice before own failed processes were explicitly cleaned. No other process was stopped. First arm in-memory gradients/RNG traces lost before reporting; initialization equality assertions and saved state/logs remain, but do NOT claim first3 RNG trace verification forseed1960. Seed1961 paired traces and first3 batch hashes compare equal. Subsequent training metadata is saved before evaluation.

Peak allocated/reserved GiB:1960 JEPA4.233/4.549,no3.339/3.795;1961 no4.513/5.086,JEPA5.412/5.828. Later serial arms retain roughly1.17GiB extra allocation; these are observed process peaks, not clean isolated architecture benchmarks. No OOM or nonfinite training detected.

Review: results are still early (1.55train epochs), seed-sensitive and physically poor; no default architecture change or blind4096continuation. Before any extension, choose a bounded common-objective/trajectory validation review using existing checkpoints and then explicitly approve continuation. Do not optimize only ADE or total loss, and do not endlessly replace the actual training with tiny probes. Collision/free-space/TTC/real driving remain unknown.
