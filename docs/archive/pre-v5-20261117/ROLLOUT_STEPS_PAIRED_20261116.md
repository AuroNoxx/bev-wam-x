# Paired rollout regularizer 4 vs 16 steps (2026-11-16)

## Scope and resource check
Authorized bounded experiment, isolated outputs; source2048 best checkpoint read-only, no test access, no default changes. Full-graph preflight ran one disposable AdamW update with b8 and16step rollout subbatch2, then discarded its model without saving. Loss0.227953, finite preclip norm11.2774, first optimizer-allocation step9.58s; allocated5.02/reserved5.10GiB. The actual training probe has higher observed memory and is authoritative for capacity.

Fresh paired arms: seeds1810/1811, each4step vs16step regularizer,64updates per arm (256 total). Same source parameters, new AdamW1e-5, weight decay0.01, batch8, BF16, clip1; coefficient0.01/every4updates/subbatch2 unchanged.128train windows; same data generators and isolated noise streams. First3FM RNG traces match within each pair; rollout asserts no buffer mutation. New latest checkpoints only; checkpoint32/64, no long training. The root config inherits legacy short_probe fields, but actual per-arm config updates steps64; max_steps64 and completion reports authoritative.

Primary evaluation identical fixed16Euler sampling,4candidates on16 repeatedly used DEVELOPMENT windows. World latent comparison is only the separately reported2window diagnostic. No untouched-validation or road safety claim.

## Results:4->16 training regularizer steps
seed1810:
- first residual5.627->5.891m/s, full1.700->1.893m/s: worse.
- mean per-trajectory peak accel14.385->12.776m/s^2, jerk89.458->69.331m/s^3: better.
- global first acceleration41.293 unchanged; global jerk206.463 unchanged.
- forward proxy17.610->17.184m; backward segments2.016->2.641; nearzero fraction0.1419->0.1513; spread4.041->4.177m.
seed1811:
- first residual6.685->6.347m/s, full2.021->1.997m/s: better.
- mean per-trajectory peak accel13.441->11.778m/s^2, jerk85.952->63.774m/s^3: better.
- global first accel39.504->35.805m/s^2; global jerk200.674->180.212m/s^3: lower but not acceptable/safe by implication.
- forward proxy19.676->21.132m; backward segments2.266->2.188; nearzero0.0719->0.0438; spread3.787->3.749m.

All64updates clipped in each arm. More flow steps reduce typical predicted jerk in both seeds, but initial/full consistency and backward motion fail paired non-regression in1810. Therefore NOT promoted to default. These mean individual peak statistics are distinct from global maxima.

JEPA latent MSE~0.2314-0.2322 vs persistence~0.1630-0.1631 on two windows remains worse. EMA changes mean absolute comparisons across models must be interpreted cautiously. No improvement to world-model acceptance demonstrated.

## Actual cost
Train+checkpoint segments:37.19/34.29s for1810(4/16),27.90/34.37s for1811. First4step run has cold/startup overhead: do not claim16steps faster. Warm1811 pair is about23%more training wall time. Evaluation~1.87-1.91s per arm. Measured training segment sum133.76s, evaluation sum7.54s, excluding outer imports/load/preflight.

Peak allocated4step5.84-5.85GiB,16step10.30GiB; reserved5.86vs10.31GiB. This includes diagnostics at first training step, not a guaranteed steady-state deployment/training requirement.

## Limits and next decision
Source OCC past-only causality uncertified; actor uses history only, future labels supervised targets only. t0forward scalar velocity assumption and absent prehistory jerk unchanged. No CARLA learned-policy closed loop, collision/TTC, safety certificate or automatic Astra RL. ADE not used as overall acceptance.

This single-variable test demonstrates partial comfort-proxy benefit, not a complete fix. Do not expand into long training or stack new penalties. Return paired tradeoffs and worst unchanged boundary case to coach/parent for one next falsifiable hypothesis. All four runs have ended, no further training launched.

Artifacts: artifacts/teacher-rollout-steps-paired/{comparison,summary,config,selection}.json plus per-arm reports/checkpoints; preflight in artifacts/rollout16-full-graph-preflight/report.json. Code: scripts/train_rollout_steps_paired.py and summarize_rollout_steps.py.
