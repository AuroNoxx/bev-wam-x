# Independent-seed and route-held-out physics-weight diagnostic

Pre-registered diagnostic comparison: existing kinematic coefficient0.1 versus candidate0.01; all remaining physical and world losses unchanged. Small masked actor width128/decoder2/heads4,967710 non-EMA parameters. Three initialization/training seeds1701/1702/1703; paired same minibatch order and bridge RNG within seed.384steps,batch8,lr5e-4 per run. Two inference seeds,4candidates,16Euler steps. BF16 eager; bitwise determinism not imposed.

Fixed selection saved before fitting:54 training windows(23stationary/20decelerating/11other),10route-held-out training-partition windows(3stationary/5decelerating/2other). One route per family selected by fixed SHA256 ranking, never by observed model results. Route overlap between new train and diagnostic heldout is empty. Categories are target-speed-defined for reporting only: stationary current absolute speed<0.5m/s and max future absolute speed<0.5m/s; deceleration current minus final speed>=1m/s; rest other.

Official validation and test remain unread. The heldout set is unseen to these freshly initialized models, but prior baseline studies may have used these train-partition files, so it is NOT a pristine final benchmark. No tuning on official holdout. Only this fixed two-candidate comparison; do not repeatedly select models against this diagnostic set without acknowledging adaptive reuse.

Report ADE, kinematic residual, signed predicted forward displacement and signed final-speed error per category. Forward displacement is useful for the stationary subset, not a universal error. Final-speed error positive means faster than recorded final speed. Inference seeds double records, not independent windows. No closed-loop/collision/free-space claims. Source temporal causality still unverified; cache time intervals nominal. No formal default changes, checkpoint creation or deletion.

Resource guard checks competing Python tasks before every run. Aggregate budget600seconds; intermediate result JSON saved after each configuration. Training timing includes batch transfer but excludes initialization/evaluation and cannot be mapped to Orin. Results follow when all runs finish.

Additional split audit: extracting `Route[0-9]+` identifiers (ignoring weather/date recording suffix) also gives empty train/heldout intersection, not merely different recording names.

## Completed results
All6runs completed.

|Weight|Heldout ADE averaged over training seeds(m)|Heldout kinematic residual(m/s)|Stationary ADE(m)|Deceleration ADE(m)|
|---|---|---|---|---|
|0.1|12.067|9.980|9.072|12.989|
|0.01|11.265|11.799|7.574|13.024|

Paired heldout ADE original→candidate: seed1701 11.478→8.727m; seed1702 14.481→17.493m; seed1703 10.243→7.576m. Candidate wins2/3 ADE comparisons, loses1/3 and consistency worsens in2/3. In seed1702 candidate deceleration ADE24.182m versus14.576m; in seed1703 candidate residual15.991m/s versus8.068m/s. No robust dominance, no default change. Stationary drift remains serious, including large forward travel in seed1702. Low training bridge loss does not guarantee good sampled trajectories.

Training throughput57.15–69.49samples/s including batch transfers; first run53.75s, subsequent44.21–44.64s each,384updates. First-run warmup affects comparison; weight change is not proven to speed training. Peak reserved0.715–0.857GiB. No weights saved; all processes exited; official validation/test untouched. These runs are short and underfit, not a final assessment of either coefficient. Recommend next diagnose instability and sampled-trajectory consistency on train before escalation to a large teacher or formal validation.
