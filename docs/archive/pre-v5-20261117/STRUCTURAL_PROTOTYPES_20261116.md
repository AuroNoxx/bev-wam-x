# Structural isolated prototypes, 2026-11-16

## Authorized bounded implementation
The latest user authorization permits the Flow/world prototype work and bounded training, superseding the earlier classical-only/no-automatic-training route for this task only. Other project safety/checkpoint requirements remain. No old weights or tests changed.

## Implemented and checked
- Genuine two-channel velocity FM, with known-current-speed longitudinal prior and zero lateral t0 assumption. Trapezoidal integration produces XY; norm/direction produce speed/motion direction. Motion direction is not independently certified body heading.
- Independent world-supervision mask arm: future positive labels 1..7 supervise future targets regardless of current occupied support. Labels 0/255 remain unknown, never free. This is positive-support supervision, NOT a calibrated occupancy/free predictor.
- Seven isolated unit tests pass: integration/roundtrip, constant velocity, axes/zero, future poison isolation, positive/new-positive mask, actor invariant under target-mask change, finite backward.
- 126-window audit: nominal frame ordering passes; source generation causality remains uncertified. Old current-occupied support drops 46.51% of future dynamic positives and 21.37% of all future positives.

## Completed 256-step comparison
Two seeds (2010/2011), baseline/velocity-only/world-only, six independent runs. Width128, two decoder layers, four heads, about0.967M active parameters; batch8, BF16, LR1e-4, warmup64, cosine horizon256 to1e-5, clip1. 84 train windows and42 same-route internal diagnostic windows; not independent validation. Shared backbone initialization and prefix action-head initialization, but two-dimensional and five-dimensional FM targets differ. Physical penalties become partly identically satisfied in velocity mode; total loss is not comparable as a quality score. First-three FM RNG traces match; full data/gradient equivalence is not claimed.

Baseline -> velocity, seeds2010/2011:
- Full kinematic residual8.69/7.94 -> approximately4e-6m/s, by construction.
- Mean per-candidate peak scalar acceleration53.15/50.10 ->28.72/29.64m/s2.
- Mean peak scalar jerk460.20/428.17 ->243.53/250.78m/s3. Still physically poor.
- Mean position label-fit6.94/6.82 ->7.04/6.95m (diagnostic, not driving score).
- Mean final-forward proxy50.96/50.33 ->50.96/44.74m; not route progress.
- No velocity candidate ends below0.5m/s; parking must be inspected separately.
World-only prediction/persistence latent MSE ratio1.982->1.968 and1.560->1.494, still worse than persistence. Dynamic proxy gains are inconsistent. Ratios use each model's own EMA space on common future-positive support; raw latent errors across separately trained target spaces are not directly comparable.

Collision, TTC, road validity, takeover, CARLA performance, automatic cloud-LLM training: unavailable/not executed. No safety claim or default replacement.

## Next bounded test
Fresh six-arm 512-step runs, same seeds/data/architecture, shared64-step warmup and512-step cosine schedule. Not strict continuation of256 schedule. Counterbalanced arm order; no combined treatment yet. Budget15min total with per-arm4min ceiling. Rolling latest/best only with VerifiedCheckpointIO and StatefulDataLoader. Best means position label-fit monitor, not best driver. Assess parking/low-speed strata, motion/jerk, diversity, forward proxy and common-world persistence comparison before scaling.

## Cleanup
Exact proposed61 old checkpoint files total106.971GiB are listed in artifacts/implementation-plan-v2/proposed_weight_cleanup_exact.json. No deletion performed. Nine major snapshots,32 unreviewed small files, all new prototype weights and all documentation/logs/configuration are retained. No external backup verified; irreversible deletion requires the owner confirmation handled by the parent.

## Approved cleanup executed
The owner explicitly confirmed permanent removal of the exact61-file proposal. On execution, paths, sizes, nanosecond modification times, keep/delete non-overlap, retained file existence and active process commands were checked. Each target was rechecked and exclusively opened before native PowerShell nonrecursive removal.61 files/114,859,632,163bytes removed, zero errors;9 major retained checkpoints remained. Actual free space change114,859,724,800bytes includes concurrent small file activity. Full audit: artifacts/implementation-plan-v2/weight_cleanup_execution.json. This does not authorize further deletion.

## Missing stop coverage and supplemental diagnostic
The original42-window diagnostic has zero label-end-stopped cases and only one initially<0.5m/s case. Therefore parking results are unknown, not failed or passed. Train-only scan, excluding all84 training+42diagnostic windows:393 initially/ending stopped,260 moving-to-stopped,277 slowing>3m/s but still moving. Frozen before model evaluation, choose eight routes per class/one middle frame per route,24 windows total. These are unseen windows, not route-independent validation. Historical labels are not unique safe targets. This supplemental evaluation runs only after all six512-step arms finish; no training updates and no test access.

## 512-step outcome and frozen stop diagnosis
All six arms completed, total484.64s including loop overhead; currently no training/evaluation process remains. Per-run about71-82s, peak allocated below0.66GiB. Same42 internal windows: baseline vs velocity label-position fit6.253->6.551m(seed2010),6.210->6.648m(seed2011); final displacement fit and speed fit improve, but consistency is structural, not evidence of safe control. Scalar acceleration/jerk still excessively large. World-only relative latent error improves overall(1.439->1.302,1.322->1.184) but remains worse than persistence, and dynamic subset is mixed(1.368->1.330,1.476->1.523).

Frozen24-window train-split supplement completed with zero updates:8 windows each initially-and-finally-stopped, moving-to-stopped, slowing-but-moving; four correlated samples per window. On initial/end-stopped group, baseline mean final speed6.35/6.25m/s, velocity3.38/3.55; velocity still moves12.76/13.40m and zero of32 candidates ends<0.5m/s per seed. On moving-to-stop group, baseline final speed6.37/6.91, velocity7.83/7.93 despite label mean final0.051m/s and initial6.37m/s. This is evidence of poor label-conditioned stopping behavior, not a safety/optimal-action label. All versions need work. Vector-acceleration and vector-jerk metrics also recorded including t0 velocity assumption; jerk before first available interval is not invented.

No combined treatment, model-size increase, default replacement or automatic longer training. Next priority: audit the84-window fitting distribution and stop-conditioned target/noise contraction before adding more objectives. Positive-world supervision must additionally be evaluated in a shared semantic target space, not only changing EMA latent spaces; calibrated free-space and closed-loop safety remain unsupported.
