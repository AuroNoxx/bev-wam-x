# FM missing-suffix causal design and isolated tests, 2026-11-16

No training/optimizer/checkpoint changes. scripts/test_fm_missing_suffix_contract.py; artifacts/fm-prefix-causality-tests-v1/report.json. Tests use a small standalone attention model, NOT production-model certification.

## Why simple replacement is insufficient
The current joint_occ_model.velocity uses self.transformer(q,context) without a temporal tgt_mask, so future trajectory tokens can influence prefix tokens. Zeroing loss on the changed-command suffix while retaining its clean target in the bridge leaks that target into valid-prefix learning. Replacing that suffix with zeros or Gaussian noise but keeping bidirectional attention also exposes a different corruption distribution; the model can infer cutoff information indirectly. Never pass valid-length, switch-time or label mask to actor/context, padding masks, online sampling or noise-conditioning inputs.

## Conservative complete-window baseline
Train ordinary full-horizon FM only on windows without an exogenous ego-command change in the supervised horizon. Same full25-token actor, no validity input. Current hold_v3 retains480/514 per split, loses all34 current-hold windows; document selection bias and lack of hold learning. This is a baseline, not a sufficient final data solution, and it does not establish that future exogenous commands are inferable.

## Prefix-capable temporal-causal FM option
All samples, full or partial, use the SAME fixed lower-triangular physical-time self-attention topology across25 future trajectory tokens. This architectural mask is independent of sample validity, contains no cutoff, and applies identically at training/inference. Cross-attention reads only historical/context-derived features. Per-token normalization only; no future-token global pooling, time-axis normalization, bidirectional convolution, spline/global coefficient reconstruction, or extra future-derived context.

On valid tokens, standard conditional FM bridge uses clean target and source noise. On unsupported suffix, remove missing/NaN target before arithmetic and leave finite source noise; no target velocity/loss assigned there. The network receives a full-shaped x_t,flow-time,current context only. Valid prefix outputs must be mathematically independent of every suffix input, so suffix corruption cannot reveal cutoff to valid predictions. Loss mask is a supervision operation after forward, with original total denominator preserved. At inference there is no data validity mask: integrate all25 tokens from standard source noise under the fixed causal vector field.

Physical decoder must also be prefix-causal. Cumulative integration of local speed/heading controls qualifies. The recent globally fitted6-knot coefficients do NOT automatically qualify: early values may depend on later/global coefficients, and underdetermined coefficients have no unique target. Do not silently combine those coefficients with this proposed FM. This design is for temporally indexed observable targets with a separately audited decoder; identifying low-speed yaw/curvature remains its own issue.

This changes the model's conditional factorization and may affect multimodal coherence/quality. It is not a proof that the new flow learns the desired complete-joint trajectory distribution. It does provide a testable absence of future-suffix paths into supervised-prefix predictions.

## Passed isolated tests (CPU, no optimizer)
- Different suffix input values (large finite pollution), identical prefix/context/noise/time: valid-prefix outputs EXACTLY equal.
- Valid-prefix-loss parameter gradients EXACTLY equal; gradients to suffix inputs exactly zero.
- Missing NaN clean labels removed before bridge arithmetic: finite inputs and targets, exactly same valid computation.
- Fixed cumulative decoder prefix unchanged under suffix pollution.
- Negative controls: bidirectional model prefix changes max1.7954; causal attention followed by global temporal pooling changes max1.1377. Tests detect real leak paths rather than vacuously passing.
- Production code, GPU precision, fused attention, EMA/buffer updates and multi-Euler-step sampling invariance still need equivalent tests before any training.

## Pre-registered comparison design (not executed)
1. First test production-isolated causal variant, including4/16-step sampled prefix invariance, finite gradients and all cross-token routes, frozen checkpoint/source hashes and actual timing. No optimizer.
2. Architectural cost: bidirectional vs fixed-causal FM on identical480complete windows, same initialized parameters/data/noise/optimizer/budget/decoded metrics. Only topology changes; no partial labels yet.
3. Partial-supervision value: hold causal architecture fixed; common draws over514windows, baseline contributes no loss from34partial windows, treatment contributes their supported prefixes. Same fixed slots/denominator, count active labels and effective gradient exposure. Additional labels are the intended intervention; do not claim equal information. Do not use dev to retune weights.
4. Evaluate unchanged480, command-changing34 valid prefixes, actual current-hold behavior, ordinary deceleration/start/steady, noise diversity and dynamics separately. Collision/TTC unknown offline. No closedloop based on aggregate score.

## Distinguish uncertainty types
Recorded external ego permission/cruise change is a specific intervention invalidating the factual suffix as a label for held current intent. This does NOT justify censoring unexpected lead motion, occlusion, unusual trajectories, multiple safe choices or high-error samples. Those are partial-observability/multimodality to model and evaluate, not discard. Generic future-world prediction may remain marginal over unknown future commands; do not reflexively erase all JEPA targets. No future switch time or hidden teacher state is an actor input.
