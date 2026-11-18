# Frozen rollout convergence and weighted gradient audit (2026-11-16)

## Scope
Zero optimizer updates; all checkpoints loaded read-only. No test data. Five frozen models (original 2048 and paired 1810/1811 baseline/rollout) evaluated on the same 16 repeatedly used DEVELOPMENT windows, four candidates per window. This is not an untouched validation set or driving acceptance. Initial speed coverage is seven windows below 2 m/s and nine at least 5 m/s; no 2-5 m/s windows in this selection.

Each window reuses exactly the same precomputed context and explicit noise tensor for 4/16/32 Euler flow steps. Flow integration time is not the physical 0.2 s trajectory sample interval. BF16 arithmetic retained. This diagnoses numerical discretization, not exact continuous-time convergence. No dropout (configured 0).

## Objective audit
The FM loss is MSE of predicted velocity against target-minus-noise at a target-informed interpolation. Original auxiliary physical terms use a single endpoint estimate from that bridge at t>=0.5, gated and normalized by gate probability. Original kinematic loss already includes t0; the previously omitted t0 was in reporting, not the training loss. A pure-noise multistep rollout therefore has a distribution mismatch with the original single-bridge physical objective.

Sampling normalizes heading and clamps negative speed to zero. The clamp can produce dead gradients below zero, but this observation does not establish it as the cause of the entire failure. Raw negative speed fractions and first raw speeds are recorded, not guessed from post-clamp zeros.

## Results
Across the five models, mean position change 16->32 is 0.30-0.38 m vs 1.11-1.28 m for 4->16. Smaller position increments do not imply better vehicle dynamics. Mean per-trajectory peak scalar jerk (m/s^3), 16->32:
- original2048: 93.13 -> 113.02
- 1810 baseline: 90.50 -> 107.69
- 1810 rollout: 88.78 -> 105.48
- 1811 baseline: 80.22 -> 99.17
- 1811 rollout: 87.77 -> 106.66

These are mean individual trajectory peaks, not global maxima. Per-window initial-speed bins, forward progress proxy, predicted/position-derived speed discrepancy, near-zero fraction and global first acceleration are retained in convergence.json. This run uses explicit per-window noise; its absolute metrics must not be directly substituted into the prior batched-noise report.

## Gradient replay
Two initial paired training batches (b=8, rollout first two examples), source2048 weights, paired seeds1810/1811, no optimizer or EMA updates. Replay computes the weighted original objectives and coefficient0.01 rollout kin/acc/jerk, not their unweighted norms. Named buffers unchanged. These are local BF16 diagnostic replays, not a claim of bitwise reproduction of every original training operation.

Initial grouped analysis: weighted FM norms2.70/5.81, grouped original other objectives11.11/6.91, rollout kin1.10/1.36, acc0.226/0.812, jerk2.22/2.40. Combined gradient10.36/14.27 implies clip-to1 factors0.0965/0.0701. FM vs original-other cosine flips from -0.365 to +0.593 between batches. Thus raw jerk norms cannot establish jerk dominance; component directions and original losses matter. Detailed original-objective decomposition is saved separately.

## Boundaries and interpretation
Five synthetic constant-speed/constant-acceleration/normalization/first-jump/left-axis tests pass. t0 velocity=(observed scalar speed,0), position=(0,0), relative heading=0 is still an explicit approximation; measured lateral velocity unavailable. Prehistory jerk remains undefined, not zero. Source OCC past-only causality uncertified. No collision/TTC/road compliance/closed-loop/safety claims; no automatic Astra training reward.

No defaults changed or training resumed. Candidate next single-variable experiment, only after review: align training regularizer integration steps to the 16-step evaluation rather than stacking more losses. First compare frozen 4-vs16 weighted gradient geometry/cost, then decide whether a bounded training comparison is warranted. This does not promise success and does not waive the initial-boundary or clamp issues.

Artifacts: artifacts/frozen-rollout-convergence-gradient/{convergence,gradient,detailed-gradient,completion}.json.

### Detailed decomposition outcome
Weighted original position gradient norms9.00/3.25 and original kinematics6.39/4.70 exceed or rival FM2.70/5.81. Position-vs-FM cosine is -0.47/+0.75. World auxiliary norms are small at these two sampled batches (latent~0.04, variance~0.01-0.02, semantic~0.08); this is not a dataset-wide claim or proof that they are unimportant. Original position and kinematics, rather than newly added jerk alone, explain most of the combined norm in the first batch. All results are local projections before global clipping; Adam moments and subsequent updates are not inferred from gradient norms alone. No process remains running beyond existing TensorBoard.
