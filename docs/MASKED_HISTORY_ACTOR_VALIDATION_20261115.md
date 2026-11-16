# Masked temporal actor, isolated implementation

New scripts: `masked_history_joint.py`, `validate_masked_history_joint.py`. No baseline code or checkpoint changes.

Actor consumes history validity, age in seconds and geometry validity. A missing frame skips recurrent state update and does not change last-valid observation time. The learned timing embedding includes age and elapsed time since last valid observation. For the current data these are nominal 0.1-second frame-index intervals, not measured timestamps. Current frame must be valid. Source 0, 255 and geometry-outside cells are all conservatively encoded as unknown, not verified free space. Future labels remain outside actor calls and only enter auxiliary losses.

This is a small dense integration diagnostic: width 128, 2 decoder layers, 4 heads, identical 4 training anchors (including recording starts), same initialization / optimizer random seed / batch 4 / 256 updates / learning rate 1e-4. Compare short 4 frames against 14 long-short frames; identical update budget is not identical FLOPs or wall time. No validation or test samples, no checkpoints. This does not establish the best teacher capacity or driving safety.

The legacy `forecast(history,state)` method is explicitly rejected because it cannot carry the masked contract. Use the new `forecast_batch` / `context` / forward / sample / overridden auxiliary_loss paths only; old representation_report is not an appropriate diagnostic for this class.

## Final operational domain vs present evidence
User target is 0–120 km/h. Use current public data coverage for foundational validation now; lack of high-speed mini samples is not a blocker. Later CARLA directed collection must cover 0–20, 20–40, 40–60, 60–80, 80–100, 100–120 km/h separately (with consistent boundary assignment), plus stopped vehicles, cut-ins, braking, overtakes and safe-following. Report current counts and held-out closed-loop outcomes per bin, never treat unobserved bins as passed. Final exact acceptance thresholds require calibrated scenario definitions; no single expert-path distance serves as safety acceptance.

Source-level causality is still uncertified. Temporal masks prevent this implementation from consuming explicit future labels but cannot undo future-fused source OCC. Simulator-derived input also requires production sensor uncertainty/domain validation. Early cloud coach remains planned, not connected in this probe.

## Measured outcome

All six contract tests passed. Non-EMA parameters: 967,710.

|History|Train meanADE4 before → after (m)|Train latent MSE after|256 update seconds|Peak reserved GiB|
|---|---|---|---|---|
|short4|52.989 → 6.453|0.6696|17.453|0.471|
|long14|52.963 → 6.641|0.6729|47.181|1.207|

Long history does NOT show a benefit in this tiny run: 6.641 m versus 6.453 m, with 2.70x optimizer-loop time. This is one seed and four anchors, not evidence long history is generally worse; neither is an overfit or deployment acceptance. Loss and gradients remained finite. All 32 prior weight hashes matched. No baseline/test/checkpoint changes.
