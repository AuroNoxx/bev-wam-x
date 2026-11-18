# Fixed-context decoder continuation (2026-11-16)

Scope: ten train-only contexts per seed, diagnostic action MSE, not a driving model or a new main training. Five empirical means of identical actor groups and five fixed deceleration windows. No dev/test/CARLA used.

## Restore contract
Both original step-256 checkpoints contain optimizer, constant scheduler (LR 1e-4), Torch/CUDA RNG and StatefulDataLoader cursor. Added isolated load hooks. Reconstructed contexts from original frozen encoder: frozen parameters exactly equal; future poisoning exact; all stored 256-step velocity outputs reproduced with maximum absolute difference 0.0. Original contexts were not separately hash-recorded, so this is not original context-byte hash certification. First restored batch and all optimizer state tensors matched before updates. Same 256-batch epoch naturally repeats, all batches contain the same ten items. FP32 + MATH, AdamW, clip 1, no architecture/data/objective change relative to the tiny unit test.

Hard limits: total step 2048 and 120 seconds per seed. Evaluated every 256. Early-stop all ten component velocity RMSE <= 0.1 m/s plus correct temporal direction, or three consecutive checkpoints with <1% MSE gain and no worst-item/direction improvement. Neither stopping condition fired; both stopped at 2048.

## Results
- Seed 7201: MSE 0.00870029 at 256 -> 0.000185334 at 2048. Direction correct for all 10 by 768. Final worst RMSE 0.2355 m/s, 0/10 pass strict 0.1 threshold; all close to threshold except a deceleration item. Best monitored MSE was 0.000102724 at 1536, retained separately; final worsened. Training monitor wall 16.38 s; peak GPU memory unavailable after reporting error, not inferred.
- Seed 7202: MSE 0.00769765 -> 0.0000353707. All 10 directions correct by 512. Final worst RMSE 0.11345 m/s; 9/10 pass. Wall 16.82 s, peak allocated 0.19594 GiB.
- Both fitted increasing start and decreasing deceleration curves, unlike the original 256-step output. This refutes a categorical claim that this decoder/context cannot express opposite trends. It does not establish convergence, full-data learnability, generalization or safe driving. Frozen contexts and action-only loss are deliberately easier than joint main training.

## Operational issue
Seed 7201 completed training and saved final checkpoint/monitor evaluation, then failed final reporting because Lightning teardown moved the model to CPU. Reporting-only device placement fixed; seed 7201 was NOT retrained. Its report uses saved 2048 monitor output, and peak memory was not recovered. Seed 7202 ran once after the fix. Original checkpoints and frozen parameters verified unchanged. Versioned latest/best retained; best means lowest observed unit action MSE, not driving quality.

## Conclusion
The 256-step failure was insufficient evidence for a structural impossibility. Optimization budget and seed dependence matter on this tiny task; the predetermined complete-fit gate still did not pass. Stop here. Next decision should separate full-data conditional optimization from data ambiguity, rather than altering architecture based on the earlier short test. No automatic main training extension.
