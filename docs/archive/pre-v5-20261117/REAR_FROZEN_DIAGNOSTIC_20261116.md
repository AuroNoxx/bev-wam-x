# Frozen deterministic rear-trajectory diagnostic, 2026-11-16

Completed two seeds x512 updates, stopped. No CARLA/test/old-weight edits; FM remains the intended research direction. Framework Lightning + StatefulDataLoader + VerifiedCheckpointIO. Frozen history256 encoder for seeds2070/2071, full480x128 context, zero-initialized Linear61440->12 (737292 trainable parameters), six acceleration/curvature knots. No coefficient target loss. Physical decoded SI losses scaled by5m,5m/s,.1rad,.1rad/s, equal normalized weights; fixed before dev. FP32 head, batch12, AdamW1e-4,wd.01,clip1,64warmup512cosine.

514train/514dev, route isolated but only one route/episode each and cruise8vs8.5 confounded. Each seed6144draws/514unique; no training-derived selection based on dev. Both source encoders/checkpoints and buffers unchanged; future-label poisoning context exact; zero head equals constant current speed straight. Feature extraction is frozen, no world updates.

## Storage error and recovery disclosure
Seed2070 failed first128-step save because new task omitted required task_state. Full latest.pending contained finite weights, AdamW/scheduler/Lightning loops and loader cursor1536. Preserved pending; checked source hash, added explicit recovery metadata, saved recovered128.ckpt through VerifiedCheckpointIO, resumed remaining384updates. Original RNG snapshot was absent: this head/update path has no stochastic ops after fixed input sequence/zero initialization, but do not call it a verified bitwise uninterrupted replay. Final both checkpoints contain optimizer/scheduler/task_state/RNG and loader cursor6144, global_step512. CPU reload predictions matched saved GPU outputs <.001m. Later saves succeeded. No repeated first128 updates.

Hot head update ~.009s; training per seed ~5s, peak GPU allocated.0823GiB excludes frozen feature extraction. Recovery execution including both feature precomputations/evaluation took21.35s; initial interrupted preparation and human debugging time are additional. Original resumed-run wall time must not be presented as total first-attempt wall time.

## Results
Mean trajectory position diagnostic m (not driving score):
- zero coefficients/constant speed2.680;
- seed2070 trained2.034;
- seed2071 trained2.680;
- original frozen FM four-candidate reference4.842/4.616 (different representation, head/objective/budget; not attributable FM ablation).

Deceleration169windows: zero-coefficient position3.746m, trained2.673/4.703m. Five-second speed MAE zero4.389m/s, trained3.102/4.907m/s. Seed2071 worsens versus simple persistence. Steady163windows position zero.473m, trained.923/1.072m. Both sacrifice steady behavior. Starting54windows position3.822m ->2.707/2.439m, but not successful driving.

Full actor-velocity dynamics including measuredt0: mean peak acceleration.555/.483m/s2, jerk.257/.290m/s3. Much smoother, but weak response/averaging remains. Initial lateral velocity discontinuity mean.00485/.00157m/s, max.01053/.00520m/s; curvature-at-t0 still not explicitly constrained. No lateral-acceleration/actuator safety certificate.

Current stationary permitted20windows: predicted five-second speeds2.328/1.741m/s versuslabel6.518; MAE4.190/4.777m/s. Seed2070 predicts negative speed on26.6% of timepoints in this subset, despite no reverse labels; seed2071 zero. Signed-speed capacity is not permission to reverse.

## Critical intent-label limitation
Frozen audit: bothtrain anddev34windows have current movement_permission=false but the logged permission switches within future5seconds. Actor correctly does NOT see the future switch. Fitting that future trajectory asks it to anticipate an exogenous command change. Predictions at0.6s for thosewindows.347/.444m/s. Cannot call this obeying current hold or a causal closed-loop pass. Such future command changes require a defined conditional-label contract, e.g. a valid supervision prefix until command changes, or recorded rollouts with current intent held. Do not leak future command/switch time into actor or simply reward anticipation. Existing labels are factual trajectories, not uniquely valid plans under current intent.

## Conclusion
No robust two-seed improvement over constant-velocity baseline, so no closedloop. Smooth output alone is insufficient; steady/decel and unintended reverse failures remain. Representation and objective changed, so this cannot disprove FM. Next should first resolve exogenous-command label validity and direction authorization, not blindly extend training. No further experiment auto-started.
