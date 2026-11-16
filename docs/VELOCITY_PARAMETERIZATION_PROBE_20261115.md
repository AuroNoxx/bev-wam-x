# Velocity-integrated trajectory parameterization diagnostic

New isolated subclass VelocityIntegratedJoint, no original model/checkpoint edits. Flow space is25x2 local velocity vectors normalized by30m/s. Initial observed velocity is(state_speed,0), heading frame initiallyforward. Positions use trapezoidal0.2s integration then x/50,y/10 scales; speed is vector norm/30 and sin/cos come from normalized velocity. At exact rest choose forward heading. Small diagnostic reuses5-channel decoder input padded withzeros and first2 output channels, not a fully optimized architecture.

Roundtrip using ground-truth velocity/heading yields per-window position ADE0.01047/0.01227/0.04981/0.05270m and FDE0.02132/0.01294/0.08989/0.08770m. These target inconsistencies alone do not explain multi-metre sampled error. Reconstructed kinematic residual1.64e-6m/s is expected algebraically; it is not independent model-quality evidence.

Compare three512-step small-model modes on same4train anchors and seed: absolute5D FM+world; velocity2D FM+world; velocity2D FM+world+position coefficient0.2. No other physical losses applied. Same seed does not yield identical bridge noise because dimensions differ. Sampling16steps/K4 over3seeds, additionally64steps for first seed. Report ADE and speed-based acceleration/jerk and backward-direction fraction. These acceleration/jerk summaries differentiate scalar speed, not complete vector lateral dynamics; cannot establish comfort. Direction<0 meanscos headingnegative; do not confuse nonnegative speed norm with prohibition of reverse motion.

Velocity integration enforces internal consistency but does not constrain acceleration, jerk, steering curvature, lane adherence or collision. Near-zero-speed heading is ambiguous. No safety/generalization/closed-loop acceptance. SourceOCC causality still uncertified. All data are the same training-only samples, officialvalidation/test untouched, no checkpoint saved.

## Results: three inference seeds, one training seed
|Model|Mean ADE(m)|Kinematic residual(m/s)|Mean scalar-speed accel(m/s2)|Mean scalar-speed jerk(m/s3)|Negative heading fraction|
|---|---|---|---|---|---|
|absolute_world|2.5968|8.605|3.8119|32.668|0|
|velocity_world|3.7814|1.6228e-06|1.9333|14.998|0.36667|
|velocity_position_world|6.5475|2.168e-06|4.5748|38.659|0.17167|

Velocity integration enforces algebraic consistency but did not improve sampled ADE in this bounded experiment. Adding position supervision also did not fix the issue; substantial lateral error appeared. Negative-heading fraction includes arbitrarily small speeds and must NOT be presented as an actual reverse-driving event rate; future tests need signed velocity with a meaningful threshold. Acceleration/jerk are scalar speed changes, not full lateral dynamics. No default architecture change justified. Next defensible diagnosis is controlled speed/curvature or acceleration parameterization with explicit rest handling and separate physical-bound checks, alongside training/normalization robustness; not larger blind training.

The report descriptive limitations were corrected after the run (numerical results unchanged) to explicitly list the three modes above, with no kinematic/heading/nonnegative-speed penalties applied. The kinematic loss is calculated for diagnostics but not optimized. No formal validation/test use, no saved checkpoints, all3runs completed.
