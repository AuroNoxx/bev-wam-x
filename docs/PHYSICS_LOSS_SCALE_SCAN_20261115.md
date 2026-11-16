# Physical loss scale, gradients and activation timing

Isolated diagnostic, same4 train windows and small967710-parameter actor,512updates each,batch4,lr5e-4,world auxiliaries always present. No baseline configuration changes or saved weights. Three inference seeds, one training seed; BF16 not bitwise deterministic. Full control repeated here differs from previous full result, reinforcing that marginal changes need repeated training seeds.

## Full-weight gradient audit
Full trainable parameter gradients measured separately per raw term, with three independent bridge seeds at initial/final model. Ratios below use configured weighted gradient norm divided by FM gradient norm; not Adam update magnitudes or proof of destructive conflict.

Initial two active-gate seeds: position0.381–0.425xFM, kinematics0.367–0.411xFM, heading0.002–0.009xFM, negative-speed0.790–1.055xFM despite its0.02 coefficient. Final: kinematics3.680–4.336xFM, cosine with FM+0.159 to+0.225. Therefore its relative scale grows substantially; not simply an opposite-gradient explanation. Final position0.464–0.582xFM, cosine-0.074/+0.521. Third bridge seed had all t<0.5 so all physical gates/gradients were0 at both snapshots. This is expected from batch4 stochastic gate, not missing autograd. Sparse gated minibatches can contribute variability. Audit includes no gradient-vector conflict decomposition for world loss.

## Weight/timing scan
full: original physical weights; all_x0p1: all four physics weights multiplied0.1; kinematics_x0p1: kinematics0.1→0.01 only; ramp_all_x0p1: first256updates no physics then linear increase to0.1×original weights over next256.

|Variant|Mean ADE over3 inference seeds(m)|Mean kinematic residual(m/s)|Measured loop(s)|
|---|---|---|---|
|full|6.188|5.036|32.35|
|all_x0p1|4.597|9.745|32.09|
|kinematics_x0p1|2.058|6.163|32.04|
|ramp_all_x0p1|3.553|8.376|32.11|

The kinematics-only0.01 coefficient is the most useful next diagnostic candidate, not an accepted default: materially lower train ADE than full weights but residual rises from5.04 to6.16m/s. Uniform downweighting and delayed activation also trade worse consistency for better ADE; neither dominates. Physical residual remains orders of magnitude above target0.005–0.109m/s. No collision, comfort, closed-loop or held-out evaluation. The small 4-anchor scan cannot establish safety/generalization, teacher-scale behavior, high-speed coverage or a globally optimal coefficient.

Recommend verify only-kinematics0.01 over multiple training seeds and a slightly broader train-only diagnostic set, then examine physical term measured on noisy bridge endpoint vs sampled trajectory. Do not remove safety checks because offline ADE improved. No formal training launched. Source causal provenance remains uncertified. Test split and old checkpoints untouched.
