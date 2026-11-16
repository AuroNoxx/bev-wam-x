# JEPA-space and initial-boundary frozen audit (2026-11-16)

## Budget and evidence
Read-only/frozen, zero updates, no test/default/checkpoint changes. Three checkpoints: original2048,1810-rollout16,1811-rollout16. Sixteen existing DEVELOPMENT windows,14from one HardBreak route and2from another; strongly correlated and not broad scenario evidence. First frozen pass11.43s and matched-mask refinement11.19s excluding imports/load setup. No long process remains.

Authoritative matched-mask report: artifacts/jepa-space-boundary-audit-v2/report.json. Initial audit kept separately. Scripts audit_jepa_space_boundary.py and _v2.py.

## Does JEPA vs persistence compare the same space?
Yes within each checkpoint: target=EMA FrameEncoder(masked future), persistence=the SAME EMA FrameEncoder(canonical current), prediction=forecast predictor output. Both predictions compare to the identical target under identical valid-token masks. Predictor and encoder have separate learned LayerNorm parameters, so they are trained to match rather than tied. Current/historical/future cached grids are intended in current ego FLU; all16metadata future offsets are10/20frames (nominal1/2s), not an accidental horizon index mismatch. Source timestamps and source OCC causality remain unverified beyond existing checks.

For original2048, target RMS~0.999 and predicted RMS~0.995. Channel-unit normalization preserves the ranking. Online-vsEMA target drift~0.00081; zero-to-unknown target recoding drift~0.00060. Across trained64step checkpoint variants, fixed-input EMA target drift from original~0.00017-0.00022. None is large enough at these observations to explain the JEPA/persistence gap by a simple norm or EMA-space mismatch.

Actual original2048 latent MSE:
-1s all-valid960tokens:JEPA0.130828 vs persistence0.042975.
-2s all-valid896tokens:0.187200 vs0.106365.
-1s dynamic-class proxy204tokens:0.163547 vs0.046658.
-2s dynamic-class proxy175tokens:0.172288 vs0.051851.
-1s changed-known-class226tokens:0.164587 vs0.049764.
-2s changed-known-class214tokens:0.206896 vs0.095351.

All three checkpoints show the same broad ordering. Dynamic-class proxy means pooled vehicle/rider class presence, not measured moving-object tracks; class change is not all motion (a car can move while class stays the same). Other-valid is NOT certified static.

Matched valid-mask target shuffle test: original true0.156567 vs shuffled0.152258 (persistence0.074860);1810 true0.156413 vs shuffled0.152109;1811 true0.157686 vs shuffled0.153450. The predictor does not beat this particular permutation on this highly correlated development sample. Predicted cross-example std~0.056 vs target~0.132 (unmasked diagnostic). This suggests weak scene-specific prediction but is NOT proof of complete representation collapse; a single permutation and two routes cannot establish that.

## The mask is limited, not a full world-dynamics objective
Actor canonicalization sends0/255/nonvalid to unknown255. Label target uses only CURRENT OCCUPIED support. Then valid-token means any future occupied semantic count. It is neither a verified visible/free-space mask nor a motion mask. Future0 inside support remains distinct encoder category0 although it is not certified free; recoding it255 changes target only~0.0006here and does not fix ranking.

Current-occupied support excludes37.46%/39.57% of future vehicle/rider-class occupied voxels at1s/2s (voxel-weighted across these16windows). It therefore cannot claim to learn all newly entering actors/occupied regions. These exclusions are not proof of sensor visibility and must not be turned into free negatives. Semantic pooling footprints and the encoder receptive fields are not identical; GroupNorm also couples spatial features. Latent errors include observation/mask effects, not only physical object motion.

## Initial-boundary mechanism
The standard model generates25future points from independent Gaussian5-channel noise; current state enters context, not a hard boundary condition. Physical t0 is used in the existing kinematic loss, but speed/position outputs remain independently predicted. Heading normalization and speed clamp occur after flow sampling; no hard initial-speed continuity is enforced.

Reproduced fixed window13/candidate2, exact prior batch/noise contract (seed8193):observed current speed8.258523m/s; offline label first speed8.517669m/s, first position(1.681664,-0.048861)m. Labels are diagnostics, not uniquely correct driving.
-original2048 raw first speed0.203959 -> acceleration-40.272823m/s^2 (not clamped).
-1810-rollout16 raw-0.469869 clamps0 -> -41.292617m/s^2.
-1811-rollout16 raw1.097513 -> -35.805050m/s^2 (not clamped).

For the negative raw output, clamp derivative w.r.t. that raw speed is0; thus the speed-only local acceleration/jerk branch cannot directly correct that raw value at that point. This is NOT absence of all parameter gradients: original FM/raw endpoint objectives and shared position/heading paths remain active. Positive raw failures also exist, so removing clamp alone does not explain or solve the full boundary issue. True measured lateral velocity remains absent; scalar-forward t0 is an explicit approximation, and prehistory jerk is unknown.

## Few falsifiable next options, not implemented
1. Before world-model training, broaden frozen route-balanced TRAIN-only dynamics audit with several target permutations, frozen EMA and identical valid masks. Predeclare whether predictor beats persistence/permutations specifically on changed-known support; report no claims outside that support. This tests whether present failure is merely the narrow two-route development selection.
2. If learning is then warranted, isolate world predictor parameterization: frozen shared target encoder, compare absolute prediction vs residual over current EMA features under identical support/budget. A zero residual equals persistence at initialization, so count only learned improvement beyond persistence and dynamic/shuffle gates, not the built-in baseline advantage. Do not simultaneously alter support masks or loss weights.
3. Boundary branch: separately test a current-speed-centered flow target/initialization or genuinely state-integrated action representation, one variable at a time. Do not conflate reparameterization with post-hoc clipping, and do not promise bounded acceleration implies safe driving. Existing integrated-output probe already showed that consistency by construction can harm progress or dynamics.

No experiment beyond this audit launched. Collision/TTC/closed-loop safety and automatic Astra training remain unavailable/unknown.
