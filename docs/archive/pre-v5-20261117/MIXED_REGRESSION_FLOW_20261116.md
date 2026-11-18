# Mixed-scene deterministic regression versus Flow learnability

Authorized small diagnostic only. No default replacement, officialtest, simulation or safety acceptance.

## Data selection
The initial attempt to select from84 historical training windows left mostly startup frames after protected-window interval exclusions; that proposal was rejected before training and retained asdesign_rejected_initial84.json. The actual design uses12 train-only moving windows, two per six scenario-directory classes, on different recordings. Both historical0.6s and future5s support intervals are disjoint from the protected42/38 diagnostic windows within each recording. Actual timestamps are nominal10Hz frame indexing, not new causal certification. Some routes/geometry overlap prior diagnostic routes, so this is learnability, not independent route-generalization.

Actual selected initial speeds2.03-12.21m/s, with acceleration/deceleration/lateral-motion targets. Directory labels are not event annotations: the selectedHardBreak windows are recovery acceleration, not certified braking events. This set is small and not a distribution-representative driving benchmark.42 existing diagnostic windows stay out of training.

## Matched setup
Fresh seeds2040/2041, counterbalanced two-arm order, same model weights initially, same fixed12-window batch,512updates, same batch/noise RNG draws, BF16, AdamW1e-4,64warmup and512cosine horizon, clip1. Width128, decoder2layers/4heads, velocity2integraldecoder/current-supportworld. Both share position reconstruction auxiliary with the same random gate andJEPA latent/variance/semantic weights0.1. Kinematic/unit/negative penalties already vanish by this representation and are omitted symmetrically.

FM arm learns target-noise velocity on the standard bridge and generates from pureGaussian noise through16Eulersteps. Regression arm predicts normalized2Dvelocity target directly with zero action-token input/time0, then uses the same physical decoder. It is deterministic and repeats of its prediction are not independent candidates. Effective use of action-input/time parameters differs; this is a diagnostic of conditional regression versus flow learning, not a claim of architecturally identical functions or production superiority. Different action losses must not be compared numerically as quality scores. Same update/sample budget is not equal inference FLOPs.

Lightning Trainer, StatefulDataLoader, VerifiedCheckpointIO, rollinglatest/best, localTensorBoard. Per-arm5min/overall15min guard. Nonfinite loss/gradient stops. Best monitors training-subset position fit only, not driving quality. Every128updates fit diagnostics; final42window diagnostics not used for checkpoint choice. Future-poison invariance asserted during evaluation.

## Interpretation gate
Compare recorded-target velocity and position fit, candidate mean/spread, vector acceleration/jerk, forward proxy and per-scene behavior. Regression's zero diversity is structural, not a quality win. No collision/free-space/TTC oractualeventsuccess labels, all remainunknown. A successful tiny fit would only show this model can learn these specific recorded sequences, not that expert labels are the sole correct actions.

## Completed outcome
All four512-update arms completed normally, total428.19s(7.14min). Peak allocated0.935-0.940GiB. First-three RNG traces matched. Latest/best present; no active experiment afterward.

FM -> regression, seeds2040/2041:
-12 fitted windows position diagnostic2.197->0.960m and1.872->1.218m; velocityRMSE1.776->1.190 and1.725->1.198m/s. Regression is easier to fit in this budget, but neither meets a strict0.5m/s velocity-fit gate.
- Fit mean peak vectorjerk239.24->83.86 and225.45->82.10m/s3. Both remain too rough for a driving claim.
-42 held windows position diagnostic7.218->7.128 and7.387->7.704m; velocityRMSE2.766->2.499 and2.904->2.821. Generalization does not consistently improve, despite the deterministic model's structural absence of sample diversity.
- Held vectorjerk232.44->67.65 and222.49->61.30m/s3. Lower stochastic roughness is not equivalent to correct interactions or safety.
- Per-scene held position improvements are mixed:2040 regression worsensHardBreak/CutIn/Exit/MergerIntoSlowTraffic and improvesV2/SequentialLaneChange;2041 improvesCutIn/Exit/MergerIntoSlowTraffic but worsensHardBreak/V2/SequentialLaneChange. No cross-scene quality winner.

Conclusion: the same conditional architecture can fit recorded trajectories better with direct regression under this short budget, which supports a flow-generation/denoising learning burden beyond a total inability to encode conditions. It does not establish adequate conditional perception, perfect train fit, superior production architecture, or generalization. Do not replace the default or continue longer automatically. Keep regression as a diagnostic baseline while separating action-mean learning from stochastic refinement; any next change needs a bounded mixed-scene test rather than returning to a parking-only criterion.
