# Matched-six deterministic capability diagnostic, 2026-11-16

Completed, no training remains running. No FM training, test data, old-weight changes, guard changes or CARLA model execution. This is an offline capability diagnostic, not evidence that deterministic control replaces the FM project.

## Data and integrity
Source: data/carla_causal_matched6_20261116_v1. Independent cache: data/carla_matched6_cache_v1. scripts/prepare_matched6_diagnostic.py checks all 3600 packets for frame/snapshot sync, pose agreement, previous-observation/current-control timing, constant current cruise/permission, navigation transform agreement (<0.000683 m). Historical actor whitelist poison checks pass at 24 points spanning six episodes. Future labels are never actor fields; model future-label poison check also exact for both seeds. Unknown remains255, mean unknown fraction approximately99.2%; semantic rays/pose/map remain simulation oracles. Cache build54.63s.

Records start after settling; anchors6..549 yield544windows per episode, 2176train/1088dev. No filtered windows. Train/dev entire episodes separated; roads388/392 versus713 remain confounded. Uniform source ranges did NOT produce matching realized cruise: train8.330..8.757 versusdev7.887..7.894m/s. No redraw or dev-based rebalancing.

All episodes have real middle stop and restart, plus final stop. Initial transient stop durations are not all >=1s. Complete5s stopped supervision is very scarce: only1devwindow satisfies current speed<.2 and all25future speeds<.2. Shorter stopped-prefix dev counts:106at.2s,91at.6s,81at1s. These are overlapping windows, not independent events.

## Frozen bounded configuration
scripts/train_matched6_regression512.py; artifacts/matched6-regression512. Fresh seeds7201/7202, each512updates,batch12,6144draws including all2176train windows (2.82passes). Existing bidirectional condition/25x2 velocity residual architecture, scales[10,.5], same trapezoid decoder, no interaction branch or12coefficient decoder.966939trainable parameters. Direct normalized q MSE instead of FM bridge objective; unchanged .2 gatedpositionSmoothL1 and .1eachJEPA auxiliary losses. Thus this is NOT a controlled new-data-vs-old-data improvement estimate or a causal estimate of FM-vs-regression superiority.

FP32+MATH, AdamW1e-4/wd.01/clip1,64warmup then original512cosine with.1floor. Lightning/StatefulDataLoader/VerifiedCheckpointIO, latest128steps,bestdevfit retained. Final512used, not bestpicked. Checkpoints include optimizer/scheduler/taskstate and correct loadercursor6144. This run verified checkpoint structure/cursor, not an interrupted-resume bitwise experiment. Sourceweights untouched because initialization fresh.

Training wall142.29/134.18s, peakallocated.9903/.9899GiB, totalwith evaluation291.34s. Normal budgetstops, no nonfinite error. OnlyTensorBoard Python processes remain afterward. Predictive velocity roundtrip maxerror8.4e-8m/s inpreflight. Fit uses representative trainstride8 (272windows), dev usesall1088.

## Results:7201 /7202
- Overall dev position-fit diagnostic1.716/2.102m versusconstant-speed2.590m andstationary10.177m. Trainstride8 diagnostic1.772/1.967m. Not a drivingacceptance metric.
- Start91windows: predicted5s speed.524/.966m/s versuslabel5.241. Currentlead already moving>.2m/s subset16:pred.567/.948 versuslabel6.521. Remaining75currentlead stationary windows have genuinely uncertain futurelead start, so their future-fit misses alone do not prove failure to react. CurrentleadGT used ONLY to stratify frozen results, not input.
- Deceleration446:pred5sspeed4.005/4.658 versuslabel1.465;speedMAE2.540/3.193m/s. Position1.975/2.538m versusconstant3.146. Better positionfit does not erase under-braking.
- Steady276:position1.134/1.251m versusconstant.769m, bothworse.
- Stopped1s prefix81:predicted speedMAE.542/.905m/s,1sdrift.492/.818m. No parkingpass.
- Meanpeakvectoracceleration includingtruecurrentt04.729/5.265m/s2. Meanpeakvectorjerk23.966/28.285m/s3 versuslabel3.995. Jerk excludes unknown pre-t0 acceleration; acceleration includes the first0->.2s transition.
- Negativepredictedlongitudinalvelocity fraction3.915%/.00735%, clearseeddependence. No physical guarantee from thedecoder. Overall speedaverages can hide this.
- Collision,TTC andactualclosedloopprogress are unknown offline. World auxiliary was trained unchanged but worldpredictionquality was NOT separately reevaluated inthis diagnostic.

## Decision
The data supports causal task setup and some learnable trajectory structure beyondconstantvelocity, but neitherseed meets useful reactive start/stop behavior and smoothness. Do NOT run these models inCARLA or extendtrainingautomatically. No threshold was loosened. Lowtrainanddevdata independence, realizedspeedshift, shortbudget andnonuniquefuturelead actions limit conclusions.

Priority for root review: distinguish immediate reaction after actually observedlead motion and currenthold prefixes from unpredictable5slead future; then decide whether one bounded longer-learning check is warranted or whether thecondition/actionpath stillrequires restructuring. Do not claim that moredata alone fixed thearchitecture, or thatFM has been disproved. No furtherexperiment initiated.
