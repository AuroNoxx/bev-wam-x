# Isolated-RNG three-seed first-boundary replication

Completed authorized six short runs: seeds1802/1803/1804 × baseline/first_penalty,96steps each. Same teacher2048 weights, fresh AdamW lr1e-5,BF16,b8,128train windows,16reused validation windows,4fixed-seed candidates/window. No test, no old weights changed, no further training scheduled. All runs exited normally. Per-arm max5min, overall15min cap not reached.

Paired data order is controlled by seed-specific StatefulDataLoader. FM bridge noise/time uses explicit seed*100000+step under fork_rng. Extra boundary probe uses seed*100000+50000+step in a separate fork with every-call assertion that it does not alter outer CUDA RNG. Recorded FM pre/post RNG hashes agree for first3steps in each pair. This fixes the earlier extra-noise-stream confound, but not BF16 nondeterminism or small-data variance. Boundary arm still has extra forward/backward work, so equalsteps != equalFLOPs.

Seed-wise first residual m/s, baseline -> penalty:
1802:6.4555 ->4.7719;1803:5.3976 ->3.9466;1804:5.5289 ->3.5440.
Mean5.7940 ->4.0875 (~29%lower).
However whole-trajectory residual mean1.5983 ->1.7471 (~9%worse). Per-trajectory peak acceleration averaged across candidates13.233 ->14.088m/s². Per-trajectory peak jerk averaged90.399 ->102.047m/s³ (~13%worse). These averaged peaks are not global peaks.
Global peak acrossall3seeds/192candidates:accel38.295 ->36.135m/s²;jerk198.431 ->192.237m/s³. Thus the very worst maxima slightly improve, while average peak dynamics worsen; neither is acceptable evidence of safe/comfortable driving.
Mean forward displacement proxy20.681 ->20.703m almostunchanged overall, but individualseed1802 drops21.727 ->16.640m and backward segments increase. Otherseeds moveforwardmore. Meanmaxlateral1.705 ->1.382m; not a road-boundary metric. Collision/TTC/routeprogress/trafficlaw/takeover remainnull.

Mean training wall35.20s baseline vs41.67s penalty (~18%more); firstbaseline includes more warmup. Peakreserved~4.55–4.60GiB vs5.54GiB. Evaluation/checkpoint costs and timing variance prevent interpreting this as pure kernel overhead.

Conclusion: local first-segment objective helps its target consistently acrossthree seeds but does not yield overall dynamic improvement. Do not promote weight.1/hinge6 or replace defaultteacher. Next change should address full-horizon consistent dynamics and initial boundary jointly, validate realistic constraints and retain progress/path evidence; no automatic next experiment executed.

Artifacts: artifacts/teacher-boundary-isolated-multiseed/{summary,aggregate,selection,config}.json and perseedarm report/latestcheckpoint. Script:scripts/train_boundary_isolated_multiseed.py. Saved task settings include noise_seed,boundary_arm; new selection SHA binds actual128/16selection. No uninterrupted gradient-equivalence claim.
