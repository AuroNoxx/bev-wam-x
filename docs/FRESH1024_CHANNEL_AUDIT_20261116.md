# Fresh1024 channel decomposition

Frozen4checkpoints,8common reuseddevelopmentwindows,one noise. Output artifacts/fresh1024-channel-audit/report.json; script audit_fresh1024_channels.py. No updates,test or old artifact overwrite.

Public sample(k1,16) and independent16Euler implementation matched EXACTLY for all4. Endpoint error=(1-t)*(predicted flow-target flow) verified maxabs<=2.5e-7. Thus no observed sampler/formula scaling mismatch.

At bridge t=.5, x physical endpoint RMSE (1960JEPA/no,1961JEPA/no)5.725/15.362/9.642/6.067m. Position-error finite-difference RMSE x11.96/19.17/20.77/10.71m/s versus y2.77/2.92/3.30/2.45m/s. Full25time-by5channel FM error tables stored, not only aggregates. Fixed50m x scale makes apparently small normalized errors physically large; first predicted point has no hard position boundary.

Train labels only:5296windows,132400points,6011negative speed points(4.54%) across1186overlappingwindows; min−0.0787299m/s, none below−0.1. Not certified as sensor noise without upstream provenance, but too small to explain tens-of-m/s failure. No relabeling performed.

Minimal next candidate, NOT adopted: encode x/y as time-dependent scaled residual to observed constant-velocity ego prior, using fixed train-only per-horizon statistics. Both FM endpoint target/noise and inference decoder must use that same representation. This changes numerical conditioning, not evidence of safety or guaranteed continuity. First perform invertibility/causal input tests and bounded paired training. Keep original5channel representation,weights andoutputs intact. Do not claim residual encoding fixes executable acceleration or jerk; test all original multidimensional metrics.
