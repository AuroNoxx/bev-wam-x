# 16-step jerk-weight hypothesis: frozen linear gradient audit

Reuses measured weighted gradient norms and dot products from artifacts/frozen-gradient-4vs16-clean/report.json. These are the actual 16-Euler-step graphs, same source2048 checkpoint, same two seed batches and explicit noise, not an inference from 4-step results. No new forward, backward, optimizer update or test read was necessary. Applies locally at the original initialization; not a claim about every trained checkpoint or sample.

Let C be the complete current weighted gradient and J the already weighted rollout jerk gradient. For jerk x0.1, C_new=C-0.9J. Measured projection supplies dot(C,J). Then norm(C_new)^2=norm(C)^2-1.8dot(C,J)+0.81norm(J)^2. The exact real-arithmetic linear combination is evaluated from stored FP32/BF16-derived dot products, rather than rerunning a potentially differently rounded backward. Clip coefficients use1/(norm+1e-6), capped at1. This is not Adam-update prediction.

Seed1810: jerk norm3.169 vs all-other10.829; cosine jerk/other -0.002 (essentially orthogonal), jerk/base -0.083. Jerk accounts for7.84% of the projection along total. Combined norm11.277->10.833, clipped-direction cosine0.9675.
Seed1811: jerk norm4.273 vs all-other12.601; cosine jerk/other+0.740, jerk/base+0.687. Jerk accounts for22.63% of total projection. Combined norm16.023->12.920, clipped-direction cosine0.9875.

The measured two points do NOT support the premise that the 16-step jerk term dominates or consistently conflicts with the remaining objectives. Reducing it tenfold still triggers clipping in both, while removing part of an aligned gradient in seed1811. A beneficial generalization effect is not ruled out, but there is insufficient evidence under the requested gate to launch that training ablation. Training therefore NOT started. No defaults/checkpoints changed. Four rollout examples are all moving (~6.69-10.13m/s), not the zero-speed worst case.

Recommendation: stop this weight hypothesis and have the coach select a boundary-specific falsifiable diagnosis, ideally on the unchanged worst first-speed/clamp case, rather than weakening the one term that improved typical jerk without evidence. This is a proposal, not an implementation or safety claim.
