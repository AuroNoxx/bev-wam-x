# Frozen 4-vs-16 rollout gradients (2026-11-16)

## Execution and controls
No optimizer/EMA updates, test reads, default changes or checkpoint writes. Source is original2048 best checkpoint. Two reproducible first training batches for seeds1810/1811, base objective batch8, rollout subbatch2, one explicit identical noise tensor per example. Four unique rollout fingerprints retained in report; initial speeds6.69-10.13m/s. This is a narrow moving-vehicle sample, NOT evidence about stopping or the previous zero-speed clamp cases.

Each arm recomputes context from identical actor-only inputs and asserts bitwise equal context. Dropout0, BF16, coefficient0.01, identical physical objective. Parameter version counters and named buffers unchanged. CPU stored gradients allow cross-arm dot products without retaining both on GPU. Source future labels only participate in the original supervised loss, never rollout actor input.

## Result
seed1810/1811 respectively:
- Combined rollout gradient 4-vs16 cosine0.159/0.871; norm16/norm4=1.471/1.538.
- Kinematic component cosine -0.008/0.736; jerk -0.055/0.800. Thus a4step surrogate can give very different directions for a16step sampler; this is not universally an opposite gradient.
- After adding the unchanged base objective, total-gradient cosine is0.916/0.985. This important qualification prevents overstating the effect on the actual overall update. Global clipping preserves direction, but Adam update direction is not inferred here.
- Clip-to1 coefficients4->16:0.09638->0.08868 and0.07020->0.06241.
- Same four trajectories show better overall kinematic residual and position-derived/predicted speed agreement at16steps, but first boundary acceleration is mixed. All have near-zero and negative raw speed fraction0; no conclusions about zero-speed behavior.

## Cost measurement and correction
The first diagnostic retained a loop-local loss reference and contaminated later peak-memory figures. Its artifacts remain for audit and are not the authoritative cost result. The clean rerun explicitly releases this reference; weighted directions/norms agree.

Authoritative clean warmed forward + one combined backward per arm:
- seed1810:4steps0.123s,16steps0.463s (3.77x).
- seed1811:4steps0.125s,16steps0.443s (3.53x).
- Diagnostic allocated GPU peak including model/base gradient:4steps3.42-3.43GiB,16steps5.22-5.24GiB.

These are one warmed measurement each, counterbalanced order, with graph retained for subsequent diagnostic component gradients. NOT end-to-end training throughput, not peak training memory with a simultaneously retained full base-loss graph/optimizer. Clean diagnostic18.75s total. No change to training budget or defaults implied.

## Next single-variable proposal
A bounded paired64-update comparison can test ONLY regularizer Euler steps4->16, retaining coefficient0.01/every4updates/subbatch2/data/noise/source and primary16step evaluation. The existing4step paired outputs provide references, but use the same evaluation noise contract for both. First verify the actual combined training graph fits memory. Keep progress, initial/full residual, acceleration/jerk, candidate diversity/near-zero and source limitations visible; no ADE-only or safety conclusion. If results remain seed-dependent or fail multi-metric non-regression, do not promote. No additional losses, weights or parameterization changes in this test.

Awaiting parent decision; no further training launched.

Authoritative artifacts: artifacts/frozen-gradient-4vs16-clean/{report,derived-summary,completion}.json. Original contaminated-memory diagnostic: artifacts/frozen-gradient-4vs16/. Scripts: diagnose_gradient_steps_clean.py and original diagnose_gradient_steps.py.
