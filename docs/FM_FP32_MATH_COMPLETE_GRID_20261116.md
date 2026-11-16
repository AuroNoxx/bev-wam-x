# FP32 + MATH complete-grid check, 2026-11-16

No parameter/optimizer updates. scripts/audit_fm_fp32_math_extended.py and artifacts/fm-fp32-math-extended-audit-v1/report.json. Source checkpoint hashes, model parameters and buffers unchanged. Original32-case grid exactly retained:2 checkpoints x4paired nonoverlapping train windows x4suffix amplitudes. Each gradient comparison repeated3times, plus same-input repeat perpair/run. All windows from ONEtrain episode, not independent scenarios. Runtime18.23s.

## Fixed-threshold outcomes
96/96gradient comparisons pass original max(1e-6,max_gradient*1e-5); largest difference3.57628e-7. Largest same-input repeat difference2.23517e-7. Prefix forward outputs exact, suffix-input gradients zero, all finite. All16forward-sampling tests (2checkpoints x4pairs x4/16Euler) latent and decoded prefixes exact/finite. This supports the chosen FP32+MATH causal isolation path on tested conditions, not universal numerical proof or learned quality.

## Local cost
Warm forward/backward includingcontext but nooptimizer/dataIO,10timed repetitions:
- batch2 median22.95ms,p9524.93ms,peakallocated0.181GiB.
- batch12 median85.94ms,p9587.14ms,peak0.694GiB.
16EulerK1context+decode,4timed repetitions:
- batch2 median37.61ms,p9538.62ms,peak0.105GiB.
- batch12 median40.52ms,p9541.27ms,peak0.256GiB.
Prior BF16Flash batch12backward86.00ms/.690GiB,sample38.83ms;BF16MATH87.23ms/.690GiB,sample42.11ms. Costs are close at this small~1Mmodel and shortbenchmark; do not claim generalFP32speedup. Small extra persistentgradientfixtures altermemoryby~.004GiB. No extrapolationto150Mteacher,Orinorvehicledeadlines.

## Operational candidate, not training launch
A bounded next comparison can pin FP32 and explicitMATH attention for ALLarms and use identical480complete windows, existing25-temporal-velocity representation, same initialization/data/noise/loss/budget. Only fixedtemporalcausal vs originalbidirectionalselfattention changes. Future labels and cutoff never enteractor; teacherloss/weightskeptsame. This separatesarchitecturalcost fromlaterpartial-supervisionintervention. Do not include12globalcurvehead or future-prefix weighting inthatfirststudy. Holdcoverageisabsentfromcompletecase480subsetandmustbedisclosed, notcountedaspassed. Actualtrainingstillrequiresnextdecision; noneautostarted.
