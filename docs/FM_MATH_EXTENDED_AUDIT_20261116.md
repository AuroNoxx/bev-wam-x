# Extended BF16 MATH causal audit, 2026-11-16

No optimizer updates, no checkpoint or default modifications. scripts/audit_fm_math_extended.py; artifacts/fm-math-extended-audit-v1/report.json. Runtime18.76s. Follow-up repeat: scripts/audit_math_embedding_repeat.py, artifacts/fm-math-embedding-repeat-v1/report.json.

## Coverage / fixed criteria
Eight train anchors36,106,176,246,316,386,456,526, with nonoverlapping history+future intervals; SAME one episode/road, not eight independent scenarios. Two saved2070/2071 history256 checkpoints, fourpaired batches, suffix normal amplitudes0.1/1/3/10. No dev/test. Same25-token fixed triangular model and cumulative2Dvelocity decoder. Original gradient threshold max(1e-6,maxgradient*1e-5) unchanged.

## Results
MATH30/32gradient comparisons pass, Flash27/32. All64prefix forward checks exact; suffix input gradients zero; all32sampling comparisons (4/16Euler) latent and physical prefixes exact and finite.
MATH failures both2071/pair0 at amplitudes1/10: encoder.embedding.weight max1.56462e-6 /1.42306e-6 >1e-6; relativeL2~1.39e-5 /1.29e-5. Flash failures range7.6294e-6--1.52588e-5 in transformer feedforward/self-attention weights, beyond respective unchanged thresholds.

Focused identical-input repeat on the affected MATH case showed BF16 parameter-gradient variation4.66e-7--9.76e-7; suffix comparisons1.42e-7--5.36e-7 in those3repeats. Thus residual embedding accumulation variation is near the fixedthreshold and the original twofails were not stable effects in that repeat graph. Do NOT discard the initialfails or call universalpass. FP32+MATH repeated samecase gives2.61e-8--7.82e-8 suffix differences; not yet the full32caseFP32 sweep.

## Speed / allocated memory (AMD7900XT, small~1Mmodel)
Full-context forward/backward, nooptimizer ordataIO,10timed repeats after2warmups:
- batch2 Flash23.42ms /MATH23.76ms median; peakallocated0.178GiB both.
- batch12 Flash86.00ms /MATH87.23ms; peak0.690GiB both.
16Euler K1sample includingcontext+decode,4timed repeats:
- batch2 Flash38.00ms /MATH40.95ms; peak0.101GiB.
- batch12 Flash38.83ms /MATH42.11ms; peak0.252GiB.
MATH about1.4%slower for these forward/backward sizes and~8%slower in sampling. Smallshortbenchmark, not trainingthroughput, not150Mteacher, notOrin, notvehiclelatency guarantee. Samepeakallocation does not imply sameworkspace under all shapes.

## Decision
Not all strictgradientchecks pass, so no automatictraining or declarationthat MATH is fullycertified. Forward causal isolation is strongly supported within testedconditions. Backend improves largerFlash-specific sensitivity but near-thresholdembeddingaccumulation variability remains. A bounded next verification could useFP32+MATH overtheexact32casegrid or investigate deterministicembeddingbackward, without relaxingthreshold. Only after a chosenpath satisfiesits declaredchecks should a bounded matched complete-window causal-vs-bidirectional study be scheduled; adding partialsupervision remainsseparate. Do not rerun untilsuccess and overwritefailures.
