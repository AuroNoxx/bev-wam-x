# Production-code fixed-causal FM isolation, 2026-11-16

No optimizer/training/checkpoint changes. Isolated subclass scripts/fixed_causal_flow.py overrides only trajectory decoder self-attention with the same fixed25-token triangular mask for every sample. Label cutoff is never a forward input. Original bidirectional implementation remains untouched. Test scripts/test_production_fm_causal_isolation.py; preserved first failure at artifacts/production-fm-causal-isolation-v1 plus v2/v3 reports.

One real2070-history256 checkpoint, two real train contexts, synthetic prefix lengths7/16 used only in scoring. Same current context/time/noise on prefixes; suffix bridge values replaced by large finite noise. Full production condition encoder/predictor, TransformerDecoder and current2Dvelocity trapezoid integration exercised. Includes action-FM and decoded-prefix-position loss gradients, no JEPA future-target loss. No12-coefficient decoder mixed in. Final state/buffers and source SHA unchanged.

## FP32
Prefix forward output and decoded trajectory exactly unchanged. Prefix derivative w.r.t suffix input exactly0. Four- and16-Euler-step latent and decoded prefixes exactly unchanged. Future-label poisoning leaves context exact. Parameter gradient max difference1.86e-8 vs original tolerance1.70e-6; identical-input repeat max3.63e-8. All finite. Original bidirectional negative control prefix change0.51435.

## BF16: forward isolation passes, strict gradient isolation FAILS
Forward/decoded prefixes,4/16-step samples and suffix input gradients remain exactly unchanged/zero. However parameter gradients differ max6.103515625e-5 vs original tolerance1.71875e-6. Maximum gradient magnitude0.171875, difference/max about0.0355%. Largest differences at encoder.conv.0.weight,conv.3.weight,conv.3.bias; temporal_gate.weight1.907e-6. Same-input repeat gradient difference only7.92e-9. Therefore this cannot simply be dismissed as repeated-input nondeterminism; suffix-dependent mixed-precision backward numerical sensitivity remains. Original bidirectional negative control prefix change0.5078125.

First strict assertion stopped at BF16; failure preserved. Subsequent diagnostic runs recorded the failure rather than relaxing threshold and calling it passed. Each run~10s. No training occurred.

## Scope and decision
FP32 isolation is supported for these tested contexts/checkpoint/backend, not universally proven for every input. BF16 does not yet meet the strict full-parameter-gradient contract, despite exact tested forward/sample causality. Do not claim all production tests passed. A next study could retainFP32 while separately investigating mixed-precision context-gradient accumulation; do not automatically train or adopt a precision change. No compile/alternative-kernel sweep, driving quality, real vehicle feasibility or full joint-flow distribution claim. Fixed causal attention is a structural change; it must be evaluated separately from adding partial labels.
