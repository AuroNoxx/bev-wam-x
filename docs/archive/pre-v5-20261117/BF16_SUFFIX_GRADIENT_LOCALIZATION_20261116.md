# BF16 suffix-gradient localization, 2026-11-16

No training, optimizer or checkpoint/default changes. Fixed same two-sample batch, same25-token triangular topology and valid-prefix scoring. Primary scripts localize_bf16_suffix_gradients.py and localize_bf16_exact_graph.py; artifacts bf16-suffix-gradient-localization-v1 and bf16-exact-graph-localization-v1.

## Reproduced original failure, without loosening threshold
Original full-scope BF16 forward/backward repeated3times: suffix perturbation causes maxparameter-gradient difference6.103515625e-5 each time, topencoder.conv.0.weight relativeL2=2.32633e-4 (~0.0233%). Original tolerance1.71875e-6. Same-input repeat max3.7e-9--4.0e-8. Prefix outputs identical and suffix-input gradients exactly0.

Forcing ONLY scalar loss/reconstruction intoFP32 while retaining original BF16forward does NOT solve it: same6.1035e-5 in3/3 runs. Thus this is not simply a low-precision final loss-reduction issue.

## Attention backend isolation
Same original graph, BF16, but explicitly MATH SDPA: maxgradient difference1.1920928955e-7 in3/3 runs, top transformer.layers.0.self_attn.in_proj_weight relativeL2=1.12529e-7; passes unchanged tolerance1.71875e-6. ExplicitFP32loss plusMATH gives same result.

CPU operator-dispatch profiler at the actual model attention shapes confirms default uses aten::_scaled_dot_product_flash_attention and aten::_flash_attention_backward; forcedMATH uses aten::_scaled_dot_product_attention_math. Dispatch evidence saved attention_dispatch.json. It identifies the executed implementation family, not a claim to know the low-level kernel instruction responsible.

## Layer/target diagnostics
Separating context/decoder precision and requesting intermediate gradients changes reduction/execution graph and some combinations pass, so they cannot replace the exact original reproduction. In the instrumentedgraph:
- output gradients and context gradients are identical across suffix perturbations;
- BF16 FM-only residual max3.8147e-6 at transformer.layers.0.linear2.weight (relativeL2~3.21e-6), still over original threshold;
- position-only max~8.15e-10;
- contextFP32/decoderBF16 max1.49e-8;
- contextBF16/decoderFP32 max1.77e-8.
All finite. The observed error is sensitive to mixed-precision backward/reduction path and amplified upstream; it is not evidence of a forward semantic suffix path under the tested causal topology. Do not describe it merely as identical-input nondeterminism: perturbation dependence is reproducible.

## Supported conclusion / limits
Evidence localizes the tested failure to the default BF16 fused-attention/backward numerical path rather than the loss mask or a bidirectional forward route. MATH attention removes the observed above-threshold discrepancy in this case. This is not universal kernel certification: one checkpoint, two contexts, chosen large finite suffix pollution, no full amplitude/data/backend sweep. Kernel internals are not proven. No threshold changed and no training began. Before any future experiment, either keep the already-testedFP32 path or separately validate BF16+MATH across ordinary noise amplitudes/morecontexts and measure its memory/throughput cost; do not silently switch backend and claim performance neutrality.
