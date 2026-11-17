# Native OCC-Qwen v2 implementation and run record

Status: isolated implementation and verification complete; formal launch pending matched v1 baseline evaluation. This is a new experiment, not an in-place modification of v1. Current authoritative plan: NEXT_STAGE_ARCHITECTURE_PLAN.md. v1final audit: OCC_QWEN_NATIVE_V1_FINAL_AUDIT_20261117.md.

## Implemented choices
- Unchanged native0.4m input,4x20x128x128, same5296train/1124validation windows and full-resolution aligned cache. No synthetic expanded coverage.
-8x8 scene lattice per frame,64tokens/frame,256historical tokens retained. Height is represented as CNN channels, not a Conv3d claim.
- First12 pretrained Qwen text blocks frozen, input gradient still reaches CNN/projectors. Chose layer truncation instead of LoRA, not both. No text generation or VQA training.
- Scene encoded once; independent3-layer256-wide World and Action decoders.20D ego/navigation enters Action through AdaLN-Zero.
- Scene->World, Scene->Action; tanh World->Action gate initialized exactly0.
- World delta target EMA(future)-stopgrad EMA(current). Deterministic zero-source flow w_t=t*delta, target velocity=delta, inference w0=0. Zero final world projection gives EXACT persistence before learning for1/3/8Euler steps. This is not Gaussian-base multimodal World generation and does not add an OCC decoder.
- Action remains Gaussian residual-velocity flow25x2 with dt0.2s integration. Chose acceleration/jerk Huber regularization, not bounded-control rollout. Endpoint estimate a_t+(1-t)*predicted_flow_velocity is converted to physical velocity; first/second velocity differences scaled by3m/s2 and10m/s3 and penalized with Huber delta1. Both weights0.001. These are soft penalties, not a guard guarantee.
- AdamW groups: CNN/projectors2e-4, remaining trainable adapters/heads1e-4, weightdecay0.01,100step warmup and cosine to10% LR. Batch8,20epochs. Native cache + channels_last, workers0, fused disabled. Full checkpoints every100steps and epochend; best by single-candidate ADE.

## Parameters and verification
Total613,389,415; trainable9,290,581; frozen Qwen604,034,048; EMA64,786. First LR group722,962; other8,567,619.

CPU model tests verify exact initial persistence, closed-gate World isolation, finite gradients and learned state sensitivity, frozen backbone, and serialization. CPU Lightning4-step/step2restore spanning epochs is bitwise exact for model/optimizer/scheduler/EMA/RNG/sample sequence and rejects v1schema. Reports in artifacts/occ-qwen-native-v2-tests and occ-qwen-native-v2-cpu-resume-test.

Real pretrained GPU batch8 gate executed4full+2resumed optimizer updates, test-only weights never used as formal initialization. Model maximum resume difference1.164e-10, optimizer3.115e-11, scheduler exact; all within predeclared1e-5relative/1e-7absolute tolerance. Peak allocated3.846GB,reserved4.003GB. Four steps including initialization/save cost16.20s is NOT a steady-state throughput claim. Evidence artifacts/occ-qwen-native-v2-gpu-gate/result.json.

## Matched validation contract
Both versions use the same1124ordered validation windows, batch8,one candidate,8Euler steps,dt0.2s,metric units and aggregation. Each batch uses an independent action source generator seed1907+batch_index, so changes in loss/world tensor shape do not shift Action noise. v1 World remains random-source while v2 World is zero-source by architecture. Re-evaluate v1 best under this common protocol; do not silently equate it to the separately reproduced v1training-validation ADE4.575159m.

World latent MSE is compared against each model's OWN same-encoder/same-lattice persistence. Different latent encoders/lattice mean raw v1-v2 World MSE values are not directly comparable. v1already beat its persistence; residual/gating remains an architectural hypothesis, not a proven fix for World failure.

No reliable CARLA closed-loop behavior or jerk<10m/s3 is claimed. v1best/latest, data, configurations and existing models remain preserved.

## Formal launch
Fresh isolated training launched2026-11-17 05:38:26+08,wrapperPID1044,20epochs/batch8. Live artifacts: artifacts/occ-qwen-native-v2-training/{status.json,stdout.log,stderr.log,launcher.json,source_snapshot.json}. No test checkpoint used to initialize trainable adapters. Common-noise v1baseline evaluated before v2launch: ADE4.46205877m,speedMAE2.16127590m/s (report v1-matched-evaluation.json). This is deliberately separate from v1's original production noise stream/ADE4.575159m. Await v2validation before any performance comparison claim.

## First complete v2 validation
Epoch1/step662: ADE5.0375638m, loss0.19746025, vector jerk mean57.74374m/s3, every trajectory exceeds10 at least once. WorldMSE0.0828653 vs ownpersistence0.1397571. Not yet better than matched v1final ADE4.4620588m; no claim of improved smoothness. First epoch~215.6s before final epoch-save overhead; estimate65-75minutes remaining after enteringepoch2. Actual trainerPID5484,wrapper1044; best/latest verified present(~1.32GB each). Continue the predeclared20epoch run without mid-run model/config edits.


## Completed run and final audit

20epochs/13240steps completed. See [final paired audit](OCC_QWEN_NATIVE_V2_FINAL_AUDIT_20261117.md) for definitive metrics. Best epoch17 ADE3.171468m, jerk8.615677m/s3; latest epoch20 ADE3.831426m, jerk9.827580m/s3. Best ADE improves28.92% against matched v1, but92.794% trajectories still exceed jerk10 and tail jerk worsens. This supersedes the live estimates above. All effective weights and data retained; no v2 cleanup deletion needed.
