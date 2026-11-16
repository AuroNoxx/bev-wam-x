# Non-LLM CNN + World/Flow baseline v2

Scope: independent non-LLM visual baseline, not a multi-head ablation. Uses exact Qwen v2 selection (336 training windows from Town12,72 validation windows from Town13;route-disjoint;test untouched). Four past/current front RGB frames and state20 only enter actor; future image summaries are training targets, never actor inputs at deployment.

CNN randomly initialized and jointly trained:3 convolution layers(3->32->64->128),strides2,adaptive2x2 pooling per frame,16 tokens with128->2048 projection. Same128width/2layers/4heads/512FFN joint Flow head.1,037,154trainable parameters. Frames initially448x252 then112x64 internally. World target is future4frames each4x4RGB,16x12 tokens projected by fixed seeded12x32 Gaussian matrix,train-only standardization; this differs from Qwen's frozen future latent target. Thus comparison is not a strict single-variable LLM ablation.

Lightning2048steps,batch32,seed9101,AdamWlr1e-4wd0.01,gradientclip1,FP32,SDPA MATH.16Euler steps,seed9188fixed candidate,no best-of-K. Last checkpoint retained,not validation-selected. TensorBoard local only. No claim of driving safety from offline error.

Scripts: scripts/train_cnn_flow_baseline_v2.py (--cache-only for CPU preparation), scripts/cnn_carla_worker_v2.py (stdin images4paths,state20;stdout ready,xy25x2,velocity25x2,model_seconds). Artifacts: artifacts/cnn-flow-pilot-v2. Training completed and results.json contains actual values.

## Completed run

2048steps completed in75.726s (cachedRGB training, excludes preparation); peak allocated GPU0.236GiB; finalbatchloss0.196695. Validation72windows: ADE3.6363m, endpoint1/2/3/5s1.0275/2.6007/4.2520/7.6709m, speedMAE1.7371m/s, meanjerk35.2517m/s3. CV ADE4.5527m. World MSE0.6730 is specific to this RGB-summary target and not numerically comparable with Qwen futurelatent MSE.

latest.ckpt is final2048step and not best-selected. worker_smoke.jsonl validates ready/finite trajectory output on four recorded current/past CARLA frames. Cold model timing8.874s, no warm-latency claim. This recorded static input produced first vx approximately-0.071m/s, so offline ADE improvement does not imply starting or driving ability. Actual CARLA comparison is separately owned by coordinating worker; no simulator was launched here.
