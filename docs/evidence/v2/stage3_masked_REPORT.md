# Stage 3 v3 — masked future patch completion

日期：2026-10-24T01:15:36.541689-07:00；gate = **False**。本次执行仅 Stage 3。

## 配置与判定口径

```json
{
  "seed": 20261017,
  "epochs": 20,
  "batch_size": 64,
  "learning_rate": 0.0003,
  "weight_decay": 0.0001,
  "workers": 2,
  "cpu_threads": 6,
  "mixed_precision": "bf16",
  "encoder_dim": 256,
  "encoder_layers": 6,
  "encoder_heads": 4,
  "encoder_mlp": 1280,
  "patch_stride": 4,
  "predictor_dim": 384,
  "predictor_layers": 6,
  "predictor_heads": 6,
  "predictor_mlp": 1536,
  "lambda_sim": 25.0,
  "lambda_var": 25.0,
  "lambda_cov": 1.0,
  "variance_gamma": 1.0,
  "variance_epsilon": 0.0001,
  "gradient_clip": 5.0,
  "validation_pairs": 4096,
  "bootstrap_samples": 2000,
  "rank_threshold": 0.75,
  "stable_epochs": 3,
  "minimum_prediction_gain": 0.05,
  "visualization_interval_steps": 400,
  "visualization_sample_ids": [
    0,
    512,
    2048,
    4096
  ],
  "visualization_seed": 20261017,
  "sdpa_backend": "AUTO",
  "residual_target": true,
  "residual_horizons": [
    0,
    1
  ],
  "dynamic_loss_weight": 2.0,
  "masked_completion_weight": 0.5,
  "mask_ratio": 0.25,
  "masked_horizons": []
}
```

SDPA = AUTO；operator flags = `{"cuda_matmul_allow_tf32": false, "cudnn_allow_tf32": false, "cudnn_benchmark": false, "cudnn_deterministic": true}`。
PyTorch 2.9.1+rocm7.2.1.gitff65f5bc / ROCm 7.2.53211-e1a6bc5663 / Radeon RX 7900 XT。
不使用 torch.compile、外部 flash-attn/xformers/Triton 或外部数据/权重；AUTO 可使用 PyTorch 自带 ROCm AOTriton 内核。
Flow 保持 hybrid：horizons 0/1 residual，其余 absolute。Completion 重建 absolute EMA targets，仅训练时启用；EMA stop-grad。
Mask 为每 sample/horizon/token 独立 Bernoulli，排除 future_bev_mask=False；actual ratio 的分母为有效且选中的 token。
最终指标用完整验证集与 10-step Euler flow；每 epoch 趋势仅前 32 batches，不能替代最终结果。
门槛：所有时域 E_norm <= 0.85，rank >= 16，var（历史字段实际为 std_mean）>= 0.10。动态子集为变化最大的 25% token。
保留现有 variance penalty（来自 stop-grad EMA target，属于无梯度监测项）；没有改变该历史训练定义。

## 最终完整验证集

| Seed | +0.5s | +1.0s | +1.5s | +2.0s | rank | var/std_mean | gate |
|---|---|---|---|---|---|---|---|
| 20261022 | 1.317618 | 1.139156 | 0.886803 | 0.708992 | 137.980 | 0.959821 | False |
| 20261023 | 1.336872 | 1.160391 | 0.970997 | 0.777577 | 157.863 | 0.921723 | False |

## Dynamic 子集

| Seed | +0.5s | +1.0s | +1.5s | +2.0s |
|---|---|---|---|---|
| 20261022 | 1.152001 | 0.927195 | 0.639523 | 0.532903 |
| 20261023 | 1.174938 | 0.970174 | 0.721263 | 0.599454 |

## 历史对比（读取原 results.json，未重跑）

| Experiment | Seed | Epochs | +0.5s | +1.0s | +1.5s | +2.0s | gate |
|---|---|---|---|---|---|---|---|
| stage3_001 | 20261022 | 10 | 2.118373 | 1.241628 | 0.984556 | 0.883594 | False |
| stage3_001 | 20261023 | 10 | 2.255713 | 1.293713 | 1.016473 | 0.900552 | False |
| stage3_residual_001 | 20261022 | 10 | 1.393907 | 1.161195 | 1.099396 | 1.093007 | False |
| stage3_residual_001 | 20261023 | 10 | 1.348792 | 1.152524 | 1.106413 | 1.112765 | False |
| stage3_hybrid_001 | 20261022 | 20 | 1.282142 | 1.140440 | 0.957189 | 0.783878 | False |
| stage3_hybrid_001 | 20261023 | 20 | 1.263194 | 1.118994 | 0.938245 | 0.766458 | False |
| stage3_masked_001 | 20261022 | 20 | 1.317618 | 1.139156 | 0.886803 | 0.708992 | False |
| stage3_masked_001 | 20261023 | 20 | 1.336872 | 1.160391 | 0.970997 | 0.777577 | False |

旧 hybrid 的完整验证 +1.5s 为约 0.95/0.94，高于 0.85；此前‘+1.5s 已过’仅符合前 32 batches 的 epoch 指标，不能用于最终 gate。
各实验 EMA teacher 随训练变化，E_norm 分母也变化；历史对比不等于固定 teacher 的严格因果消融。新增 mask 抽样改变 RNG 消耗；Rerun 抽样使用独立 RNG，不扰动训练。

## 每 epoch 趋势（32 batches）

### Seed 20261022

| Epoch | flow | mask | +0.5s | +1.0s | +1.5s | +2.0s | ratio | rank | var | gate | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.54956 | 1.17990 | 12.1045 | 4.6038 | 7.2796 | 4.8420 | 0.24992 | 202.16 | 0.9717 | False | 77.5 |
| 2 | 0.94935 | 0.23581 | 3.9949 | 2.1734 | 3.4732 | 2.3891 | 0.25023 | 149.91 | 1.0587 | False | 74.6 |
| 3 | 0.58699 | 0.09886 | 2.3101 | 1.6292 | 1.7369 | 1.2461 | 0.24973 | 124.33 | 1.1192 | False | 75.0 |
| 4 | 0.46015 | 0.07999 | 1.6912 | 1.4256 | 1.2759 | 0.9485 | 0.24976 | 115.84 | 1.1433 | False | 75.4 |
| 5 | 0.41317 | 0.07410 | 1.6655 | 1.4078 | 1.1168 | 0.8373 | 0.25026 | 114.06 | 1.1464 | False | 75.3 |
| 6 | 0.38826 | 0.07001 | 1.8512 | 1.5559 | 1.1129 | 0.8313 | 0.24989 | 115.29 | 1.1388 | False | 75.5 |
| 7 | 0.37392 | 0.06613 | 1.4730 | 1.2597 | 0.9602 | 0.7328 | 0.24963 | 117.13 | 1.1260 | False | 75.7 |
| 8 | 0.36032 | 0.06236 | 1.3617 | 1.1885 | 0.9276 | 0.7174 | 0.25037 | 119.08 | 1.1093 | False | 75.6 |
| 9 | 0.34692 | 0.05941 | 1.4495 | 1.2590 | 0.9292 | 0.7097 | 0.25023 | 121.14 | 1.0912 | False | 75.6 |
| 10 | 0.33761 | 0.05613 | 1.4306 | 1.2274 | 0.9536 | 0.7292 | 0.24985 | 123.03 | 1.0731 | False | 75.6 |
| 11 | 0.32360 | 0.05345 | 1.6010 | 1.3115 | 0.8839 | 0.6823 | 0.24981 | 124.80 | 1.0544 | False | 75.3 |
| 12 | 0.31517 | 0.05087 | 1.2523 | 1.0823 | 0.8214 | 0.6413 | 0.25024 | 126.31 | 1.0353 | False | 75.6 |
| 13 | 0.30645 | 0.04845 | 1.4996 | 1.2276 | 0.8402 | 0.6471 | 0.24997 | 127.56 | 1.0174 | False | 75.4 |
| 14 | 0.30021 | 0.04642 | 1.3968 | 1.1797 | 0.7992 | 0.6170 | 0.24962 | 128.54 | 1.0014 | False | 77.1 |
| 15 | 0.29420 | 0.04488 | 1.3618 | 1.1513 | 0.8144 | 0.6312 | 0.24945 | 129.38 | 0.9870 | False | 76.2 |
| 16 | 0.28669 | 0.04348 | 1.2725 | 1.1008 | 0.8079 | 0.6268 | 0.25000 | 130.02 | 0.9751 | False | 75.1 |
| 17 | 0.28176 | 0.04247 | 1.2734 | 1.0971 | 0.7837 | 0.6115 | 0.24981 | 130.47 | 0.9652 | False | 75.6 |
| 18 | 0.28155 | 0.04181 | 1.3088 | 1.1238 | 0.7860 | 0.6138 | 0.24982 | 130.80 | 0.9576 | False | 75.6 |
| 19 | 0.27815 | 0.04148 | 1.2979 | 1.1128 | 0.7802 | 0.6102 | 0.24990 | 130.99 | 0.9525 | False | 75.6 |
| 20 | 0.28360 | 0.04132 | 1.2931 | 1.1075 | 0.7805 | 0.6080 | 0.24994 | 131.12 | 0.9495 | False | 75.6 |

### Seed 20261023

| Epoch | flow | mask | +0.5s | +1.0s | +1.5s | +2.0s | ratio | rank | var | gate | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.53881 | 1.13120 | 11.9751 | 4.5463 | 7.4974 | 4.9946 | 0.25024 | 205.48 | 0.9812 | False | 75.4 |
| 2 | 0.93647 | 0.21515 | 4.5029 | 2.3517 | 3.4392 | 2.3534 | 0.24987 | 167.79 | 1.0496 | False | 75.8 |
| 3 | 0.56321 | 0.08714 | 2.1575 | 1.5698 | 1.7471 | 1.2479 | 0.24967 | 148.59 | 1.0874 | False | 74.8 |
| 4 | 0.44114 | 0.06983 | 1.7994 | 1.4834 | 1.2968 | 0.9648 | 0.24986 | 140.44 | 1.1026 | False | 74.9 |
| 5 | 0.40170 | 0.06467 | 1.4060 | 1.2270 | 1.1190 | 0.8609 | 0.24991 | 138.04 | 1.1033 | False | 75.4 |
| 6 | 0.37604 | 0.06120 | 1.7607 | 1.4523 | 1.1432 | 0.8533 | 0.24974 | 138.16 | 1.0935 | False | 75.3 |
| 7 | 0.36015 | 0.05836 | 1.4621 | 1.2389 | 1.0048 | 0.7781 | 0.24971 | 139.05 | 1.0811 | False | 75.3 |
| 8 | 0.34292 | 0.05537 | 1.4355 | 1.2183 | 0.9647 | 0.7490 | 0.25018 | 140.49 | 1.0654 | False | 74.8 |
| 9 | 0.33316 | 0.05245 | 1.4485 | 1.2392 | 0.9574 | 0.7444 | 0.24965 | 142.00 | 1.0467 | False | 74.8 |
| 10 | 0.31622 | 0.05007 | 1.3451 | 1.1798 | 1.0316 | 0.7955 | 0.24976 | 143.47 | 1.0272 | False | 75.0 |
| 11 | 0.31094 | 0.04786 | 1.4846 | 1.2415 | 0.9449 | 0.7453 | 0.25011 | 144.75 | 1.0081 | False | 74.9 |
| 12 | 0.30332 | 0.04578 | 1.3812 | 1.1663 | 0.8736 | 0.6906 | 0.24969 | 145.90 | 0.9898 | False | 74.6 |
| 13 | 0.29420 | 0.04400 | 1.3511 | 1.1479 | 0.9021 | 0.7051 | 0.25006 | 146.84 | 0.9733 | False | 74.6 |
| 14 | 0.28499 | 0.04251 | 1.3679 | 1.1525 | 0.8565 | 0.6778 | 0.25018 | 147.67 | 0.9578 | False | 74.7 |
| 15 | 0.27894 | 0.04109 | 1.2642 | 1.1008 | 0.8506 | 0.6742 | 0.25025 | 148.25 | 0.9437 | False | 74.7 |
| 16 | 0.27732 | 0.04017 | 1.2992 | 1.1047 | 0.8379 | 0.6584 | 0.25000 | 148.75 | 0.9313 | False | 74.4 |
| 17 | 0.26700 | 0.03938 | 1.3242 | 1.1279 | 0.8490 | 0.6665 | 0.24988 | 149.02 | 0.9214 | False | 74.5 |
| 18 | 0.26808 | 0.03886 | 1.3204 | 1.1279 | 0.8529 | 0.6681 | 0.24998 | 149.28 | 0.9136 | False | 74.7 |
| 19 | 0.26732 | 0.03855 | 1.3336 | 1.1341 | 0.8538 | 0.6668 | 0.24967 | 149.43 | 0.9084 | False | 74.6 |
| 20 | 0.26429 | 0.03854 | 1.3097 | 1.1241 | 0.8530 | 0.6681 | 0.25002 | 149.52 | 0.9052 | False | 74.5 |

## 结论与下一步

Gate **False**；最差时域 +0.5s，seed 20261023，E_norm=1.336872。
若不通过：保留负结果，先做数据覆盖与采集器审计（短时域动态变化、时空对齐、有效 future 覆盖），再决定是否采集；不要继续盲目叠加损失。Completion loss 降低不证明 history-only flow 生成改善。
目标/损失方向仅作为待验证假设：可见 future 辅助任务与无 future 的生成任务可能存在差异；本实验不能单独认定数据或目标为根因。若没有明确覆盖或目标缺陷，接受本轮负结果。
本次停止于 Stage 3 报告；未启动 Stage 4/5–9。

## 复现与交付

```bash
/home/auronox/Desktop/mini-wa-jepa/.venv/bin/python scripts/run_bev_wa_stage3.py --output-dir artifacts/bev_wa_alignment/stage3_masked_001 --residual --residual-horizons 0 1 --dynamic-loss-weight 2.0 --masked-completion-weight 0.5 --mask-ratio 0.25 --epochs 20 --batch-size 64 --rerun-save
```

源码 commit：`58fedae458021bb37212442868288f95947288a8`（dirty 工作区；逐文件 hash 见 run_manifest.json）。
GOAL SHA256：`513ce800f086557e63d9bd95a6b84e47cd8b9fe5c85c186d33fdbce0b552f6b0`。
总耗时：3089.2s。TensorBoard：`/home/auronox/Desktop/mini-wa-jepa/artifacts/bev_wa_alignment/stage3_masked_001/tensorboard`。Rerun：`/home/auronox/Desktop/mini-wa-jepa/artifacts/bev_wa_alignment/stage3_masked_001/rerun`。
SDPA fidelity：`{"candidate_backend": "AUTO", "max_abs_error": 0.00390625, "relative_l2_error": 0.0021475653629750013, "is_allclose": false, "shapes": [16, 4, 64, 64], "dtype": "torch.bfloat16"}`。

- Seed 20261022: checkpoint `/home/auronox/Desktop/mini-wa-jepa/artifacts/bev_wa_alignment/stage3_masked_001/flow_latent_seed_20261022.pt`；耗时 1551.5s；mask_ratio_actual 0.249921；峰值 allocated VRAM 8.15 GiB。
  Epoch JSONL：`epochs_seed_20261022.jsonl`；RRD：`rerun/stage3_seed_20261022.rrd`；原始可视化数组/图：`figures/seed_20261022/`。
- Seed 20261023: checkpoint `/home/auronox/Desktop/mini-wa-jepa/artifacts/bev_wa_alignment/stage3_masked_001/flow_latent_seed_20261023.pt`；耗时 1535.4s；mask_ratio_actual 0.249921；峰值 allocated VRAM 8.14 GiB。
  Epoch JSONL：`epochs_seed_20261023.jsonl`；RRD：`rerun/stage3_seed_20261023.rrd`；原始可视化数组/图：`figures/seed_20261023/`。
