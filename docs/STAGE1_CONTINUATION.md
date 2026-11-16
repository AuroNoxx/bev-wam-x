# 第一阶段继续训练：物理一致性与轨迹指标

2026-11-15。当前只推进阶段一，不启动世界模型、控制器或 CARLA 闭环。

本轮已执行完成：15 轮正常早停，最佳第 5 轮，整体改善但急刹车退步；完整配对验证见 [STAGE1_V2_RESULTS.md](STAGE1_V2_RESULTS.md)。以下保留可复现的实验配置，不表示训练仍在运行。

## 已测到的问题与证据边界

首版训练在第 34 轮 / 22508 步正常早停。最佳 FM 验证 loss 权重在第 26 轮 / 17212 步，完整保留在 `artifacts/lightning/flow/best.ckpt`，不覆盖。

同一最佳权重在 1124 个验证窗口、每次 8 个候选、16 个 Euler 步、5 秒完整时距上的测量：候选平均 ADE 3.55187 m，oracle minADE8 1.76598 m，平均 FDE 7.09623 m。oracle 选择依赖未来真实答案，不能用于部署。

预测位置差分与预测速度的平均差为 2.01441 m/s，真实数据对应差约 0.17834 m/s。新增向量运动学残差将位置差分与相邻时刻速度/朝向的梯形积分比较，并包含当前位置、速度和朝向边界；预测残差 2.37360 m/s，标注本身约 0.11759 m/s。两种残差口径不同，不混用。

打乱 OCC 后候选平均 ADE 4.16604 m，打乱导航后 5.61736 m，说明模型使用了两类条件；这种输入扰动不是重新训练的因果消融。16→64 步采样未改善平均 ADE，但对运动学残差有改善，不能将所有误差归结为采样精度。训练只有 42 条 Town12 路线，窗口大量重叠，当前 OCC 前方约 25.6 m，约 30% 的未来轨迹点超出该范围。不能由这些事实直接断言唯一根因，也不能把“世界模型没训练”当成生成轨迹质量差的解释。

原标准 Flow Matching 并不是错误算法；它优化归一化速度场误差，与米级轨迹误差和可执行性不是同一个指标。旧报告中“纵向权重 1/25”仅描述归一化误差的尺度，不足以证明实际误差的因果来源。

## 本次实验

- 保留原模型结构、全部数据、split、图像外的 OCC 输入、完整 25 点 / 5 秒轨迹，以及标准 FM 主目标。
- 在流时间 t≥0.5 的局部端点估计上加入：以米等权的 XY Huber 误差、位置/朝向/速度一致性、单位朝向与非负速度软约束。低 t 的端点存在较强多模态不确定性，因此不强制其辅助目标。
- 端点估计为 `x_t + (1-t) v_theta`，不是完整采样轨迹；最终效果必须由完整采样验证。运动学项是数值一致性约束，不是已标定的轮胎/车辆动力学，也没有假装完成控制器可行性检查。
- 固定权重 FM 1、位置 0.2、一致性 0.1、朝向 0.02、负速度 0.02；位置尺度 5 m、速度尺度 2 m/s。这是首个可复现实验配置，不宣称最优。
- 从首版 best **仅初始化模型权重**，新建优化器、调度器和本轮步数；不是恢复旧目标的 optimizer。学习率 5e-5，两轮 warmup 后余弦衰减。配置 `configs/lightning-stage1-v2.json`。
- 每轮完整训练 5296 窗口（batch 8，662 updates），完整验证 1124 窗口。保留测试集 2485 窗口不参与选模。
- 最佳模型按 `trajectory/meanADE8_m` 最小值选择；早停同指标，至少 15 轮，连续 10 轮未改善 0.01 m 时停止，60 轮上限沿用原配置。
- 每轮同时记录 oracle ADE/FDE、平均 ADE/FDE、横纵误差、速度误差、运动学残差、候选分散程度、六类场景误差。不能仅靠压缩候选分布或牺牲某类场景宣称改进；最终与首版逐项对比。

## 测速、检查与运行

100 个计时训练 step、10 个 warmup：0.05052 s/step，158.34 samples/s，PyTorch 峰值 allocated 0.517 GiB / reserved 0.715 GiB。完整独立验证约 17.90 s，包含首次启动开销；预计每轮约 51 s 加 checkpoint 开销，正式整轮结果优先。

仍使用 Lightning 2.6.6、ROCm PyTorch、BF16、原生 SDPA、fused AdamW、channels-last、2 个 DataLoader worker。完整模型 compile 此前实测较慢，保持关闭。没有安装或修改全局依赖。

检查入口：`test_stage1_objective.py` 校验匀速/加速轨迹、一米 XY 误差、梯度；`check_stage1_v2.py` 对真实 OCC 校验编码器/导航/输出梯度和权重更新；`test_lightning_resume.py` 校验轮内恢复、优化器和调度器，以及保存失败保留原断点。检查与测速权重不作为正式起点。

```powershell
# 本轮首次启动：从首版 best 的权重开始
.\scripts\Start-Stage1.ps1
# 中断后恢复本轮：恢复模型/优化器/采样器/RNG/步数
.\scripts\Start-Stage1.ps1 -Resume
# 本地监控
.\.venv-train\Scripts\python.exe -m tensorboard.main --logdir logs/tensorboard/v1 --host 127.0.0.1 --port 6006
```

TensorBoard `http://127.0.0.1:6006/`，run `lightning-flow/stage1-consistency-v2`。基线评估为后缀 `-assessment` 的运行，测速为 `stage1-v2`，不混作正式训练。

新断点 `artifacts/lightning/flow-v2/latest.ckpt` / `best.ckpt`，每 50 步及每轮更新 latest，只保留这两个逻辑断点。每个断点保存并校验训练设置和数据哈希，目标或设置变更不能伪装成无缝恢复。状态 `artifacts/lightning/flow-v2/status.json`，启动器 `logs/stage1-v2/stage1-launch.json`，完整日志 `logs/lightning/flow-v2.log`。结果以实际验证为准；完成本轮不自动推进阶段二。
