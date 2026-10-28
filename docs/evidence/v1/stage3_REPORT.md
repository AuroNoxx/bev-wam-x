# 阶段 3：确定性单步潜变量预测与 VICReg

正式运行：`vicreg_001`；已执行 12/12 epochs；运行完成。
**阶段 4 未启动。验收指标来自原始潜变量，不在白化/PCA 后计算。**

编码器 6,213,952 参数，预测器 11,017,600，总计 17,231,552。
训练相邻对 99,595；固定验证对 4,096，预测时间间隔 0.1s。
按预定验证规则选择 epoch 10；训练峰值分配显存 1.902 GiB。

## 验收结论

- PASS: full_run
- PASS: rank_stability
- PASS: test_rank
- PASS: prediction_gain
- PASS: significant_gain
- PASS: training_memory

只有门槛全部满足且用户确认后才能进入下一阶段；报告完成不等同于门槛通过。

| 指标 | 选中验证 checkpoint | 完整测试 |
|---|---:|---:|
| 内容协方差有效秩 / 256 | 0.929246 | 0.853944 |
| 整帧池化有效秩 / 256 | 0.038284 | 0.032935 |
| 通道标准差均值 | 1.003031 | 0.984727 |
| 通道标准差 P10 | 0.991610 | 0.971470 |
| 预测 MSE | 0.000783 | 0.000486 |
| 复制当前帧 MSE | 0.002858 | 0.002237 |
| 预测/复制 MSE | 0.274064 | 0.217373 |

测试相邻对 9,978；episode 数 22。
按 episode 成簇 bootstrap 的配对 MSE 差 95% 区间：[-0.0019481902657592293, -0.0015619496554501455]；比值区间：[0.17270880175208994, 0.2615102091311718]。

| 测试子集 | 帧对数 | 预测 MSE | 复制 MSE |
|---|---:|---:|---:|
| 全部 | 9978 | 0.000486 | 0.002237 |
| 扰动 | 2667 | 0.000497 | 0.002347 |
| 恢复 | 3510 | 0.000579 | 0.002347 |
| 地图 CSC | 4990 | 0.000710 | 0.002407 |
| 地图 SCS | 4988 | 0.000262 | 0.002067 |

## 口径、限制与可复现性

- 六层 d=256/four-head 编码器；两层 stride-2 Conv 后为 32×32，再以 4×4 patch 投影成 8×8 token；没有缩小输入 BEV 或使用预训练权重。
- 直接对共享编码器的真实 token 做 VICReg，两分支都有梯度；不使用 EMA/额外投影头。预测器只看到当前 BEV、当前速度与从过去计算的角速度。
- 主有效秩来自先按固定空间位置扣除全验证/测试集均值的协方差谱，理论通道上限 256；另报未校正 token rank 和帧池化 rank。浮点稳定性使用 float64 Welford 累积，未通过加噪声或白化抬高秩。
- 256 是协方差矩阵的代数通道维度上限，不是独立 episode 数。token/相邻帧存在相关性，不将池化后的低秩隐藏为“全局表示健康”。
- 严格使用 v2 的同 episode、连续 step 对；扰动/恢复/终止目标均不按动作掩码过滤。未跨 reset、未将未来 ego state 或专家导航作为输入。
- 固定 4096 对验证用于选择 checkpoint；测试不用于调参。管线冒烟的 3-batch 评测单独保留，不计入正式得分。
- 预测目标在训练中可共同变化，因此必须同时审查原始 copy MSE、标准差与有效秩；低 MSE 本身不能证明世界模型有效。
- 95% 区间按 episode 重采样，不把相邻帧当成独立观测；单次 seed 实验仍有局限。主秩门槛须在末三次验证及选中测试 checkpoint 上均满足。
- 周期图/Rerun 中的全验证秩和 PCA 参考来自最近一次完成的验证；图中的预测/目标是该 optimizer step 的真实输出。逐 epoch 曲线和最终 PCA 使用对应 checkpoint 的验证统计。
- 潜变量热图/PCA 是特征可视化，不是预测未来像素。阶段 3 无轨迹头，ADE/FDE、路线完成率、碰撞率和闭环推理延迟均未产出。
- 没有运行旧同进程 Panda3D/Pygame 窗口；阶段 2 的键盘验证与 Rerun 截图退出 panic 原样保留。本阶段离线训练不生成虚构的第三人称 RGB。
- 方法、预先固定的阈值及权重见 [协议](../../docs/STAGE3_PROTOCOL.md)；正式配置与源码快照保存在 artifacts/restart_stage3/formal_protocol.json。

## 证据

- [完整训练/验证/测试记录](vicreg_001/report.json) / [逐步日志](vicreg_001.log) / [TensorBoard](vicreg_001/tensorboard)
- [完整测试原始误差与协方差](vicreg_001/test_raw.npz) / [实际潜变量样本与 Rerun](vicreg_001/samples)
- [固定测试地图样本与实际预测/目标](vicreg_001/selected_test_diagnostics/manifest.json) / [帧对/episode/seed 清单](dataset_pair_inventory.json) / [Rerun 内容验证](rrd_contents_check.txt)
- [有效秩/标准差/预测比曲线](health_curves.png) / [协方差谱](covariance_spectrum.png) / [共享 PCA](latent_pca.png)
- [验收视频](stage3_acceptance.mp4) / [Marimo 指标](metrics.json) / [Marimo 导出](dashboard.html)
- [训练源码快照](formal_training_source.tar.gz) / [ROCm 验证](rocm.json) / [新增测试](unit_tests.log)
- [稳定统计修正前的失败测试](unit_tests_initial_cancellation_failure.log) / [舍入问题最小复现](covariance_cancellation_mre.py) / [复现结果](covariance_cancellation_mre.json)
- [全部 33 项测试](tests_all.log) / [前阶段源码](pre_stage3_source.tar.gz)

