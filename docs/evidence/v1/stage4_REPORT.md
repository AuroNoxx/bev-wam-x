# 阶段 4：确定性世界/动作联合学习验收

状态：正式训练、固定测试及两组各 20 次闭环评测完成。总体验收：**通过**。停止于本阶段，等待用户确认；没有实施阶段 5。

## 实测比较

| 模型 | 测试 ADE / FDE (m) | 闭环路线完成率 | 成功 / 碰撞 / 出界率 | 横向误差 (m) | 完整管线 P95 (ms) |
|---|---|---|---|---|---|
| joint | 0.130488 / 0.223658 | 0.969774 | 95% / 5% / 0% | 0.156093 | 14.396 |
| action_only | 0.130400 / 0.225772 | 0.297146 | 25% / 0% / 0% | 0.100434 | 10.601 |
| resnet | 0.179834 / 0.273087 | 0.985050 | 100% / 0% / 0% | 0.140896 | 11.269 |
| constant_turn | 0.714692 / 1.731936 | 未评测 | 未评测 | 未评测 | 未评测 |

开环使用完全相同的 9600 个完整轨迹锚点，来自 22 个 episode。ResNet 冻结权重的重算结果复现阶段 2 分数。

## 门槛

| 指标 | 通过 |
|---|---|
| formal_training | True |
| matched_cohort | True |
| ade_below_resnet | True |
| fde_below_resnet | True |
| closed_route_retention | True |
| full_pipeline_p95 | True |
| parameters_15_to_30m | True |

路线下限 0.886544680；完整管线 P95 上限 25ms；参数 22,372,200。
硬件具备 20GB 显存，且数据集采用 mmap 机制按需加载，无需人为设置 14GB 限制。显存峰值门槛已按指示移除。实测两组训练 PyTorch allocator 最大峰值为 10.1779 GiB，进程 DRM 最大采样值为 14.2830 GB（13.3021 GiB），完全处于 20GB 硬件预算内。显存 allocator、reserved、进程 DRM 及 RSS 原始测量值均完整保留备查。

## 表示健康与差异的不确定性

- joint：选中 epoch 12；内容 rank/D 0.853002；池化 rank/D 0.033891；std 0.972474；world/copy MSE 比 0.272902。
- action_only：选中 epoch 12；内容 rank/D 0.743539；池化 rank/D 0.046700；std 0.735551；world/copy MSE 比 3.876366。
- joint_minus_resnet ade_m：按 episode 成组重采样的差值 95% 区间 [-0.05577884424861896, -0.0436583329765163]。负值表示联合模型误差更小。
- joint_minus_resnet fde_m：按 episode 成组重采样的差值 95% 区间 [-0.05948042346645234, -0.04011388155025977]。负值表示联合模型误差更小。
- joint_minus_action_only ade_m：按 episode 成组重采样的差值 95% 区间 [-0.0006642962342494182, 0.0009148825304220599]。负值表示联合模型误差更小。
- joint_minus_action_only fde_m：按 episode 成组重采样的差值 95% 区间 [-0.004447885978165011, -8.588778613517439e-05]。负值表示联合模型误差更小。
- joint_minus_constant_turn ade_m：按 episode 成组重采样的差值 95% 区间 [-0.6214112961817528, -0.5509236805279397]。负值表示联合模型误差更小。
- joint_minus_constant_turn fde_m：按 episode 成组重采样的差值 95% 区间 [-1.6080680300758445, -1.4190608398454962]。负值表示联合模型误差更小。

## 方法与限制

- 两组各 12 轮、batch 128、相同 seed/初始模型 SHA256/批次顺序；只按固定 4096 验证锚点的 ADE 选权重。冻结后才评估测试集，没有根据测试结果重训。
- 输入为过去四帧及可观测速度/角速度；保留全部空间 token。输出四帧未来潜变量和十个 0.2s 路点。启动时重复首帧，reset 清空缓存；没有导航输入或专家接管。
- action_only 只在当前微调中关闭世界预测/方差/协方差损失，两组都保留本项目阶段 3 初始化。该对照不等于删除全部世界模型预训练。
- 与 ResNet 相比，同时改变了历史输入、结构、参数量及恒速转弯先验，因此不能把 ResNet 差距全部归因于世界损失。constant_turn 为相同运动先验，无拟合参数。
- 仅一次训练种子；按 episode 的 bootstrap 只描述本次样本的不确定性。门槛按预先固定点估计判断；需结合 joint−action_only 区间审查联合目标的作用。
- 世界 MSE 的目标编码器会随训练改变；必须结合标准差、内容秩、池化秩及 copy 对照阅读。低潜变量 MSE 不能单独证明驾驶能力。
- 闭环使用相同 20 种子、dt=0.02s 及控制器，决策与完整管线分别计时；排除每 episode 前20步、reset和可视化/磁盘。过程中没有并发 GPU 训练。
- 闭环录像的橙色轨迹和潜变量目标来自同一次策略运行随后实际观测；尾部缺失保持掩码。潜变量热图是特征，不是未来像素。
- 既有同进程 Panda3D/Pygame 故障和 Rerun 截图后退出 panic 原样保留。阶段 4 的录制没有验证这些故障已修复，也没有把 Rerun 时间轴当作车辆键盘控制。

## 证据

- [固定协议](../../docs/STAGE4_PROTOCOL.md) / [冻结配置和源码](formal_protocol.json) / [门槛原始值](acceptance.json)
- [联合训练](joint_001/report.json) / [动作对照训练](action_only_001/report.json) / 两组目录内 TensorBoard、validation NPZ、周期 samples PNG/NPZ/RRD
- [完整测试及置信区间](test_eval/report.json) / 同目录每锚点原始误差与预测 / [分组与相同 cohort](test_eval/paired_cohort.npz)
- [联合闭环](joint_closed_loop/report.json) / [动作对照闭环](action_only_closed_loop/report.json) / 各目录原始 latency NPZ、实际目标 NPZ、RRD 和视频路径
- [训练比较](training_comparison.png) / [开环比较](open_loop_comparison.png) / [闭环比较](closed_loop_comparison.png)
- [阶段视频](stage4_acceptance.mp4) / [Marimo 指标](metrics.json) / [Marimo 导出](dashboard.html)
- [41 项测试](tests_all.log) / [ROCm 验证](rocm.json) / [训练源码快照](formal_training_source.tar.gz) / [结果清单](results_manifest.json)
