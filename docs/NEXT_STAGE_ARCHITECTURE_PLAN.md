# 下一阶段架构诊断与 v2 路线（计划，尚未实现）

记录日期：2026-11-17（本机 UTC+08）。来源：用户本轮提供的诊断、指标与后续路线。
本文件仅归档计划，不修改当前训练，也不表示下述假设已被验证。

## Phase 1：当前运行保护与状态

- 当前入口：`scripts/train_occ_qwen_native_optimized.py`。
- 当前配置：`configs/occ-qwen-native.json`。
- 状态来源：`artifacts/occ-qwen-native-training/status.json`。
- 本次只读快照：phase=training，epoch=14/20，global_step=8750；最新状态中的 val/action_ade_m=5.085054874420166，val/loss=0.1808346062898636。这是状态文件快照，不是训练完成或最终 checkpoint 审计。
- 运行期间不停止、不暂停、不干扰当前训练；不修改 `occ_qwen_native_flow.py`、active data loaders、任何 active configs；不启动竞争 GPU 作业。
- 现阶段仅只读状态、保存此计划。等待既定 20 轮自然完成。
- 完成后先只读审计 `best.ckpt` 和 `latest.ckpt`，核实完整性、实际 epoch/global_step、保存指标及恢复状态，再进入隔离模块的 v2 实现。Phase 2 不在本次执行。

## 证据等级与边界

当前工程已经实现原生 0.4m OCC 输入、冻结 Qwen 文本主干和 Scene/World/Action 联合结构；World 为 latent-only。未来 3D OCC decoder、OccWAM scene-to-ego 第二阶段不是当前实现，不能把本计划当成这些功能的完成声明。
下列历史指标来自用户本次提供，尚未在本次归档任务中重新核验。对原因、因果关系和改善幅度的描述均为诊断假设，不是当前实验已证明的结论。

## 用户提出的诊断（待验证）

### 1. 空间压缩与前视范围

- 输入为 4 帧 `[B,4,20,128,128]`，原生体素 0.4m，XY 范围 `[-25.6m,25.6m)`。
- 128×128 平面压缩为 4×4 token lattice，几何覆盖相当于每 token 12.8m。用户假设这种压缩损失关键细节；不能直接把 token 覆盖范围当作已经测出的检测误差或实际感受野。
- 对称范围的前视只有 25.6m。用户指出这对高速情况下的未来 5s、25 点轨迹覆盖不足。具体速度与可观测性影响需审计，不伪造更远范围观测。

### 2. OCC 与冻结文本主干的模态对齐

- 用户诊断：从零初始化的 3D OCC CNN/projector 直接接全冻结 Qwen text；当前没有 LoRA 或截层。
- 用户提供的历史 `qwen-visual-pilot-v2` matched-split 结果：frozen Qwen + joint flow，val ADE 4.109m、jerk 46.59m/s³；1.03M 参数三层 CNN，ADE 3.636m、jerk 35.25m/s³。
- 当前单全局 LR 1e-4。模态对齐不足是待验证解释，历史比较不能自动证明当前新模型的失败原因。
- 命名澄清：当前输入保留高度维，但实际 encoder 将高度嵌入到 CNN 通道后用 Conv2d，并非已实现原生 Conv3d 网络。

### 3. World 目标与跨流噪声

- World 的 +1s/+2s 目标为原始 latent 回归，latent-only，没有 OCC decoder，也没有残差目标。
- 用户提供的历史结果：World MSE 1.10–1.83，对照 persistence 0.25–0.60。需重新确认归一化、目标编码器、时间和 mask 相同后才可公平比较。
- Scene/World/Action 双向交互可能让早期 World 噪声污染 Action，是假设，尚无本轮隔离因果证据。

### 4. 状态导航与动力学约束

- 用户诊断：state_dim=20，nearFLU/50 状态/导航 token 可能被稀释；表现为横向修正弱、纵向速度平坦。
- 用户提供历史轨迹 jerk >35–46m/s³，认为与轨迹/速度缺乏约束有关；因果待核验。
- 用户提供的 `carla-smoke-v2/v3` 结果：全部候选触发 safety guard（jerk 10m/s³、lateral bound 1.2m），最终全制动。这里归档用户提供的历史记录，不将其视为本次原生 OCC 模型的实测结果。

## Phase 2 前置：训练结束后的审计

先只读最终权重与日志，再在不改变训练权重的评估中统一以下口径；涉及 GPU 的评估必须等本轮训练完成后进行：

1. 核实 best/latest 可读取、参数有限、实际 epoch/global_step、optimizer/scheduler/EMA/RNG/loader 状态及最佳指标来源。
2. 单候选 ADE，不能用 best-of-N/minADE 替代部署候选质量。
3. 1s、2s、3s、5s endpoint error。
4. Speed MAE。
5. 分别报告位置差分推导的 jerk 与速度差分推导的 jerk，注明坐标、dt、单位及聚合方法，避免混淆。
6. World latent MSE 对照同口径 static persistence；统一目标编码器、空间/时间对齐、归一化、已知区域 mask 与样本集。
7. 历史用户提供结果与本轮直接实测结果分别标注，不混合训练轮数、输入、采样步数或候选口径。

## 隔离 v2 路线（尚未实施）

### A. 更细的空间 token

- 采用 8×8 lattice，每帧 64 token，4 帧共 256 token；每 token 几何覆盖 6.4m。
- 保留原生 OCC、真实源数据和已知/未知语义；不通过放大或填充伪造更远范围。
- 明确这改变 token 数与计算量，尚未做显存、速度或收益验证。

### B. 强化 ego/navigation 条件

- 将 20D ego/navigation 通过 AdaLN-Zero 注入 Action Flow decoder，而非只依赖共享 token 交互。
- 具体归一化与零初始化行为在实现时验证，不能只凭名称宣称条件已被有效使用。

### C. 残差 World 与受控耦合

- 目标改为 `Δz = z_future - stop_grad(z_now)`。
- 输出头零初始化，设计目标是初始预测等同 persistence；须核验 flow 参数化、积分路径与解码重建是否确实满足该性质，而非直接假定零初始化就保证。
- 保留 Scene→World、Scene→Action 条件路径；World→Action 使用零初始化 tanh gate，初始隔离 World 噪声，之后可学习打开。
- 仍是 latent 路线；本计划没有自动授权加入未来 3D OCC decoder 或 OccWAM 第二阶段。

### D. 主干适配与优化分组

- Qwen 截取浅/中层，或在 q_proj/v_proj 加 rank=8 LoRA；二者待实现时选择，不在当前 run 上改动，不宣称两者都已决定或已完成。
- AdamW 参数组：OCC CNN/projector LR=2e-4；Action head/adapters LR=1e-4。
- 未明确列出的参数组在隔离实现时显式记录，不悄悄继承不明学习率；保留可复核的训练与恢复契约。

### E. 动力学约束

在实现时选择并明确记录以下路线，而非当前同时启动多组实验：

- 预测有界纵向加速度以及 yaw rate/curvature，通过可微 rollout 得到轨迹；或
- 对速度离散一阶、二阶差分（加速度、jerk）加入 Huber 惩罚。

目标是满足 CARLA guard 的 jerk <10m/s³，但软惩罚不保证硬阈值。必须在统一 dt/单位下测量实际分布与越界率；即便加速度和 yaw rate 有界，也不能据此直接宣称 jerk 必然合格或闭环安全。

## 执行顺序与验收限制

1. 现在仅保存计划、只读监控；不更改任何运行模型/数据加载/配置，不执行 GPU 工作。
2. 等 20 轮自然结束，先做只读最终审计并保留 best/latest。
3. 然后才在隔离模块实现 v2，不覆盖本轮实现、历史基线、数据或权重。
4. 实现时验证各项初始化、因果标签隔离、参数冻结/训练分组、动力学单位、完整断点恢复与资源边界，再报告实际证据。
5. 本文件不是实现完成记录，不预先宣称性能提升、可实时运行、达到 guard 阈值或 CARLA 闭环可靠。

## 用户后续授权补充

用户随后说明即将休息，要求本轮训练结束后继续按其提供的方案实现并训练新版本，并与当前版本采用相同验证条件比较、主动汇报，无需再次等待批准。该授权不改变运行期间的保护规则：本次仍只完成 Phase 1 文档归档；必须先等本轮自然结束并完成最终审计，才能执行隔离 v2 实现与训练。比较时保持同一验证 split/窗口、单候选、相同采样步数及噪声控制、坐标/dt/单位和指标聚合；架构变化与仍不可比的部分单独披露，不用历史不同口径指标冒充公平对照。
