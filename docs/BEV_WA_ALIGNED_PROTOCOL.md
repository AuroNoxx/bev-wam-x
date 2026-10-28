# BEV WA-JEPA 方法对齐执行协议 (BEV WA-JEPA Aligned Protocol v1.0)

**生效日期：** 2026-10-22（用户时区）  
**状态：** 阶段0冻结协议（Frozen Execution Contract）  
**基线 HEAD：** `58fedae458021bb37212442868288f95947288a8`  
**前置文档：** `docs/GEMINI_BEV_WA_JEPA_EXECUTION_PROMPT.md`、`docs/BEV_WA_JEPA_ALIGNMENT_REPORT.md`、`docs/BEV_WA_JEPA_CLEANUP_GUIDE.md`  
**历史目标快照：** `docs/legacy/GOAL_before_bev_wa_jepa_alignment_20261022.md`、`docs/legacy/AGENTS_before_bev_wa_jepa_alignment_20261022.md`

---

## 0. 核心定位与路线变更说明

本项目借鉴 WA-JEPA（World-Action Joint Embedding Predictive Architecture）的世界—动作联合建模思想，将视觉输入替换为自车语义 BEV（鸟瞰图），独立实现端到端闭环驾驶。

### 0.1 路线核心变更
1. **主架构演进：** 彻底停止以“Flow候选生成 → 独立后置75M WAM打分 → 风险硬门控截断”作为主路线。转向在**统一共享预测器内部迭代交互生成「未来场景latent」与「驾驶轨迹」**的联合世界—动作流匹配（Joint World-Action Flow Matching）架构。
2. **渐进式重构：**
   - **阶段1：** 保留环境渲染、四通道BEV标准与底层控制，增量回归控制接线与时间对齐。
   - **阶段2：** 盘点并复用有效原始数据，构建覆盖2.0s的未来BEV与轨迹配对样本，重训公平基线。
   - **阶段3：** 重建未来潜空间预测预训练（升级为多时刻长期未来Flow生成，引入EMA目标教师）。
   - **阶段4：** 核心重做联合世界—动作生成模型，显式定义非对称梯度与内部交互，设立四臂公平对照。
   - **阶段5：** 复用退化评估框架，对新模型与基线重新进行开闭环压力测试。
   - **阶段6：** 重新定义为联合模型“生成动作与预测未来”的因果后果一致性诊断，取消强制串接独立大WAM。
   - **阶段7：** 重建主推理链，由联合模型直接输出可执行轨迹，完成导航、跟车、超车回归、弯道控速与恢复的闭环驾驶。
   - **阶段8/9：** 严格未启动，须经显式用户授权。
3. **独立实现与科学纪律：** 不使用 WA-JEPA 原版代码、预训练权重或克隆库；坚持 ROCm / PyTorch 原生 functional SDPA；保留所有历史失败证据与对照实验；每阶段提交报告后主动停止，等待用户明确指令。

---

## 1. 时空对齐、坐标系统与数据契约

### 1.1 时间轴定义 (Time Horizon)
控制频率固定为 **10Hz** ($dt = 0.1\text{s}$)。模型面向未来 **2.0秒** 的驾驶规划时域：
- **历史观测时域：** 4个历史时刻，包含当前步及过去0.3秒：
  $$t - 0.3\text{s}\;(\text{lag}-3),\quad t - 0.2\text{s}\;(\text{lag}-2),\quad t - 0.1\text{s}\;(\text{lag}-1),\quad t\;(\text{current})$$
- **未来场景 BEV 目标：** 4个离散时刻，步长 0.5 秒（即 $dt_{bev} = 5$ 步）：
  $$t + 0.5\text{s}\;(\text{step}+5),\quad t + 1.0\text{s}\;(\text{step}+10),\quad t + 1.5\text{s}\;(\text{step}+15),\quad t + 2.0\text{s}\;(\text{step}+20)$$
- **未来驾驶轨迹：** 10个连续路点，步长 0.2 秒（即 $dt_{traj} = 2$ 步）：
  $$t + 0.2\text{s}, t + 0.4\text{s}, \dots, t + 2.0\text{s}\quad (10 \text{ waypoints})$$
- **时间掩码与边界：**
  - 采集器记录 $N$ 帧 BEV 与 $N+1$ 个状态。
  - 对于靠近终止边界的样本（$t > N - 20$），未来帧可能缺失。严格采用布尔掩码 `future_bev_mask: (4,) bool` 与 `future_traj_mask: (10,) bool`。
  - 严禁向未来复制当前帧或跨越 Episode 边界伪造未来数据。
  - 历史时刻对于 $t < 3$ 的起始步，采用复制第 0 步填充并置 `history_valid: (4,) bool` 掩码。

### 1.2 坐标系与真值隔离 (Coordinate Frames & Isolation)
- **输入 BEV 坐标系：** $128 \times 128$，四通道 CHW uint8，覆盖 $64\text{m} \times 64\text{m}$。自车固定居中，车头朝上（Ego-centered Heading-up）。
- **未来 BEV 目标坐标系：** 各未来时刻的原始观测 BEV，即各个未来时刻自车居中朝上的局部视点。模型必须学习由于自车运动带来的视点变化隐式转换。
- **未来轨迹坐标系：** 固定在当前时刻 $t$ 的自车坐标系（以 $t$ 时刻自车后轴中心为原点，前向为 $+x$，左向为 $+y$，逆时针航向角）。
- **真值严格隔离：** 仿真器内部交通参与者全局位姿、高精地图几何与未来真值仅用于离线标签构造与评估，严禁注入模型推理输入。

### 1.3 观测输入字典 (Observation Profile)
推理阶段模型仅接收以下合法观测：
1. `bev`: `[B, 4, 4, 128, 128]` uint8，连续4帧语义BEV。
2. `ego_motion`: `[B, 4, 2]` float32，历史4步实测线速度与角速度 $[v_x / 10.0, \omega_z]$。
3. `applied_action`: `[B, 4, 2]` float32，历史4步底盘实际执行的动作 $[\text{steering}, \text{throttle}]$。
4. `route_points`: `[B, 8, 2]` float32，导航路径点（自车局部系，归一化除以 40.0）。
5. `route_valid`: `[B, 8]` bool，导航点有效掩码。
6. `vehicle_params`: `[B, 4]` float32，自车物理尺寸与参数 $[\text{length}/5., \text{width}/2., \text{wheelbase}/3., \text{max\_steer}/1.]$。
7. `maneuver_allowed`: `[B, 2]` bool，静态车道许可 $[\text{allow\_left}, \text{allow\_right}]$。

---

## 2. 候选模型架构与系统预算

### 2.1 整体参数与延迟预算 (Single Orin-X 对标)
- **部署参数总量硬上限：** $\le 100\text{M}$ 参数。
- **推荐目标范围：** **60M–80M** 参数。
  - 共享 BEV 时空编码器：~10M–15M
  - 联合世界—动作预测器：~50M–65M
  - 轨迹输出与几何层：~1M–2M
- **训练专用模块（不计入部署参数）：**
  - EMA 目标教师编码器（参数量与编码器完全相同，评估模式且不产生反向梯度）。
- **实时推理延迟目标：**
  - 10Hz 闭环控制下，完整端到端决策耗时 **P95 $\le 50.0\text{ms}$**。
  - P95 统计范围必须涵盖：BEV uint8 转张量预处理、编码器前向、Flow ODE 多步积分采样、运动学合法性检查与控制器追踪计算。严禁使用仅测量模型头的“head-only”虚假延迟。

### 2.2 核心模块设计
```
[History BEV (4, 4, 128, 128)] 
           │
           ▼
┌─────────────────────────┐
│ Online BEV ST-Encoder   │ ──(EMA Update τ=0.996)──► ┌─────────────────────────┐
│ (Stem + Transformer)    │                           │ EMA Target BEV Encoder  │
└─────────────────────────┘                           └─────────────────────────┘
           │                                                       │ (stop-grad)
           ▼                                                       ▼
   [History Tokens H]                                      [Target Latent Y_scene]
           │                                                       ▲
           │                                                       │
           ▼ (Conditioning)                                        │ Flow Match Loss
┌────────────────────────────────────────────────────────┐         │ L_scene
│ Joint World-Action Predictor                           │ ────────┘
│ (Joint Transformer with Interactive Attention)         │
│                                                        │
│ Inputs:                                                │
│  - Noisy Scene Tokens:  Z_scene(s)                     │
│  - Noisy Traj Tokens:   Z_traj(s)                      │
│  - Condition: [H, ego, route, vehicle, permissions]    │
│  - Flow Step: s in [0, 1]                              │
│                                                        │
│ Outputs:                                               │
│  - Predicted Scene Velocity:  V_scene                  │
│  - Predicted Traj Velocity:   V_traj                   │
└────────────────────────────────────────────────────────┘
                                   │
                                   ▼ Flow Match Loss L_traj
                         [Target Trajectory Waypoints]
```

1. **BEV 时空编码器 (Online BEV Spatiotemporal Encoder)：**
   - 卷积 Stem 将各 128×128 BEV 图降采样至 $16 \times 16 = 256$ 个空间 Token（或通过 Patch Embedding 映射为 $8 \times 8 = 64$ 个大 Patch），通道维度 $D=256$。
   - 包含时空位置编码与 4–6 层 Transformer Encoder 块，将 4 帧历史编码为条件 Token 序列 $H$。
2. **EMA 目标教师编码器 (Target Teacher Encoder)：**
   - 结构与 Online 编码器一致，权重由 $\theta_{EMA} \leftarrow \tau \theta_{EMA} + (1-\tau) \theta_{online}$ 缓慢更新（$\tau = 0.996$）。
   - 接收 4 个未来 BEV 帧，生成未受污染的目标潜变量 $Y^{target}_{scene}$。严格置于 `eval()` 模式并截断梯度。
3. **联合预测器 (Joint World-Action Flow Predictor)：**
   - 包含 8–12 层共享 Transformer 块，隐藏层维度 $D=512$，多头数 8，前馈网络 2048。
   - 带噪未来场景 Token $Z^{scene}_s$ 与带噪轨迹 Token $Z^{traj}_s$ 在预测器内部进行全交互注意力（Joint Cross/Self-Attention）。
   - 生成物理时间与流匹配时间独立编码：流生成时间 $s \in [0, 1]$ 采用 32 维正弦位置编码；物理未来时间点采用可学习物理时序嵌入。
4. **非对称梯度设计 (Asymmetric Gradient Contract)：**
   - **场景损失路径：** 场景预测损失 $\mathcal{L}_{scene}$ 计算在 $V^{scene}$ 与真实流速度之间。反向传播时对输入轨迹 Token 路径施加 `stop-gradient`，防止动作噪声破坏世界表征稳定性。
   - **动作损失路径：** 轨迹预测损失 $\mathcal{L}_{traj}$ 作用于 $V^{traj}$。允许动作梯度自由回传通过预测器内部的场景交互层，促使世界表征对驾驶动作提供最具预测性的信息。

### 2.3 流匹配参数化与采样器 (Flow Matching & Solvers)
- **流插值公式：** 采用线性最优传输（Optimal Transport Flow Matching）：
  $$Z_s = (1 - s) \epsilon + s X,\quad V^* = X - \epsilon,\quad s \sim \mathcal{U}[0, 1],\; \epsilon \sim \mathcal{N}(0, I)$$
- **ODE 采样算法：**
  - **Euler 采样器：** 固定 5–10 步。
  - **Midpoint 采样器：** 固定 3–5 步（每步 2 次前向）。
  - 推理时 Candidate 0 采用零噪声初始化作为确定性主干，其余候选引入温度 $\sigma \in [0.5, 0.85]$ 的高斯噪声。

---

## 3. 四主对照臂与消融体系

为了严格、科学地评估联合建模的真实收益，阶段4将冻结完全相同的原始数据、导航条件、训练轮数与控制器，构建四个同场竞技的主对照臂：

| 对照臂代号 | 架构描述 | 核心作用与隔离目的 | 参数量预算 |
|---|---|---|---|
| **Arm A (Deterministic MSE)** | 确定性端到端轨迹模型 (Joint/ResNet 匹配基线) | 标定确定性回归在当前任务上的表现与计算开销 | ~12M–15M |
| **Arm B (Action-only Flow)** | 纯动作流匹配模型 (仅对轨迹加噪积分，条件仅输入历史BEV) | 隔离“流匹配生成机制”对轨迹平滑性与多模态的增益 | ~50M–65M |
| **Arm C (Decoupled Flow)** | 分离式级联预测：先独立预测未来Latent，再串联送入轨迹生成 | 检验非联合、松耦合注入未来特征的传统级联方案 | ~60M–75M |
| **Arm D (Joint WA-Flow)** | **新主线：** 未来场景Latent与驾驶轨迹在预测器内联合生成 | 验证场景与动作在去噪过程中持续交互的核心价值 | **60M–80M** |
| **Ablation D-0** | 联合架构，但将未来场景监督损失置零 ($\mathcal{L}_{scene}=0$) | 严格分离“跨注意力结构收益”与“世界潜变量监督信号” | 60M–80M |

**训练与随机性规范：**
- 每个对照臂至少独立运行 **2 个独立训练随机种子**（`seed=20261022` 与 `seed=20261023`）。
- 严禁将推理时的流匹配噪声种子或 Dropout 种子混淆为训练重复。

---

## 4. 数据划分、示范资格与 Seed 隔离契约

### 4.1 数据集与有效样本
1. **历史合规导航/交通数据池：**
   - 训练集：`mixed_train_010`（48个Episode，13,952步），包含 `navigation_train_009`（24局）与 `traffic_train_009`（24局）。
   - 验证集：`mixed_validation_010`（24个Episode，7,371步），包含 `navigation_validation_009`（12局）与 `traffic_validation_009`（12局）。
2. **示范资格严格过滤 (Demonstration Qualification)：**
   - **世界模型预训练：** 所有因果观测有效的 BEV 步均可参与（100% 原始感知数据利用）。
   - **动作模仿损失：** 必须通过 `eligible_navigation_anchors()` 过滤：
     1. 10 个未来轨迹路点全部有效。
     2. 至少 2 个导航引导点有效。
     3. 未来 2.0 秒（20 步）内**无碰撞（crash）且无越界（off-road）**，且处于无随机扰动状态。
   - 经实测审计，`mixed_train_010` 中动作合格率为 **87.34%**（12,186 / 13,952 步），`mixed_validation_010` 中为 **87.55%**（6,453 / 7,371 步）。不合格步严格置动作损失权重为 0。

### 4.2 Seed 命名空间与隔离协议
为防止测试集泄露与已见场景冒充盲测，严格定义以下种子命名空间：

| 类别 | Seed 范围 | 场景数 | 用途与限制 |
|---|---|---|---|
| **训练导航集** | `4200000–4200023` | 24 | 模型正式训练 |
| **训练交通集** | `4200300–4200323` | 24 | 包含跟车、超车、变道干扰的复杂交通训练 |
| **开发验证集 (Nav)** | `4300000–4300011` | 12 | 开发期超参调优、模型选择与开环评估 |
| **开发验证集 (Traffic)**| `4300300–4300311` | 12 | 交通场景闭环诊断与行为消融 |
| **历史 Pilot10 回归集**| `4300000–4300005`, `4300300–4300303` | 10 | 历史已见场景，**仅用于防倒退回归测试，严禁作为盲测成果** |
| **绝对未见盲测池** | **`4400000–4400999`** | 1000 | **阶段0–7严禁访问**，专门保留给阶段8统一泛化验收 |

---

## 5. 阶段门槛与预登记验收指标 (Pre-registered Gates)

所有指标必须在实验启动前写入协议，严禁事后根据测试数字修改门槛。

### 5.1 阶段3：BEV未来生成预训练门槛
1. **预测误差相对基线 (Normalized Latent MSE)：**
   - 相比简单复制当前帧基线（Copy-Current），在 $+0.5\text{s}, +1.0\text{s}, +1.5\text{s}, +2.0\text{s}$ 各时域的归一化误差必须满足：
     $$E_{norm}(t) = \frac{\|\hat{Y}_t - Y_t\|^2}{\|Y_{curr} - Y_t\|^2} \le 0.85$$
2. **表征非塌缩度 (Representation Health)：**
   - 空间 Token 维度有效方差 $\text{Var}(Y) \ge 0.10$。
   - 潜变量矩阵有效秩（Effective Rank） $\ge 16$。
3. **因果与泄漏检查：**
   - 构造时间反转与未来替换冒烟测试，确保无未来信息回传到历史编码器。

### 5.2 阶段4：联合世界—动作训练门槛
1. **开环轨迹精度：**
   - 验证集 10 点平均位移误差 **$\text{ADE} \le 0.25\text{m}$**。
   - 2.0s 最终位移误差 **$\text{FDE} \le 0.60\text{m}$**。
   - 负速度预测比率 $\le 0.01\%$。
2. **闭环基础驾驶能力 (开发集)：**
   - 常规导航场景完赛率 **$\text{RC} \ge 90\%$**。
   - 无碰撞到达率 **$\text{Arrival} \ge 80\%$**。
   - 无故停车与死锁步数比例 $\le 3\%$。
3. **世界模型科学增益判定：**
   - Arm D（联合生成）相对 Arm B（Action-only）在相同预算下，路线完成度差异具备统计显著性：
     $$\Delta \text{RC} = \text{RC}_D - \text{RC}_B > 0\quad (p < 0.05)$$
4. **推理延迟与算力预算：**
   - 完整系统在 AMD RX 7900 XT 上端到端决策耗时 **P95 $\le 50.0\text{ms}$**。

---

## 6. 清理安全、保护清单与 Dry-Run 机制

### 6.1 清理白名单 (Dry-run Candidate Whitelist)
以下文件和目录经核验可由源文件完全重新生成，列为可清理候选载荷（逻辑容量约 14.59 GiB）：
1. `outputs/data/stage2_v2_100k/baseline_cache/` (7.35 GiB，可由 `dataset/baseline.py:build_cache` 重建)
2. `artifacts/restart_stage7/flow_features_001/` (3.67 GiB，可由旧冻结编码器提取)
3. 导航 Pilot 训练与验证缓存 `tokens.npy`（合计 2.62 GiB）
4. 历史风险诊断特征 `risk_features_014`, `risk_features_015`, `risk_features_hd03`（合计 0.94 GiB）

### 6.2 绝对保护清单 (Strict Protection Whitelist)
**严禁删除以下任何文件：**
- 所有原始数据集：`outputs/data/stage2_v2_100k/{train,val,test}/*.npz`、`artifacts/stage7_integrated/navigation_train_009/`、`traffic_train_009/`、`mixed_train_010/`。
- 所有关键模型权重：`best.pt`，历史对照与评估检查点。
- 历史成败证据与 MRE：`INVALID_CONTROL_WIRING.json`、`control_wiring_mre_001/`、`review_014_001/`、`review_015_001/`、`docs/STAGE7_FAILURE_ANALYSIS.md`。
- 所有评估视频、RRD 录像、TensorBoard 日志与指标文件。
- 项目源代码、`.git`、环境配置及所有规格文档。

**执行规则：** 阶段0期间**仅执行 Dry-run 审计**，不执行任何实质删除。后续即使获得明确清理授权，也必须逐文件核验 SHA256，生成 `cleanup_result.json`，严禁使用通配符或递归删除。

---

## 7. 阶段推进与停止约定

- 严格遵循仓库统一规则：**每阶段报告完成后立即停止，等待用户对下一阶段的明确指令**。
- 未收到下一阶段授权前，不自动启动数据采集、预训练或闭环测试。
- 阶段8与阶段9在阶段7正式验收并通过前严格封存。
