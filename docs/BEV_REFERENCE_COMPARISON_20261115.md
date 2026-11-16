# bev-wam-x 与当前 OCC + Flow 的实现对照

2026-11-15。只读检查同级 `E:/ai/projects/bev-wam-x`，不修改旧项目、不启动训练、不读取当前保留测试集。使用旧结果与当前已保存的验证预测；不重做状态/导航基线。

## 已找到的成功参照

- 旧 v1 Stage 4：9,600 个完整轨迹锚点、22 个 episode；joint ADE/FDE 0.130488/0.223658 m；20 局闭环成功率 95%，路线完成率 0.969774。它是确定性联合模型，不是当前随机候选 Flow，且该阶段未使用导航。证据：`../bev-wam-x/docs/evidence/v1/stage4_REPORT.md`、`../bev-wam-x/models/joint.py`。
- 旧 v2 Action-only Flow（B）：8,147 个有效锚点，两训练种子 2s ADE 0.1675/0.1767 m，FDE 0.3751/0.4111 m。该项不依赖联合未来场景生成，适合作为 Flow 生成路径参照。证据：`../bev-wam-x/artifacts/bev_wa_alignment/stage4_repair_execution_001/open_loop/REPORT.md`。
- v2 联合模型 D 也有 2s ADE 0.1693/0.1769 m；它并非在所有闭环场景都通过。保留旧项目成功的已验证范围，不将旧项目所有阶段合并成一次成功或失败。

## 实际发生的变化

| 项目 | 旧 BEV v2 / B | 当前 OCC / Flow v3 |
|---|---|---|
| 任务来源 | MetaDrive 语义 BEV 数据 | Bench2Drive 多场景、跨地图划分 |
| 场景表示 | 明确 drivable、boundary_static、other_actors、visibility 四通道 | 20 层高度、语义标签 embedding；源未填充与 unknown 分开编码 |
| 历史编码 | 逐帧空间 token，时间位置编码，独立历史运动 token | 四帧 × 高度 × 类别 embedding 展平成通道后卷积 |
| 规划条件 | 8 个导航点及掩码、历史运动/执行动作、车辆参数、变道许可 | 当前状态、近/远导航点及命令组成 20 维向量，线性投影后加到空间 token |
| 解码主干 | 默认 384 维、8 层；有 encoder 初始化入口 | 192 维、4 层；v3 从随机权重训练 |
| 轨迹 | 10 点，0.2–2.0s；x/y/v/heading | 25 点，0.2–5.0s；归一化 x/y/sin/cos/v |
| Flow 目标 | 有效 mask 加权 Smooth L1 速度场损失 | MSE 速度场损失 + 局部 endpoint 位置/一致性等辅助项 |
| 推理起点 | `z_traj = torch.zeros(...)`；单条确定性输出 | `torch.randn(...)`；8 条候选，16 步 Euler |

这些变化是代码/配置事实，不是已完成因果归因。当前也保留 8×20=160 个空间 token 和坐标通道，**不能说当前将整个 OCC 池化成一个向量、彻底丢掉左右位置**。早期通道融合也不等于理论上无法表达时序，是否造成学习困难仍待直接验证。

旧 B `solve_euler` 虽接受 noise_seed/generator，但轨迹起点实际为零，因此其报告中的低 ADE 不证明随机多候选轨迹质量或多样性。训练仍用高斯噪声，不能把旧推理写法当作标准采样正确性的证明，或直接照搬后就宣称问题解决。

旧 v1 的低误差还有恒速/恒转率运动先验加残差结构；与 v2 B 的零起点 Flow 是两条不同参照，不能混为一谈。当前之前已做过匀速参考残差 10 轮对照，减速和风险门槛未通过，不重复包装成新发现。

## 原始源码来源核验

从旧开环 results.json 的 `provenance.source_files` 提取哈希，与现存文件比较：

- `models/bev_wa_joint_flow.py`：相同；零起点、条件投影及 B 解码器差异有对应评估版本依据。
- `models/latent.py`、`dataset/bev_wa_dataset.py`：相同。
- `models/bev_wa_latent_flow.py`：不同；当前逐帧编码实现可读，历史文档也记录该合同，但尚不能宣称整个依赖源码已精确恢复。完整重跑前需追溯此文件版本。

核验原始值保存在 `artifacts/diagnostics/bev-reference-comparison/source-match.json`。本次没有加载旧模型重跑，历史低误差仍是原始结果证据，不是本机新复现成绩。

## 对齐时距后的当前误差

复用当前 v3 已保存预测与验证标签：1,124 窗口、9 路线、3 个采样种子、每次 8 候选；没有新训练或读取测试集。

| 当前输出规则 | 前 2s ADE / FDE，m | 完整 5s ADE / FDE，m |
|---|---:|---:|
| 固定候选 0 | 1.1704 / 2.3771 | 3.5661 / 8.0269 |
| 每时刻 XY 中位数 | 0.9698 / 2.0345 | 3.1071 / 7.1005 |
| 8 候选平均 ADE | 1.1861 | 3.5808 |

前 2s 仍有明显误差，因此不能用“现在预测 5 秒”将问题完全解释掉。但旧新数据来源、速度、机动、划分和训练预算不相同，不能将约 0.17 和约 1.17 的比值解释成 OCC 导致的性能退化比例。中位数不是原始单条 Flow 样本，其可执行性局限见既有诊断。

计算结果：`artifacts/diagnostics/bev-reference-comparison/current-horizon-metrics.json`。源预测为 `artifacts/diagnostics/flow-v3-assessment/flow_v3.npz`，标签为 `artifacts/diagnostics/20261115-contract/evaluation-labels.npz`。3 种子汇总口径与训练期单次验证指标不同。

## 当前判断与下一项直接检查

已确认：当前并不是在成功旧链路上仅替换输入表示，而是同时重新实现了表示、条件融合、轨迹时域、主干和采样合同。现有证据不足以将失败归为 OCC 本身或 Flow Matching 方法本身。

最小的下一项检查是冻结当前权重和输入，比较当前随机起点与旧 B 的零起点推理，统一输出前 2s/完整 5s 的轨迹误差、停车前冲、减速时机和日志框风险。它只检查已有推理差异，不训练、不重做状态/导航基线、不增加模块。零起点若只改善 ADE 而损害风险或候选覆盖，不应采用；若无改善，就不能继续归因于随机采样。

随后再围绕已知旧链路作输入适配，而不是直接搬整套世界模型：导航数据不足以凭空造出 8 个路线点，OCC 标签也不能未经语义核对就冒充 drivable/visibility。具体要复用哪一层，须由上述直接证据决定。本次没有改模型实现或宣称根因已经锁定。
