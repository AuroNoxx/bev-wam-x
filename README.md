# bev-wam-x

**基于 BEV 的世界模型与轨迹规划研究项目。** 围绕一个问题展开：预测未来场景，能否帮助模型做出更好的驾驶决策？

项目在 MetaDrive 中搭建从语义鸟瞰数据、未来潜空间预测，到轨迹生成、控制执行和开闭环评测的实验链路。v1 保留确定性预测及后续生成式扩展，v2 探索在共享 Transformer 内联合建模未来场景与驾驶轨迹。代码、对照实验和失败案例一并保留，便于理解设计演进与复核结果。

[v1 技术文档](docs/V1.md) · [v2 技术文档](docs/V2.md) · [复现说明](docs/REPRODUCIBILITY.md) · [结果证据](docs/evidence/README.md) · [目标与协议](GOAL.md) · [任务记录](TASKS.md)

## 实现了什么

- **数据与仿真**：采集四通道语义 BEV，对齐历史观测、实际执行动作、导航条件与未来目标；通过标签掩码区分有效监督与缺失数据。
- **世界与动作建模**：实现潜空间预测、Flow Matching 和场景—轨迹联合模型，设置确定性回归、纯动作 Flow、分离式与联合模型等对照。
- **规划与控制**：模型输出未来 2 秒的显式轨迹，由 Pure Pursuit + PID 控制器执行，连接开环预测与 MetaDrive 闭环诊断。
- **实验可视化与复核**：结合 Rerun、TensorBoard 和 Marimo 查看训练、预测及评测；保留配置、种子、原始指标和失败证据。

首次阅读可从下方架构和结果概览开始；实现细节见 [v1](docs/V1.md) / [v2](docs/V2.md)，运行条件见 [复现说明](docs/REPRODUCIBILITY.md)，指标来源见 [证据索引](docs/evidence/README.md)。

## 两代实现

| 版本 | 主要设计 | 当前定位 |
|---|---|---|
| **bev-wam-x v1** | BEV 编码与确定性潜空间/轨迹联合学习；后续扩展 Flow 候选、独立动作条件 WAM、风险评分 | 历史基线与失败分析；代码归档在 `archive/v1_legacy_20261024/code/`，共享基础模块仍在根目录 |
| **bev-wam-x v2** | 历史 BEV 和导航等合法条件；未来场景 token 与轨迹 token 在共享 Transformer 内联合流匹配 | 当前研究主线；源码在根目录 `models/`、`dataset/`、`scripts/` 等 |

两代均使用 BEV。这里的 v1 / v2 指架构代际，与数据 schema 版本、单个实验修订号分别记录；历史命名的对应关系见 [命名与发布说明](docs/NAMING_AND_RELEASE.md)。

## v2 数据与模型路径

```text
4 帧历史语义 BEV [B,4,4,128,128]
  + 历史自车运动 / 实际执行动作 / 导航 / 车辆参数 / 静态车道许可
                  |
           BEV 编码与条件投影
                  |
       共享 World-Action Transformer
       /                              \
未来场景 latent                    10 个轨迹路点
t+0.5/1.0/1.5/2.0s                 (x,y,v,heading), t+0.2...2.0s
                                      |
                           Pure Pursuit + PID, 10Hz
                                      |
                                MetaDrive 闭环
```

未来 BEV 经 EMA teacher 生成训练目标，推理只使用历史观测和允许的条件输入。Stage 3 还探索 residual/hybrid 与训练期 masked future patch completion。对照包括 A 确定性回归、B Action-only Flow、C 分离式级联、D 联合流匹配及 D0 消融，具体比较条件与尚未完成的验收项见 [v2 文档](docs/V2.md)。

## 结果概览

**当前取得了阶段性开环结果，复杂交互场景的闭环安全和世界模型的额外收益仍待验证。** 以下汇总截至 2026-10-27 的既有实验记录，本次文档整理没有新增训练。

| 范围 | 已有证据 | 结论边界 |
|---|---|---|
| v1 Stage 3/4/6 | 历史 acceptance 记录通过；Stage 5 完成评测协议 | 不等于 v2 通过；Stage 5 的通过标记不保证正向鲁棒性收益 |
| v1 Stage 7 | `scientific_passed=false` | 生成/风险/规划科学门槛未通过 |
| v2 Stage 3 masked | 两种子、20 epochs、完整验证；+2s E_norm 0.7090/0.7776 | +0.5/+1/+1.5s 未全部达到 0.85，整体 **未通过** |
| v2 Stage 4 开环 | 8,147 个动作合格锚点；D 两种子 ADE 0.1693/0.1769m，FDE 0.3798/0.3964m | 有效标签口径通过开环门槛；不代表闭环或额外世界模型收益 |
| v2 Stage 4 闭环诊断 | D22 六局无车直线 6/6 到达；另四局中三局越界 | `development_only=true`，`stage4_accepted=false` |
| v2 Stage 5/6/7 | 有历史压力测试、一致性诊断与闭环报告 | 不构成连续通过；Stage 7 成功结论已勘误撤回 |
| Stage 8/9 | 无本次启动或验收 | 未见场景盲测、3D 游戏迁移仍待后续研究 |

其中 ADE 表示预测轨迹的平均位置误差，FDE 表示终点位置误差；开环评估在记录数据上计算预测误差，闭环评估则让模型参与仿真控制，两者分别报告。结果分母、种子、失败项、控制周期和可核验 JSON 见 [v2 文档](docs/V2.md)。

## 研究范围

这是个人研究与工程验证项目，采用 JEPA 潜空间预测、Flow Matching 等方法独立实现实验系统，不使用官方 WA-JEPA 的代码或预训练权重，也不宣称完成其官方复现。尚未完成通用闭环驾驶验收或赛车游戏迁移。

模型学习输出轨迹，转向、油门和制动由代码控制器执行。当前实验主要运行于 **AMD RX 7900 XT / ROCm**；Orin-X 是历史设计参考，尚无该硬件的部署测试结果。

## 阅读与运行

```bash
git clone https://github.com/AuroNoxx/bev-wam-x.git
cd bev-wam-x
# 阅读源码和 docs/evidence 不需要下载模型或数据。
```

运行依赖包括 Python 3.12、PyTorch/ROCm、MetaDrive/Panda3D；训练和评测还需要未随 Git 发布的数据与权重。原 `scripts/bootstrap.sh` 使用作者本机共享环境配置，**不是通用的一键安装脚本**。迁移后的环境与路径问题、测试命令及历史复现方法见 [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。

## 目录

- `models/`：BEV 编码器、未来潜空间 Flow、联合 World-Action 模型及对照。
- `dataset/`：样本时序、掩码、行为资格与导航数据。
- `env/`、`controller/`：MetaDrive/语义 BEV、车辆控制与路线约束。
- `evaluation/`、`scripts/`、`tests/`：度量、训练/诊断入口与回归测试。
- `mini_wa_jepa/`：保留的 Python 兼容包名；展示名与仓库名统一为 `bev-wam-x`。
- `archive/v1_legacy_20261024/`：v1 源码与设计归档；大型压缩包仅在本地。
- `chapters/`：v1 为主的交互教学材料，不能作为 v2 成绩来源。
- `docs/`：两代说明、冻结协议、可发布证据与复现限制。

## 命名与兼容

GitHub 仓库和本地规范目录已更名为 `bev-wam-x`。历史实验目录、checkpoint 字段、冻结协议、旧 Git 快照及 `mini_wa_jepa` 导入路径保留原名，避免破坏溯源、权重加载与现有脚本。旧目录兼容链接不包含另一份数据。详见 [命名与发布说明](docs/NAMING_AND_RELEASE.md)。
