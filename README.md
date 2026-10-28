# bev-wam-x

**基于 BEV 的世界模型与轨迹规划研究项目。** 在 MetaDrive 中构建语义鸟瞰输入、未来潜空间预测、世界与动作联合生成，以及可追溯的开环/闭环评测。原项目名为 Mini-WA-JEPA。

[v1 技术文档](docs/V1.md) · [v2 技术文档](docs/V2.md) · [复现说明](docs/REPRODUCIBILITY.md) · [结果证据](docs/evidence/README.md) · [目标与协议](GOAL.md) · [任务记录](TASKS.md)

## 项目边界

这是个人研究与工程验证项目，尚未完成通用闭环驾驶验收或赛车游戏迁移。项目采用 JEPA 潜空间预测、Flow Matching 等方法独立实现实验系统；不宣称复现官方 WA-JEPA，不使用其代码或预训练权重。

模型学习输出轨迹；转向、油门/制动由 Pure Pursuit + PID 等代码控制器执行。当前证据主要来自 AMD RX 7900 XT / ROCm，**不是 Orin-X 部署测试**。

## 两代实现

| 版本 | 主要设计 | 当前定位 |
|---|---|---|
| **bev-wam-x v1** | BEV 编码与确定性潜空间/轨迹联合学习；后续扩展 Flow 候选、独立动作条件 WAM、风险评分 | 历史基线与失败分析；代码归档在 `archive/v1_legacy_20261024/code/`，共享基础模块仍在根目录 |
| **bev-wam-x v2** | 历史 BEV 和导航等合法条件；未来场景 token 与轨迹 token 在共享 Transformer 内联合流匹配 | 当前研究主线；源码在根目录 `models/`、`dataset/`、`scripts/` 等 |

两代均含 BEV 实现。旧归档说明把 v1 简称为 RGB/camera 不准确：保留代码 `models/latent.py` 的输入明确是四通道 BEV。架构代际、数据 schema v2 和 Stage 3 v3 实验修订是不同概念，不能混用。

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

未来 BEV 经 EMA teacher 生成训练目标，推理不读取未来真值。Stage 3 另含 residual/hybrid 与训练期 masked future patch completion 实验。对照包括 A 确定性回归、B Action-only Flow、C 分离式级联、D 联合流匹配及 D0 消融；已有运行并不意味着每项公平性约束已经全部验收。

## 当前结果：按版本和证据范围解读

截至 2026-10-27 的文件审计（没有新增训练）：

| 范围 | 已有证据 | 结论边界 |
|---|---|---|
| v1 Stage 3/4/6 | 历史 acceptance 记录通过；Stage 5 完成评测协议 | 不等于 v2 通过；Stage 5 的通过标记不保证正向鲁棒性收益 |
| v1 Stage 7 | `scientific_passed=false` | 生成/风险/规划科学门槛未通过 |
| v2 Stage 3 masked | 两种子、20 epochs、完整验证；+2s E_norm 0.7090/0.7776 | +0.5/+1/+1.5s 未全部达到 0.85，整体 **未通过** |
| v2 Stage 4 开环 | 8,147 个动作合格锚点；D 两种子 ADE 0.1693/0.1769m，FDE 0.3798/0.3964m | 有效标签口径通过开环门槛；不代表闭环或额外世界模型收益 |
| v2 Stage 4 闭环诊断 | D22 六局无车直线 6/6 到达；另四局中三局越界 | `development_only=true`，`stage4_accepted=false` |
| v2 Stage 5/6/7 | 有历史压力测试、一致性诊断与闭环报告 | 不构成连续通过；Stage 7 成功结论已勘误撤回 |
| Stage 8/9 | 无本次启动或验收 | 未见场景盲测、3D 游戏迁移仍待后续研究 |

结果分母、种子、失败项、控制周期和可核验 JSON 见 [v2 文档](docs/V2.md)。本仓库不把小场景到达率、单时域误差或潜变量余弦相似度解释为完整驾驶能力。

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
