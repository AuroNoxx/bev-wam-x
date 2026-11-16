# 帧修复、行为盘点与高速专家准备

2026-11-15。用户授权先修复数据、盘点场景，并允许删除旧错帧权重；仍不进行长训练或批量采集。

## 配对修复范围

原始 LiDAR 对 OCC 和 actor 标注的交叉证据见 [专项核查](FRAME_ALIGNMENT_20261115.md)。当前发布版本采用 `OCC 文件 ID = annotation/pose ID - 1`，不平移车辆状态、导航或未来轨迹。证据来自 15 条训练/验证路线的原始点抽查，不能推广为所有上游版本的编号规律；生产代码精确来源和原始绝对时间戳仍未补齐。

新增 `scripts/build_aligned_cache.py`，建立独立 `data/occ_v3`，严格保留原来的 5,296 个训练窗口、1,124 个验证窗口及顺序。原缓存/manifest 不覆盖，不新增边界窗口，不改变网格、mask 类别、轨迹时距、归一化或目标函数。记录每个 OCC 文件 ID、对应位姿 ID、源文件哈希及旧缓存哈希。未来 OCC 仍只作为监督，历史位姿 ID 不超过当前帧。测试集不读取、不重建、不评估，v3 中空 test 列表表示封存，不能解释成测试集丢失。

构建过程对每个临时 NPZ 重新加载，检查 shape/dtype、状态/轨迹/位姿逐数组不变及来源 ID，完成后原子替换。最终产物是否完成以 `artifacts/diagnostics/alignment-v3/final-verification.json` 的 `pass` 为准；测得耗时与体积也记录在那里。

本轮最终验收已通过：6,420 对新旧缓存哈希全部核验，另复核 102 个路线首尾窗口的数组与索引。实际使用两 worker 读取了 Flow 的 8 样本批次，也读取了世界模型的未来 OCC 标签。完整重建耗时 638.88 秒，新增缓存 210,968,407 字节（约 201.20 MiB）。这是 CPU 数据处理成本，不是训练/模拟采集成本；未测其峰值内存。

51 条路线各一个窗口的小测试通过；9 个验证窗口与之前独立旁路修正结果逐数组一致。检查了 -1 OCC 文件边界、当前前向填充仍为 unknown、v3 loader 拒绝混入 v2 样本。数据读取支持由配置显式指定 manifest 和版本。新配置为 `configs/lightning-stage1-v3.json`，启动器使用独立 `flow-v3` run，不再隐式载入旧权重。

训练继续暂停，启动器及直接 Lightning CLI 的暂停检查均已测试。没有更改模型架构、优化器或断点实现。现有 `test_lightning_resume.py` 在隔离目录执行了旧 loader 兼容性回归：完整/恢复分支模型参数差为 0，优化器步数与调度器一致，优化器浮点状态最大差 2.39e-9（容差内），模拟原子替换失败保留原断点。此为数步回归，未延续正式 run，也未将其权重作为新模型初始化。

## 实际行为初筛

读取训练 42 条、验证 9 条路线的全部 9,327 个原始标注帧。不是用场景名称代替行为，而是使用车辆 ID、同向航向、车道、相对位置及实际车速提取候选。计时来自 10 Hz 名义帧索引，原始 timestamp 缺失。

| 行为候选 | 训练事件 / 涉及路线 | 验证事件 / 涉及路线 | 判定摘要 |
|---|---:|---:|---|
| 持续跟车 | 31 / 19 | 6 / 4 | 同一前车、同道路/section/车道，中心前向距离 2–60 m，连续至少 3 秒 |
| 慢速排队 | 5 / 4 | 0 / 0 | 自车和前车低于 10 km/h，前方 50 m 内至少两辆同向慢车，连续至少 3 秒 |
| 停车再起步 | 14 / 8 | 3 / 2 | 低于 1 km/h 至少 1 秒，前后各 10 秒内超过 10 km/h；不自动等于拥堵 |
| 车道变更 | 19 / 13 | 2 / 2 | 同一道路/section 的相邻同向车道，变更前后各稳定至少 1 秒 |
| 超过同一运动目标 | 0 / 0 | 0 / 0 | 同向、连续目标由前方超过 5 m 变为后方超过 5 m，排除目标跳变 |
| 完整超车 | 0 / 0 | 0 / 0 | 上项加初始同车道、自车换道、超过后回到目标车道并稳定至少 1 秒 |

这些均为阈值筛出的候选，不是人工精标。严格同 road/section 条件会漏掉跨路段行为，短片段和目标可见范围会截断事件。未检出超车不证明源数据完全没有超车；需要检查片段再决定补采数量。相邻窗口不算独立 episode；同一段可同时属于跟车/排队/启停，不可把事件时长相加当独立驾驶时长。微小负速度仍保留，慢队列规则不把负值自动裁成零。

结果为 `artifacts/diagnostics/behavior-screen/report.json`；每条路线逐帧 CSV 位于 `timelines/`；速度×行为矩阵在 `speed-behavior-matrix.json`，单格去重帧数并附路线/道路 section 数。候选区间包含必要的变更前后上下文。输入筛查利用了带特权信息的标注目标，不能把这些额外目标当作模型的可见 OCC。

`review-examples.html` 展示每类最长候选的速度曲线，并可展开逐帧车道、前车 ID 和距离。示例按最长时长选取，不代表随机或完整质量验收。合成正/负控制验证了真实完整超车可检出、道路边界重编号不误算变道、慢队列可检出。

## 高速专家发现与下一步

本机已有 CARLA 0.9.15 和对应 Python API；尚未安装/接入 PDM-Lite 的完整采集栈，原 CARLA 环境缺 scipy、ujson、shapely、srunner、leaderboard 等导入。本轮未安装环境或启动模拟器。

核对固定 commit `48e6a6dbcb133220333bd65e242fbc2b492bb15b` 后发现，PDM-Lite 的 [autopilot.py 第 352 行](https://github.com/autonomousvision/carla_garage/blob/48e6a6dbcb133220333bd65e242fbc2b492bb15b/team_code/autopilot.py#L352) 将初始目标速度封顶为 72 km/h，并使用道路限速的 0.72 倍。它不是开箱可用的 120 km/h 专家。其配置中的检测距离、跟车时距、横向前视和拟合动力学也需要高速审查，不能只去掉速度上限。原代码当前内容与固定 commit 对照一致；证据摘录见 `expert-source-evidence.json`。

准备了 `configs/highway_expert_probe.json`，状态为未执行/专家未合格。下一步先独立检查地图路线和车型动力学、直线加减速与弯道保持，再检验跟车、切入、变道及超车；逐档 80、100、120，60 秒小案例，记录实际达到的速度、持续时长、碰撞、车身越界、车距/TTC、加速度/jerk、控制饱和和传感器同步。具体阈值先根据车型标定锁定，不把未验证的参数写成安全保证。

先保留并修复已有低中速交互数据；优先补持续高速、验证中的拥堵，以及完整超车/取消的证据缺口。暂不下载大数据，不启动大规模专家采集，不变更 OCC 范围或训练世界模型。世界模型反事实仍需同初始状态下可干预分支，单条日志不够。

## 旧权重清理实际结果

用户已明确授权删除。核对了旧正式训练、测速以及旧训练派生的恢复测试共 15 个断点，逻辑文件大小合计 837,957,065 字节（约 799.14 MiB）。两个人工/合成回归断点不在目标清单。

自动审批审查在进程创建前拒绝删除操作，实际返回 `blocked by policy`，没有提供更具体理由。删除 0 个、释放 0 字节；未换工具重试。15 个目标仍存在，路径/大小/SHA256 已写入 `artifacts/diagnostics/alignment-v3/checkpoint-deletion-blocked.json`。原有 50 个断点/评估 JSON 的哈希仍一致。新回归测试自身的少量隔离断点单独保留，不会用于正式训练。

旧数据、日志、TensorBoard、配置、split 和报告全部保留。旧权重因工具限制仍在磁盘，不意味着会再次用作训练初始化。删除后的旧模型重跑对照原本就不可用；已有数值报告仍可作为历史记录，不能替代同随机种子、相同预算的严格配对重训。

## 复现入口

```powershell
.\.venv-train\Scripts\python.exe scripts/build_aligned_cache.py --smoke
.\.venv-train\Scripts\python.exe scripts/test_alignment_v3.py
.\.venv-train\Scripts\python.exe scripts/build_aligned_cache.py
.\.venv-train\Scripts\python.exe scripts/verify_aligned_cache.py
.\.venv-train\Scripts\python.exe scripts/screen_logged_behaviors.py
.\.venv-train\Scripts\python.exe scripts/render_behavior_review.py
```

构建脚本会重新计算指定产物；已有完成的缓存无需无故重跑。模型效果、控制器在 120 km/h 的表现以及闭环安全性尚未完成验证，不能由本轮数据工程通过推断。
