# Stage 1 v3：60 轮训练结果

2026-11-15。修正 OCC 文件与标注的配对后，从随机权重训练 Flow，固定原有 5,296 个训练窗口和 1,124 个验证窗口、网格及 v2 目标函数。最终测试集没有读取或参与选模。

已完成 60 轮、39,720 次优化器更新，训练进程正常退出（exit code 0）。第 54 轮原本触发早停；用户明确要求跑完 60 轮，随后从 latest 恢复，保留模型、优化器、学习率调度器和数据进度，仅在此次恢复中关闭 EarlyStopping。配置本身未改写，覆盖参数记录在 runtime.json 和 finish60-authorization.json。续训前独立回归验证了步数连续、恢复参数一致和保存失败保护。

## 最终选模

预先指定的选模指标为验证集 `trajectory/meanADE8_m`，越低越好；不是按测试集或事后任意指标选模型。第 60 轮刷新 best，最终 best/latest 均为第 60 轮，相同内容 SHA256，已重新加载验证模型有限值、优化器/调度器状态及数据契约哈希。优化器步数均为 39,720。

| 指标 | 第 44 轮（追加前最佳） | 第 60 轮（最终 best） |
|---|---:|---:|
| 8 候选平均 ADE，m | 3.6323 | 3.5994 |
| minADE8，m | 1.6520 | 1.7421 |
| 8 候选平均 FDE，m | 7.9315 | 8.0528 |
| 位置/航向/速度运动一致性残差，m/s | 0.6836 | 0.5236 |

追加到 60 轮令选模指标改善约 0.033 m，但并非所有指标都改善：平均 FDE 和 minADE 都略变差。这是本次单种子、已用于选模的开发验证结果，不能当作统计显著提升，也不能把所有变化归因于一帧修复；新旧训练预算、初始化历程不同。

第 60 轮训练 loss 0.05319、验证 loss 0.06145、minFDE8 3.3387 m、速度 RMSE 2.3858 m/s、候选 spread 指标 2.4661 m。minADE/minFDE 是利用未来真值事后选候选的离线上界，不能当作可部署选择器的驾驶成绩。候选平均指标也不是单条已选择轨迹的成绩。运动一致性残差只是离线诊断，不能替代车辆跟踪或碰撞/越界测试。

## 场景差异

第 60 轮平均 ADE：HighwayCutIn 2.444 m、HighwayExit 2.260 m、MergerIntoSlowTraffic 2.277 m、MergerIntoSlowTrafficV2 6.689 m、SequentialLaneChange 4.560 m、HardBreakRoute 4.300 m。各组分别只有 2/1/2/1/1/2 条验证路线，窗口数为 256/54/161/37/135/481；邻近窗口高度相关，速度、道路和机动也存在混杂，不能把组间差异当作单一因素因果解释。

## 留存与边界

- 逐轮完整记录：`artifacts/lightning/flow-v3/epoch-metrics.csv`。
- 汇总：`artifacts/lightning/flow-v3/training-summary.json`。
- 断点核验：`artifacts/lightning/flow-v3/completion-verification.json`。
- 原始日志：`logs/lightning/flow-v3.log`；TensorBoard run：`lightning-flow/stage1-alignment-v3-fresh`。
- 原始数据、旧缓存、旧评估和日志未清理；先前旧权重删除仍受工具策略阻止，本轮没有重试。
- 未启动世界模型训练、最终测试集评估或 CARLA 闭环。现有数据仍缺少 80–120 km/h 样本，不能由本轮训练宣称覆盖 0–120 km/h 驾驶能力。

后续应固定第 60 轮模型，检查单条/可部署选择器、分时距与覆盖内外误差及轨迹可执行性，再决定下一项单因素实验；本次监控没有自动启动新实验。
