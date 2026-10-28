# BEV方法对齐交接：哪些可以清理

本文件是清理建议，不是已执行记录。盘点日期为2026-10-22（用户时区）；仓库可能继续变化，实际执行须重新盘点。此次仅读取目录/文件大小、链接和部分生成器源码，尚未逐字节证明每个缓存可重建。

原则：**退出新主链不等于可以删除。优先清理可再生缓存，保留原始数据、关键权重和实验成败证据。** 新模型更换encoder后旧latent不适合作为新训练缓存，但仍可能用于复算旧报告，因此删除必须保持可追溯和可恢复生成能力。

## A. 优先候选：经核验后可删除的缓存载荷

| 路径/范围 | 此次逻辑容量 | 可删范围与前提 |
|---|---:|---|
| `outputs/data/stage2_v2_100k/baseline_cache/` | `.npy`约7.35 GiB | 原始train/val/test分片、manifest和normalization完整且通过核验后，可删除展开缓存；生成器在`dataset/baseline.py:build_cache` |
| `artifacts/restart_stage7/flow_features_001/` | 全部`.npy`约3.67 GiB | 核对manifest、旧encoder、原始数据及缓存脚本；仅删确认可生成的数组，不整目录删元数据。再细分大token和小标签，优先仅删大token |
| 有效导航训练目录下`train/tokens.npy`、`validation/tokens.npy` | 合计约2.62 GiB | 见下表；确认其manifest的raw_paths、encoder哈希、源数据与调用方，当前任务不再依赖且能重建时可删token载荷 |
| `risk_features_014/015`中的`histories.npy/prediction.npy/target.npy` | 各约0.146 GiB | 保留原生分支、生成权重和源快照；先复算并归档分组诊断；保留所有标签、有效位、分组字段和概率验证产物 |
| `artifacts/restart_stage7/risk_features_hd03/`同名三类数组 | 约0.633 GiB | 条件同上；优先保留训练/验证/测试标签与历史指标；不得为了清理重跑新科学测试 |
| 项目内`__pycache__`、`.pytest_cache`等工具缓存 | 未统计 | 仅限定仓库范围和已确认可再生类型；不递归删除环境、模型或数据目录 |

有效导航token目录为：

- `artifacts/stage7_integrated/navigation_head_pilot_002/`
- `artifacts/stage7_integrated/navigation_head_pilot_003/`
- `artifacts/stage7_integrated/navigation_head_pilot_004/`
- `artifacts/stage7_integrated/resnet_navigation_pilot_005/`
- `artifacts/stage7_integrated/mixed_joint_010/`
- `artifacts/stage7_integrated/mixed_resnet_010/`

**只指定上述目录内两个split的`tokens.npy`，不删除整个训练目录。** 它们还含权重、TensorBoard、实际预测、图和报告。不同模型可能共享这些缓存，特别要检查Flow训练及历史复算脚本的引用。

前三项全量候选逻辑容量上限约13.64 GiB；实际可释放量取决于依赖审计、保留小数组、文件系统块、硬链接和活跃mmap。不是已承诺释放空间，也不是一次性删除命令。

## B. 暂不删除：不能证明冗余的续训检查点

`artifacts/restart_stage6/runs_orinx/{conditioned,unconditional}_{20261019,20261020}/last.pt`四个文件合计约3.68 GiB，是较大的候选，但**不纳入默认可删清单**。

`last.pt`通常带优化器/调度器/随机状态，`best.pt`不一定能替代续训功能。只有确认旧run完成、无人续训、best及历史复现所需权重齐全，且用户明确接受丢失精确续训能力，才能另行列入删除授权。不要把所有`last.pt`或epoch checkpoint用通配符扫掉；某些可能是OOM/失败恢复的唯一证据。

## C. 必须保留

- `outputs/data/stage2_v2_100k/{train,val,test}/`原始分片、manifest、normalization；不要整删7.5G数据根目录，主要空间实际来自可再生cache。
- 009/010导航与交通有效原始数据，以及被merged manifest引用的源目录。合并集可能引用原文件，不能把原目录误认为重复副本。
- Stage6原生动作分支、Stage7风险原始分支、真实`episode.npz/actual.npz`、事件/掩码、历史动作与未来观测。
- Stage1验收样本/记录；Stage2–7正式报告、原始评估结果、TensorBoard、周期预测/真值、视频、RRD、失败MRE、日志、源码快照和协议。
- 有效stage3/4 encoder、Joint和Action-only、ResNet等基线`best.pt`；历史stage6/7关键对照权重和校准阈值；尤其旧报告明确引用的checkpoint。
- 融合001无效目录及`INVALID_CONTROL_WIRING.json`，旧失败运行、未完成分区与异常记录。失效训练产物是失败证据，不能因不能用于新训练而默认删除。
- risk缓存中的`risk/risk_mask/valid/seed/group/is_flow/is_normal/command/realized/waypoint_valid`及`rows.json/raw_hashes.json/manifest.json`；015分组错误的审计证据。
- 现有源代码与测试、旧GOAL/协议快照、docs/legacy、TASKS、所有未提交工作、`.git`、`.venv`与环境配置。

大RRD不是缓存，仅因为文件大不能删。若以后要节省这部分空间，应另做可验证的外部归档，当前指南未授权删唯一录像/回放。

## D. 执行清理时必须处理的具体问题

1. **先列清单。** 对每个候选记录绝对路径、真实路径、类型、大小、hash、链接数、生成来源、重建入口、活跃依赖、保留替代项和判定。清单不合并成按名称递归删除的危险规则。
2. **确认可重建。** 源分片/权重/hash正确，保留旧生成源码和环境；对确定性展开缓存在临时新目录验证样本相等。对模型特征至少保留一个重建核对样本，不能只因目录名字含features就认定可再生。若重建需缺失数据或权重，移入保护清单。
3. **链接不跟随删除。** 此次发现`risk_features_001 → risk_features_hd03`、`risk_runs → risk_runs_hd03`、`risk_eval → risk_eval_hd03`、`flow_20261021/22 → runs_orinx/flow_...`。链接和目标分开登记，不能用多个别名重复统计或删除目标。硬链接按inode去重。
4. **不要让缓存伪装完整。** baseline loader只看`complete.json`。删数组时必须将旧标记备份到清理审计后从缓存位置撤下，否则loader会误以为缓存完整。其他manifest保留历史原件，另写`CACHE_EVICTED.json`说明被移除载荷和重建方法；必要时让读取器给出明确缺缓存提示，不能静默混用新旧特征。
5. **不可中断活跃任务。** mmap/训练/评估可能仍引用文件。发现使用中则跳过，不强制杀进程；磁盘大小不等于当下可释放大小。
6. **审计前后。** 实际删除仅限用户确认的逐文件白名单；保存`cleanup_plan.json`与`cleanup_result.json`，记录成功、跳过、错误、实际空间变化以及全部保护文件hash变化检查。不要覆盖旧清理记录。

禁止使用`rm -rf artifacts`、对所有`*.pt`或`*.npz`全局删除、`git clean -fdx`、重置工作区，或根据旧训练已失败就整目录清理。本文没有执行任何删除。
