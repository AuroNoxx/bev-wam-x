# Current project direction (updated 2026-11-17)

The user's later 2026-11-16/17 instructions supersede the 2026-11-15 classical-only direction quoted below. Current work is the isolated native v2 experiment described in docs/OCC_QWEN_NATIVE_V2_20261117.md; it completed20epochs and final audit (docs/OCC_QWEN_NATIVE_V2_FINAL_AUDIT_20261117.md); its recorded status is artifacts/occ-qwen-native-v2-training/status.json. Native v1 is completed and preserved. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first, then [the native-input implementation/status record](docs/OCC_QWEN_NATIVE_INPUT_20261116.md).

Native v1 has completed20 epochs/step13240 and final audit; isolated v2 implementation and subsequent training are authorized by the later user instructions in docs/NEXT_STAGE_ARCHITECTURE_PLAN.md. Older pause statements are historical. Do not duplicate or disturb an active trainer. Preserve best/latest, full resume state, existing data and historical classical code. World predicts future latent targets only: future 3D OCC decoding and OccWAM scene-to-ego correction are NOT implemented or authorized by the current architecture decision. No reliable closed-loop driving result is established for this new model.

## Earlier project instructions (architecture-direction item superseded above; other safeguards remain applicable)

# 项目级 Codex 规则

本文件仅适用于 highway-occ-flow，不能把本项目的清理规则推广到其他项目。

## 固定架构与环境

- 遵循 docs/ARCHITECTURE.md 的最新决策：2026-11-15 用户明确改为先不使用 Flow Matching 和世界模型。当前路线为历史 OCC → 障碍运动估计/短时预测 → 行为选择 → 轨迹搜索/优化及可执行性检查 → Pure Pursuit + 速度 PID；实施与验收见 docs/OCC_CLASSICAL_PLAN_20261115.md。旧模型、数据和报告保留；不自动训练，不沿用其他项目方案。
- 不自行加入 RGB 感知训练、独立 BEV 分支或直接踏板动作生成。
- Python、虚拟环境、模拟器、下载及缓存均放在本项目中；不更改全局驱动或其他项目环境。
- 使用本地 TensorBoard；不使用 SwanLab，不自动租卡、调用付费 API 或上传数据。

## 训练断点滚动保留

- 对每个活动训练运行、每个独立训练模块，默认仅保留 `latest` 和 `best` 两个逻辑断点。不按每步或每轮无限积累 checkpoint。
- `latest`：轮内按配置的间隔更新，且每轮完成后更新。间隔依据可接受的重算时间与保存成本设置，不能只在很长的一轮结束时才保存。
- `best`：仅在预先指定的验证集指标改善时更新；选择方向、阈值和指标必须记录。不能因进入下一轮就删除更早但仍最优的模型，也不能用保留测试集选择 best。
- 断点至少包含模型、优化器、调度器（若使用）、AMP scaler（若使用）、epoch/global_step、随机状态、采样器/数据进度以及配置和数据 split 标识；不支持精确轮内续训时明确记录实际恢复粒度。
- 先写同文件系统的临时断点，完成落盘和可加载性/必要字段校验，再原子替换 latest/best。只有替换成功后，才清理本运行已被替代的旧断点；失败时保留原有可恢复版本。
- 临时保存期间允许短暂超过两份；不能为了硬凑两份而先删除唯一可恢复断点。latest 与 best 相同时可共用安全引用，不强制复制。
- 只清理本运行登记且已被替代的 checkpoint。不得递归清空父目录或使用宽泛通配符误删其他运行、模块或用户手动保留的模型。
- 原始训练/验证/测试数据、预处理缓存、日志、TensorBoard 曲线、配置、split 清单和结果报告不属于 checkpoint，不得按轮清理。
- 当前正式入口采用 Lightning Trainer、ModelCheckpoint 与 VerifiedCheckpointIO，恢复数据位置采用 TorchData StatefulDataLoader；测速不等于正式训练。修改训练器时须验证保存失败保留旧断点，以及恢复后优化器和步数连续。现有回归入口为 `scripts/test_lightning_resume.py`。
- 若工具执行层拒绝清理，记录实际错误和仍保留的文件；不换工具绕过，不宣称已释放空间。
