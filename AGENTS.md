# highway-occ-flow 项目规则（2026-11-17，v5）

## 当前唯一主线

### 阶段评估范围

2026-11-19 用户要求按全项目能力汇报。跨阶段总结覆盖速域、巡航、减速刹停、弯道、变道、平顺性、World 和闭环，同时保留停车与起步回归。报告样本数与路线数，说明训练侧、独立验证或未测；按整体证据确定优先级。当前汇总见 `docs/V5_CAPABILITY_SCORECARD.md`。

### 最新两步执行要求

2026-11-18核验发现subset2000原第2个框架epoch为空，原finished状态仅500更新／4000抽样。修复采样器边界及轮末状态恢复后，本次补跑入口为`python -m v5_qwen_drive_world.complete_three_passes --phase subset2000 --resume`，保留原750步调度并补齐至750更新／6000抽样。不要并发重复启动；按实际更新数判断完成，不能仅按epoch编号。原误报状态已保存在subset2000/empty-epoch-audit。用户要求先完成本次2000×3，再评估后续补采或输出头修正。

2026-11-18用户明确授权使用项目磁盘空间加速当前训练：当前续训入口为`python -m v5_qwen_drive_world.resume_disk_cached --phase subset2000 --resume`，仅对冻结ViT/前24层启用持久特征缓存；原像素、可训练部分、EMA教师更新、微批1/累积8、数据顺序及损失保持不变。允许为此在完整断点落盘并核验后切换本次v5训练进程，不影响其他旧任务。缓存身份校验、actor/未来教师分目录、128GiB磁盘上限及3GiB RAM上限见`configs/v5-qwen-drive-world/frozen-prefix-disk.json`。真实联合损失及176项梯度对比已通过。必须使用带UpdateBoundaryCheckpoint的入口，防止恢复后首个未提交微批重复保存；原train_disk_cached仅保留实现，不再直接用于续训。

用户授权补采本机缺失高速RGB/轨迹/占用数据。依次运行64条×30轮过拟合、2000条（1400B2D＋600Town04高速）×3轮SFT。两步指标与13GiB均只观测，不设质量达标门槛；这不是GRPO。真实六视角与EMA教师入口`train_supervised.py`先用微批1/累积8/有效batch8，保留官方processor原尺寸；三相机诊断微批4成绩不能替代。只运行一份训练，不重复启动。详情见`docs/V5_TWO_STAGE_SFT.md`，训练状态以该阶段status/epochs/validation日志为准。

### 2026-11-17 最新性能改造授权（优先于下文旧实现描述）

保留官方ViT及其原生processor像素尺寸、完整32层VLM与8组真实planner KV；前24层完全冻结且不checkpoint，仅后8层注意力投影挂PEFT LoRA r16。允许ViT输出额外2×2空间平均池化及为850 token上限进行显式预算分配；保留当前帧全部已有视角、历史仅前视主相机。Token合并是新后训练适配，不宣称与原模型等价。World训练分支禁止Conv3d，改用官方perception可迁移2D参数、BEV query cross-attention与4级Conv2d残差上采样；不得宣称新解码器已预训练。允许测试微批2/4，累积4/2保持有效batch8；通过实测后才能设为默认。SFT以3/5轮估算，不承诺收敛。冻结的官方ViT自身patch embedding保持原组件；World的Conv3d禁令不要求替换ViT。

三档容量实测已各通过两次联合更新，当前默认微批4/累积2/有效batch8，热态0.8194秒/样本。PyTorch allocated峰值14.79GiB不等于全卡占用（Windows监控峰值19.57GiB），不能承诺所有数据或长期运行不OOM。旧3D World和低分辨率记录仅为历史，当前工程入口为`scripts/Verify-V5Lightweight.ps1`。正式manifest/SFT尚未完成，不把诊断损失更新称作正式训练。

先读 `docs/ARCHITECTURE.md`，再读 `docs/QWEN_DRIVE_HIGHWAY_120_ARCHITECTURE.md` 和 `docs/V5_IMPLEMENTATION_STATUS.md`。
用户授权建设 `v5-qwen-drive-world`：官方 Qwen-Drive-1.0-4B 原生历史 RGB/状态/导航 → 因果推理 → 官方 planner-rl 控制量 Flow 扩展 + 官方 perception 动作条件 World 扩展 → World Critic → 5Hz 规划 / 20Hz 控制，目标连续覆盖 0–120 km/h。
旧 OCC-CNN/4×4 token、native v1/v2 和 classical-only 是历史，不得作为本轮输入架构；不按速度拆多个模型。
新增代码在 `v5_qwen_drive_world/`，配置在 `configs/v5-qwen-drive-world/`，数据/产物/日志使用同名独立目录。旧源码不修改。`scripts/verify_qwen_drive_components.py` 只作新模块入口。

## 执行与清理

- 先只读检查后台任务；不停止、暂停或修改运行中旧任务读取的代码/配置/权重，不重复启动训练。
- 用户本次明确确认“删除已完成运行的旧权重，保留日志和结果记录”；删除登记见 `artifacts/v5-qwen-drive-world/weight-cleanup.json`。此授权不扩大为未来自动删除任意权重。
- 旧文档保存在 `docs/archive/pre-v5-20261117/`。旧结果中“权重保留”仅代表当时状态；已删除权重不能恢复训练。数据、缓存、日志、历史结果和其他任务未提交文件保留。
- 环境、模型、模拟器、下载和缓存限本项目，不改全局驱动和其他项目；仅本地 TensorBoard，不用 SwanLab、付费 API、租卡或上传数据。
- 清理被工具拒绝时记录实际错误，不换工具绕过。

## 必须使用现成训练框架与算子

- 训练循环、梯度累积、精度、调度、保存和恢复交给 PyTorch Lightning；LoRA 用 PEFT；数据状态用 TorchData StatefulDataLoader。不得另写完整训练框架。
- 优先复用官方组件和 PyTorch/ROCm 已有向量化、foreach/fused、SDPA、grid_sample、checkpoint 算子。加速选项必须在本机实测正确且可求导；不启用不兼容 CUDA 扩展或静默断梯度。禁止手写替代 ViT/OCC 输入编码器。
- 训练禁止 INT4/NF4。`bf16_autocast_fp32_master`：可训练参数与优化器状态 FP32，BF16 autocast；Flow ODE 和运动学积分禁用 autocast 并 FP32。
- 所有标准 Transformer attention 使用 SDPA MATH，checkpoint 重算也须处于相同上下文；Qwen3.5 原生线性注意力不是 SDPA，保留官方实现并独立检查因果梯度，不能宣称 MATH 自动修复它。
- 用户在重启后明确授权微批=1、梯度累积=8，单卡有效 batch=8；配置与报告须同时记录三个数，不能把累积通过写成物理 batch8 通过。支持24层配置，但缓存组与官方32层 planner 不匹配时不得复制/伪造缓存；可选官方全层+gradient checkpointing。
- 用户未授权降低图像分辨率。验证保持官方原生processor尺寸；不能用128×128或其他缩小输入替代原生尺寸验收。已有低分辨率记录保留为历史工程诊断，不能用于宣称用户要求已通过。
- 只有同机 CARLA 闭环允许 VLM torchao int8_weight_only；planner/World 保持 BF16/FP32。训练器不得加载量化权重。

## 不增加无意义模块或多余对照组

- 不增加用户未要求的兜底策略、速度锁、静态偏移锁、额外安全网络或多套驾驶策略。只实现规范要求的 World Critic、动力学代价与速度自适应介入。
- 不生成多余消融组、不同种子对照训练或旧模型横向比较。仅保留规范要求的静态 Persistence 验证指标；它是零训练计算，不是额外模型/训练组。
- 工程验证限于真实接口、梯度、精度、数据时间/坐标、保存恢复和最终闭环所必需内容。随机小模型通过不代表官方模型已通过。
- 未来 RGB/OCC/轨迹只能作标签/教师目标，不能进入 actor prompt/cache。World 残差基底 stop-gradient。路线/相同初态分支必须同 split。
- 112m 网格、160m RGB、5Hz、120km/h 是目标，不是已验证能力；网格外不能当 free，运动学积分不等于真实车辆侧滑保证。

## 训练断点滚动保留

- 每活动运行/模块只保留 latest/best；latest 轮内定期及每轮更新，best 按预定验证指标选择，不读 test。
- 包含模型、优化器、调度器、scaler（若使用）、epoch/step、RNG、采样器/数据位置、配置与 split 身份；明确真实恢复粒度。
- Lightning ModelCheckpoint + VerifiedCheckpointIO：同盘临时写入、落盘、可加载性/字段验证后原子替换；失败保留原断点，不能先删唯一可恢复版本。
- 临时保存允许短时多一份；只清理当前运行已替代且登记的断点，不删除数据、缓存、日志和结果。
- 修改恢复逻辑须验证保存失败保留旧断点、恢复后优化器/步数/数据位置连续；历史入口 `scripts/test_lightning_resume.py` 可参考，不依赖已删除历史权重。
