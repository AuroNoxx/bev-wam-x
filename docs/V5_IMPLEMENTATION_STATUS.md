# v5 实施状态（2026-11-17）

2026-11-19 控制标签一致性复核已完成。将实际 SFT 控制标签输入训练积分器，2,000 训练窗和 128 验证窗的轨迹重建 ADE 分别为 1.806 m、1.011 m。17 个仿真验证巡航窗的横向控制限幅将 5 秒横向绝对误差中位数从 0.015 m 推高到 0.985 m。仿真巡航与实车巡航的 750 输出平均偏向相反，缺省导航、噪声反号和实车历史状态差分均未在各自的配对检查中解释横向偏差。优先核验有界控制目标的可实现性，再决定是否补训。[标签与实车同窗诊断](V5_REAL_ROAD_REPLAY.md)

2026-11-19 已完成真实道路回放。四城 8 段日志、39 个窗口，冻结 162 / 750 的 ADE 分别为 4.16 / 4.27 m，匀速为 3.49 m。两个版本在 14/14 个直线减速窗的终点均偏左，原始 750 在 12/12 个巡航窗也输出正向平均横向加速度。当前暂缓扩训，优先核查直线段的控制标签和输出分布。[实车指标与坐标复核](V5_REAL_ROAD_REPLAY.md)

2026-11-19 工作优先级转至全速域评测。固定第 162 步复算现有验证：近稳速巡航 ADE 3.142 m（匀速 1.548 m），两个重叠的制动刹停窗口均有明显纵向过冲；训练和验证均缺 60–80 km/h，80–120 km/h 缺独立验证。下一批优先补过渡、高速巡航、跟车减速与变道的独立路线。[全速域能力盘点](V5_CAPABILITY_SCORECARD.md)

2026-11-19 第 162 步已在 10 条未用于本轮拟合、选模和此前 v5 低速诊断的道路进行冻结复验。起步由 3/10 升至 10/10，严格停车维持 1/10，停车 ADE 由 0.422 增至 0.536 m。停车预测在第 3–5 秒持续增加。维持本地修订，暂缓云端扩训。[新道路复验与逐秒诊断](V5_GEOMETRY_REPAIR.md)

2026-11-18 新增 26 对同路段停车与驶离窗口，23 对进入拟合，3 对整路留出。配对补训共 648 次更新，开发集选中第 162 步。固定 128 窗口整体 ADE 3.164 m，停车 ADE 0.697 m，明确起步 5/7，三项本地判据均通过。严格停车仅 2/16，留出 3 对的起步 3/3、严格停车 0/3。保留该检查点作为独立复验候选。详细范围与不足见[配对补训结果](V5_GEOMETRY_REPAIR.md)。

2026-11-18 历史帧独立检测及框尺度对齐找回两例快速驶离。固定规划权重后，两例自车 5 秒预测位移分别从 0.53、2.69 m 增到 0.88、4.83 m。128 窗口整体 ADE 3.236 m，停车 ADE 0.975 m，明确起步 4/7，未达到 5/7。继续采用原第 750 步规划器。[逐帧核对与规划复验](V5_GEOMETRY_REPAIR.md)

2026-11-18 纯 RGB 尺度跟踪在 36 个 CARLA 验收窗口的前车位移 MAE 为 0.087 m，静止误报 0/12。纵向头补训 552 次更新，固定 128 窗口整体 ADE 从 3.410 降至 3.246 m，停车 ADE 从 2.485 降至 0.991 m；明确起步仅 3/7，低于原模型的 5/7。继续保留第 750 步规划器。两例前车已明显驶离，但跨帧光流跟踪中断。[分项复验与逐窗口结果](V5_GEOMETRY_REPAIR.md)

2026-11-18 前车位移读出使用 16 个新路段完成 192 次更新。在另外两张地图的 12 个路段验收，精确率 75%、召回率 87.5%，位移 MAE 0.806 m，高于零位移基线 0.741 m。本次读出未接入纵向控制。[补训与独立验收明细](V5_GEOMETRY_REPAIR.md)

2026-11-18 完成 Town03/05 共 12 个新路段的低速分支配对。冻结读出在 12/12 对中给驶离分支更高的位移预测，按原阈值命中 11/12 次驶离，误报 4/12 次静止，精确率 73.3%，未达到接入纵向控制的判据。[新道路配对验证](V5_GEOMETRY_REPAIR.md)

2026-11-18 复核了前车框的原图运动与历史相机外参。完整姿态及有方向的特征匹配使内部留路线位移 MAE 降至 0.537 m，固定阈值下仍有 11 次误报和 4 次漏报，未接入纵向控制。[历史图像对齐与误报复核](V5_GEOMETRY_REPAIR.md)

2026-11-18 已完成按历史时序估计前车位移的内部留路线验证，48 个窗口中命中 11 次、误报 8 次、漏报 3 次，未通过接入纵向控制的判据。[前车驶离验证](V5_GEOMETRY_REPAIR.md)

2026-11-18 本地修订已在原 128 个窗口配对验证。World 前向走廊 AP 从 0.0841 升至 0.1214。两次纵向修订分别出现停车漂移或起步漏判，目前保留原规划器。结果与各组样本数见[停车、起步与 World 定位复验](V5_GEOMETRY_REPAIR.md)。

第 750 步已完成[停车/起步多种子回放、指令干预、积分检查与 World 变化区诊断](V5_STOP_WORLD_AUDIT.md)。同一目标和近似初速下，模型在训练窗口也难以拉开停车与起步的纵向位移；World 对变化格有小幅收益，尚未达到扩训门槛。

## 第 750 步逐窗口诊断（2026-11-18）

以最终断点、同一验证顺序、每窗口同一个 Flow 噪声种子和 10 步 ODE 重放 128 条轨迹。逐窗口 ADE 与 5 秒 FDE 对原日志的最大差值为 0 m。模型 1/3/5 秒终点误差依次为 0.825/3.869/8.289 m，匀速为 0.757/3.502/7.001 m。逐窗口模型 ADE 胜出 51/128，平均差 +0.391 m；先在各路线内取均值再等权汇总 9 条路线，差值为 +0.087 m。

| 按专家轨迹或控制划分（分组可重叠） | 窗口 | 模型 ADE / 匀速 ADE（m） |
|---|---:|---:|
| 5 秒内距初态最远不足 0.5 m | 16 | 2.485 / 0.246 |
| 初速 ≤3 km/h 且 5 秒内距初态最远至少 0.5 m | 8 | 5.262 / 7.264 |
| 前 3 秒均值纵向加速度低于 −0.5 m/s² | 3 | 5.828 / 7.279 |
| 3 秒时横向位移绝对值 ≥1.2 m（含弯道） | 37 | 3.716 / 3.592 |

近静止组来自两条路线，目标巡航速度均为 0。模型在 16 条上全部输给匀速，1/3/5 秒终点误差为 0.385/2.158/8.662 m，5 秒末平均向前行驶 8.74 m、速度 5.00 m/s。该组贡献全部窗口相对匀速的净 ADE 差额的 71.5%；组内平均仅约 0.3% 的动作分量被裁剪。全验证集裁剪比例与模型超出匀速的 ADE 差额的 Pearson r 为 0.071。裁剪仍会使专家轨迹回放失真，但修裁剪无法解释近静止组的主要误差。训练清单中按同一位移口径有 112/2000 条近静止样本，6,000 次已提交抽样中有 281 次来自该组。

优先在训练集内复现并区分保持停车和低速起步的纵向预测，检查 Flow 从噪声采样到输出控制的分布及目标速度条件的影响。现有验证每个窗口只有一个固定噪声种子；减速组只有 3 条，横向位移也不能直接识别真实变道。当前数据不足以归因到单个模块或说明高速能力。

证据：[逐窗口结果及实际帧号](../artifacts/v5-qwen-drive-world/subset2000/final-window-audit/windows.json) · [分组与逐路线统计](../artifacts/v5-qwen-drive-world/subset2000/final-window-audit/summary.json) · [逐时刻预测轨迹](../artifacts/v5-qwen-drive-world/subset2000/final-window-audit/predictions.npz)。

## 2026-11-18 14:31（北京时间）第二阶段完成

最终 latest 断点核验：优化器全局步数与 EMA 步数均为 750，采样器累计抽取 6,000 个窗口（Bench2Drive 4,200、高速 1,800）。补训段完成 250 次更新，训练器记录 3,880.6 秒，含最终验证。128 个独立验证窗口的 ADE 为 3.410 m，5 秒终点误差为 8.289 m，逐轨迹峰值 Jerk 均值为 49.43 m/s³。World / Persistence Brier 为 0.014790 / 0.013944。按既定[扩训判据](V5_SFT_SCALE_DECISION.md)，轨迹与 World 整体指标均未达标；80–120 km/h 独立验证仍为 0 条。当前结果支持先诊断标签和分场景误差，暂缓付费全量联合 SFT。

2026-11-18 已在第 750 步最终验证前确定[大规模 SFT 扩训判据与 Colab 双卡预算](V5_SFT_SCALE_DECISION.md)。先核验本轮实际更新和独立验证，再按轨迹、World、分速域结果决定是否扩训。

## 2026-11-18：发现空轮，正在补齐第三个有效训练轮次

本次核验发现原进程虽标记finished，实际只有500次更新／4000次抽样；epochs.jsonl的第2个框架epoch为0条样本，不能宣称2000×3完成。第500步独立验证128条：ADE4.0757m、5s FDE10.1828m、逐轨迹最大jerk的均值55.5259m/s³；World Brier0.0137597，Persistence0.0128615，尚差约6.98%。验证集没有80–120km/h样本。

原因是采样器在最后一次yield之后重置轮内位置，Lightning按已知长度结束时不会执行该重置。已改为下一次迭代开始重置；另显式恢复轮末采样器状态，防止Lightning在epoch边界跳过loader恢复而重播初始顺序。CPU Lightning/TorchData契约检查已通过：2000+2000后恢复再2000，750次更新、6000次顺序完全一致，70/30来源配比及轮中恢复均通过。证据：`sampler-epoch-contract/result.json`；该检查不涉及驾驶模型训练组。

当前补齐入口：`python -m v5_qwen_drive_world.complete_three_passes --phase subset2000 --resume`。从完整step500恢复原模型、176项FP32优化器状态、EMA、RNG与数据顺序；原750步调度、原生像素、微批1/累积8及缓存身份保持不变，仅允许额外一个框架epoch补空轮，达到750步停止。完成状态改按实际更新数核验。原错误状态保存在`subset2000/empty-epoch-audit/`，不覆盖历史日志。此阶段结束前不补采、不扩大数据集。

真实恢复已通过：step500→501前8条样本顺序完全一致，继续到step502，峰值allocated14.462GiB，缓存namespace仍为c28f51f45a10735f0cf0eafe。补跑首次启动未产生更新：切换Lightning为max_steps模式改变了轮末恢复行为，故保持原epoch模式并通过回调在750步停止；原checkpoint SHA256未变，实际受影响断点的loops元数据CPU重放也已验证恢复后恰好2000条／250次更新。

## 2026-11-18：磁盘缓存加速

64条×30轮已完成；2000条×3轮从step80完整断点切换为冻结层磁盘缓存，分辨率、微批1/累积8、模型参数和损失不变。真实样本冷计算21.83秒，磁盘命中约0.50秒，联合损失与176项梯度一致性通过，续训前8条顺序一致并完成step81。缓存按需填充，不能把约43倍单样本热缓存加速套用到整段训练。剩余计算初估7.2小时＋验证/保存，预计缓存58GiB。新续训入口及checkpoint边界修正见[V5_DISK_CACHE_ACCELERATION.md](V5_DISK_CACHE_ACCELERATION.md)；实时进度以当前status为准。

最终入口`resume_disk_cached`已在step88/704条断点恢复，首微批之后完整checkpoint SHA256未变，后续8条顺序一致，第89次更新已完成。原生像素与微批1/累积8继续保持，实测峰值allocated14.46GiB。此为加速与恢复验证完成，第二步3轮训练尚在运行。

## 最新：补采与恢复检查通过，真实两步SFT运行中

已按用户要求完成数据清理及Town04实测高速补采，建立64条过拟合集与1400B2D＋600高速子集；具体标签范围、缺桶、独立高速验证缺口、训练实现见[V5_TWO_STAGE_SFT.md](V5_TWO_STAGE_SFT.md)。真实六视角缓存输出/KV和40项LoRA梯度一致性通过；实际SFT两次更新与保存恢复通过，176项优化器状态FP32、EMA步数一致、恢复后8条样本逐条吻合，故意保存失败后旧checkpoint SHA256不变。证据`overfit64/resume-step1.json`、`resume-step2.json`。

已启动`run_two_stage.py`顺序运行64×30轮、2000×3轮；活动阶段以`artifacts/v5-qwen-drive-world/two-stage-run.json`及各阶段`status.json`为准。原生六当前视角＋历史前视、微批1/累积8；恢复检查峰值allocated14.46GiB/reserved15.25GiB。首次冷缓存约13秒/样本，尚不能套用三相机诊断的0.82秒估算全程。两步指标只观测，不以13GiB或0.2m阻止进入第二步。下面“尚无manifest/训练器/高速数据”均为此前状态。

本页区分当前实测与历史；官方底座路径为 `models/v5-qwen-drive-world/Qwen-Drive-1.0-4B`。真实SFT已启动；GRPO和闭环评测尚未启动。

## 最新结果：2D World与后8层LoRA（2026-11-17）

用户最新授权已实施，默认`default.json`与`lightweight.json`采用同一条主线：**原生像素、当前全视角/历史前视、额外2×2视觉池化、32层完整前向/后8层LoRA、2D World、微批4/累积2/有效batch8、最多5轮**。未修改官方源代码、预训练权重、全局驱动，也未启动额外训练组。

### 改造前10步分项计时

`profile-before-lightweight.json`：含加载共82.25秒，完成10微批；后9个有效热态计时均值如下（去掉首次微批）。使用GPU同步计时，以真实KV梯度回传分离Flow/VLM反向；只有内存内诊断更新，不写权重文件。原配置快照`config-before-lightweight.json`用于复现。

| 阶段 | 秒/微批 |
|---|---:|
| 冻结ViT前向 | 0.6894 |
| VLM前向 | 0.9088 |
| VLM反向与checkpoint重算 | 2.1416 |
| Flow前向 | 0.0685 |
| Flow反向 | 0.1409 |
| optimizer＋裁剪（第8微批一次） | 0.1753 |
| 整个微批，含更新摊销及检查 | 3.9970 |

VLM反向约占53.58%，并非未经实测的75–85%。官方混合主干只有8层标准全attention，另24层为线性attention，不能按32层全部L²计算。此前确实出现Conv3d workspace OOM及选算子等待，但最后一次3D联合失败点为deformable slice_backward；不能泛化为所有ROCm Conv3d必然OOM。

### 已落地改造和正确性

- `components.py`保留32层及8组真实planner KV；layers[0:24]冻结无checkpoint，layers[24:32]每层都有PEFT LoRA。前6组KV无需梯度、后2组有梯度；无伪造缓存或层截断。
- `lightweight_input.py`对保留图片逐值复用原patchifier结果，当前三相机均保留，历史只保留前视。原生当前图704×800、历史384×416（宽×高），未设小分辨率。官方demo从12图/3400tokens变成6图/836tokens，其中555视觉、281文本。image占位及MROPE网格同步调整，ViT用原始网格。
- 所有输入视角都保留当前帧；额外相机不局限于官方三相机常量。8当前相机的CPU接口检查也达到850tokens，但复用RGB只验证接口，不能当成8相机GPU吞吐或标定验证。若固定2×2池化仍超预算，显式自适应平均池化，而不是少保留当前相机。`lightweight-input-contracts.json`确认保留像素和状态逐值相同、导航/历史文本完整。
- `world2d.py`的World有2,561,761参数、0个Conv3d。复用官方语言投影/归一化及重采样BEV query，新增cross-attention、动作条件残差和四级Conv2d上采样，输出[B,3,128,128]。8×8 latent与输出均为[Y,X]，物理范围保持不变；残差基底detach。新解码器没有预训练占用精度，未声称World<350MiB或反向<15ms。
- `torch.backends.cudnn.benchmark=False`；Lightning管理混合精度/累积/裁剪/更新，AdamW foreach、PEFT、TorchData、FLA与SDPA MATH继续启用。Flow积分FP32，训练参数和174项Adam状态FP32。
- `lightweight-causal-gradient.json`：实际32层预训练混合主干12文本token的后缀梯度为0，前24层参数无梯度，后8层各LoRA梯度有限非零。该检查包含checkpoint重算，不替代全场景因果和质量评估。

### 连续两次联合更新

三个容量档均在同一官方演示场景上完成两次Lightning优化器更新；第二轮累积包含已驻留的优化器状态。前若干微批不提前改参数，到更新边界LoRA/Flow/World均改变，第二次更新动作条件网络梯度非零。没有OOM或进程异常；不是长期稳定性/驾驶精度承诺。

| 微批 | 累积 | 有效batch | 热态秒/样本 | 秒/更新 | PyTorch allocated峰值GiB | reserved峰值GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 1.1341 | 9.0731 | 14.0785 | 14.9141 |
| 2 | 4 | 8 | 0.8904 | 7.1234 | 14.1219 | 15.3457 |
| 4 | 2 | 8 | 0.8194 | 6.5552 | 14.7910 | 15.6992 |

证据`lightweight-mb1.json`、`lightweight-mb2.json`、`lightweight-mb4.json`及同名stdout/stderr/资源日志。选择最快的4×2作为当前默认。**<16GiB只在PyTorch allocated/reserved口径通过**；Windows全卡专用占用峰值分别18.68/19.15/19.57GiB，包含运行库、桌面与其他应用，不能宣称整张显卡占用也低于16GiB。微批4时系统可用RAM最低3360MB。

### 3轮 / 5轮计算时间

`training-time-estimate-lightweight.json`按第二次更新的完整累积时间÷8计算，包含联合反向和优化器更新，排除首次编译。

| 微批×累积 | 假设7690有效样本：1轮 | 3轮 | 5轮 |
|---|---:|---:|---:|
| 1×8 | 2.42小时 | 7.27小时 | 12.11小时 |
| 2×4 | 1.90小时 | 5.71小时 | 9.51小时 |
| 4×2 | 1.75小时 | 5.25小时 | 8.75小时 |

这是同场景重复、诊断标量损失、三相机输入的工程计算估计，**不是完整SFT总工期**。7690是旧数据原始训练帧数，尚未转换为有效v5时序样本清单。正式manifest、REASON/PLAN监督、真实未来占用/EMA、正式SFT与保存恢复尚未实现；IO、验证、保存、教师计算、采集和Stage2不包含在表内。未达到0.4–0.6秒/样本，也不能承诺3–5轮充分收敛。没有后台正式长训练。

以下内容是改造前历史记录，保留用于追溯；其中“World未通过”“全32层重算”和旧耗时不再代表上述新2D路径。

## 历史结果（2026-11-17 16:52）

- 项目内FLA与HIP/TinyCC编译链已补齐；24层原生线性attention、因果卷积、gated RMSNorm已启用，值/梯度/因果检查通过。标准attention保留SDPA MATH，使用完整上下文的query分块和checkpoint；AdamW保留foreach与梯度裁剪。
- 原生RGB→Flow、微批1/累积8连续2次更新通过。第二轮4.01054秒/样本、32.0843秒/更新，allocated峰值14.5918GiB；相较原55.32秒/样本估计约快13.79倍。
- 联合World只完成过1个微批前后向，第2微批OOM，0次优化器更新；进一步的细粒度CPU卸载在前向原生退出0xC0000409，已撤回该改动。事件1000/1001记录ucrtbase.dll，不能从故障模块名断言根因。证据`granular-offload-native-exit-events.json`。系统未重启（启动时间仍14:51:49）。未继续启动联合长训练，验证进程已全部退出。
- 当前代码保留已验证的原生Conv3d深度分块与原生块checkpoint；CPU激活卸载不是默认启用项。NDHWC全网布局与expandable_segments未启用。World整系统尚未通过，不能将单算子结果视为可训练承诺。
- 新耗时文件`training-time-estimate-fla.json`：仅RGB→Flow同场景成本，20轮×1000有效样本约0.93天；假设旧7690帧全部有效则约7.14天。缺少World稳定更新、EMA、真实时序样本清单、IO/验证/保存/Stage2，完整训练ETA仍为空。正式数据入口和SFT训练器也尚未完成，未伪造正式训练运行。

## 2026-11-17 加速依赖与编译环境补齐

项目独立overlay已安装flash-linear-attention0.6.0、fla-core0.6.0、einops0.8.1，沿用现有torch/ROCm与triton-windows3.6.0.post26，不修改全局驱动。`runtime.configure()`为本进程指定已有ROCm SDK的HIP_PATH和Triton自带TinyCC，HIP驱动初始化与gfx1100内核编译执行已通过；不需要全局Visual Studio安装。可复现安装入口为`scripts/Setup-V5Accelerators.ps1`及固定版本requirements。

`fused-operator-verification.json`：FLA chunk gated delta rule、Triton causal_conv1d、FusedRMSNormGated的BF16输出/输入梯度与原生参考通过预设容差，后缀因果梯度为0；AdamW fused与foreach的FP32参数/状态更新对比通过。首次算子编译和验证合计363.26秒，不计作每步训练时间。随后实际4B全32层混合主干短文本因果反向也通过，后缀梯度最大值0。

首次接入融合AdamW在第8微批遇到Lightning BF16 precision插件对融合优化器梯度裁剪的拒绝，未执行优化器更新。保留原始失败证据`accumulation-native-fused-adamw-clipping-failure.json`；当前恢复foreach AdamW并保留梯度裁剪，FLA线性attention/卷积/归一化继续启用，标准Transformer attention仍是SDPA MATH。增量推理算子尚未验证，明确保留Torch实现。原生分辨率、微批1/累积8的连续两次更新与World联合复验结果后续记录。

下面“未安装FLA/编译未就绪”等内容为本次补齐前的历史诊断，不代表当前依赖状态。

FLA+foreach首轮更新通过后，第9微批因SDPA MATH临时706MiB分配触及16.99GiB限额；证据`accumulation-native-cap85-updates2-fla-foreach.json`。尝试PyTorch expandable_segments在本机加载阶段引起严重主存压力（可用455MB），已停止本次独立进程并撤回该选项，不能启用为默认；保留`-segments.json`及资源日志。随后以原生SDPA MATH按256个query分块、checkpoint每块，仅减临时矩阵，不裁剪任何图像/token/key；因果绝对位置掩码、非因果、boolean和additive掩码的值/梯度与原生算子对比通过，因果后缀梯度0（`math-query-tiling.json`）。完整模型复验首微批峰值13.04GiB；最终连续更新结果另记。

RGB→Flow原生12图/3400tokens已经完成16个微批、连续2次Lightning更新，132项优化器状态全部FP32且step=2，前7微批不更新，LoRA/Flow梯度有限非零。第二轮累积32.0843秒，即4.01054秒/样本；fit67.12秒，含加载/因果检查102.89秒。峰值allocated14.5918GiB、reserved15.3281GiB。证据`accumulation-native-cap85-updates2-fla-foreach-tiled.json`。与旧55.32秒/样本的外部监控估计相比约13.79倍，但单场景工程验证不能代替正式数据吞吐。

World原生联合反向仍受原生NCDHW卷积workspace限制；对18个原生3D残差块使用PyTorch checkpoint wrapper后仍未通过。针对真实失败尺寸[1,192,16,128,128]、核[64,192,3,3,3]验证MIOpen channels_last_3d布局，峰值2.971→0.627GiB，输出差0，输入/权重梯度相对L2差0.00238/0.00789，通过BF16容差。使用现成`nn.utils.convert_conv3d_weight_memory_format`转换原生卷积布局，本进程设置`PYTORCH_MIOPEN_SUGGEST_NHWC=1`，不变更网络/权重值/图像/网格。证据`conv3d-layout.json`；完整联合复验另记。

**后续联合结果否决了NDHWC默认启用**：在编码器反向选算子中长时间等待，py-spy原生栈停留MIOpen FindConvBwdDataAlgorithm / CK3D / hipEventSynchronize；已停止自身验证进程，证据`joint-ndhwc-offload-stall-stack.txt`，不能将单算子通过当作整系统通过，也不能仅据此断言驱动根因。当前改用NCDHW的原生Conv3d深度分块（4层输出、完整邻域halo，不删任何体素）。真实编码器[1,256,16,128,128]输出差0、输入/权重/偏置梯度相对L2最大0.00812；峰值4.138→1.860GiB（`conv3d-depth-tiling.json`）。

原生8相机联合首次微批前后向115.41秒、峰值16.146GiB，但第2微批保留旧梯度后在deformable attention的slice_backward申请190MiB失败，0次更新，`-trace-offload-fla-foreach-tiled-blocks-convtiles-encoder.json`。随后试验细粒度CPU卸载在前向原生退出，已撤回至整段encoder/decoder checkpoint；该失败见上方当前结果。TorchAO CPU梯度卸载[官方实现及限制](https://github.com/pytorch/ao/blob/main/torchao/optim/README.md)不支持梯度累积且不支持梯度裁剪，没有启用或更改累积语义。

## 保持分辨率后的进展与耗时

用户继续授权保持分辨率训练后，仅将本进程allocator上限由75%调整为85%（约16.99GiB），未降低图像尺寸、未截断层、未量化。官方原生processor的12张历史图像、3400 tokens，在微批1/累积8下完成1次Lightning AdamW更新：前7批不更新，第8批LoRA/Flow参数更新，132项优化器状态FP32且step=1；peak allocated15.4650GiB、reserved16.5996GiB。证据 `accumulation-native-cap85.json`。同一演示场景重复8次、诊断标量损失，不是正式SFT或质量收敛证据；也尚未证明优化器状态驻留后的第二轮累积。

包括模型加载共475.16秒（7分55秒）；外部监控估计计算442.56秒，约55.32秒/样本，监控采样约2.3秒。操作系统统计的共享GPU内存峰值约2.57GiB，包含桌面与其他应用，不等于PyTorch allocated。正式v5 manifest不存在，不能启动名义正式训练。

按20轮、同样样本成本计算，`天数=20×有效样本数×55.32/86400`：1000样本约12.81天；若现有train的7690帧都成为有效样本，约98.47天；若7690个有效B2D样本再按70/30补齐高速数据，约10986样本、140.68天。原始帧数不是已经制作好的v5时序样本数。这些估计均未含World、EMA未来目标、数据IO、验证、保存恢复、采集或Stage2；同一场景8微批不能确立整个数据集的平均速度。**完整系统总耗时尚无法可靠给出**。计算记录 `training-time-estimate.json`。

当前实际框架为Lightning2.6.6、PEFT0.21.3、TorchData0.11；已用AdamW foreach、SDPA MATH、checkpoint、grid_sample和index_add。Qwen3.5线性注意力实际调用 `torch_chunk_gated_delta_rule`，融合线性注意力和causal_conv1d均未启用；不能宣称全套融合加速。原始perception CUDA JIT不可用，使用官方PyTorch路径。核对记录 `operator-memory-verification.json`。

用户追问未启用原因后的实查：`fla`/`causal_conv1d`均未安装；现有triton-windows3.6.0.post26包含AMD后端，但默认驱动初始化编译找不到HIP头文件。头文件实际存在于项目环境，临时设置本进程HIP_PATH与venv Scripts PATH后，错误推进为 `stdlib.h not found`，并提示未找到MSVC/Windows SDK。说明当前加速依赖与编译工具链尚未就绪，**没有证明7900XT或Windows/AMD本身不支持融合内核**；SDPA MATH也不要求原生线性注意力使用回退实现。全局环境未改，尚未完成融合内核前后向/因果验证。证据 `acceleration-readiness.json`。

World内存优化保持所有相机/射线/网格：体素聚合将紧随其后的相机求和提前合并，并分块加权index_add；查询独立的deformable attention按256查询调用同一官方grid_sample路径并checkpoint。CPU/ROCm数值和梯度验证通过：deformable输出最大差0，GPU梯度最大差9.54e-6；体素FP32值/梯度匹配，BF16相机求和顺序改变，诊断最大差0.03125、梯度差0。证据 `perception-memory-operators.json`。这些不是额外训练组。

原生8相机896×512、3654 tokens联合诊断曾在World grid_sample申请760MiB时触及16.99GiB限额，0次更新；失败已单列 `accumulation-native-joint-cap85-grid-sample-failure.json`。查询分块后联合前向通过，首次反向仍失败；要求的两次更新均未完成。

使用PyTorch dispatcher定位到 `aten.convolution_backward.default`：输入[1,192,16,128,128]、核[64,192,3,3,3]、BF16的占用解码器3D卷积，反向申请约2.53GiB时触及16.99GiB allocator限额。原生backend选择检查中BF16/FP32均为Miopen，不能误称该卷积没有使用GPU库。证据 `accumulation-native-joint-cap85-updates2-trace.json`。试用现成 `torch.autograd.graph.save_on_cpu(pin_memory=True)` 仍在同一算子失败，系统可用内存最低3.247GiB，故未启用为默认配置；证据同名 `-trace-offload.json` 和资源日志。当前联合SFT仍未启动，正式manifest和训练器也尚未完成。所有本轮诊断进程已退出，系统启动时间仍为14:51:49，没有再次重启。

## 14:51重启事件：停止自动GPU压力测试

原生分辨率、batch8的容量测试最后写入 `requested_batch_backward`（每场景3400tokens）。用户报告电脑卡死并手动重启；系统启动时间14:51:49，事件41/6008确认非正常重启。这次测试是**中断/未通过**，不是成功，也没有捕获到可确认的OOM异常或峰值显存。

刚重启后系统报告中还出现AMD/LiveKernelEvent，但附件指向11月3日旧dump，不能据此断定本次根因。现有证据不足以区分显存/共享内存压力、ROCm算子/驱动卡死等原因。重启后没有Python/CARLA残留进程，没有自动续测/训练。

`check_memory.py` 已改为默认只预览，默认路径不导入torch、不加载模型；原物理batch8压力执行已禁用。用户随后明确授权微批1、梯度累积8继续验证，配置与AGENTS已同步，单卡有效batch=8。新验证使用Lightning、原生权重与独立证据记录，设置75% PyTorch allocator上限（约14.99GiB）；该上限不覆盖所有驱动分配，也不保证驱动一定不会卡死。

## 微批1 / 累积8 验证

**用户纠正：没有授权降低分辨率。以下128×128记录仅保留为工程诊断，不能算本次原生分辨率要求通过。入口默认已恢复官方processor尺寸，拒绝缩小图像；现有原生processor尺寸也不等于未来1920×1080采集目标全部验证。**

`verify_accumulation.py` 使用Lightning自动优化、BF16 mixed、SDPA MATH、PEFT、StatefulDataLoader和AdamW foreach，8个微批后仅执行1次更新，不保存新权重。官方真实12张历史RGB降至128×128、538 tokens；同一场景重复8次，仅用于工程验证，不是8条独立训练样本，也不是正式SFT。

首次测试暴露官方from_pretrained默认eval而验证入口未显式train，导致checkpoint未生效；在allocator限额内报OOM，证据另存 `accumulation-128-eval-mode-failure.json`。入口改为显式train并断言后通过：前7批LoRA/Flow参数未变，第8批完成更新；两支梯度有限非零，132项AdamW状态均FP32且step=1。峰值allocated14.0045GiB、reserved14.4473GiB，包含模型加载总耗时61.51秒。证据 `accumulation-128.json`。这是RGB→Flow的累积更新通过，World联合检查另记，不等于完整系统或高分辨率通过。

随后128×128联合诊断未通过：首次World编码的体素聚合请求额外2GiB时触发14.99GiB allocator限额（当时PyTorch allocated13.46GiB，设备仍空闲5.91GiB），0次更新，详见 `accumulation-128-joint.json`。这是受限额触发的OOM，不能说整张20GB卡已经用满，更不能据此认定重启前整机卡死的原因。

恢复官方processor尺寸后的微批1/累积8验证：12张历史图像、3400 tokens，首个微批的SDPA MATH前向申请706MiB时触发14.99GiB allocator限额，0次优化器更新；设备仍空闲4.72GiB。PyTorch peak allocated13.6822GiB、reserved14.5684GiB。约2.3秒采样一次的系统监控未显示内存耗尽（最低可用约11GiB）。这是受限额影响的未通过，不足以判断无此限额时20GB能否容纳完整训练。程序正常退出，无整机重启。证据 `accumulation-native.json`、`memory-diagnosis.json` 及 `logs/v5-qwen-drive-world/accumulation-native-memory.jsonl`。不继续重复加压，也不以降低分辨率解决。

## 已完成

- 检查旧训练：没有活动trainer，保留TensorBoard及其他任务的未提交源码。36个旧权重文件已删除，逻辑大小20,072,937,901字节，清单 `artifacts/v5-qwen-drive-world/weight-cleanup.json`。
- 164份历史文档归档；README、ARCHITECTURE、AGENTS切换到v5。框架/算子、不加无意义守卫、不做额外对照组已写入AGENTS。
- 递归清理最初被工具执行层拒绝，未换工具执行；用户运行 `scripts/Clear-Obsolete-V5.ps1 -Apply`，22个明确旧目录已删除，逻辑大小20,409,829,313字节，实际记录磁盘空闲增加19,780,202,496字节。详见 `user-cleanup-20261117-140701.json`。原始数据、日志、模拟器、训练环境保留。
- 官方主干、planner-rl、perception共11,658,738,446字节已下载，固定revision `28484089a7cc8c335cf5089fb0745cf7c49b6eaa`，三份SHA256全部验证。官方源码commit `28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0` 未修改。
- 当前torch2.11.0+rocm7.14.0、gfx1100；PEFT0.21.3与Ninja1.13.2安装于独立runtime/v5-qwen-drive-deps，未替换旧训练环境依赖。

## 工程证据

| 项目 | 实际结果 | 范围限制 |
|---|---|---|
| 三份官方组件加载 | VLM723个键核对、planner严格加载、perception严格加载 | 不是整系统训练完成 |
| PEFT LoRA | 3,145,728可训练参数，FP32，梯度有限非零 | full-attention q/k/v/o；原生线性attention保留 |
| 因果性 | SDPA MATH和预训练混合主干短序列后缀梯度最大值0 | 短文本检查，不是所有数据因果认证 |
| 24层 | 文本前向有限 | 6组KV不兼容未修改32层planner所需8组；当前采用全32层checkpointing |
| FP32运动学/Euler | 0、5、20、60、100、120km/h数值与梯度检查通过 | 不等于实车动力学或闭环验证 |
| 原生RGB→控制Flow | 官方真实demo12历史图像、538 tokens、50×2控制速度场，梯度回到LoRA | batch1/图像128×128/未接World；非正式训练，0次优化器更新 |
| 同次RGB检查显存 | peak allocated12.294 GiB，前后向9.33s | 不含初始化和完整训练优化器；不能推算batch8/5Hz |
| 70/30与速度分桶 | StatefulDataLoader恢复后5批逐值一致，20批来源比例0.3，标签阈值检查通过 | 工程索引数据；不代表高速数据已存在 |
| 感知算子 | 官方PyTorch deformable-attention路径和PyTorch index_add体素聚合的GPU梯度通过 | 原始CUDA JIT缺少本机CUDA工具链，未编译成功 |

证据：`component-verification.json`、`native-rgb-gradient.json`、`contracts.json`，均位于 `artifacts/v5-qwen-drive-world/`。

## 历史3D World实施（已由world2d.py替代）

`v5_qwen_drive_world/world.py` 已实现官方FPN/DepthNet/UVTR/BEVTransformer/Occupancy decoder的扩展，新增部分仅为动作条件未来残差头。128×128走廊的几何重新投影、BEV query物理迁移、位置嵌入重采样、60→160m深度bin初始化逐项记录于migration。新增远距bin未训练，不宣称远距识别成立。

现成PyTorch算子实现ROCm可求导路径，不修改官方源码。BF16三维BatchNorm的MIOpen错误改用现成FP32归一化；FP32 master/BF16特征scatter类型不一致通过保留FP32 LayerNorm、对齐scatter源dtype解决。8路真实标定相机、128×128图像、128×128走廊的World前后向已通过：输出[1,3,128,128]，残差头梯度有限非零，峰值10.625GiB，45.96秒。VLM与World分开驻留验证，没有联合训练/优化器更新。证据 `world-verification.json`，失败历史单独保留。

SFT联合损失、footprint占用采样、TTC/动力学/推进奖励和组内相对优势算式在 `objectives.py`；未建立完整训练器和数据输入链。World之外的当前/历史目标速度跟踪与车道输入仍需接入，不能从单张占用图假造障碍速度。

## 数据缺口（已查实）

遍历现有60路线共9,327帧原始标注，速度分桶为2,640 / 5,883 / 793 / 11 / 0 / 0（0–20…100–120km/h）；没有80–120km/h样本。原始RGB在保留的Bench2Drive归档内，旧clips主要提取了anno/OCC。原OCC仅±25.6m，不能当成112m走廊真值。

`configs/v5-qwen-drive-world/collection.json` 已列Town04/Town06、80/100/120km/h、六类工况、传感器和split契约。**尚未采集新的高速专家数据**。采集器、标定和高速专家跟踪需完成后才能执行70/30正式训练。证据：`data-audit.json`。

## 下一实施顺序

1. 容量门控已完成：原生像素＋压缩token＋后8层LoRA＋2D World＋Flow，三档微批各两次更新。正式数据接入后仍需核对真实长度、相机数与数据批次峰值。
2. 接入REASON/PLAN特殊token及12类意图监督；联合计算通过不等于完整四层训练链已完成。
3. 建立真实RGB/未来占用数据读取、与新2D latent一致的EMA教师、完整高速采集与速度均衡训练manifest；变长batch需bucketing或明确mask支持，当前重复演示场景没有验证该项。
4. 用Lightning实现正式SFT、3–5轮warmup+cosine、best/latest与完整恢复；以真实数据验证一次保存恢复连续性后再启动正式训练。
5. 验证规定的Persistence指标；建立连续Flow的真实GRPO概率/转移分布，再启动Stage2。当前只有优势和奖励算式，没有伪装成GRPO的加权FM训练。
6. 接5Hz/20Hz闭环和用户指定动态介入，验证六类工况与全速域。当前无v5闭环结果。

## 当前可运行入口

```powershell
.\scripts\Setup-V5Accelerators.ps1
.\.venv-train\Scripts\python.exe -m v5_qwen_drive_world.verify_lightweight_inputs
.\.venv-train\Scripts\python.exe -m v5_qwen_drive_world.verify_lightweight_causality
.\scripts\Verify-V5Lightweight.ps1 -Microbatch 4
.\.venv-train\Scripts\python.exe -m v5_qwen_drive_world.estimate_lightweight
.\.venv-train\Scripts\python.exe -m v5_qwen_drive_world.audit_data
```

以上是明确工程入口，没有正式训练脚本的占位启动器，也没有后台长训练。`Verify-V5Native.ps1`、`verify_world`、`verify_accumulation --with-world`和旧estimate_time保留为历史3D诊断入口，不是当前主线验收。
