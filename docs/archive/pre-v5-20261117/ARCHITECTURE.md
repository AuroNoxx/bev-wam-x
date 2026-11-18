# Completed experiment: isolated native v2

Native v1 completed20epochs and its final weights are preserved. Native v2 completed20epochs and final paired checkpoint audit: 12frozen Qwen layers,256historical scene tokens,AdaLN-Zero Action conditioning,zero-gated World-to-Action and deterministic residual World. See [v2 implementation/run record](OCC_QWEN_NATIVE_V2_20261117.md) and [next-stage plan](NEXT_STAGE_ARCHITECTURE_PLAN.md). Active status is `artifacts/occ-qwen-native-v2-training/status.json`. No future3D OCC decoder or OccWAM second stage has been added. The v1 and older classical records below are retained for history, not silently applied to v2.

Final results: [v2 final audit](OCC_QWEN_NATIVE_V2_FINAL_AUDIT_20261117.md). Best ADE3.171468m; not established as closed-loop safe. No further trainer started.

# Native v1 architecture record (2026-11-17)

The current authorized experiment supersedes the classical-only decision of 2026-11-15. The classical pipeline remains preserved as a historical baseline, not the current training architecture.

**Current path:** historical native Bench2Drive OCC + ego state/navigation -> OCC encoder/projector -> frozen Qwen text backbone, with bidirectional Scene/World/Action token interaction -> Action Flow future ego trajectory and World future latent prediction.

- Input: four historical 3D OCC grids, `[B,4,20,128,128]`, native 0.4 m voxels, XY `[-25.6,25.6)` m and Z `[-5,3)` m. No preprocessing downsample-then-enlarge or invented forward coverage. Height is encoded into CNN channels; learned feature compression produces a 4x4 spatial token lattice, not a future 3D voxel output.
- Historical frames align to current ego coordinates; out-of-support cells remain unknown. Unfilled is not asserted to be verified free. Future observations are training labels only, not actor inputs. This does not certify upstream label-generation causality.
- World output is **latent only**. Future 3D OCC decoding and OccWAM scene-to-ego second-stage correction are **not implemented and not part of the current authorized experiment**.
- Native v1 completed all20 epochs atstep13240 on2026-11-17; see [final audit](OCC_QWEN_NATIVE_V1_FINAL_AUDIT_20261117.md). Isolated v2 implementation is now authorized under [the next-stage plan](NEXT_STAGE_ARCHITECTURE_PLAN.md). Do not duplicate any active run or overwrite v1weights. Main runtime entry: `scripts/train_occ_qwen_native_optimized.py`; configuration: `configs/occ-qwen-native.json`; live status: `artifacts/occ-qwen-native-training/status.json`.
- Active execution options: native aligned cache and channels_last, workers=0. Fused AdamW was excluded after its numerical gate. Full best/latest checkpoints retain optimizer, scheduler, EMA, RNG and exact loader state. Training/validation progress is not proof of CARLA closed-loop reliability.

Implementation details, authorization chronology and verification: [OCC_QWEN_NATIVE_INPUT_20261116.md](OCC_QWEN_NATIVE_INPUT_20261116.md). Initial joint-flow design: [OCC_QWEN_JOINT_FLOW_20261116.md](OCC_QWEN_JOINT_FLOW_20261116.md). Preserved classical baseline: [OCC_CLASSICAL_PLAN_20261115.md](OCC_CLASSICAL_PLAN_20261115.md).

---

## Historical architecture record (below, through 2026-11-15; not the current direction)

# 当前架构决策

2026-11-15 最新用户指令：“不用flowmaching和世界模型，先把自动驾驶做出来”。此决策优先于下方历史v1目标。

当前目标：CARLA 中由当前/历史 OCC、自车状态和导航/车道地图驱动的自动闭环。历史 OCC → 障碍运动估计与短期多假设预测 → 行为选择 → 带时间轨迹搜索/优化 → 占据碰撞、道路及执行约束检查 → Pure Pursuit＋速度PID。无需先训练规划神经网络；不加载FM/world权重，不加入RGB训练，不移植旧BEV项目方案。

新实施顺序、输入权限、验收和现有缺口见 [OCC_CLASSICAL_PLAN_20261115.md](OCC_CLASSICAL_PLAN_20261115.md)。这是已选定的实现路线，不是已经实现/验收的驾驶能力。旧模型、缓存、配置和评估保留。原输入坐标、unknown、数据隔离、环境和断点安全要求继续适用。

## 历史固定架构 v1（停止作为当前实施目标）

决策日期2026-11-14。此文是实现目标，不代表已经完成。

## 模块职责

1. 输入历史OCC、自车速度/加速度/航向/角速度，以及导航目标、可行驶区和车道约束。未来真实数据只作标签。
2. Conditional Flow Matching生成8条候选轨迹，每条5秒，间隔0.2秒，共25个未来点。点包含x/y/yaw/speed；学习时角度用sin/cos表达并校验归一化。不输出油门/刹车/转向。
3. 使用与执行端一致的Pure Pursuit和速度PID，在校准自行车动力学近似中模拟跟踪，得到预计实际轨迹。检查制动距离、加速度、转角及变化率、横向加速度和跟踪误差。车辆参数按车型读取/实测，不凭空假设。
4. 轨迹条件世界模型输入历史OCC、自车状态、预计实际轨迹，预测1–5秒未来动态占用与不确定性。静态地图按坐标关系保持；标准接口不要求踏板序列。
5. 碰撞、不可执行、越界等先作为硬约束，之后比较进度、目标速度、舒适性和频繁变道。检查车辆外形扫掠体积，不只检查中心点；对未知空间和不确定性保守评分。
6. 选中轨迹后，Pure Pursuit计算转向，速度PID计算纵向需求并经标定映射为油门/刹车；处理饱和、积分累积和同时踩两踏板。
7. 初始目标规划5Hz、控制20Hz，每次只执行到下一规划周期。无可行候选/超时触发保守车道保持和制动；不声称所有状态下保证安全。

深刹/浅刹由带时间速度曲线和车辆能力决定，控制器模拟用于提前检查能否执行；不能默认车辆完美跟随目标轨迹。

## 输入与坐标契约

- 统一当前自车x前/y左/z上，单位米、秒、弧度。CARLA坐标转换必须覆盖位姿、向量、yaw、角速度及转向符号。
- 保存frame id、时间戳、位姿、体素原点/分辨率/尺寸、类别与有效性掩码，缓存附来源哈希和预处理版本。
- occupied、verified-free、unknown分开。Bench2Drive的unfilled不等于已验证空闲，包围盒代理标签不等于精细表面。
- 历史对齐当前自车；未来OCC也输出到当前固定坐标，附有效覆盖掩码。源未来自车坐标OCC用标签位姿转换，视野外不填为空闲。旧测速采用未来逐帧自车坐标，尚未满足新契约。
- 未来观测不能泄漏到当前输入。候选驶出已知覆盖不能假设世界模型知道未观测环境。
- 旧测速4×128×128、前后各25.6米仅作性能参考。正式高速范围须依据制动距离、遮挡与资源确定；短视距阶段限速，不宣称90–120km/h能力。
- 运行时是CARLA几何/标注构造的OCC，不是RGB感知预测；新旧引擎的类别和几何必须重新对齐。

## 世界模型训练边界

先使用实际专家轨迹监督未来OCC。这只能建立日志分布预测器，不能证明它准确预测未尝试动作和其他车辆的反应。随后按闭环失败补可复现的干预数据；同一初始条件的分支属于同一split，日志回放不能冒充真实反事实响应。

记录候选来源、执行近似误差，后续混合实际闭环输入，减少真值条件与推理条件差异。分别评测动态车辆、未知覆盖和风险校准。

## 实施顺序

1. CARLA版本和OCC适配验证；控制器先独立跟踪预定义轨迹。
2. OCC-only数据、FM轨迹生成、规则评分基线。
3. 轨迹条件世界模型与执行模拟，接入预测评分。
4. 固定候选数/随机种子比较有无世界模型评分。
5. 定向补数据；不因一次失败换成直接动作或RGB架构。

首轮白天干燥、多车道巡航/跟车/变道/超车/取消。出口和汇入数据可辅助学习，但复杂城市与匝道不列入首轮闭环必达目标。

## 测试可视化

训练后评测提供CARLA自由/跟车视角及独立OCC窗口，见 EVALUATION_VIEWER.md。显示实际输入、候选/选中轨迹和不同预测时距；与控制异步运行，不增加感知模型。

## 当前实现补充（2026-11-15）

新入口见 STAGES.md：Flow 与轨迹条件世界模型已实现并通过真实参数更新测速；坐标缓存升级为 preprocess_version=2。公开标注的 angular_velocity 与位姿差分交叉检查，单位为 rad/s，转为 FLU 时只翻转 z 符号，不再次做 degrees→radians 转换。

阶段 1/2 离线训练网格为 20×64×160，完整保留原生高度层，XY 聚合到 0.8 m。当前源观测只有约前后 25.6 m，扩展未来输出网格不代表新增感知范围。闭环的 90–120 km/h 目标仍须扩展有效观测并验证制动距离，当前不能宣称成立。

2026-11-15 配对修正：当前 pinned Bench2Drive 数据实测为 annotation/pose i 对应 OCC 文件 i-1。独立 `data/occ_v3` 保留原训练/验证窗口，只修正 OCC 文件索引；位姿、状态、导航、未来轨迹索引不变，记录 OCC/pose 来源 ID。旧 `data/occ_v1` 保留作审计；正式修复验收以 `artifacts/diagnostics/alignment-v3/final-verification.json` 为准。该修复不代表 0–120 km/h、远距观测或闭环驾驶已通过。

第一批正式 split 为 Town12/Town13/其他地图的 42/9/9 段，防止重叠道路跨 split。源码、数据及模型旧分支不会被新入口读取。控制器标定、实际可执行轨迹条件构造、风险评分及完整闭环仍属于后续阶段，未被此次离线模型测速冒充完成。
