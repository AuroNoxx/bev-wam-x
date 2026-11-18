# Native OCC-Qwen v2：CARLA 0.9.15 闭环诊断（2026-11-17）

## 结论与本报告的主运行

**模型能够从静止起步并驱动车辆，但当前没有通过可靠闭环驾驶验证。** 最终三场采用完整语义 LiDAR 扫描、原 Bench2Drive 在线 OCC 生成器、同一 v2 best 和同一控制器。转弯和行人场景发生静态物体碰撞；前方障碍场景虽未碰撞，却有明显驶出行车道，不能认定完成了有效避障。

唯一主结果目录：`artifacts/native-v2-carla-20261117/model-upstream-fullscan/`。每场 29.9500004 秒有效仿真、150 次规划；另有 2 秒真实观测预热。不是 Bench2Drive 官方 ScenarioRunner 的完整场景或排名成绩，而是使用其 XML 路线几何的三个可审计仿真诊断。早期其他目录均保留为开发/接口排错证据，不能与主结果混合。

| 场景 / 官方 XML Route | 实际行驶距离 | 路线投影进度 | 行车道外帧占比 | 碰撞与实际行为 | 保护接管 |
|---|---:|---:|---:|---|---:|
| 跟路/左转 / Town03 26990 | 158.54 m | 100.00%* | 48.50% | 起步、持续加速、切弯并驶过目标区域，20.80 s 撞墙；未满足终点到达条件 | 0 |
| 前方静止车辆 / Town03 26456 | 59.97 m | 47.45% | 50.17% | 初期缓慢起步，随后从前车侧方经过；无碰撞，但路线偏移最大 5.28 m，不能算合格避障 | 0 |
| 独立定时横穿行人 / Town03 27515 | 24.64 m | 29.64% | 49.50% | 行人实际移动，未记录行人碰撞；车辆偏离路线，19.75 s 撞电杆，之后受阻 | 0 |

*投影进度是单调最近路线索引，不是成功完成率。左转场景在偏离路线时索引已投影到末点，但距离/横向条件不满足，最终以 30 秒上限结束。不能把此处的 100% 写成完成路线。三场均以仿真时限结束；没有推理超时、非有限值或保护强制刹车。

碰撞事件保留全部回调，墙 176 次、电杆 202 次分别是持续接触产生的重复事件，不是 176/202 个不同物体。左转首次碰撞时自车世界坐标约 `(14.4155, -125.3685, 0.0266)` m；行人场景首次撞杆时约 `(-68.2816, 150.0027, 0.1294)` m。CARLA CollisionEvent 不提供接触点；日志记录回调时自车位置、对方 actor/type/location、冲量。静态物体 actor id 可能为 0、位置为零，不能把其零坐标当真实接触位置。

## 权重、版本、数据划分和保护

开始前检查了 Windows 进程和 GPU 计数器：未发现训练器、CARLA 或其他已识别的 GPU 计算任务；有现存 TensorBoard，未停止或修改。`setup-audit.json` 保存进程快照、配置、manifest、原生缓存校验及文件哈希。

- 主权重：`artifacts/occ-qwen-native-v2-training/best.ckpt`，epoch 17 / step 11254，SHA256 `d31bbce06427fa1dd7f9685a5d67dd170b117f72ddf75f09f8b76b8b3a65f354`。
- latest 保留：epoch 20 / step 13240，SHA256 `5a29b10009fe02e0cfed2906d1df7d480f215a3af03dabd972230789430186f2`。
- 两者均为 1,320,079,131 字节；结束校验与开始相同，见 `checkpoint-preservation.json`。无删除、覆盖、训练或优化器更新；既有训练数据、缓存和 v1/classical 代码保留。
- 实际加载 `NativeOccQwenV2`，严格加载全部 288 个 state_dict 条目；配置与 checkpoint 内 `task_state.settings` 完全相等，schema 为 `native-occ-qwen-v2-lightning-v1`。
- 本次训练 manifest SHA256 `96549c6bf1d4d131138933e5aa9ee64b250eae7aba19ed329e8deec1c7aa8b1d`，5296 个 Town12 训练窗、1124 个 Town13 验证窗；缓存构建/读取身份通过核对。本次新增 Town03 三条路线不在训练或验证路线清单内，证据在 `setup-audit.json` 的 `scenes/train_routes/validation_routes`。
- 这是对本次 v2 微调训练集的空间隔离证明，不是对 Qwen 预训练内容的证明。训练集中也有 HardBreak 等同名场景家族，因此不能声称所有场景语义均未见过。没有将这三场用于选择/训练 checkpoint，也没有使用离线 test 标签。

归档来源：[v2 实现记录](OCC_QWEN_NATIVE_V2_20261117.md)、[v2 最终离线审计](OCC_QWEN_NATIVE_V2_FINAL_AUDIT_20261117.md)、[原生输入约定](OCC_QWEN_NATIVE_INPUT_20261116.md)。本次完整配置在 `configs/native-v2-carla-20261117.json`。

## 输入和执行链路

### 明确的真值来源与因果边界

输入是 **CARLA 仿真真值辅助 OCC**，不是从 RGB 或真实 LiDAR 学习推算的感知：语义 LiDAR 的类别/实例 ID、当前/过去自车与对象位姿、当前对象包围盒来自模拟器；导航来自 HD map/XML。RGB 不进入模型。未添加 RGB 分支、未来 OCC decoder、OccWAM 二阶段或新的模型架构。

1. CARLA 0.9.15，Town03，`vehicle.lincoln.mkz_2020`，晴朗中午，固定物理步长 0.05 s。
2. 语义 LiDAR：安装 `x=-0.39,z=1.84` m，64 线、85 m 原始射线范围、600000 points/s、10 Hz 旋转、上/下视角 10/-30°。每物理步取同帧 packet。
3. 20 Hz 物理步下每 packet 只有约 180°。最终入口把相邻两帧的 XYZ 根据各自传感器位姿转换到当前传感器坐标后合并，类别/实例原始 uint32 位保持不变。原始双帧 ID/时间戳随每个 OCC 保存；仅用当前及过去 0.05 秒的射线。主运行记录的最大方位空隙指标对应覆盖跨度：左转最小 355.10°、中位 359.62°，障碍最小 359.10°，行人最小 358.87°。这是实际返回点的角覆盖，遮挡/无返回仍存在，不代表所有方向空间已观测。
4. 合并扫描进入本地 `runtime/Bench2Drive-0.0.4/tools/occ_online_carla.py::OccOnlineCARLA.occ_from_carla_tick`，保留其动态实例历史对齐、地面填充、hollow/downward fill 等正文。源码 SHA256 `855c8b9c6c613f7de31bac5fb9978e042ae6e1902e1bb3801ffd239a16dd9690`。因为现有 Py3.13 推理环境没有 open3d，载入时仅从内存 AST 移除两个经 AST 检查确实未被引用的 import（open3d、OccDatasetGen）；未伪造模块、修改 upstream 文件或改生成器函数正文。两帧独立探针先通过，再接入闭环。
5. 使用已有 `b2d_native_occ.semantic_grid` 做相同语义归组；Y 翻转到 FLU，保存 `[Z,Y,X]=(20,128,128)`、0.4 m，XY `[-25.6,25.6)`、Z `[-5,3)`。85 m 射线仅裁入该原生范围，不扩画布、不缩小再放大；0 是 unfilled，255 是 unknown，均不宣称 verified free。
6. 每 0.1 秒产生当前原生 OCC，取实际历史年龄 `[0.6,0.4,0.2,0]` 秒、按旧到新顺序，并刚体对齐到当前自车。输入 `[1,4,20,128,128]`。不同位姿对齐的源支持外为 255；没有未来帧融合。
7. state20 与训练标度一致：有符号纵向速度/30，局部 IMU ax/10、-ay/10、-gyro_z（rad/s）；近远导航 XY 变到当前 ego，Y 取负、均除 50；两个 6 类 one-hot。没有把 Actor 的 deg/s 当 IMU rad/s，也没有重复转角度。
8. 同一保存的 best，单候选、8 Euler steps，每场 generator seed `1907+planning_index`。`sample` 输出已经是物理米制 FLU 25×2 轨迹，间隔 0.2 秒、总 5 秒；速度为 `residual*10+[state_speed*30,0]`，不再次对 XY 乘训练归一化系数。
9. 轨迹变到世界坐标后以 5 Hz 更新、20 Hz 执行 Pure Pursuit + 速度 PI（Kp .2、Ki .1）。纵向目标是模型原始有符号 vx，对应时刻插值；不强设最低速度、最高速度、加速度或舒适性阈值。转向使用当前物理参数中的最大轮转角、转向速度曲线及实测 2.86048 m 轴距。

导航坐标和类别编码已验证，但**训练采集器近远点选择算法尚未完全对齐**。当前取官方 XML 稠密路径前方 5/30 m 和 CARLA GlobalRoutePlanner 指令。106 个训练窗状态抽查中，近点距离中位 3.50 m，远点中位 28.70 m，近远距离随帧变化；因此当前在线导航不能标记为训练采集器逐项复现。见 `navigation-domain-comparison.json`。

在线生成严格 past/current-only，但公开离线归档的上游生成调用链/时序仍未完全认证；跑通相同生成器正文不等于证明原训练标签全部来自这条在线路径。全扫描数据已修正初版“约 99% unfilled”的严重稀疏偏差，但仍不能据总体填充比例认证分布完全一致。

### 运行保护与接管

未加入 jerk、横向偏移、道路偏离、碰撞预测或人为低速否决；没有其他规划器替换模型、候选重排、自动避障或发生碰撞就强制刹车。jerk 只随原始预测写日志。

程序保护包括形状/有限值、同帧时间顺序、10 秒 worker 回复超时、合法控制范围、600 秒单场墙钟上限。配置/同步拥有者异常会拒绝运行。主三场均未触发这些保护，`takeovers=[]`；撞墙/撞杆后继续记录模型目标和 PI 控制直到仿真时限，因此部分帧仍有油门，不能解释成保护制动。预热刹车和开始等待首个计划的 0.05 秒在逐帧日志单独标记。

## 输入、规划与控制的归因

### 已修复的接口错误

- **车轮坐标/轴距**：初版误将世界坐标 X 跨度当轴距。CARLA 0.9.15 返回世界厘米坐标，现先转到车辆坐标，再取前后轴中心差。同步模式下刚 spawn 的 Actor 缓存姿态可能仍为单位矩阵，现使用已提交的 spawn 姿态，避免首次 tick 前读取的陈旧缓存。参考 [CARLA 0.9.15 车辆实现](https://raw.githubusercontent.com/carla-simulator/carla/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/CarlaWheeledVehicle.cpp) 和本地 `wheel-coordinate-probe.json`。
- **扫描覆盖**：隔帧只取一帧会固定漏掉半边视野；最终合并两个真实相邻半扫描，而非扩大/伪造 OCC 覆盖。单帧半扫目录 `model-upstream` 已被主全扫目录替代。
- CPU 契约测试覆盖与训练 `align_native` 的逐值一致性、平移/旋转、unknown 边界、历史时间、state20、FLU 转向、位姿旋转不变轴距、扫描移动补偿及语义/实例位保持；验证高 jerk/小速度不会触发轨迹拒绝或强制刹车。

### 控制器与模型行为分别记录

最终控制器在同一 26990 路线、4 m/s 地图参考轨迹校准中约 21.75 秒抵达终点附近，零碰撞、零行车道外帧、最大路线中心偏移 0.89 m，见 `calibration-final/turn-controller-calibration/analysis.json`。此参考轨迹仅用于独立控制器校准，不混入模型运行。

主三场每个计划 0.2 秒后的实际位置与该计划第一个点的平均误差：左转 0.193 m、障碍 0.066 m、行人 0.080 m。1/2 秒比较也保存，但包含后续重新规划，不能全部归为控制器误差。转弯模型实际达到 16.12 m/s，4 m/s 校准不足以认证整个高速/大曲率范围；转弯高速度段跟踪误差增大，可能有规划不可执行与控制器跟踪的共同贡献。

**左转失败**：原始轨迹及实际运动均存在提前切弯、加速和离开目标路线，最后撞墙。坐标、轴距错误已修正后仍失败，不能只归因于早期接口错误；但近远导航选择差异和高速度跟踪能力仍是混杂因素，不能声称唯一根因已确定。

**障碍场景**：在距初始约 22.9 m 的静止车前没有表现出稳定停车。约 23.15 s 从其侧方通过，车辆中心相对路线偏离约 5.22 m，且约一半时间不在 Driving lane。没有碰撞不等于成功避障。日志中的 `obstacle_surface_gap_m` 实际为中心距减两车纵向半长的诊断近似，侧向通过时不是精确几何净空，不能作为验收依据。

**动态场景**：行人按固定 12–20 秒、1.3 m/s 跨越，独立于 ego；实际最大速度 1.300 m/s，最近自车/行人中心距约 4.46 m。本次没有行人碰撞，但随后撞杆，不能据此声称具备行人避让能力；行人横穿时序也不是完整覆盖各种冲突时间的测试。

### 冻结输入诊断（不执行、不训练）

主运行中 7 个保存的输入窗口同噪声离线重放，原始 XY 与在线输出最大差为 0。人为移除自车区域外、前方车辆/行人语义体素（填 unfilled，非 verified free），或将导航项置零；这些都是诊断输入扰动，不冒充真实场景干预，结果保存在 `frozen-fullscan-diagnostics/results.json`。

- 障碍场景 10.05/20.05 s：移除前方动态类体素后，5 秒终点仅改变 0.113/0.031 m，首段 vx 基本相同；导航置零时终点改变 10.56/2.72 m。
- 行人场景 11.05/11.85/15.05 s：前方动态类移除后终点改变约 0.082/0.102/0.042 m；导航置零改变 8.15/9.37/10.88 m。
- 左转 13.05/15.05 s：导航置零改变终点 5.60/7.48 m。对应前方没有可删除动态体素，不能把零变化解释成障碍敏感性。

这些有限窗口说明当前轨迹对所测试的动态占用扰动响应很弱、对导航响应明显，并不证明模型对所有 OCC 都无依赖，也不能唯一定位到编码器/World/Action 某层。**下一步优先对齐训练同款导航构造，并在固定权重、输入和噪声下验收道路/障碍对近端轨迹的影响；优先解决路线上行驶和风险响应，先不优化 jerk/舒适性。**

## 耗时、日志和视频

模型推理中位数：左转 55.77 ms，障碍 58.32 ms，行人 50.94 ms；P95 分别 67.43/66.03/61.94 ms。完整 OCC 生成中位数约 0.441/0.463/0.366 s，P95 约 0.566/0.605/0.539 s。单场约 30 秒仿真对应 179.94/194.40/164.63 秒墙钟，包括采集、CPU OCC、推理和保存。同步世界等待程序处理，因此这是慢速纯仿真闭环，**没有证明整条链路实时运行**。

每个主场景子目录含：

- `raw-occ/raw-*.npz`、同名 JSON：合并后的原始语义扫描、两个源帧的时间戳、传感器矩阵、当前 bbox/pose 输入；`grid-*.npy` 为当前生成网格。
- `inputs/*.npz`：每次规划实际送入模型的 history/state，只有 actor 输入字段；无未来轨迹/未来 OCC 标签。
- `plans.jsonl`：原始 25 点轨迹、速度/残差、噪声 seed、输入 SHA、历史帧 ID、state20、导航点/指令、世界轨迹、推理耗时、jerk 统计。
- `steps.jsonl`：20 Hz 位姿/速度/加速度、计划 ID、控制器未裁剪需求、目标速度、油门/刹车/转向、接管原因、路线位置、交通对象真值评估记录。
- `report.json`、`analysis.json`、`plan-analysis.json`、`road-check.json`：碰撞、时限、完成率、车道覆盖、逐计划实际/预测误差。
- `trajectory-replay.mp4`、`trajectory-overview.png`：根据真实记录绘制的俯视示意回放；灰=参考路线，红=实际，蓝=模型预测，绿=交通对象。**不是 CARLA 相机录像，也不是灰线宽度所示的道路边界。**

相机/渲染尝试遇到两次 Low 画质 shader 编译崩溃及一次 Epic/DX11 RPC 挂起，见 `infrastructure-crashes.json`、各 `server-*.json` 和早期 calibration 失败报告。最终 Epic/DX12、no_rendering 模式仍有实时语义 LiDAR 返回，避免了相机故障；没有修改全局驱动或安装全局依赖。

早期 `model-runs/model-final` 是稀疏 OCC 代理，`model-upstream` 是半扫描，`calibration-corrected-wheelbase` 是首次修正时误读 pre-tick 缓存姿态的失败测试。这些均不用于最终三场结论。为防误用，所有阶段保留原目录和失败原因，没有清理证据。

## 复现

在项目根目录打开 PowerShell；先确认无其他训练/GPU/仿真拥有者。使用新输出目录，脚本拒绝覆盖已有场景目录。启动命令：

```powershell
Start-Process -FilePath 'E:\ai\projects\highway-occ-flow\runtime\CARLA-0.9.15\WindowsNoEditor\CarlaUE4.exe' -ArgumentList '-RenderOffScreen','-quality-level=Epic','-dx12','-carla-rpc-port=2125','-nosound','-ResX=640','-ResY=360','-unattended' -WindowStyle Hidden
.venv-train/Scripts/python.exe scripts/test_native_v2_carla_contract.py
.venv-train/Scripts/python.exe scripts/audit_native_v2_carla_setup.py --output artifacts/native-v2-carla-reproduction/setup
.venv-b2d0915/Scripts/python.exe -u scripts/run_native_v2_carla.py --port 2125 --output artifacts/native-v2-carla-reproduction/main --seconds 30 --no-camera --occ-source upstream
.venv-b2d0915/Scripts/python.exe scripts/summarize_native_v2_carla.py --root artifacts/native-v2-carla-reproduction/main --port 2125
.venv-train/Scripts/python.exe scripts/diagnose_native_v2_carla_inputs.py --runs artifacts/native-v2-carla-reproduction/main --output artifacts/native-v2-carla-reproduction/frozen-diagnostics
```

独立控制器校准使用 `--calibration-only --no-camera`，另选输出目录。已有地图、CARLA API 位于 `.venv-b2d0915`（Python 3.10）；模型推理用 `.venv-train`（Python 3.13），通过只含已记录输入的文件/JSON 通道通信。没有在不兼容 Python 间直接导入 CARLA。源文件清单和最终哈希见 `source-final.json`；测试、仿真和诊断均未启动新的训练。
