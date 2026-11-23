# Stage 3.1 本机验收交接

## Stage 3.1b 重训

Stage 2.6 使用正前和左右侧三路 RGB。Stage 3.1 从六路环视 RGB 与 64 线 360° LiDAR 学习新的感知表征。正前相机为 640×360，左前、右前、左后、右后和正后相机均为 320×180。六路共享 CameraFPN，多高度投影后与自车坐标系下的 LiDAR BEV 融合。

Stage 3.1 初版使用 40×30 空间特征和 64 个时序 token。3.1b 使用 80×60 空间特征和 128 个带位置锚点的时序 token，车道和占用输出为 160×120。驾驶指令、导航锚点和自车速度进入轨迹头。时序感知状态 `z_t` 只读取传感器特征、上帧感知状态和自车位姿变化。Stage 3.2 的动作条件动力学是下一阶段工作。

3.1b 的权重、优化器和调度器从头初始化。正式训练使用新数据包 `stage31b-v2-pack` 和运行目录 `stage31b-v2-run`，中断续训时校验架构及数据包指纹。训练包包含原有 48,564 帧和新增的 5,000 帧 Town12 宽车道数据，共 53,564 帧。

本机 RX 7900 XT 完成 10,000 次更新，物理 batch 8，梯度累积 2。最低验证损失为第 7,968 步的 2.2043。使用从 Drive 读回的最佳断点，在 10,642 个互异验证帧上以置信度大于 0.5、中心距离小于 3 m 一对一匹配，命中 13,165 / 20,805 个真车框，召回 63.28%，精确率 38.15%。前方 0–40 m、横向 ±18 m 内的原始可行驶区域 IoU 为 0.921，虚线 0.228，实线 0.464，路沿 0.415。该断点通过可行驶区域目标，车辆召回和虚线 IoU 仍低于验收要求。

SR 的 CHASE 预设为 13° 俯角、向车尾右侧偏 12°。ELEVATED 保留 52° 俯角供检查道路拓扑。LIVE 模式用连续曲线绘制道路和标线，车辆按检测头给出的三维位置、尺寸和朝向生成网格，用深度缓冲处理遮挡。离线试验中，车道时序与图像证据修正后虚线 IoU 为 0.232。实验性的车辆后处理将全范围精确率提高到 59.51%，召回降至 60.33%，因此默认视图保留原始车辆。

用最佳断点、Town12 真实输入、BF16、B=1 重复 100 次。AOTriton 的 Eager / HIP Graph 墙钟 P95 为 13.57 / 7.20 ms，MATH 为 13.89 / 7.86 ms。两种后端的 Eager 与 Graph 对各自所有输出的最大绝对差均为 0。推理峰值分配分别为 244 / 266 MiB，整卡采样占用为 0.72 / 0.76 GiB。控制周期配置为 100 ms。

CARLA 1.0、异步 Town04_Opt 零样本感知诊断采用三个四车道高速点和三个城区点，每处 16 帧，0 条驾驶路线。每帧 0.10 s LiDAR 累积覆盖 72/72 角度扇区。三个高速点的模型在全部 48 帧各命中至少两辆前车，高速点虚线 IoU 为 0.113、0.081、0.165。城区三岔口 16/16 帧命中前方多车，十字路口和镇内弯街各为 1/16。该最佳断点尚未达到多场景感知验收要求。

另用同机 CARLA 1.0、Town04_Opt 静止自车与 16 辆 NPC 测量实时 SR。六路相机和 LiDAR 输入平均 35.99 Hz。道路修线、原始车辆与 960×600 离屏渲染读回一起运行时，10 Hz、20 Hz 和不限频三组实际吞吐分别为 10.05、15.24 和 15.15 Hz。不限频组从传感器全部到齐至图像读回 P95 为 96.1 ms，从仿真快照回调计时为 152.8 ms。单独使用 Town12 第 45893 帧、1280×832、4×MSAA、连续重建 100 帧，GPU 绘制 P95 为 0.091 ms，含 CPU 几何生成的绘制调用 P95 为 16.13 ms（关闭读回）。

Town12 增量入口逐帧核对 120 km/h 非路口路段相邻同向驾驶车道宽度、航向及有效同向车辆框，逐集保留计数，并执行传感器和标签质检。增量打包会校验并复制已有分片，固定已有 episode 的训练和验证归属，对增量执行完整质检及打包回读。

```powershell
.venv-carla-ue5/Scripts/python.exe -m v3_world_model.collect_wide --frames 5000 --calibration .cache/stage31-r2/cal-Town12/calibration.json --out .cache/stage31b/wide-Town12
.venv-carla-ue5/Scripts/python.exe -m v3_world_model.pack_colab_v3 --source .cache/stage31b/wide-Town12 --base-manifest .cache/stage31-r2/pack-full/manifest.json --out .cache/stage31b/pack --verify-pack
.venv-train/Scripts/python.exe -m v3_world_model.evaluate_sr --all --dataset .cache/stage31b/dataset --checkpoint .cache/stage31b/run-stage31b-a100/best.ckpt --out .cache/stage31b/val-all.json
```

## 2026-11-23 SR 验收阻断：当前权重漏车、漏车道

此前完成训练与 ROCm 数值一致性验证不代表 SR 感知验收通过。CARLA 1.0、Town04_Opt、异步、三个四车道静态位置的同渲染器对照中，参考数据可以显示四车道和四辆车；模型原始栅格已缺失车道，整圈 LiDAR 输入后仍无置信度大于 0.5 的车辆。参考数据仅用于评估对照，没有进入模型输入或训练数据。

对打包验证集均匀抽取 24 clips / 96 帧（Town12 80 帧、Town10HD_Opt 16 帧），best（7826 updates）的近场 IoU 为：可行驶区域 0.752、虚线 0.126、实线 0.238、路沿 0.234。车辆按阈值 0.5、类别无关且中心误差小于 3 m 的一对一匹配：143 个真值实例，24 个预测，15 个匹配；召回 10.5%、精确率 62.5%。latest（10000 updates）同样只匹配 15 个，不能靠换断点解决。该抽查不是全量评估，也不是检测 AP。不能仅凭 val_objects 约 0.24 或整体 loss 宣称车辆检测可用。

代码侧修复：`async_lidar.full_revolution` 在异步帧率下累积一圈、按包补偿自车运动并拒绝缺包/未来帧；最旧包按各激光通道回波顺序近似裁剪，不是逐点时间戳去畸变。实机三个位置均覆盖 72 个 5° 方位分箱。SR 保留目标置信度，未检出时显示“未检出前车”，不再显示 Clear；HUD 缓存随目标数、置信度和前车距离更新。渲染器没有“两车道”或车辆数量上限，未添加真值回填或降低部署阈值。

可重复运行：

```powershell
.venv-train/Scripts/python.exe -m v3_world_model.evaluate_sr --dataset .cache/stage31-r2/dataset-full --checkpoint .cache/stage31-r2/run-stage31-a100/best.ckpt --clips 24 --out .cache/stage31-r2/sr-diagnostic/validation-audit.json
```

对照图与实机报告在 `.cache/stage31-r2/sr-diagnostic/`，未保存 Town04 训练帧。缺失车道和漏车尚未修复，下一步需针对感知训练与数据分布定位、修复并重新验收；不得将当前 SR 当作合格闭环感知，也不得重新启用 Traffic Manager 自车驾驶来冒充模型闭环。

分支为 `feat/v3-stage3-prep`。从仓库根目录运行模块。输出示例写入 Git 忽略的 `.cache/stage31-r2`。后视外参、观测键和 tar 成员已更新，采集需使用新标定报告，Colab 需使用 `bev-wam-x-stage31-v2` 包和新权重目录。每台模拟器只保留一个同步时钟驱动进程。

## 已执行的提交前验收

- 全部 Python 源文件通过 `py_compile`，Notebook 的三个 Python 单元通过语法检查，9 项 `unittest` 全部通过。
- CPU 自测覆盖 518,400 像素、六相机光轴、5 层投影、BEV 坐标、v2 符号下的 `ego_delta` 和 `Town04_Opt` 采集拦截。新外参的 5–30 米静态几何覆盖为 100%，最大角向空隙为 0°。自遮挡、环带和亚像素过滤、gamma 显示校正在合成测试中通过。
- 模型有 **25,242,253** 个参数。感知解码从 `z_t (B, 64, 256)` 读出，稠密解码通过 `40×30 → 80×60 → 160×120` 两级上采样。分别切换 `cmd`、`nav_anchors`、`ego_speed` 时，感知输出和 `z_t` 最大差值均为 `0.0`。车道线使用 `[1, 8, 8, 3]` 加权 BCE + Dice，当前和未来占用使用 `20` 加权 BCE + Focal + Dice。前向和反向梯度全有限。
- 16 帧合成采集通过 QC、分片、SHA-256 和解包校验，打包内非 RGB 数组最大差值 `0.0`，路线和初态交集均为空。CPU Lightning 完成 1 次更新，best 和 latest 读回逐张量一致。从 latest 续训至第 2 次更新时，学习率按原 warmup 和 cosine 配置继续。
- 合成 3D SR 通过软件离屏像素检查。Windows RX 7900 XT 使用 OpenGL 4×MSAA 渲染 100 张 960×600 合成帧，GPU P95 为 0.041 ms，每帧重建网格的无读回循环约 280 FPS。HUD 英文默认值为 `Follow`。
- Windows RX 7900 XT、ROCm 7.14.60850、BF16、batch 1、随机初始化权重，每种后端 50 次。AOTriton 与 MATH 的 Eager/HIP Graph 逐输出最大差值均为 `0.0`。同步端到端 Eager/HIP Graph P95 分别为 23.37/6.73 ms、25.04/7.20 ms。分配峰值约 246 MiB，整卡用量快照约 709–741 MiB。控制周期设定为 100 ms。
- 本轮尝试启动 Windows CARLA 的离屏和窗口模式，两次均未建立 RPC 连接，未生成新版实机标定结果。以下命令用于两图标定、500 帧试采、实景 SR 和实采帧 ROCm 回归。50k 采集、A100 训练和自动释放运行时留待实机与云端验收。

## 1. 环境、CPU 复验与安全门禁

`$simPy` 指向带匹配 CARLA 1.0 Python API 的环境，`$trainPy` 指向已有 ROCm、PyTorch 的环境。若当前 venv 名称不同，调整两个变量。

```powershell
$simPy = './.venv-carla-ue5/Scripts/python.exe'
$trainPy = './.venv-train/Scripts/python.exe'
$run = '.cache/stage31-r2'
$port = 2460

uv pip install --python $simPy 'numpy>=1.26,<3' 'PyYAML>=6,<7' 'scipy>=1.12,<2' 'opencv-python-headless>=4.9,<5'
uv pip install --python $trainPy -r v3_world_model/requirements_v3.txt

$files = @(Get-ChildItem -Recurse -File v3_world_model -Filter '*.py' | ForEach-Object FullName)
& $trainPy -m py_compile @files
& $trainPy -m unittest discover -s v3_world_model/tests -v
& $trainPy -m v3_world_model.sr_viewer --backend software --out "$run/sr_preview_cpu.png"
```

CARLA 使用与 Python API 同一构建的 CARLA 1.0 / UE 5.8，DX12，High，启用渲染。在独立 PowerShell 控制台从仓库根目录启动。若已经启动匹配服务，沿用其 RPC 端口。

```powershell
& './runtime/Carla-1.0-Win64-Shipping/CarlaUnreal/Binaries/Win64/CarlaUnreal-Win64-Shipping.exe' -windowed -ResX=640 -ResY=360 -dx12 -nosound -carla-rpc-port=2460 -quality-level=High
```

## 2. 两图标定、500 帧试采、质检及回读

```powershell
foreach ($calTown in @('Town12', 'Town10HD_Opt')) {
  & $simPy -c "import carla; c=carla.Client('127.0.0.1', $port); c.set_timeout(120); print(c.load_world('$calTown').get_map().name)"
  if ($LASTEXITCODE) { throw "Map load failed for $calTown" }
  & $simPy -m v3_world_model.calibration_probe --map $calTown --port $port --frames 20 --out "$run/cal-$calTown"
  $pass = $LASTEXITCODE -eq 0
  $report = Get-Content -Raw "$run/cal-$calTown/calibration.json" | ConvertFrom-Json
  $report.cameras.back | Format-List
  if (-not $pass -or $report.passed -ne $true) { throw "Calibration gate failed for $calTown" }
}

$town = 'Town12'
& $simPy -c "import carla; c=carla.Client('127.0.0.1', $port); c.set_timeout(120); print(c.load_world('$town').get_map().name)"
if ($LASTEXITCODE) { throw 'Pilot map load failed' }

& $simPy -m v3_world_model.collect_v3 --map $town --port $port --calibration "$run/cal-$town/calibration.json" --pilot-frames 500 --out "$run/pilot-$town"
if ($LASTEXITCODE) { throw 'Pilot failed' }
& $simPy -m v3_world_model.qc_v3 --source "$run/pilot-$town" --out "$run/qc-$town"
if ($LASTEXITCODE) { throw 'QC failed' }
& $simPy -m v3_world_model.pack_colab_v3 --source "$run/pilot-$town" --out "$run/pack-$town" --verify-pack
if ($LASTEXITCODE) { throw 'Pack verification failed' }
Get-Content "$run/qc-$town/maneuvers.json"
Get-Content "$run/pack-$town/verification.json"
```

后视外参为 `[-2.85, 0.00, 1.65, 180, -12, 110]`。逐张检查经 gamma 0.5 显示校正的 `surround_bev_*.png`。地面 5–30 m 环带门禁为覆盖至少 99.5%、零角向空隙。每台相机至少保留 25 个环带内 RoadLines LiDAR 点，剔除投影落入自车遮挡或按 0.15 m 标线宽度估算小于半像素的点。最近语义标线重投影 P95 至多 4 px，实测外参最大元素偏差至多 0.001。报告列出每路全图和底部 30 行的自车遮挡比例，以及剔除计数。后视的最大自车遮挡比例要求为 0。五个站点采样时请控制现场车辆干扰。门禁报告绑定地图、CARLA 构建和传感器 SHA-256。

查看 `qc.json` 以及 `box-*.png`。绿色是 3D 框重投影，橙色是同帧 LiDAR 命中。应能在 RGB 上辨认实际车辆。每个 episode 的标线存在帧比例至少 90%，虚实线像素交集为零，采集丢帧率为零。任一事件桶空缺时，`maneuvers.json` 会记录告警。优先检查脚本实际驾驶轨迹，慢车超车按同一前车 ID 从自车前方移到侧后方判定。Town10HD_Opt 分开标定，可用 `--scenarios straight multi_car curve left_change right_change` 选择地图支持的剧本。

目前匝道候选按有明显转向的路口连接段与车道束变化筛选，请用 Town12 俯视或 SR 回放确认确为匝道。虚线使用基于 OpenDRIVE `s` 的 3 m/9 m 栅格样式，复合线做单通道归类，冲突像素保留实线。局部道路立交通过高差筛选，复杂坡道或多层路面需要重点目检。当前占用由同帧 LiDAR 回波和交通框合成。采集使用六路 RGB 和一路 LiDAR，10 Hz 为仿真步进频率。episode 轮换晴天、多云、湿地、小雨和日落，初始生成点被占用时重选路线，元数据记录天气和变道执行区间。核对无触发片段全程为 `follow`，触发帧起输出方向指令，在目标车道横向偏差 ≤0.45 m 且航向差 ≤8° 连续三帧后恢复 `follow`。未完成的请求 5 秒后超时并恢复 `follow`。

## 3. 真实 SR 层与 ROCm 回归

选取 QC 有对应框图的驾驶帧。下面用首个多车 episode 中第 21 帧演示命令。回放转换器把采集标签转换成与模型输出相同的渲染字典。线上由 `prediction_scene` 转换输出，`SRViewer.render` 只接收该字典。

```powershell
$episode = Get-ChildItem "$run/pilot-$town" -Directory | Where-Object {
  (Get-Content -Raw "$($_.FullName)/episode.json" | ConvertFrom-Json).scenario -eq 'multi_car'
} | Select-Object -First 1
$frame = (Get-ChildItem $episode.FullName -Directory | Sort-Object { [long]$_.Name } | Select-Object -Skip 20 -First 1).FullName

& $trainPy -m v3_world_model.sr_viewer --frame $frame --backend gpu --occupancy --out "$run/sr_preview.png" --benchmark-frames 250
& $trainPy -m v3_world_model.sr_viewer --frame $frame --backend gpu --window --out "$run/sr_preview.png"

& $trainPy -m v3_world_model.verify_rocm --frame $frame --sdpa aotriton --iterations 100 --out "$run/rocm-aotriton.json"
& $trainPy -m v3_world_model.verify_rocm --frame $frame --sdpa math --iterations 100 --out "$run/rocm-math.json"
```

窗口支持方向键、左键拖动和滚轮调整视角。按 O 显示未来占用，按 Esc 退出。验收看 3D 朝向、虚线与实线、路沿、轨迹带和占用重影。GPU 时间以 P95 < 3 ms、循环 ≥40 FPS 核对，离屏读取 PNG 单独发生。ROCm 脚本在同一 BF16、SDPA、相机和上一帧隐状态下逐输出比较。默认端到端 HIP Graph P95 门槛为 100 ms，整卡用量快照预算 20 GiB。输出分别列出推理时间、100 ms 控制周期、整卡用量、模型分配/保留峰值。Colab 权重回到本机后加 `--checkpoint` 重跑。随机初始化的模型输出只用于接口和性能验收。

## 4. 全量打包与 Colab

本机标定、试采、SR 和 ROCm 门禁通过后再采集 50k。可按地图分配帧数，例如 Town12 40k，Town10HD_Opt 10k。下列命令假设已经为两张地图各取得通过的标定报告。采集与打包使用新目录。提前为原始 RGB 和本地 LiDAR 审计保留空间，并逐个加载对应地图。

```powershell
& $simPy -c "import carla; c=carla.Client('127.0.0.1', $port); c.set_timeout(120); c.load_world('Town12')"
& $simPy -m v3_world_model.collect_v3 --map Town12 --port $port --calibration "$run/cal-Town12/calibration.json" --frames 40000 --out "$run/full-Town12"
& $simPy -c "import carla; c=carla.Client('127.0.0.1', $port); c.set_timeout(120); c.load_world('Town10HD_Opt')"
& $simPy -m v3_world_model.collect_v3 --map Town10HD_Opt --port $port --calibration "$run/cal-Town10HD_Opt/calibration.json" --frames 10000 --scenarios straight multi_car curve left_change right_change --out "$run/full-Town10HD_Opt"

& $simPy -m v3_world_model.qc_v3 --source "$run/full-Town12" "$run/full-Town10HD_Opt" --out "$run/qc-full"
& $simPy -m v3_world_model.pack_colab_v3 --source "$run/full-Town12" "$run/full-Town10HD_Opt" --out "$run/pack-full" --verify-pack
```

将 `pack-full` 的 `manifest.json` 和 `stage31-*.tar` 放入 Drive 中与 Notebook 的 `PACK` 对应的目录。每片上限 1.5 GB。传感器 SHA-256 按 UTF-8 和 LF 计算。路线标识使用地图与 road ID 序列，初态标识使用地图与起始 ego 位姿。同路线或同初态会通过连通分量归组，即使它们由不同天气、种子或驾驶分支产生。v2 tar 包包含 6 路 WebP、逐帧 `obs.npz` 和每集 `targets.npz`，本地 `audit.npz` 供应 QC。WebP quality 90 有损，包中的非 RGB 观测和标签按位校验。

打开 `colab_stage31_train.ipynb`，选择 A100，填写 Drive 的 `PACK`、`RUN` 后依次运行。Notebook 校验分片 SHA-256，展开到 Colab 本地盘，使用 BF16 和 `cuda_flash`。默认 `BATCH=8`、`ACCUMULATE=2`，每次更新 16 clips / 64 帧，`NUM_WORKERS=4` 可调。仅 train split 为同一 clip 的所有帧和相机应用共享亮度、对比度和色彩抖动。默认先线性 warmup 500 更新，再按 10,000 更新总周期 cosine 退火到基础学习率的 5%。续训按原物理 batch、累积、数据集、warmup 和周期恢复，`MAX_UPDATES` 控制累积停止点。

Drive 保存 `best.ckpt`、`latest.ckpt`、`training-config.json`、`validation.jsonl`、`training-result.json`。Notebook 读回两份断点和结果，核对 pack 指纹及更新数。校验通过后默认 `AUTO_UNASSIGN=True` 释放 GPU 运行时。交互调试时可设为 `False`。训练锁按主机运行时互斥，同一个 Drive 运行目录只开一个 Colab 运行时。下载权重和日志后按本机门禁验收。

训练和部署使用六目 RGB、LiDAR BEV、在线定位提供的 `ego_delta` 和上一隐状态。观测内的 `nav_anchors (3, 3)` 由预定 SD 路线的 12、24、36 m 前瞻点转换到当前 FLU 坐标，`ego_speed (2,)` 为速度 m/s 和偏航角速度 rad/s，`cmd (3,)` 为逐帧 one-hot。三者仅注入轨迹查询。Stage 3.2 可用 `decode_perception(z_t)` 直接解码感知。
