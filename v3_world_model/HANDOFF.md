# Stage 3.1 本机验收交接

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
