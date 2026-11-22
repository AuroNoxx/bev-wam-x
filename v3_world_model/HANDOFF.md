# Stage 3.1 本机验收交接

分支为 `feat/v3-stage3-prep`。从仓库根目录运行模块。所有输出示例写入 Git 忽略的 `.cache/stage31`。真实 CARLA 试采前先执行该地图对应的标定。每台模拟器只保留一个同步时钟驱动进程。

## 已执行的提交前验收

- 全部 Python 源文件通过 `py_compile`。Notebook 的三个 Python 单元通过语法检查。
- CPU 验收覆盖锁定的 518,400 像素、六相机光轴、5 层查表、BEV 单元边界和往返换算、v2 符号下的 `ego_delta` 反推、`Town04_Opt` 在建立 CARLA 连接前被拒绝。六个配置外参与本机 CARLA API 的 `Transform.get_matrix()` 最大元素差为 `8.75e-8`。
- 车道线互斥。32 个真值检测槽位编码往返最大差值为 `0.0`。25,008,525 参数模型的 CPU 梯度全有限，`z_t` 为 `(1, 64, 256)`。相同传感输入切换左右指令，所有感知输出和 `z_t` 的最大差值为 `0.0`。
- 合成采集 16 帧走过 QC、分片、SHA-256 校验和解包。抽样比对覆盖 16 帧，非 RGB 数组最大差值为 `0.0`。训练与验证之间路线和初态交集均为空。Lightning 使用解包后的四帧片段完成 1 次 CPU 更新，物理 batch 为 1，梯度累积为 1，best 和 latest 读回逐张量一致。从 latest 继续训练，累计更新数达到 2。
- 合成 3D SR 通过软件离屏像素检查。在 Windows RX 7900 XT 上，OpenGL 4×MSAA 离屏渲染 100 张 960×600 合成帧，GPU P95 约 0.041 ms，无读回的总循环约 278 FPS。该测量包括每帧重建合成场景网格。
- Windows ROCm 7.14 合成输入比较使用随机初始化权重、batch 1、BF16 和每种后端 50 次测量。AOTriton 与 MATH 的所有输出 Eager/HIP Graph 最大差值均为 `0.0`。同步端到端推理 P95，AOTriton 分别约 31.8 和 15.7 ms，MATH 分别约 22.6 和 7.0 ms。模型分配峰值约 246 MiB，整卡用量快照约 657–689 MiB。控制周期设定为 100 ms。下面的命令用实采帧和实际运行条件复核。真实 CARLA 标定、500 帧试采、实景 SR、A100 训练均待验收。

## 1. 环境、CPU 复验与安全门禁

`$simPy` 指向带匹配 CARLA 1.0 Python API 的环境，`$trainPy` 指向已有 ROCm、PyTorch 的环境。若当前 venv 名称不同，调整两个变量。

```powershell
$simPy = './.venv-carla-ue5/Scripts/python.exe'
$trainPy = './.venv-train/Scripts/python.exe'
$run = '.cache/stage31'
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

## 2. Town12 标定、500 帧试采、质检及回读

```powershell
$town = 'Town12'
& $simPy -c "import carla; c=carla.Client('127.0.0.1', $port); c.set_timeout(120); print(c.get_available_maps()); print(c.load_world('$town').get_map().name)"
& $simPy -m v3_world_model.calibration_probe --map $town --port $port --frames 20 --out "$run/cal-$town"
Get-Content "$run/cal-$town/calibration.json"
if ((Get-Content -Raw "$run/cal-$town/calibration.json" | ConvertFrom-Json).passed -ne $true) { throw 'Calibration gate failed' }

& $simPy -m v3_world_model.collect_v3 --map $town --port $port --calibration "$run/cal-$town/calibration.json" --pilot-frames 500 --out "$run/pilot-$town"
if ($LASTEXITCODE) { throw 'Pilot failed' }
& $simPy -m v3_world_model.qc_v3 --source "$run/pilot-$town" --out "$run/qc-$town"
if ($LASTEXITCODE) { throw 'QC failed' }
& $simPy -m v3_world_model.pack_colab_v3 --source "$run/pilot-$town" --out "$run/pack-$town" --verify-pack
if ($LASTEXITCODE) { throw 'Pack verification failed' }
Get-Content "$run/qc-$town/maneuvers.json"
Get-Content "$run/pack-$town/verification.json"
```

逐张检查标定目录下的 `surround_bev_*.png`。地面 5–30 m 环带覆盖至少 99.5%，连续角向空隙至多 2°。每台相机需要至少 25 个路面标线 LiDAR 对应点，最近语义标线像素的重投影 P95 至多 4 px，实测外参最大元素偏差至多 0.001。五个固定站点采样时请保持现场车辆干扰可控。环带门禁之外，检查车身近处自遮挡、画面接缝、高坡和立交。门禁报告绑定地图、CARLA 构建和传感器 SHA-256。

查看 `qc.json` 以及 `box-*.png`。绿色是 3D 框重投影，橙色是同帧 LiDAR 命中。应能在 RGB 上辨认实际车辆。每个 episode 的标线存在帧比例至少 90%，虚实线像素交集为零，采集丢帧率为零。任一事件桶空缺时，`maneuvers.json` 会记录告警。优先检查脚本实际驾驶轨迹，慢车超车按同一前车 ID 从自车前方移到侧后方判定。Town10HD_Opt 分开标定，可用 `--scenarios straight multi_car curve left_change right_change` 选择地图支持的剧本。

目前匝道候选按有明显转向的路口连接段与车道束变化筛选，请用 Town12 俯视或 SR 回放确认确为匝道。虚线使用基于 OpenDRIVE `s` 的 3 m/9 m 栅格样式，复合线做单通道归类，冲突像素保留实线。局部道路立交通过高差筛选，复杂坡道或多层路面需要重点目检。当前占用由同帧 LiDAR 回波和交通框合成。真实采集耗时取决于 12 路相机、地图和磁盘，10 Hz 定义为仿真步进频率。

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

本机标定、试采、SR 和 ROCm 门禁通过后再采集 50k。可按地图分配帧数，例如 Town12 40k，Town10HD_Opt 10k。下列命令假设已经为两张地图各取得通过的标定报告。采集与打包使用新目录。提前为原始 RGB、真值深度和 LiDAR 保留空间，并逐个加载对应地图。

```powershell
& $simPy -c "import carla; c=carla.Client('127.0.0.1', $port); c.set_timeout(120); c.load_world('Town12')"
& $simPy -m v3_world_model.collect_v3 --map Town12 --port $port --calibration "$run/cal-Town12/calibration.json" --frames 40000 --out "$run/full-Town12"
& $simPy -c "import carla; c=carla.Client('127.0.0.1', $port); c.set_timeout(120); c.load_world('Town10HD_Opt')"
& $simPy -m v3_world_model.collect_v3 --map Town10HD_Opt --port $port --calibration "$run/cal-Town10HD_Opt/calibration.json" --frames 10000 --scenarios straight multi_car curve left_change right_change --out "$run/full-Town10HD_Opt"

& $simPy -m v3_world_model.qc_v3 --source "$run/full-Town12" "$run/full-Town10HD_Opt" --out "$run/qc-full"
& $simPy -m v3_world_model.pack_colab_v3 --source "$run/full-Town12" "$run/full-Town10HD_Opt" --out "$run/pack-full" --verify-pack
```

将 `pack-full` 的 `manifest.json` 和 `stage31-*.tar` 放入 Drive 中与 Notebook 的 `PACK` 对应的目录。每片上限 1.5 GB。传感器 SHA-256 按 UTF-8 和 LF 计算，跨 Windows 与 Colab 使用同一标识。路线标识使用地图与 road ID 序列，初态标识使用地图与起始 ego 位姿。同路线或同初态会通过连通分量归组，即使它们由不同天气、种子或驾驶分支产生。按路线、初态分别检查零泄漏。WebP quality 90 有损，标签、深度编码、姿态和 LiDAR 数组按位相等。回放不必解包到训练目录。深度以 uint16 厘米保存在监督文件中，此版本五组损失未使用深度。

打开 `colab_stage31_train.ipynb`，选择 A100，填写 Drive 的 `PACK`、`RUN` 后依次运行。Notebook 会验证每片 SHA-256，展开到 Colab 本地盘，使用 BF16 与 `cuda_flash` 训练。运行目录默认为 Drive，保存 `best.ckpt`、`latest.ckpt`、`training-config.json`、`validation.jsonl`、`training-result.json`。再次打开时有 latest 就自动续训，保留原物理 batch、梯度累积及数据集，调整 `MAX_UPDATES` 控制累计更新上限。训练锁按主机运行时互斥。同一个 Drive 运行目录只开一个 Colab 运行时。下载权重和日志后按本机门禁验收。

训练侧的网络端口仅为六目 RGB、LiDAR BEV、由当前及过去位姿获得的 ego 增量、上一隐状态及可选交互命令。部署由在线定位提供 `ego_delta`。单向约束由网络拓扑保证，命令仅进入轨迹分支。输出 `z_t` 固定为 `(B, 64, 256)`。
