# CARLA GUI 安装与运行记录

日期：2026-11-14。机器：Windows，Ryzen 7 7800X3D，约 32GB RAM，RX 7900 XT 20GB。

## 项目自包含安装

- CARLA：`runtime/CARLA-0.9.15/WindowsNoEditor/`
- Python 运行时：`runtime/python/`
- 虚拟环境：`.venv/`
- 安装包和缓存：`.cache/`
- Python 依赖：`requirements.txt`（CARLA 0.9.15、pygame 2.6.1、numpy 1.26.4）
- Python 版本：3.10.22，CARLA 0.9.15 Windows wheel 不支持已有 Python 3.13。
- 上述运行时、环境和缓存均由 `.gitignore` 排除。

官方来源：https://downloads.carlasim.com/Windows/CARLA_0.9.15.zip

下载长度：7755135185 bytes。

本机计算的 SHA256：`2650B7B3FBE21FEFD7C7CD2C0D53A0E5E2DF9E51A4E0EEC9BCF9AC0291697C39`。这是下载后记录的本地摘要，不是官方签名认证。7-Zip 解压退出码 0。

## 使用

在项目目录的 PowerShell 中：

```powershell
.\scripts\Start-Carla.ps1
# 手动驾驶启动（默认是 CARLA autopilot 演示）：
.\scripts\Start-Carla.ps1 -Manual
# 在当前终端持续显示 2 分钟真实帧号、位置和速度：
.\.venv\Scripts\python.exe -u .\scripts\check_preview.py --seconds 120
# 结束当前项目的模拟器和驾驶客户端：
.\scripts\Stop-Carla.ps1
# 在本目录重新安装依赖与 CARLA：
.\scripts\Install-Carla.ps1
```

两个窗口：CarlaUE4 为模拟器观察窗口，Highway OCC Flow - CARLA Town04 为驾驶相机/控制窗口。

- WASD / 方向键：加速、制动、转向。
- P：切换 CARLA 自带 autopilot。不是训练出的学生策略。
- Tab：切换相机位置。
- H：帮助。
- Esc：退出驾驶窗口；模拟器可单独保留，需要完全退出时运行 Stop-Carla。
- 默认启动 CARLA 自带 autopilot；P 切换后可手动驾驶。车辆从 Town04 的多车道高速路段出发，默认 autopilot 后续可能驶离高速，并非高速闭环专用策略。当前只生成一辆车，没有背景交通。
- 同步仿真步长 0.05 秒，客户端以非忙等待方式限制为 20 Hz，避免以数倍现实速度运行和不必要的 CPU 忙等。

## 已验证与问题

1. CARLA API 0.9.15 导入、RPC 握手与 Town04 加载成功。
2. DX11 可显示场景，但启动相机客户端后发生 `D3D device being lost, 0x887A0020 INTERNAL_ERROR`。崩溃记录 `logs/dx11-crash.xml` 确认为 RX 7900 XT / D3D11，`bIsOOM=0`。不能将此错误直接解释为显存不足。
3. 切换项目启动参数为 DX12 后，成功显示高速道路、Tesla Model 3 和 HUD。一个截图观察到客户端约 62 FPS、服务端约 208 FPS；不是持续基准，启动命令的 t.MaxFPS 参数未证实生效，不可当作资源限制。
4. pygame 的系统字体枚举遇到非字符串注册表值而失败。`scripts/manual_preview.py` 仅在本进程使用已知字体，未改注册表、系统字体或下载的 upstream 脚本。
5. 使用 pythonw 正常显示驾驶窗口；后台进程的 Hidden 启动方式会隐藏该 GUI，因此驾驶客户端明确以可见方式启动。
6. 初次演示后来客户端退出，服务端仍在监听但 RPC 超时；没有新增崩溃记录，原始挂起根因未确认，不能归因于用户最小化。重启时另发现停止进程后未等待退出的竞态，现已增加退出等待并按项目路径识别服务端。重启后 RPC、画面和车辆运动恢复，实时检查记录在 `logs/preview-health.jsonl`。短时检查不等于训练可用或长期稳定。
7. 未修改 AMD 驱动、全局 Python 或现有训练依赖。未启动模型训练或云 API。
8. 修复后完成 120 秒只读健康检查，退出码 0。启动稳定后（墙钟 10–120 秒）帧号 474 → 2654，即约 19.82 FPS；仿真时间推进 109 秒。车辆位置持续改变，记录速度含停车及最高约 87 km/h。GUI 截图同时显示 Server/Client 20 FPS。检查期间无新增连接失败，服务端工作集约 1.82 GiB、驾驶客户端约 0.11 GiB（均为系统内存，不是显存），系统可用内存约 18.87 GiB。只验证单车短时预览，未验证交通密集场景或模型训练。

## SwanLab 卸载与监控选择

按用户要求，已卸载 VS Code 的 `swanlab.swanlab` 扩展，并从以下两个环境卸载 swanlab 0.9.8 / swanboard 0.1.10b2，复查均不存在：

- `D:\ai\apps\UnslothStudio\unsloth_studio`
- `E:\ai\projects\Snap3D-Coder-4B\.venv`

VS Code 将扩展旧目录标记为 obsolete，实际目录可能等待 VS Code 重启清理。没有发现运行中的 SwanLab 服务。

保留原始 SwanLab 训练日志和模型。自动审批以 `blocked by policy` 拒绝删除桌面 SwanLab 快捷方式和 `D:\ai\apps\SwanLab` 启动脚本，未提供具体原因，因此这些残留尚未删除；不宣称完全清理。

早先新建的 `D:\ai\envs\carla-0915` 删除操作同样被自动审批拒绝，仍有残留。CARLA 安装包已迁移至本项目，实际运行的 CARLA、Python 和虚拟环境均来自本项目，不使用该 D 盘环境。uv 为最初安装下载的 D 盘 Python/cache 也未做跨项目清理。

新项目拟用 TensorBoard 看 loss/学习率/吞吐等曲线，Rerun 看图像、占用与轨迹的时序对齐；本轮尚未安装这两项或伪造训练日志。

SwanLab 旧离线看板官方声明仅做简单维护：https://docs.swanlab.cn/guide_cloud/offline-board.html
Rerun 时序视图：https://rerun.io/docs/howto/visualization/visible-time-ranges
