# Highway OCC Flow · v1

v1 是在 **CARLA 0.9.15** 的 Town04 和 Town06 上训练的单帧驾驶模型。本次发布固定第 22 轮权重对应的规划网络与控制器，作为后续架构的比较基线。

## 模型如何开车

模型把 640 × 360 前视图像投影到地面，与 LiDAR 高度、密度、强度和导航走廊拼成一张 7 通道、128 × 128 的鸟瞰图（BEV）。图像投影覆盖前方 45 米，完整网格覆盖车后 16 米至车前 112 米、左右各 25.6 米。

- **蓝色导航带**是模型收到的第 7 个通道，表示沿所选路线、目标车道应走的方向。
- **绿色轨迹**是模型依据当前画面、实测车速和保持／左移／右移指令给出的规划。每 0.1 秒一个点，共 40 点。控制器跟踪绿线，在保持车道时用近端导航带约束前瞻点。
- **彩色占用栅格**表示模型预测的动态障碍物位置，对应未来 0.5、1、2、3、4 秒。第三人称跟车界面在右上角叠加合成 BEV，可与路面上的预测对照查看。

鼠标面板可请求左、右变道和调整目标速度。变道请求由控制器持续评估相邻车道空隙。目标速度用于纵向控制，模型输入使用当前实测车速。

## CARLA 0.9.15 实测记录

| 场景 | 第 22 轮基线结果 |
| --- | --- |
| Town04，18 辆 NPC 的固定路线 | 45 秒行驶 401.39 米，路线横向偏差峰值 0.377 米，HIP Graph 推理 P95 为 3.31 毫秒 |
| 独立右岔口，1 条路线 | 分岔后 40 米观察窗内，37/37 帧处于指定分支 |
| 高速直道 | 118–122 km/h 连续居中 2.20 秒，全段指定车道横向误差 P95 为 0.513 米。验收目标分别为至少 10 秒和低于 0.35 米，仍待达标 |

## 在本机运行

从项目根目录打开桌面启动器，选择已安装的模拟器、地图、本地模型和 NPC 数量，点击“开始驾驶”。复现上表的基准需要 CARLA 0.9.15、匹配的 Python 客户端、ROCm 推理环境以及本地模型包。

```powershell
py -3 -m v6_dual_system.desktop_launcher
```

已有 0.9.15 环境时，也可以从命令行启动 Town04 跟车界面。

```powershell
& .\v6_dual_system\Start-FastPilot.ps1 -SimulatorVersion 0.9.15 -Town Town04 -Mode interactive -Scenario traffic-right-fork -ModelDirectory .\models\carla-driver\v6-unified-r22 -NpcCount 18 -InitialSpeed 40
```

发布标签为 `v1`。为兼容已安装的桌面快捷方式及本地模型目录，Python 包名沿用 `v6_dual_system`。本地模型包包含 `best.ckpt`、`open-loop.json` 和 `graph.json`。Git 仅发布源码和文档，运行需要另备模型与环境。此版权重的 SHA-256 为 `2bf80d4c55a24a158be502f0a15803be86d3556c0440be2104537278d6f3cab6`。

[规划网络](v6_dual_system/fast_model.py) · [控制器](v6_dual_system/controller.py) · [跟车画面与 BEV 渲染](v6_dual_system/chase_view.py) · [训练入口](v6_dual_system/train_fast.py) · [早期 Qwen Drive 架构文档](docs/QWEN_DRIVE_HIGHWAY_120_ARCHITECTURE.md)
