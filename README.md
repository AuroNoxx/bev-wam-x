# Highway OCC Flow

固定架构：历史 OCC + 自车/导航 → Flow Matching 带时间候选轨迹 → 控制器/动力学可执行检查 → 轨迹条件世界模型 → 评分 → Pure Pursuit + 速度 PID → CARLA。

Windows / RX 7900 XT 20GB。只训练 OCC 模型；不增加 RGB 感知模型、语言模型或直接踏板策略。所有环境、数据和模拟器保留在项目内。本地 TensorBoard，不使用 SwanLab、云 API 或租卡。

## 当前入口

- [本轮阶段一结果：整体改善，急刹车退步，尚未通过验收](docs/STAGE1_V2_RESULTS.md)
- [当前第一阶段续训：原因、约束、指标和运行方式](docs/STAGE1_CONTINUATION.md)
- [每个阶段做什么、配置和验收](docs/STAGES.md)
- [Lightning 框架、加速实测与迁移](docs/FRAMEWORK.md)
- [新架构真实测速与耗时口径](docs/SPEED_REPORT.md)
- [ModelScope / HF / Mirror 下载实测](docs/DOWNLOAD_SOURCES.md)
- [完整架构与坐标契约](docs/ARCHITECTURE.md)
- [数据准备约束](docs/DATA_TRAINING_PLAN.md)
- [CARLA 与 OCC 网页操作](docs/EVALUATION_VIEWER.md)
- [UE5 安装验证证据](docs/CARLA_UE5_VALIDATION.md)
- [清理实际结果](docs/CLEANUP_STATUS.md)
- [项目规则及断点保留](AGENTS.md)

## 训练

```powershell
# 当前阶段一 v2：中断后恢复本轮（首次启动不带 -Resume）
.\scripts\Start-Stage1.ps1 -Resume
```

首批 60 段公开数据按 Town12 训练 / Town13 验证 / 其他地图测试隔离，42/9/9 段。当前第一阶段 v2 从首版最佳正式权重继续改进，仅初始化权重，新建优化器和运行；不使用测速权重。启动器状态见 `logs/stage1-v2/stage1-launch.json`。

2026-11-15：训练/验证/测试窗口分别为 5296/1124/2485。首版第 34 轮正常早停，最佳权重为第 26 轮，保存在 `artifacts/lightning/flow/`。阶段一 v2 已完成 15 轮 / 9930 步续训和三种子配对验证：整体平均 ADE 3.55→3.23 m，急刹车 3.92→4.12 m；最佳为本轮第 5 轮。训练已结束，阶段一尚未通过验收，第二阶段未启动。详细取舍见结果报告。

当前框架为 PyTorch Lightning 2.6.6。实际训练状态：`artifacts/lightning/flow-v2/status.json`；运行日志：`logs/lightning/flow-v2.log`。每运行仅保存 `latest.ckpt` / `best.ckpt` 两份可恢复断点。训练期间关闭 CARLA。原 `train_v1.py` 仅作历史参考，正式入口不再调用其训练循环。

```powershell
.\.venv-train\Scripts\python.exe -m tensorboard.main --logdir logs/tensorboard/v1 --host 127.0.0.1 --port 6006
```

本地监控 http://127.0.0.1:6006/ ，选择 `lightning-flow/stage1-consistency-v2`；`lightning-flow/formal` 和 `v1-flow` 为首版历史。具体测速及配置见续训报告；离线误差不代表已具备闭环驾驶能力。

## 测试观察界面

```powershell
.\scripts\Start-OccViewer.ps1
.\scripts\Stop-OccViewer.ps1
```

左网页 http://127.0.0.1:8765/ ，右 CARLA。跟车环视已实现：按住 CARLA 内左键看四周，镜头跟随车辆，松手平滑回到车后方；用户要求暂缓实际拖拽验证。当前驾驶仍是 Traffic Manager 演示，显示 LiDAR 表面占用，未连接训练策略；正式模型测试必须先完成阶段 3/4 的适配与验收。
