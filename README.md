# Highway — Qwen Drive World v5

原生 Qwen-Drive-1.0 历史 RGB/自车状态/导航，联合控制量 Flow 与动作条件未来 World，目标 0–120 km/h。

- [架构入口](docs/ARCHITECTURE.md)
- [完整 v5 规范与接口差异](docs/QWEN_DRIVE_HIGHWAY_120_ARCHITECTURE.md)
- [实际实施状态](docs/V5_IMPLEMENTATION_STATUS.md)
- [项目规则](AGENTS.md)
- [旧文档归档](docs/archive/pre-v5-20261117/)

新模块：`v5_qwen_drive_world/`。本轮不沿用旧 OCC 输入网络。已完成旧运行权重按用户指令清理；历史日志、数据和报告保留。目标性能与训练/闭环完成状态以实际状态文档为准。

当前性能改造：原生像素＋≤850视觉/文本token、前24层冻结/后8层LoRA、纯2D动作条件World。微批4、累积2的两次联合更新已通过；官方三相机演示场景0.819秒/样本，PyTorch allocated峰值14.79GiB。真实数据SFT尚未启动。工程复验：`scripts/Verify-V5Lightweight.ps1 -Microbatch 4`。
