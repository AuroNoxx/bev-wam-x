# Highway OCC Flow · v2

v2 在 CARLA 0.10.0（UE5）中采集三目 RGB 和 360° LiDAR，用于无高精地图输入的驾驶模型。观测由三路 384 × 224 相机、六通道 LiDAR BEV、道路级粗导航和战术指令组成。高精地图只用于制作监督标签。

本分支完成 100 帧三目标定试采，白线重叠误差 P95 为 2.236 格，左右近场覆盖约 79%。正式批采等待几何门禁通过，以及 Town12 和 Town04_Opt 的兼容地图就绪。Town04_Opt 专供后续零样本闭环测试。

[校准报告与五张实拍图](v2_mapless_wa/PHASE1_REPORT.md) · [v2 工程](v2_mapless_wa/) · [v1 冻存版本](https://github.com/AuroNoxx/highway-occ-flow/tree/v1)
