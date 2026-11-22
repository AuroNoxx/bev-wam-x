# Highway OCC Flow · v2 (Stage 2.6)

![Stage 2.6 Epic Demo](docs/images/v2/stage26-epic-demo.webp)

*CARLA 1.0 / UE 5.8 · Epic 画质（`1824×1310` 3D 场景 + `712×1310` 战术 BEV）· 40 Hz 帧锁视角 + 20 Hz 感知规划 · 零样本 `Town04_Opt` 多车环线（12 NPC）巡航与右并线（[高清截图](docs/images/v2/stage26-epic-shot.jpg)）*

`v2`（源码目录 [`v2_mapless_wa/`](v2_mapless_wa/)）面向 CARLA 1.0 / UE 5.8，基于三目裸传感 RGB（前 `768×448` / 左右 `384×224`）与 64 线 360° LiDAR BEV，结合 GNSS/IMU 与粗粒度 SD 道路导航及变道意图，在不输入高精地图的条件下完成车道感知、未来动态占用预测与闭环控制。OpenDRIVE 仅用于生成训练监督与离线/闭环评测判分。

---

## 1. Stage 2.6 核心改动

- **时序地图记忆与自车运动补偿（`Phase26`）**：模型共 `15,862,810` 参数，在 `v2`（Stage 2）基础上引入跨帧 `prev_map_tokens` 与连续 IMU/速度推导的 `ego_delta` 帧间位姿变化，并以 `clip_len=4` 连续片段训练，抑制分界线闪烁并保持匝道出口目标一致性。
- **语义增强 SD 导航与单向注意力隔离**：在 20/40/60 m SD 导航点外增加 `route_instruction`（follow / exit / turn / roundabout）与 `route_dist`；保持地图查询与交互指令的严格单向隔离（离线对照中改变左/右指令，地图折线、存在性、分割与占用最大差值均为 **`0.0`**）。
- **物理控制与安全包络重构**（[`phase2_control.py`](v2_mapless_wa/phase2_control.py) / [`phase2_live.py`](v2_mapless_wa/phase2_live.py)）：
  - **C¹ 连续平滑变道与曲率前瞻**：改用带符号道路曲率 + 速度自适应 Hermite 接续长度，修复接缝处虚假曲率尖峰与单帧虚线漏检减速，变道横向偏移恢复更平顺。
  - **38 m LiDAR 包围盒预过滤 + 聚类障碍检出**：在 `project_corridor` 前做无损 AABB 预筛选（32,000 点合成扫描耗时从 `20.03 ms` 降至完整 `step()` `6.56 ms`，提速 `3.05x`，`max_gap_diff=0`）；动态占用预测采用按预估车速 `v * t` 对齐的时空危险区。
  - **巡航 SD 对齐与 40 Hz 帧锁双窗口演示**：新增 Option 3A 无地图巡航对齐；[`phase2_demo_async.py`](v2_mapless_wa/phase2_demo_async.py) 采用 40 Hz 后台渲染节拍线程（`FIXED_DT = 0.025`）+ 一帧运动学前瞻相机锁定，独立于主线程 HIP Graph 与 OpenCV HUD。

---

## 2. 关键量化指标

### 训练与本机离线验收

- **云端训练**：Google Colab L4，PyTorch Lightning `bf16-mixed` + `cuda_flash` SDPA + `foreach` AdamW；以本机冻存的 Stage 2 基线（详见 [技术报告](v2_mapless_wa/doc/issue.md)）热启动，`clip_len=4`，`physical_batch=8`，`gradient_accumulation=2`，10 轮共 `actual_updates=7,500`（6 个动作桶各 `2,000` clip/epoch，耗时 `7,421 s`）；按 `0.25 * lateral_near_m + 0.15 * val_exit_near_m - map_miou` 选定 **Epoch 4（`selected_updates=3,000`）**。
- **本机验证环境**：AMD RX 7900 XT（ROCm BF16 + `aotriton` SDPA），36 层 BatchNorm 折叠 + BF16 HIP Graph（`batch=1`，250 次迭代）与 Eager 最大绝对误差 **`0.0`**；单帧推理耗时 **`p95_wall_ms = 5.86 ms`**（`run_cpu` P95 `5.92 ms`，`p50_device_ms = 5.07 ms`，模型分配显存峰值 **`0.174 GiB`**）。

| Town12 独立验证集（2,000 帧） | Stage 2（第 14 轮） | Stage 2.6（第 4 轮） | 说明 |
| --- | ---: | ---: | --- |
| 0–24 m 横向 L1 | 0.101 m | **0.074 m** | 521 帧出口近段 `0.086 m` / 418 帧弯道近段 `0.045 m` |
| 24–60 m 横向 L1 | 0.829 m | **0.553 m** | 远场引导精度提升 |
| 7 条地图折线平均误差 | 0.596 m | **0.538 m** | — |
| 地图 mIoU | **68.39%** | 68.38% | 持平 |
| 未来占用 IoU | 43.88% | **44.21%** | 5 个未来时刻均值 |

### 闭环评测（CARLA 1.0 / UE 5.8 · DX12 High · 640×360 · 20 Hz 同步）

> 说明：下表汇总 **3 个独立会话、共 9 条路线、10,989 帧、0 次碰撞**的闭环回归结果（`Town04_Opt` 未参与任何训练）。该轮实测于平滑控制升级之后、最新 `38 m LiDAR 包围盒 3x 加速` 补丁之前，故当时整卡控制周期 `loop_p95_ms` 仍高于 50 ms 实时线（改后整套 9 路线时延待在 CARLA 同步评测中复测）。

| 路线 | 帧数 | 碰撞 | 横向 P95 / 目标道 P95 | 关键行为与分项目标 | 推理 P95 / 控制周期 P95 |
| --- | ---: | ---: | ---: | --- | --- |
| `town12_val_freeway_exit`（独立验证区） | 1,500 | 0 | 0.987 m | **岔口命中 1/1**（Stage 2 为 `0/1`） | 7.01 ms / 91.99 ms |
| `town04_ring`（零样本环线） | 1,251 | 0 | **0.337 m** | 正常到达终点 | 7.58 ms / 63.19 ms |
| `town04_ramp_exit` | 1,450 | 0 | 1.100 m | 岔口 `1/1`；紧弯跨线 `60/60` | 6.71 ms / 60.54 ms |
| `town04_ramp_tight_bend` | 789 | 0 | 3.777 m | 岔口 `1/1`；紧弯跨线 `58/58` | 6.67 ms / 60.12 ms |
| `town04_ring_right_request` | 1,296 | 0 | 目标道 **0.247 m** | 58 ticks 入道，偏出 `0/1,202`，变道刹车 `0` 帧，保持至终点 | 6.74 ms / 64.57 ms |
| `user75`（75 km/h 右并线） | 1,007 | 0 | 目标道 0.389 m | 43 ticks 入道，偏出 `0/879`，最低速 `18.7 -> 29.9 km/h` | 6.77 ms / 64.90 ms |
| `user50`（50 km/h 两次右并线） | 1,294 | 0 | 1.540 m / **0.275 m** | 两段偏出均 `0`，最低速 `18.4 -> 31.6 km/h` | 6.81 ms / 66.82 ms |
| `early50`（早弯右并线） | 1,285 | 0 | 目标道 **0.281 m** | 59 ticks 入道，偏出 `0/1,189`，变道刹车 `0` 帧 | 6.67 ms / 66.31 ms |
| `steer_recovery`（1.20 s 转向扰动） | 1,117 | 0 | 目标道 2.306 m | 扰动越线后回到终点，偏出 `62/989`（未持平全段） | 7.88 ms / 64.43 ms |

- **控制耗时的拆分**（上表 9 路线实测）：`control_p95_ms` `25.16–28.10 ms`，`sim_tick_p95_ms` `26–54 ms`。
- **显存实测（3 会话分开报告）**：行驶期 UE + 模型联合峰值 **`9.42–9.54 GiB`**，整卡行驶峰值 **`13.13–13.43 GiB`**（纯 UE 加载峰值 `10.63–12.83 GiB`，整卡全程峰值 `14.49–16.59 GiB`）。

### 双窗口 Epic 实时演示模式

- **全环线演示**（`Town04_Opt` `town04_full_ring` 3 圈 `9.21 km`，`12 NPC`，UE5 Epic `1824×1310` + BEV HUD `712×1310`）：连续行驶 `3,402.7 m`、`4,191` 控制帧、UE5 平均 **`40.89 FPS`**、`0` 碰撞、车道偏移 P95 **`0.450 m`**。
- **上方 GitHub 动图录制**（附加一路 `960×688` 录制相机，Epic + 12 NPC + 自动平滑右并线）：UE5 平均 `32.48 FPS`、`271` 帧、行驶 `127.55 m`、`0` 碰撞、车道偏移 P95 `0.188 m`。

---

## 3. 快速启动双窗口 Epic 演示

```powershell
# 一键启动 Town04_Opt 3 圈 9.21 km Epic 实时演示（2560x1440 分屏适配）
.\scripts\Start-Stage26Demo.ps1

# 或运行独立闭环验收（20 Hz 同步）
.\.venv-train\Scripts\python.exe -m v2_mapless_wa.phase2_live --out data/v2-mapless-wa/eval --weights <path-to-stage26.pt>
```

快捷键：`A / D` 左右并线 · `W / S` 巡航车速 `±5 km/h` · `Space` 暂停 · `R` 重开 · `Esc / Q` 退出。

---

## 4. 下一步：Stage 3 路线图

- **多模态时空轨迹与隐空间世界模型**：探索在编码器隐空间 + BEV 动态占用中结合 **Flow Matching** 多模态轨迹规划与**动作条件世界模型（Action-Conditioned World Model）**，联合输出带速度规划的时空轨迹 `(x(t), y(t), v(t))`，减少规则限速。
- **自主战术决策与拟人手感指标**：补齐“自主变道/借道超车”战术数据与闭环机制，并在闭环验收中正式引入**变道最大掉速、方向盘转角与转向角速度 `|d(steer)/dt|`、横/纵向急动度（jerk）与跟车时距（THW）分布**等拟人化门禁。

---

[Stage 2.6 技术架构与闭环问题报告](v2_mapless_wa/doc/issue.md) · [Phase 1 三目标定报告](v2_mapless_wa/PHASE1_REPORT.md) · [v1 冻存版本](https://github.com/AuroNoxx/highway-occ-flow/tree/v1)
