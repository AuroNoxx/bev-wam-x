# Bench2Drive 原始标签契约与无学习基线（2026-11-16）

## 范围与复现
这是现有公开数据子集的数值契约检查和解析基线，不是官方模型复现。固定原有 Town12 42 条 train 录制、Town13 9 条 validation 录制；42 条 train 对应41个路线组，其中 HighwayExit 有同路线不同录制。5296 train、1124 validation 缓存窗口。此轮仅3个锚点读取train/val，基线遍历全部1124val；不读取test，不训练，不安装依赖、不修改默认配置。

官方数据 revision `d84ae18cf20a88b9220df1bc15c073d8824b7ac1`。现有manifest SHA前后不变。脚本：`scripts/audit_bench2drive_official_baselines.py`；权威结果：`artifacts/bench2drive-official-contract-baselines-v3/{plan,anchor_contract,report,rows,units,synthetic_formula_tests}.json` 和 `baseline_metrics.csv`。CPU全量约2秒。v1因NumPy布尔JSON汇总失败，v2保留，v3修正速度幅值和有符号误差分别报告；没有修改原数据。

## 官方来源与不能混淆的定义
- 官方Zoo固定commit [prepare_B2D.py](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/498c1f799dd90faf840dedb3f0d3234ec2e567db/mmcv/datasets/prepare_B2D.py)：使用S=diag(1,-1,1,1)转换world/ego；导航world y翻转；ego_yaw=-theta+pi/2；ego_vel=[speed,0,0]；accel翻y；rotation_rate取负。theta是compass约定，不能直接当CARLA yaw。当前缓存以raw world2ego得到当前ego FLU，而不是直接把theta当yaw。
- 官方 [B2D_vad_dataset.py](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/498c1f799dd90faf840dedb3f0d3234ec2e567db/mmcv/datasets/B2D_vad_dataset.py) 用frame_idx/10作为timestamp；原JSON无真实timestamp，因此这里只是名义10Hz。官方VAD默认轨迹为当前LiDAR坐标逐步差分，sample_interval5/future_frames6；本项目当前ego FLU、0.2秒×25点，不能声称与官方VAD标签/模型逐项复现。
- 官方consumer之外，现已补齐gyro生产链：data_collect.py从IMU[3:6]原样保存angular_velocity，sensor_interface直接取gyroscope.xyz，CARLA0.9.15 InertialMeasurementUnit.cpp使用GetPhysicsAngularVelocityInRadians并转换到actor/传感器局部，明确rad/s。无需deg/rad再次换算。speed按官方ego_vel有符号纵向量使用。这仍不是独立时钟/IMU标定认证。全val当前gyro与过去0.1秒pose yawrate绝对差中位0.00616、P95 0.25567 rad/s；speed与过去位移速度绝对差中位0.0664、P95 0.3218 m/s。瞬时测量与区间估计并不完全一致，不能为提高val分数调符号、单位或平滑器。
- expert_assessment文件i−1动作对齐是官方consumer另一契约，不能用来证明OCC偏移。此轮检查原anno内throttle/steer/brake/reverse，精确控制施加前后时序仍未知。
- [occ_online_carla.py](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/tools/occ_online_carla.py) 的online历史队列与offline centered21帧路线并存；发布实际调用链尚未认证。既不能宣称已证未来泄漏，也不能保证past-only。原值0未填充不等于free，2映射unknown；缺显式free/可见性真值。

## 三个固定代表锚点
1. train HighwayCutIn Town12 Route23683，frame41。
2. train HardBreakRoute Town12 Route22720，frame160。
3. validation SequentialLaneChange Town13 Route3219，frame73。

均取路线可用窗口中位索引，未按基线好坏挑点。检查raw pose逆矩阵、actor位置、compass关系、原导航到本egoFLU、未来pose→缓存轨迹、控制字段、OCC class/unknown及i−1 source网格重建。三点轨迹归一化误差最大3.78e−8、导航2.79e−8，当前OCC重建逐值相等。全1124val缓存位置与独立rawpose重建最大误差2.97e−6m。

这证明实现对应和数值一致，不证明传感器真实采样同步或源OCC生成因果。顶层x/y与ego actor位置不是完全相同字段，不替代world2ego。源OCC i−1配准沿用既有rawLiDAR几何证据，不凭文件名认定时间。

speed是有符号纵向值：val有90个轻微负值，最小−0.0462m/s，没有<-0.2；reverse控制flag有4帧，彼时速度仍正，不能据flag翻转速度。位置基线用有符号speed；报告同时列速度幅值MAE和有符号纵向MAE。

## 冻结无学习基线
静止：全轨迹原点。恒速：当前有符号纵向速度、零yawrate。CTRV：同速度加当前annotation gyro_z的FLU负号，不使用未来位姿导数。公式已通过零转率、左右对称、反向速度、静止测试。CTRV是无侧滑恒转率简化，不能代替真实车辆动力学；没有反推steer标定。

平均终点位置误差，单位m；列对应1/2/3/5秒：

|基线|1s|2s|3s|5s|
|---|---:|---:|---:|---:|
|静止|6.436|13.141|19.993|34.497|
|恒速|0.901|2.503|4.270|8.631|
|当前gyro CTRV|1.170|3.981|7.820|18.259|

速度幅值MAE，恒速和CTRV相同：1.274/1.763/1.929/2.513m/s；静止：6.633/6.793/6.973/7.525m/s。CSV另外提供各horizon prefixADE、位置P95、每条路线/场景族及静止/减速分层。

- 当前静止208窗；完整5秒保持静止72窗。这72窗静止终点误差0.012/0.020/0.024/0.028m，恒速0.016/0.037/0.053/0.088m。
- 减速169窗，恒速终点误差1.396/4.138/6.514/11.768m；CTRV1.822/6.061/11.074/23.650m。
- 加速415窗。各组用未来标签事后分类，不进入基线输入；窗口高度重叠，不能当独立驾驶事件。

## 结论
本地标签坐标/导航/缓存转换在这些检查中一致，公开数据已有可复现的低门槛无学习参考。当前gyro恒转率外推在该固定val上比恒速差，已抽样确认当前瞬时gyro与过去区间平均差异可由快速左右变向解释，不能据此认定错帧；本轮不据此修改gyro、挑参数或训练。位置/速度拟合仅是离线诊断，不是官方模型效果、碰撞率、闭环安全或驾驶质量验收。

补充生产链：[data_collect.py](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/tools/data_collect.py)、[sensor_interface.py](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/leaderboard/leaderboard/envs/sensor_interface.py)、[CARLA0.9.15 IMU实现](https://github.com/carla-simulator/carla/blob/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Sensor/InertialMeasurementUnit.cpp)。B2D传感器零安装旋转，gyro noise std0.001；瞬时与区间平均不同不应当作单位错误。
