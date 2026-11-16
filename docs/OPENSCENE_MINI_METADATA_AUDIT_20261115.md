# OpenScene mini 元数据审计（2026-11-15）

## 获取与范围

仅下载官方 getting_started 文档列出的 v1.1 mini metadata，未下载相机、LiDAR或OCC包。HuggingFace链路较慢，停止本次自己的下载，改用官方列出的ModelScope直链；镜像约27秒完成。没有登录、注册或点击接受条款。归档534,344,190字节（509.59MiB），SHA256 `c2e08867e062ba916eccd56df191e728d42fb94de854c28f2226eed14cecd8be`，与镜像响应X-Linked-ETag一致。

原始归档位于 `data/public/openscene-mini-audit/openscene_metadata_mini.tgz`。HF未完成的约30MB `.part` 保留，不作有效数据使用；未擅自永久删除。

审计脚本 `scripts/audit_openscene_mini_metadata.py` 流式读取tar中的pickle元数据，仅允许numpy数组重建，不执行任意自定义pickle对象、不释放路径到文件系统。结果 `artifacts/openscene-mini-metadata-audit-20261115.json`。不读取当前项目保留测试数据。

## 实测结果

- 64个日志，51,867帧，token全唯一。
- timestamp中位间隔0.500003秒；P95=0.500332秒；最大约1.00072秒，33处间隔不在0.4–0.6秒之间，不能当完整无缺帧连续记录。
- 日志时间跨度85.50–565.50秒，总跨度约7.20小时；总跨度不是去掉缺口后的严格连续暴露时长。
- 地点：Las Vegas strip 48段，Singapore 6段，Boston 5段，Pittsburgh 5段。地点/速度不是道路等级标注。
- 50,265帧有OCC路径字符串，1,602帧没有。尚未下载OCC，不能验证路径实际存在、label质量或mask。

按 `ego_dynamic_state` 前两维范数作为m/s速度并换算km/h（官方字段定义待研究核对，已用位姿差分交叉验证）：

|统计|km/h|
|---|---:|
|中位数|10.01|
|P90|38.35|
|P95|43.83|
|P99|50.09|
|最大|57.74|

**>=60、80、100、120 km/h 均为0帧。** 与ego全局位姿、真实timestamp差分速度的绝对偏差中位0.192km/h、P95=1.180km/h，支持上述速度量级。不是仅凭数据集名字推断覆盖。

因此mini样本不能补齐本项目高速驾驶数据缺口。它可以用来核验长历史元数据契约，但不能据此自动下载其6.7GB OCC，或宣称整个OpenScene都没有高速；完整集分布仍需单独审计。

## 因果及许可边界

官方README明确每帧OCC使用20秒LiDAR累积。尚未核实其时间窗方向与未来融合细节，因此累计OCC不能直接当实时因果actor输入；可研究作为未来监督标签，actor仍需要当前可观测或严格历史因果重建输入。长记录不能消除这点。

数据许可为CC BY-NC-SA及nuPlan非商业数据许可，代码Apache2.0不能替代数据许可。当前仅公开mini元数据研究审计，没有接受额外协议页面；正式使用前仍需依据实际用途确认许可，不能假定适合商业/车端产品训练。

来源：
- https://github.com/OpenDriveLab/OpenScene/blob/main/docs/getting_started.md
- https://github.com/OpenDriveLab/OpenScene/blob/main/README.md
- https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
- https://www.nuscenes.org/terms-of-use
- https://www.modelscope.cn/datasets/OpenDriveLab/OpenScene/resolve/master/openscene-v1.1/openscene_metadata_mini.tgz

## Velocity field evidence update
Official NAVSIM consumption code maps `ego_velocity = np.array(ego_dynamic_state[:2])` and acceleration from remaining entries: https://github.com/autonomousvision/navsim/blob/main/navsim/common/dataclasses.py . Upstream nuPlan schema defines ego_pose vx/vy as local m/s: https://github.com/motional/nuplan-devkit/blob/master/docs/nuplan_schema.md . This agrees with measured pose/timestamp finite-difference checks above. Original OpenScene export assignment has not been traced; evidence is consumer schema + upstream units + numerical cross-check, rather than a claimed exporter audit.
