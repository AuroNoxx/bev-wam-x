# 两步真实数据SFT（2026-11-17）

完成第二步后，按[扩训判据与 Colab 预算](V5_SFT_SCALE_DECISION.md)核对第 750 步结果；其中的质量门槛用于决定下一阶段的算力投入。

2026-11-18空轮修正：原2000×3进程结束时仅500更新，第二个框架epoch抽样数为0，实际只完成两个有效轮次。采样器边界与轮末恢复已修正并通过CPU框架契约检查；当前以`complete_three_passes --phase subset2000 --resume`从step500补齐至750，原配置和学习率计划保持不变。详细证据见[V5_IMPLEMENTATION_STATUS.md](V5_IMPLEMENTATION_STATUS.md)。后文`resume_disk_cached`为此前入口，本次修复后的补跑须使用新入口；框架epoch4代表第三个有效轮次。

2026-11-18更新：第一步已完成，第二步已启用用户授权的冻结层磁盘缓存加速；原3GiB RAM缓存仍作一级缓存。当前续训入口为`python -m v5_qwen_drive_world.resume_disk_cached --phase subset2000 --resume`，包含恢复后累积边界保存修正。详见[V5_DISK_CACHE_ACCELERATION.md](V5_DISK_CACHE_ACCELERATION.md)。以下无磁盘缓存描述为初始实现。

用户要求先运行64条×30轮过拟合，再运行2000条×3轮子集微调。显存13GiB、ADE0.2m、Persistence改进等均只记录，不作为流程门槛。这两步都属于SFT，不是World-Guided GRPO。仅运行这一条训练主线。

## 已准备的真实数据

- 64条：45条低速且前方存在车辆的Bench2Drive窗口，19条实测100–120km/h高速窗口，其中10条在3秒内实际换道。窗口相关且来自少数路线，数量不等于独立场景数。
- 2000条：1400条Bench2Drive、600条Town04补采。实测速度0–117.36km/h；60–80km/h缺桶，尚未达到完整0–120覆盖。Town06未安装、未采集。
- 独立验证128条来自原有隔离Bench2Drive路线，最高43.51km/h；目前没有独立80–120km/h验证数据，高速验证统计必须报告空值。所有同初态Town04分支归训练集。
- 当前六视角＋前视四时刻，使用官方processor原像素预算，不设置缩小尺寸。原始RGB、标定、位姿、实际速度、帧时间与标签分开保存。未来RGB只进EMA标签教师，不进入当前actor。
- 未来50个10Hz真实位姿构成5秒轨迹；当前FLU网格为[时间,Y,X]。占用标签仅含车辆/行人足迹，不涵盖完整静态场景。B2D动态标签有效半径保守限定90m，unknown有独立mask。
- 运动学控制标签由实际速度/姿态差分得到，按既定控制范围裁剪；保留raw_controls与裁剪比例，真实轨迹不改。64条平均16.17%的控制分量发生裁剪，2000条平均13.45%；模型控制范围和低速运动学近似可能限制轨迹拟合，不能承诺0.2m。
- 将64条裁剪后真值控制直接按既定积分器回放，平均轨迹重建ADE为3.26m（B2D1.66m、高速7.04m）。这不是模型成绩；部分高速变道超出控制范围，表明动作标签约束与真实轨迹并不完全相容。按用户最新指示保留实测结果继续两步流程，不修改真实轨迹来制造达标。

## 训练实现与记录

`train_supervised.py`使用Lightning自动优化、PEFT、TorchData StatefulDataLoader、BF16 mixed、FP32训练参数/优化器/运动学、SDPA MATH及已经验证的FLA算子。真实六视角＋三个EMA教师时刻的训练先采用微批1/累积8/有效batch8，不能套用三相机诊断的微批4吞吐。

新增REASON/PLAN标记及12类意图监督，Flow匹配、真实轨迹/动力学损失、未来动态占用Focal和EMA残差潜变量监督。额外当前占用Focal训练同一共享解码器，以便Persistence复制当前预测。当前模型在记录的真实动作条件下评估World，不把它等同未执行动作的反事实真值。未知官方四通道指令暂填零并明确标记，真实导航类别、FLU点和巡航速度仍输入。

冻结ViT及前24层只在有缓存时复用，RAM缓存上限3GiB，不生成大规模磁盘特征。保留8组真实planner KV，后8层LoRA始终重新计算。真实六视角检查已确认缓存前后hidden/KV逐值相同，40个LoRA梯度张量一致，序列850tokens、原像素未改。

每步损失与本地TensorBoard、每轮训练均值、每5轮过拟合评估、每轮子集独立评估落盘。ADE/FDE通过10步Flow ODE采样计算，记录横向加速度/jerk、未来World与Persistence同支持mask的Brier。64条评估复用训练样本，只是过拟合，不是泛化。

每阶段仅latest/best（验证ADE选择）两个固定文件，保存可训练参数、EMA、优化器、调度器、RNG、Lightning循环及StatefulDataLoader位置；固定官方底座由版本/SHA身份引用，不重复复制9GB底座。VerifiedCheckpointIO临时写入、验证、fsync、原子替换，失败保留旧断点。恢复粒度为完整梯度累积更新边界。

第二步从第一步最后权重继续，重新建立第二步优化器/调度器；同一阶段恢复则完整恢复优化器/数据位置。实际两次SFT更新与恢复检查通过：176项FP32优化器状态，EMA更新步数一致，恢复后8条样本逐条吻合；故意保存失败后旧checkpoint SHA256不变。修正了提前结束fit后保存会失去轮内位置的问题，改用更新边界的ModelCheckpoint。证据`overfit64/resume-step1.json`与`resume-step2.json`。第1轮恢复片段的训练均值按fragment标记，之后完整轮指标正常记录。

`run_two_stage.py`已顺序启动两步流程，活动阶段和是否结束以`artifacts/v5-qwen-drive-world/two-stage-run.json`为准；各阶段samples/epochs/validation日志给出实测，不以启动替代完成声明。

## 清理与补采证据

逐文件SHA256核对旧提取OCC与保留源归档后，清理12333个重复/已登记临时文件，实际释放11.55GiB；原始归档、RGB、标注、日志、环境和官方权重保留。登记：`artifacts/v5-qwen-drive-world/duplicate-cleanup-result.json`。

高速采集使用本项目CARLA0.9.15，Epic/DX12实际稳定。旧Low/DX11尝试发生D3D设备丢失，保留错误记录。官方BasicAgent执行离线变道专家，实测道路45从6道变5道，命令时约108km/h，变道时约111km/h；它只产生训练数据，不是部署模型的兜底驾驶器。采集结束关闭自身CARLA进程后再训练。
