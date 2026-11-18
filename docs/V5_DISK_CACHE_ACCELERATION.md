# 冻结层磁盘缓存加速（2026-11-18）

用户明确授权使用剩余项目磁盘空间加速正在运行的2000条×3轮训练。原始路径仅有3GiB RAM缓存，实际缓存命中率约7%，最近100条约14.9秒/条，许多冻结前向被反复计算。

新入口`python -m v5_qwen_drive_world.resume_disk_cached --phase subset2000 --resume`复用原Lightning训练器、损失、优化器、StatefulDataLoader和checkpoint实现，只替换冻结特征的存储方式，并限定checkpoint在实际完整更新后保存。实施在独立模块，没有改动当时正在运行的训练源文件。

保存冻结ViT＋前24层的BF16输出、位置、当前actor所需前6组真实KV及当前输入元数据。后8层LoRA、另外2组KV、EMA适配器与World输出实时计算，不缓存过期可训练输出，不量化，不缩图，不改变数据采样顺序。future_teacher和actor目录及key分离。

缓存身份包含底座SHA、数据清单SHA、预处理/冻结层实现SHA、训练配置、特殊token嵌入SHA、torch/ROCm版本；每条key另包含actor JSON与图片路径/大小/修改时间。输入变化会失效。写入使用同盘pending＋fsync＋原子替换。配置允许3GiB RAM及最多128GiB项目磁盘缓存；按实际访问逐步生成，不另外预训练、不先跑完整预计算。

缓存会在首次访问时增加，不会一启动就全命中。当前train＋validation最多2128个actor缓存，训练有4328个不同未来教师窗口；具体磁盘需求与加速比以生成后的实际统计为准。

存储层的逐值读回、角色隔离、输入变化失效、写入失败保留原文件检查已通过，记录`artifacts/v5-qwen-drive-world/disk-cache-storage-check/result.json`。真实原生模型联合loss/全部可训练梯度对比及实际续训也已通过，详见下面实测。

## 实测（2026-11-18）

第80次参数更新、640条抽样处核验完整断点后切换。实际预训练模型＋真实B2D六视角样本，冷计算21.8309秒；强制清空RAM后从磁盘读取，连续两次分别0.5103/0.4942秒。联合损失均为5.0024542809，176项可训练梯度按rtol1e-5/atol1e-7逐项通过。测试0次优化器更新，原像素未修改。此约43倍是该样本全部特征已缓存的对比，不是整个训练的加速倍数。

首次恢复后的8条抽样与未中断采样器从头重放的第641–648条逐条一致；176项FP32优化器状态及EMA step=80均恢复，第81次更新完成。后续checkpoint检查发现Lightning回调未恢复_last_global_step_saved，恰好从8的倍数step恢复时可能在第一个未提交微批重复保存；`UpdateBoundaryCheckpoint`在启动时记录已恢复step，并只在真实累积边界调用原ModelCheckpoint。旧入口不再用于之后的续训。

已在step88、704条抽样的完整断点上切入最终`resume_disk_cached`入口。回调单元检查通过；实际恢复后的首微批执行后，原完整checkpoint仍为step88/704条，SHA256保持不变（`checkpoint-boundary-live.json`）。随后第705–712条顺序核对通过，并完成step89更新（`disk-resume.json`）。此前磁盘缓存namespace保持一致并成功复用，实际训练仍在继续；本次没有缩图、量化或重置优化器。

按step80处采样器确定顺序重放，剩余1360/2000/2000次抽样，需新增约5762个冻结场景编码。用旧日志B2D/高速冷计算中位数21.02/5.81秒、磁盘命中0.502秒估算，剩余纯计算约7.20小时，缓存约58GiB。四分支平均分摊冷开销是近似；验证、保存及系统负载另计，初步总剩余7–9小时，不是完成时间承诺。证据`subset2000/disk-time-estimate.json`。

切换只在原v5下一个完整checkpoint验证成功后进行，保留模型、176项FP32优化器状态、EMA、调度器、RNG和采样器位置。记录`subset2000/disk-cutover.json`、`disk-resume-expected.json`、`disk-cache-contract.json`与`disk-resume.json`。断点之后尚未提交的微批可能重算；不丢弃已保存的完整参数更新。
