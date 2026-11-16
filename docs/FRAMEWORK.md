# 训练框架与加速迁移

2026-11-15：按用户要求，训练循环迁到 PyTorch Lightning 2.6.6。保留 OCC / Flow Matching / 世界模型任务定义，Trainer 接管自动反向、优化器步进、验证调度、混合精度和恢复；使用原生 EarlyStopping、ModelCheckpoint、LearningRateMonitor、ThroughputMonitor 和 TensorBoardLogger。不使用自写 epoch/batch 训练循环。

## 为什么选 Lightning

当前模型是自定义 OCC 编码器、连续轨迹条件流与未来占用预测器。Lightning 的 LightningModule / DataModule 接口直接容纳这些任务。HF Trainer 同样能训练自定义 torch.nn.Module，但需遵守模型输出、loss 和标签接口；它与 Transformers 预训练模型生态更紧密。本项目没有 tokenizer 或语言模型，因此选择 Lightning 是减少适配工作，不是宣称框架本身更快。

参考：[Lightning Trainer](https://lightning.ai/docs/pytorch/stable/core-api/trainer)、[HF Trainer 自定义模型约束](https://huggingface.co/docs/transformers/main_classes/trainer)。

## 环境和编译

复用项目 `.venv-train`，保留 PyTorch 2.11.0+rocm7.14.0、ROCm 7.14.0、Python 3.13.17。新增 Lightning 2.6.6、TorchData 0.11.0、Triton Windows 3.6.0.post26 及其依赖，没有修改系统驱动、全局 Python 或其他项目。

项目内 `configure_runtime()` 设置编译缓存、临时目录、HIP_PATH 和 CC，仅对本训练进程及子进程生效。编译器使用 Triton wheel 自带的 TinyCC；HIP 头文件来自已有项目 ROCm SDK。没有安装系统级 Visual Studio/Windows SDK。

实测处理过的错误：默认 Inductor 编译子进程触发 `pass_fds not supported on Windows`，改为 `TORCHINDUCTOR_COMPILE_THREADS=1`；随后补齐进程级 HIP_PATH，并选择 wheel 自带 TinyCC，基本 `torch.compile` 前向和反向测试通过。最终是否启用编译取决于同模型完整训练测速，不以 import 成功为依据。

PyTorch CPU profiler 记录到 `aten::_scaled_dot_product_flash_attention`、`aten::_flash_attention_forward`、`aten::_flash_attention_backward`，说明该环境的代表性 BF16 注意力前后向已使用内置 Flash 内核，不需要额外的 flash-attn 包。该检查不等于模型中所有算子均使用 Flash。

BF16 没有 GradScaler，但 Lightning 默认 mixed-precision 插件仍拒绝 fused AdamW 配合梯度裁剪。通过官方 `LightningModule.configure_gradient_clipping` 扩展点调用原来的 PyTorch `clip_grad_norm_`（阈值 1、检查有限值）；未修改 Lightning 源码，也没有取消梯度裁剪。这个扩展点显式拒绝带 GradScaler 的配置。

## 恢复与保留规则

旧运行在第 16 轮结束（10592 步）暂停，原 `artifacts/v1/v1-flow/latest.pt` 和 `best.pt` 保留作迁移来源。只进行一次轮边界格式迁移，不重置优化器动量、学习率或全局步数；第一次迁移后的 shuffle 序列改由 TorchData 管理，不能称为旧循环的逐位等价延续。

新运行使用 `artifacts/lightning/flow/latest.ckpt` 和 `best.ckpt`。原生 ModelCheckpoint 每 50 步更新 latest，每轮更新 latest，并按验证 loss 保存 best（最小值）；原生早停使用 min_delta=1e-4、patience=10，最多累计 60 轮，至少累计 15 轮。旧 best 也转换并保留，避免迁移后丢失历史最佳结果。

使用 TorchData StatefulDataLoader 保存轮内采样及 worker 进度，任务钩子保存 Python/NumPy/Torch/GPU RNG。项目原子保存要求通过 Lightning CheckpointIO 扩展实现：框架写临时文件，重新加载检查字段和有限参数，fsync 后原子替换。框架管理训练及断点时机，扩展只处理本地存储校验。

测速和迁移核验记录放在 `artifacts/lightning/`；旧测速与旧训练日志不冒充 Lightning 运行结果。当前配置为 `configs/lightning.json`，模型与数据契约仍为 `configs/train_v1.json`。

## 最终实测与选择

同一 Flow 模型、完整数据清单、batch 8、BF16，10 步热身后计时 100 步，测速权重不用于正式训练：

| Lightning 配置 | 样本/秒 | 每步秒 | PyTorch reserved 峰值 GiB |
|---|---:|---:|---:|
| 基线：普通 AdamW、NCHW、workers=0 | 135.59 | 0.05900 | 0.641 |
| fused AdamW + channels-last、workers=0 | 134.36 | 0.05954 | 0.715 |
| fused AdamW + channels-last、workers=2 | **164.72** | **0.04857** | **0.715** |

最终采用第三组，实测相对基线约提升 21.5%；收益来自整套配置，不能独立归功于 fused AdamW。该配置不改变 batch、模型、精度或完整数据窗口。第一、二组启用了卷积自动调优；第三组和正式运行关闭自动调优，因此表格用于比较实际配置，不能把全部差异归因于单个因素。

完整模型的 Inductor/Triton 编译能进入前后向，但观察到约 3 秒/步；关闭卷积自动调优后，连续日志点仍为 3.02 秒/步。两次均提前停止慢速测速，未标成完成 100 步，也未用于正式训练。**完整 torch.compile 不启用**，不以“已安装 Triton”冒充训练加速成功。

轮内恢复集成测试：从 10592 步开始连续更新 4 步，对比在 10594 步保存后恢复到 10596 步，最终权重最大差为 0，优化器 step 与调度器一致。动量最大绝对差 7.13e-10，满足 rtol=1e-5/atol=1e-8 的明确浮点容差，不宣称优化器动量逐位相同。模拟原子替换失败后，旧断点内容完全保持。结果见 `artifacts/lightning/contract-test/result.json`。

迁移验证：latest 为第 16 轮/10592 步，历史 best 为第 14 轮/9268 步；权重逐项相等，114 组优化器状态与 scheduler.last_epoch 保留。见 `artifacts/lightning/flow/migration.json`。新框架从第 17 轮继续，而不是拿测速权重或重新初始化训练。

```powershell
# 恢复第一阶段；不会启动第二份重复训练
.\scripts\Start-Stage1.ps1 -Resume
# 当前第二阶段入口，需在第一阶段停下后单独执行
.\.venv-train\Scripts\python.exe scripts\train_lightning.py --mode world
```

TensorBoard 地址不变：http://127.0.0.1:6006/ 。新 run 为 `lightning-flow/formal`（Windows 上有时显示反斜杠）；旧 run 保留用于比较。新日志是 `logs/lightning/flow.log`，实时状态是 `artifacts/lightning/flow/status.json`。

## 正式恢复验证

2026-11-15 01:27 检查：Lightning 正式运行已完成第 17、18 轮，正在第 19 轮；latest 已前进到 12300 步，保存了 114 组优化器状态及 TorchData loader 状态。活动目录只有 latest/best 两份 ckpt。原生 ThroughputMonitor 最近记录约 168.66 样本/秒，PyTorch reserved 约 0.672 GiB（不是全系统显存），第 18 轮 validation loss 约 0.045830。新最佳为第 17 轮的约 0.045632；此前第 14 轮最佳在原运行中仍保留。

TensorBoard 的 train/validation loss、lr、吞吐、内存标量均已通过本地 API 读到；核验记录见 `artifacts/lightning/flow/live-verification.json`。GPU FLOPs/利用率未由当前监控获得，没有把吞吐或显存冒充 GPU 利用率。以上是当时快照，后续状态以运行日志为准。
