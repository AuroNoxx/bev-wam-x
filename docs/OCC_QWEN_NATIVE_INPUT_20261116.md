# 原生 Bench2Drive OCC 输入更新（2026-11-16）

最新用户选择：先不做高速，只使用 Bench2Drive 原始范围/分辨率，不再先缩小后扩大画布。

当前新架构入口：`scripts/occ_qwen_native_flow.py` / `NativeOccQwenJointFlow`。原 `occ_qwen_joint_flow.py` 保留为历史实现和共享主干函数，旧数据不删除，不再作为本轮原生OCC输入入口。

## 数据契约

- 原始每帧 `(128,128,20)` 对应内部 `(Z,Y,X)=(20,128,128)`；历史 `[B,4,20,128,128]`。
- XYZ格距均0.4m；x/y范围[-25.6,25.6)，z[-5,3)。没有64×64预先聚合，没有160列前向扩展。
- `scripts/b2d_native_occ.py` 直接读取已下载Bench2Drive原始occ/seg文件，旧occ_v3仅提供已验证split/帧索引/state/trajectory，不读取旧history/future体素。
- 每个源voxel保留空间分辨率，语义类别沿用7类映射；源occ=0仍是unfilled，不称为已验证free；源occ=2和对齐后超出源支持均unknown=255。原始occ/seg文件未改。
- 历史和训练目标按各帧ego pose，在同样0.4m固定当前ego FLU网格做最近邻刚体对齐。旋转/移动产生的视野外区域为unknown；不补远距观测，不融合未来到历史。最近邻对齐仍有离散采样误差，不等于增加信息。
- actor加载模式 `include_labels=False` 不读取未来pose/OCC/轨迹标签；label模式未来只用于loss。
- 保留anno i / OCC i−1已审计配对；报告记录具体源文件及SHA256。上游归档生成过程因果性仍未完全认证，不能因此宣称真实部署感知。

## 编码器与目标

输入不降采样不扩画布。进入模型后，学习型CNN仍以stride2压缩特征，再生成16场景token，这属于特征编码，不是把输入OCC缩小再扩成伪大范围。20层高度作为通道保留，不新增独立BEV输出。

Qwen/Action/World三方交互不变；World仍只输出latent。**没有新增未来3D OCC解码，没有OccWAM第二阶段，没有正式训练。**

## 验证

`test_occ_qwen_native_flow.py --pretrained --device cuda` 两个真实train路线样本通过：原生current-frame转换精确相等、actor模式与label模式历史一致、完整前后向、六向跨流梯度、EMA、一次无标签采样。optimizer_steps=0。

额外只读合成检查通过：identity、0.4m横向位移、未知边界、unfilled/unknown/车辆映射。

报告 `artifacts/occ-qwen-native-smoke.json`：GPU allocated峰值3,571,567,104字节，reserved3,584,032,768字节（batch2有界验证，不含CARLA，不是训练吞吐或实时保证）。主干参数与可训练参数未变化。

## 继续工程核查：2026-11-16

- `configs/occ-qwen-native.json` 为独立配置，formal_training_paused=true。旧TRAINING_PAUSED.json只记录旧flow-v3授权，不能授权此模型。
- `occ_qwen_native_data.py` 提供train/validation Dataset和StatefulDataLoader，不提供正式Trainer.fit；正式训练入口继续暂停。
- `audit_occ_qwen_native_contract.py` 在51条train/validation路线各取1个中段窗口，51/51通过；5296训练窗、1124验证窗清单互斥、路线互斥。只对51个窗口做源文件内容检查，不能声称6420窗完整逐文件审计。
- loader保存状态后重建，恢复的下一批所有tensor逐值相同。尚未运行优化器/调度器/checkpoint训练恢复链路，不冒称正式续训已完成。
- 补充历史[-6,-4,-2,0]、未来标签[10,20]、finite state/pose/trajectory、规范路径与禁止test输入检查。
- 修复原生32×32特征图展平后1D池化成16条带的问题：现在先做二维4×4池化，再展平16 token，保留纵横两个空间索引；world监督support对应4×4。原来压缩会把整行横向信息混进同一token，不能仅用shape正确掩盖。`test_native_spatial_pool.py`验证两个X位置落入不同token。
- 修复后真实Qwen/GPU前后向、六向跨流梯度、EMA、无标签采样再次通过，结果为`artifacts/occ-qwen-native-smoke-spatial.json`；batch2峰allocated3.57GB/reserved3.58GB。没有优化器更新或权重训练。
- 数据审计报告`artifacts/occ-qwen-native-contract-audit.json`。source-level因果性仍未认证，World仍latent-only，没有新增OCC解码或scene-to-ego阶段。

## Trainer and complete resume integration: verified engineering test

Implemented `scripts/train_occ_qwen_native.py`: Lightning Trainer, trainable-only AdamW, per-step ExponentialLR, post-optimizer EMA, local TensorBoard, validation single-candidate ADE, rolling best/latest through existing VerifiedCheckpointIO. Full checkpoints include frozen backbone, EMA, optimizer, scheduler, loops/step, RNG, StatefulDataLoader and exact settings/manifest identity. Default CLI is preflight-only; --fit was tested and refused before model loading by the independent pause guard. Formal training remains paused.

`test_occ_qwen_native_resume.py` ran a bounded CPU FP32 test using random tiny Qwen and four real native train samples: continuous 4 updates compared with checkpoint at step 2 followed by 2 resumed updates. Six TEST updates total; no pretrained Qwen or production weights changed. Full model including EMA matched exactly (max difference 0), all AdamW tensors matched exactly, scheduler and final step matched. Encoder updated; frozen backbone did not. Simulated atomic-replacement failure and nonfinite checkpoint both preserved old valid bytes. Local CSV logs were written.

Report: `artifacts/occ-qwen-native-engineering-test/result.json`; test checkpoints approximately 1.89MB, explicitly test_only. Real pretrained GPU forward/backward was separately verified. Large-Qwen BF16 optimizer+checkpoint resume, long multi-epoch best/latest behavior, convergence and driving quality have NOT been tested. No future OCC decoder or scene-to-ego stage was added.

## 正式训练启动：2026-11-17

用户最新指示继续，不再停在验证。仅本实验解除暂停；正式run为`artifacts/occ-qwen-native-training`，batch8、20轮、5296训练窗/1124验证窗、warmup100步后cosine学习率（最低0.1倍）。旧逐步ExponentialLR会过快衰减，已在正式启动前换为按完整训练长度的调度。

真实预训练主干BF16 GPU已验证6次测试更新：连续到step4与step2恢复到step4模型/EMA差0；optimizer step精确，moment最大差5.44e-9，满足项目rtol1e-5/atol1e-8；scheduler一致。完整断点2,877,876,089字节。报告`artifacts/occ-qwen-native-pretrained-resume-test/result.json`。此测试头没有作为正式初始化，正式adapter/encoder按seed71新初始化，Qwen主干保留预训练且冻结。

正式进程于本地02:51启动。status.json每10步更新loss/速度/显存/ETA，完整验证结束记录ADE，best/latest原子滚动保存。ETA前期不含完整验证开销，不当作承诺。模型仍只输出未来latent与轨迹，没有未来OCC解码或scene-to-ego阶段。

## 2026-11-17 performance resume, step 1150
Native aligned cache built all 6420 windows. Full settings/manifest and 5296/1124 split ordering verified. No new decoder/OccWAM stage and no input resampling added.
CPU migration found TorchData shared sampler/base-seed RNG rewind across epoch on raw-to-raw as well. Independent ExactResumeDataLoader fixes this with actual generator state and strictly validated legacy reconstruction. Tiny Lightning model/optimizer/scheduler/EMA/RNG/batch sequence exact; integer 5296/batch8/seed71 tests cover batch2/650/662, end iterator, and repeated new-format restore across two subsequent epochs.
Protected full step1150 checkpoint at pre-optimization.ckpt; old wrapper37320 and actual27032 confirmed stopped before GPU tests. Few unsaved steps after checkpoint may be recomputed; exact count unavailable, not claimed zero repetition.
GPU channels_last loss difference0, gradient relative L2 2.158e-9: pass. Fused AdamW needed old moment-layout conversion; after that parameter max diff2.384e-7 exceeded predeclared1e-7 gate although states allclose. Fused excluded instead of relaxing threshold. No production weights changed by tests.
Optimized runner launched 03:25:54 +08 from same step1150, native cache+channels_last, workers0 and original AdamW. workers0-to2 loader state migration is incompatible and excluded. Await actual resumed-step and throughput evidence; no speedup claim yet.
Verified resumed global step/optimizer/scheduler all1150, then new saved checkpoint1550 contains full teacher, optimizer/scheduler1550 and exact loader generator. Second full validation ADE5.175456m, loss.350139, improved from first5.911079m. Atstep1530 measured latest100steps .425887s/step including periodic saves, full validation+epoch save102.703s. ETA about1h54 remaining at03:30 local, allow10-20min variation. Non-save intervals~.247s/step; do not present this alone as end-to-end speed. Training continues to20epochs in optimized-launcher PID31284; stdout/status in same artifact directory.
