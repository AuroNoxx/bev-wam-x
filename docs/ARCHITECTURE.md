# 当前架构：v5-qwen-drive-world

2026-11-17 用户最新指令取代 native v1/v2 和 classical-only 路线。

当前授权主线：原生尺寸RGB（当前全部已有视角、历史前视）+ 连续自车状态 + FLU导航 → 官方ViT与额外2×2空间池化（总序列≤850）→ 完整32层VLM（前24层冻结无重算，后8层PEFT LoRA r16）→ 官方planner-rl控制Flow + 纯2D动作条件BEV World → World Critic → 5Hz规划 / 20Hz控制。REASON/PLAN、EMA与真实SFT链路已接入并完成首次更新；两步训练、完整恢复及质量结果单列在[V5_TWO_STAGE_SFT.md](V5_TWO_STAGE_SFT.md)。GRPO与闭环尚未完成。

World当前使用官方perception可迁移的2D投影/归一化/BEV query权重，新增cross-attention与四级Conv2d上采样；不再训练旧Conv3d分支。三相机工程诊断默认微批4/累积2；真实六视角＋EMA教师SFT入口采用微批1/累积8，有效batch同为8。新解码器没有预训练占用能力，工程诊断耗时不能直接外推真实SFT。

完整规范：[QWEN_DRIVE_HIGHWAY_120_ARCHITECTURE.md](QWEN_DRIVE_HIGHWAY_120_ARCHITECTURE.md)。实施程度：[V5_IMPLEMENTATION_STATUS.md](V5_IMPLEMENTATION_STATUS.md)。附件原文：[V5_USER_SPEC_RAW.txt](V5_USER_SPEC_RAW.txt)。历史：[archive/pre-v5-20261117](archive/pre-v5-20261117/)。

世界模型现已授权预测 +1/+2/+3s 动作条件占用，旧“latent only”限制被本次指令取代。输入不用旧 OCC-CNN/4×4 token。保留原始数据，不把旧 ±25.6m OCC 扩画布当作远距监督。
