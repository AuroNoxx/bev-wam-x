# OCC-Qwen joint latent flow：隔离实现，2026-11-16

状态：代码与有界冒烟验证，不是训练完成或闭环可用的驾驶模型。旧 classical 入口、旧模型代码、数据和报告保留。本次用户最新选择优先于 11-15 的 classical-only 方向，但不切换默认驾驶入口、不启动正式训练。

## 本次实现

`history OCC [B,4,20,64,160] + state20(含导航)` → 逐帧语义 OCC CNN、时间融合、16 个场景 token → 可训练线性投影 → 冻结 Qwen3-VL-2B 的 text backbone。

每次 Flow 求速度场时，将场景 token、自车状态 token、25 个 Action token、2×16 个 World token 一起送入 Qwen 28 层。显式 4D 全零 additive attention mask 解除因果遮挡，三个 token 组每层双向交互。Qwen 参数冻结，但不使用 no_grad 切断输入梯度。它不是旧 v4 那种预先缓存、只单向提供特征的 Qwen。

- 不加载图像塔和词表 embedding，不生成文字。
- Action head 预测归一化残差速度的 Flow velocity，积分得到 25×0.2 秒、5 秒的自车 XY 轨迹，不是油门/刹车。
- World head 预测未来 1 秒、2 秒 OCC target encoder 的 latent Flow velocity。无 OCC 解码器、不输出未来体素。
- World target 是 OCC encoder 的 stop-gradient EMA 副本；仅 loss API 接受未来 OCC/轨迹。推理 sample API 只接受历史 OCC、state、随机噪声。
- 未来 unfilled/unknown 不标成已验证空闲，仅 occupied support 参与 world loss。该 latent 不等于碰撞概率或可靠风险评分。
- 暂用 16 个池化 token，含义尚待训练。OCC→Qwen projector 未训练，冻结 Qwen 不会自动理解这些输入。
- 一个共享 Qwen 主干和两个投影头，而非三个独立大模型或复刻官方 MoT 专家。全向注意力也改变了原始因果预训练用法，必须重新验证收益。

## 数据和修正

复用 `data/occ_v3` 当前 ego FLU 坐标、历史对齐、OCC文件比pose索引小1的修复。测试对 history/future OCC ID 与 pose ID 的关系作断言；只取 train 样本，不读取 test split。

这与 OccWAM scene-to-ego 第二阶段不是一回事。本实现没有 decode→按预测轨迹变换未来OCC→encode，也没有声称16个聚合token可直接空间warp。若未来做不解码OCC的细化，需要改成有规则XY坐标/valid mask的空间latent，并验证候选轨迹变换，不等价原论文。

## 文件与验证

- `scripts/occ_qwen_joint_flow.py`：独立模型、严格本地 text 权重加载、训练损失、EMA更新和无标签采样。
- `scripts/test_occ_qwen_joint_flow.py`：两个真实 train 样本，正反变换、前后向、冻结梯度检查、六条跨流梯度依赖、一次无标签采样。没有 optimizer step。
- `artifacts/occ-qwen-smoke-tiny.json`：小型随机 Qwen 结构测试结果。
- `artifacts/occ-qwen-smoke-pretrained.json`：若存在且 passed=true 才表示真实预训练主干测试通过。

运行小型结构测试：
```powershell
.venv-train\Scripts\python.exe scripts/test_occ_qwen_joint_flow.py --output artifacts/occ-qwen-smoke-tiny.json
```
运行有界真实主干测试：
```powershell
.venv-train\Scripts\python.exe scripts/test_occ_qwen_joint_flow.py --pretrained --device cuda --output artifacts/occ-qwen-smoke-pretrained.json
```

无正式训练命令、无自动接CARLA。下一步正式训练需另获指令，并接入项目原 Lightning/VerifiedCheckpointIO 恢复与 best/latest 策略。当前不能由冒烟测试推导驾驶效果、实时性、latent不坍缩或场景理解能力。

监督限制：occupied-support mask 不代表 latent 仅依赖该局部已占据格，GroupNorm 的空间统计仍受其他格的 unknown/unfilled 编码影响。

## 已完成验证

最终版 tiny CPU 与真实本地 Qwen3-VL-2B text backbone / RX7900XT GPU 均通过，报告已保存。真实主干测试包含两个 train 样本的一次 loss.backward、六向跨流梯度、EMA 更新数值断言、无未来标签的一步 Flow 采样。optimizer_steps=0，无权重训练或保存。

真实配置总参数 1,414,366,950，可训练参数 4,879,764；torch 峰值 allocated 3,467,963,904 字节。该值不包括完整进程/驱动占用，不是 CARLA 并发峰值。报告 seconds 是整组测试时间，不是单步推理延迟。未进行完整多步采样测速、闭环或驾驶质量测试。

首次工具受限运行发生 hipErrorInvalidImage；正常用户环境执行成功。未修改驱动或任何全局设置。

最新输入更新：以 OCC_QWEN_NATIVE_INPUT_20261116.md 为准。新入口 occ_qwen_native_flow.py，0.4m/20×128×128原生OCC；本文原20×64×160输入描述和对应资源数字仅为历史记录。
