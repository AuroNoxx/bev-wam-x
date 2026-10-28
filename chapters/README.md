# bev-wam-x 开发者入门课程

面向会基础 Python、尚未做过自动驾驶/世界模型项目的开发者。建议按 0→7 学习。
统一入口为仓库根目录的 `app.py`，每章也可单独打开。

```bash
./start.sh
# 查看并编辑 Marimo 单元格：
./start.sh --edit
# 单独学习某章：
.venv/bin/marimo edit chapters/chapter3_latent_vicreg/app.py
```

## 每章要解决的问题

| 章节 | 学完应能回答 | 可以动手改变什么 |
|---|---|---|
| 0 硬件与张量 | 为什么输入很小，训练仍占很多显存？ | Batch、Token 数、dtype、梯度累积 |
| 1 BEV 与坐标 | 米、像素和四个通道分别是什么意思？ | 目标位置/朝向、道路宽度 |
| 2 监督学习与轨迹 | ADE/FDE 怎么算，缺失标签怎样处理？ | 手工预测偏差/噪声、有效点数、种子 |
| 3 潜变量与 VICReg | 为什么预测全零也可能得到很低误差？ | 特征幅度、共享比例、预测噪声、方差权重、样本数 |
| 4 联合世界/轨迹 | 先验、学习残差、历史和未来怎样连接？ | 速度、角速度、手工残差、高亮未来时刻 |
| 5 感知退化 | 遮挡、边界缺失和丢帧分别丢掉什么信息？ | 遮挡宽深、边界缺失、连续丢帧与查看帧 |
| 6 方案与后果 | 同一个起点，给不同方案会发生什么？ | 32个真实场景、四种方案、时间进度、方案条件与错方案对照 |
| 7 生成与选择 | 多条候选怎样变成一次实际驾驶决定？ | 真实 Flow 积分、两组训练种子、真实记录逐步回放、同屏候选/评分/拒绝原因/执行 |

每章分成三个小实验，一次只解释一个问题。控件紧邻当前图示，拖动立即更新；
聚焦滑块后可按左右键逐步查看。切换章节保留参数与小节位置。
中文注释代码、深入解释和正式实测记录默认折叠，按需要展开。

## 文件怎么读

- `app.py`：各章独立的响应式入口，分开创建控件、展示控件和读取值。
- `ui.py`：`make_controls()` 创建 Marimo UI 字典；`compute(values)` 执行可复现实验；
  `render(values, controls)` 组织讲解与图表。图表不会在每次拖动时重建神经网络。
- `src/`：保留章节内原有模型、数据和训练实现。教学演示不会修改已验收权重。
- `chapters/learning.py`：只复用显示、字体、代码阅读与合成场景工具，不参与正式训练。

```python
# 创建 cell：UI 字典必须是 Marimo 可追踪的顶层变量。
controls = lesson.make_controls()

# 消费 cell：在另一个 cell 中明确读取 .value，参数变化才会触发重算。
content = lesson.render(controls.value, controls)
```

普通 Python 对象里的滑块属性不能代替全局 UI 依赖；
也不要在读取值的同一个 cell 里重新创建滑块，否则操作会重置。

## 示例和实验结果的边界

合成 BEV、手工轨迹、小矩阵与显存公式均用于解释，不是训练模型的成绩。
第 4 章用正式数学先验做演示，并另行读取带来源的阶段 4 指标文件。
第 5 章用输入退化演示解释信息损失，折叠区读取正式阶段 5 结果。
第 6 章读取正式阶段 6 的真实分支与冻结模型输出：路线是实际执行记录，
模型预测的是未来特征；绿色匹配框是离线核对，不代表模型自主选路或生成了未来图片。
第7章展示实际冻结模型的11个 Flow 积分状态及同一条实际 NPZ 的滚动规划记录。
全部候选被拒绝时的 STOP 明确标为规则刹停；没有执行的候选没有反事实真值，不填造未来图像或零风险。
指标报告与科学失败独立呈现，回放可用不代表生成或规划门槛通过。参见 [第7章说明](chapter7_candidate_planning/README.md)。

正式模型/数据/训练仍在根目录的 `models/`、`dataset/`、`training/`、`scripts/`。
章节模型是可修改的教学实现，**与正式检查点不保证兼容**，尤其不能将原阶段 4 权重直接加载到章节版模型。
旧章节训练入口位于 `src/`：

```bash
.venv/bin/python -m chapters.chapter2_baseline.src.train --help
.venv/bin/python -m chapters.chapter3_latent_vicreg.src.train --help
.venv/bin/python -m chapters.chapter4_joint_world_action.src.train --help
```

Chapter 5 的旧 `src/benchmark.py` 仍待接齐正式模型/测试集接口，
不能把它的输出直接用于阶段 5 验收。当前所有阶段状态以 `TASKS.md`、`AGENTS.md` 为准。

验证方法与本次改动记录见 `docs/LEARNING_WORKBENCH.md`。

