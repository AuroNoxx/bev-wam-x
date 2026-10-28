# bev-wam-x 复现与环境边界

## 本次检查

2026-10-27 在迁移后的 checkout 使用现有 `.venv` 执行 `python -m pytest -q`：**97 passed / 5 failed**。这是源码测试，不是模型重新训练或驾驶验收。

失败项：

1. `tests/test_bev_wa_dataset.py` 的 3 项真实数据集测试：manifest 仍指向 `/home/auronox/Desktop/mini-wa-jepa/.../episode.npz`，当前路径不存在。
2. `tests/test_semantic_bev.py` 的四通道测试及 `tests/test_stage2_v2.py` 的路缘测试：当前共享解释器无法导入 `metadrive`。

测试未改成跳过，也未修改历史清单、数据或权重以制造通过。发布验证记录会保留最终执行命令与摘要。

## 源码阅读与不依赖数据的回归

```bash
git clone https://github.com/AuroNoxx/bev-wam-x.git
cd bev-wam-x
# 已准备好 PyTorch 等依赖后
python -m pytest -q tests/test_bev_wa_metrics.py tests/test_bev_wa_joint_flow.py tests/test_bev_wa_masked_completion.py tests/test_trajectory_tracker_dt.py
```

这些命令不启动训练。完整测试命令为 `python -m pytest -q`，真实数据测试需先恢复数据和 MetaDrive 环境。原 `unittest discover` 无法覆盖全部 pytest 风格测试。

## 环境准备注意点

- 历史主环境为 Linux、Python 3.12、PyTorch ROCm；ROCm PyTorch 中 `cuda` 设备 API 名称不代表用了 NVIDIA CUDA。
- `requirements-runtime.txt`、`requirements-training.txt`、`requirements-visualization.txt` 是已有环境冻结记录，不承诺所有平台或当前公共索引都可直接安装。
- `scripts/bootstrap.sh` 依赖作者本机 `/home/auronox/.local/share/ai-rocm/shared-env.sh` 与共享 ROCm venv；外部使用者应先提供自己的环境，不应原样运行这一本机启动脚本。
- MetaDrive 历史固定源码 revision 为 `fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1`（见 bootstrap）；其源码、资产与 EGL/Panda3D 支持需要另行准备。
- 已有 `.venv` 是机器环境链接，不进入 Git；此次不重装/替换共享 ROCm 环境。
- `start.sh` 启动历史教学工作台，`marimo.sh` 打开实验看板；页面能打开不等于数据、模型或历史教学入口均可运行。

## 数据与权重

本仓库不发布原始数据集、NPZ、checkpoint、视频、Rerun 文件及 TensorBoard 日志。仅有代码和小型结果文件，无法仅靠 clone 重现训练数字。

恢复时先核对文件 SHA-256、split/seed/时间轴和控制资格；如需重定位 manifest，应保存原件，生成独立的新清单与路径映射，不能原地改写带 provenance 的历史证据。不要为了迁移路径重生成或覆盖原数据。

Stage 4 开环复评脚本的可检查入口：

```bash
python scripts/reevaluate_bev_wa_stage4_open_loop.py --help
# 数据、环境及原检查点齐备后，以新的输出目录执行：
python scripts/reevaluate_bev_wa_stage4_open_loop.py \
  --manifest /absolute/path/to/relocated-manifest.json \
  --checkpoint-dir /absolute/path/to/stage4_remediated \
  --output-dir artifacts/local_open_loop_review_new \
  --arms B D --seeds 20261022 20261023
```

该脚本只评估既有权重；输出目录不得已存在。运行任何新训练或 Stage 8/9 前仍需按项目协议独立授权，本次命名/文档整理不改变阶段门槛。

## v1 恢复

旧完整快照仍在 Git 历史，可用单独目录检出：

```bash
git worktree add --detach ../bev-wam-x-v1 58fedae458021bb37212442868288f95947288a8
```

该命令恢复源码快照，不恢复未发布的实验大文件。`archive/v1_legacy_20261024/code/` 是后续整理的部分模块归档，不能当作完整环境直接运行。保留在本地的 `.tar.zst` 应先列出内容，再恢复到独立目录，避免覆盖 v2。
