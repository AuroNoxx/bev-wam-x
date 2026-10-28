# Evidence index

这些文件是已有运行记录的逐字节副本，不是本次重新训练或重新评估的结果。
来源、字节数和 SHA-256 见 [manifest.json](manifest.json)。v1 来源为旧 Git 快照 `58fedae`，v2 来源为本地既有 artifacts。

- [v1 技术与结果说明](../V1.md)
- [v2 技术与结果说明](../V2.md)
- [复现与当前测试限制](../REPRODUCIBILITY.md)

原始记录中可能包含旧目录、旧项目名、历史机器路径、已撤回的成功结论或指向未发布大文件的链接；这些字段为溯源而保留。当前解释以两份版本说明为准。特别是 v2 Stage 7 的原报告正文与其勘误相冲突，不能引用正文中的“全部通过”。

GitHub 只发布小型报告和 JSON。数据集、checkpoint、视频、Rerun、TensorBoard 和 `.tar.zst` 归档保留在本地，未随本次提交发布。
