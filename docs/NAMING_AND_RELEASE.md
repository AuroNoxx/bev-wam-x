# 命名与发布范围

- 项目展示名、GitHub 仓库名：`bev-wam-x`。
- GitHub：`https://github.com/AuroNoxx/bev-wam-x`；核对 API 后发现原 `Ant1mage/mini-wa-jepa` 已重定向至 AuroNoxx 账户。默认分支统一为 `main`，保留旧 `master` 历史。
- 本机规范目录：`/mnt/ai/projects/bev-wam-x`；`/mnt/ai/projects/mini-wa-jepa` 是指向新目录的兼容符号链接，不复制大文件。
- `mini_wa_jepa` Python import、历史实验目录、冻结协议和 checkpoint 字段保留，避免破坏已有对象路径与溯源。新的文档和 UI 使用 `bev-wam-x`。
- v1/v2 为架构代际；`BEV_WA_ALIGNED_V1` 是历史协议版本号，`stage3_masked`/Stage 3 v3 是实验修订号，均不批量篡改。
- 用户授权本次同步现有 v1 归档、v2 源码与文档；不发布数据集、权重、大体积运行产物。新证据副本集中在 `docs/evidence/`。
- 已被本地归档移出的旧 tracked 文件从当前树整理移除，历史提交仍保留；本次没有删除本地数据和大型归档，也没有重写远端历史。
- 保存了整理前分支 `backup/pre-bev-wam-x-20261027` 及工作区 diff 备份。
