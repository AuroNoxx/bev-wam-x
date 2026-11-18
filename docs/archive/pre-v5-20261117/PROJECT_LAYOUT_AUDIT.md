# 项目目录审计（2026-11-14）

## Python不是三份重复程序

runtime/python/cpython-3.10-windows-x86_64-none 是 Junction，指向同目录的 cpython-3.10.22-windows-x86_64-none；旧 .venv/pyvenv.cfg 正在引用此别名。真实3.10运行时约57MiB。

runtime/python-train/cpython-3.13 是真实3.13.17运行时，约63MiB，.venv-train 与 .venv-carla-ue5 共用它。不能将目录链接重复计入物理空间。

## 占用与处理

| 目录 | 逻辑文件大小GiB | 用途/处理 |
|---|---:|---|
| runtime/CARLA-0.9.15 |17.925|旧模拟器，UE5验证前保留|
| .cache/downloads |7.223|旧CARLA压缩安装包，可清理；本轮删除被审批阻止|
| .venv-train |4.012|实际训练依赖，含PyTorch/ROCm；TensorBoard和下载正在使用|
| .venv |0.323|旧CARLA0.9.15客户端，升级验证前保留|
| .venv-carla-ue5 |0.040|新版客户端隔离环境，尚待安装包内carla wheel|
| data/public |1.848|公开原始数据及解压文件，保留|
| data/raw |1.585|旧自采数据，已停用；此前删除被审批阻止|
| data/benchmark_cache |0.193|旧测速缓存，含RGB；新管线不应继续使用|
| artifacts/benchmark |0.361|旧试验权重和报告；报告保留，旧视觉权重待清理|
| .cache/torch |0.044|旧视觉分支ResNet权重，待清理|
| .cache/uv |0.249|包安装缓存，可能有硬链接，逻辑大小不等于可释放空间|

.cache/carla-ue5 是用户正在下载的分段和残留，不在下载期间清理。包校验、解压验证后再删除不再引用的分段和安装包。

2026-11-14尝试删除 .cache/downloads/CARLA_0.9.15.zip 被自动审批以 blocked by policy 拒绝，命令未执行，无文件删除或空间释放。未改用其他机制绕过。

后续收敛目标：新CARLA验证成功后再移除旧版模拟器、旧3.10环境/运行时及采集分支；保留一份3.13运行时、训练环境和新版模拟器客户端环境。环境分开是为避免CARLA与ROCm依赖冲突，不是每个脚本建立一个环境。
