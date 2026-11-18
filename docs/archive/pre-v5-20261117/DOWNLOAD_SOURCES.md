# 下载来源与实测速率

2026-11-15，在本机保持现有网络/代理路线不变。测速时暂停本项目下载进程，结束后已恢复；没有改系统代理。

## ModelScope 是否有

公开搜索 API `/api/v1/dolphin/datasets?Query=Bench2Drive` 返回社区镜像 [thomas/Bench2Drive](https://modelscope.cn/datasets/thomas/Bench2Drive) 和一个评测复现仓库。对 `Bench2Drive-V0.0.4`、`Bench2Drive V0.0.4`、`Bench2Drive OCC` 三种查询均返回 0 项。

镜像清单共 1002 项（含说明/属性文件），与当前选择的 60 个 v0.0.4 路线包没有同名交集。镜像测试包的大小和 LFS SHA256 与 HF 原版 `rethinklab/Bench2Drive` 相同，因此确认是原版数据的镜像，不能视为 v0.0.4 替代。

[上游 v0.0.4 更新说明](https://github.com/Thinklab-SJTU/Bench2Drive/blob/0.0.4/docs/v004_update.md) 指明该版新增 3D OCC。本项目继续使用 [rethinklab/Bench2Drive-V0.0.4](https://huggingface.co/datasets/rethinklab/Bench2Drive-V0.0.4)，固定 revision `d84ae18cf20a88b9220df1bc15c073d8824b7ac1`。以上为本次搜索结果，不宣称所有未被搜索索引的镜像都不存在。

## 三源同文件对比（原版，仅测速）

文件 `HighwayCutIn_Town06_Route298_Weather20.tar.gz`，151733498 字节，LFS SHA256 `53dfc41cf3a8cb052bf708344a4e5b7f80ee6ca26f3773acb8eb75c0313d4a54`。

第一轮读取字节 0–8388607，第二轮读取 8388608–16777215。顺序 MS→Mirror→HF、HF→Mirror→MS。每次均返回 HTTP 206 和完整 8 MiB；同轮三份内容 SHA256 完全一致。

| 来源 | 第一轮 MiB/s | 第二轮 MiB/s | 总字节/总耗时 MiB/s |
|---|---:|---:|---:|
| ModelScope | 4.62 | 10.00 | 6.32 |
| HF Mirror | 0.88 | 1.60 | 1.13 |
| Hugging Face 官方 | 0.92 | 1.19 | 1.04 |

速度包含 DNS/连接/重定向/首字节等待，不是带宽上限。小范围测速会受缓存与网络波动影响。MS 实际下载主机为 `cdn-lfs-cn-1.modelscope.cn`，Mirror 为 `cas-bridge.xethub.hf.co`，HF 为 `us.aws.cdn.hf.co`；Mirror 并非本次全程国内 CDN。

## 当前 v0.0.4 同文件对比

文件 `HighwayCutIn_Town13_Route23830_Weather9_02-07-17-47-17.tar.gz`，同样两轮各 8 MiB、反序测试；四个片段均与本地已完成文件相应范围校验一致。

| 来源 | 第一轮 MiB/s | 第二轮 MiB/s | 总字节/总耗时 MiB/s |
|---|---:|---:|---:|
| HF Mirror | 1.25 | 1.41 | 1.33 |
| Hugging Face 官方 | 1.09 | 1.36 | 1.21 |
| ModelScope | 未找到同版文件 | — | 无法比较 |

保留当前 Mirror 优先、官方源失败回退。不会为了下载更快而换成缺少所需 OCC 的旧版数据。测速片段留在日志目录，不纳入训练数据。

原始记录：`logs/v1/source-speed/results.json`、`all-sources-original.json`、`modelscope-search.json`。

## 用户要求追加：官方源持续传输与代理路径

随后在暂停后台下载的条件下，对同一个 v0.0.4 文件的相同 32 MiB 范围（16777216–50331647）分别测试一次。两个来源均 HTTP 206、返回 33554432 字节，与已完成的本地文件对应片段 SHA256 一致。

| 来源 | 总耗时 | 含首字节等待平均速度 | 首字节等待 |
|---|---:|---:|---:|
| Hugging Face 官方 | 13.664 秒 | 2.34 MiB/s（2.46 MB/s） | 3.540 秒 |
| HF Mirror | 15.957 秒 | 2.01 MiB/s（2.10 MB/s） | 2.884 秒 |

此轮官方比 Mirror 快约 17%，与前面的短片段排序不同，不能用单次测试断言长期固定快慢。原始数据分别见 `official-32MiB.json` 和 `mirror-32MiB.json`。

实时只读核对：环境没有 HTTP_PROXY/HTTPS_PROXY/ALL_PROXY，Windows 系统代理 ProxyEnable=0，但 FLClash 配置的 TUN 启用。两次 curl 的本地地址均为 198.18.0.1，目标为 198.18.2.x Fake-IP，证实经过 TUN。实际 CDN 分别是 `us.aws.cdn.hf.co` 与 `cas-bridge.xethub.hf.co`；当前配置 `DOMAIN-SUFFIX,hf.co` 和 `DOMAIN-SUFFIX,huggingface.co` 均指向 AI 代理组。因此不能把 Mirror 这次下载解释为全程国内直连；未查询到实时连接链，未宣称具体出口节点/IP。curl 的 proxy_used=0 只说明没有显式 curl 代理，不代表绕过 TUN。

没有改代理、节点或 IP。测速结束后后台下载进程均恢复。本轮网络诊断没有改变数据版本或在途任务的来源。

## 下载完成记录

2026-11-15 00:50 左右，60 个路线包全部下载完成，总计 19426372751 字节（19.43 GB / 18.09 GiB）。持续观测的多连接总速率通常约 6.5 MB/s，因此不能用单次短片段速度直接推算整批数据耗时。

其中 `SequentialLaneChange_Town05_Route24339_Weather7_02-12-00-10-52.tar.gz` 为 2434592295 字节。该包 Mirror 持续约 1.1 MB/s，于已完成 1026678784 字节时，使用下载器已有的 HF 官方回退及 `curl -C -` 继续下载，保留在途部分。完成后完整 SHA256 为 `61ef7a0a8254abacd8538208218038a16d154d625708100d966cb044c5da5c9e`，与固定 revision 的官方 LFS SHA256 完全一致。其余包未因本次切换重启。

下载清单及每包摘要：`data/occ_v1/downloads.json`；核验记录：`artifacts/v1/download-complete.json`。下载完成不等于预处理或训练完成，训练实际状态仍以启动器和训练日志为准。
