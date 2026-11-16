# 一帧偏移专项复核

2026-11-15；接续 CONTRACT_AUDIT_20261115.md。本轮只检查时间配对，未恢复训练。

**结论更新：在抽查的训练/验证数据中，已通过原始语义LiDAR点指纹确认：标注 i 与原始 lidar_seg i 同时，而对应 OCC 文件是 i-1。当前按同名整数交集读取的实现发生一帧错配。** 这是对当前抽样文件的实测结论，不是对全部Bench2Drive发布版本的保证。生产采集代码中减一的具体保存语句仍未找到，不归责某个上游代码行。

高速自动驾驶包含低速拥堵、汇入、切入和急刹等过程，这些场景仍然适用。车速支持上限限制可宣称的验证范围，不能据此否定场景或整批数据；本轮不再据车速提出换数据集。

方法：从本地原始归档只读提取 lidar_seg NPZ（xyz/obj_tag/obj_idx），取上一轮固定108个验证时刻，另选6类各1条训练路线×12时刻=72个。没有读取测试集。源点先在原生0.4m栅格体素化，对每个体素-类别组合去重，分别与OCC i-2…i+2比较是否占据且类别相同；排除ego actor，不用学习模型或未来轨迹。另用原始点的actor ID与标注车辆ID精确对应，通过完整world2vehicle变换、bbx_loc/extent和0.1m容差，验证同名标注的匹配。动态点与静态pole/sign分开报告，避免地面大面积占据主导结果。

| 检查（每时刻比例再平均） | 验证集 | 训练抽样 |
|---|---:|---:|
| 原始车辆点 i → OCC i-1 | 99.6436%（105有效时刻） | 99.7679%（65有效时刻） |
| 原始车辆点 i → OCC i | 73.8587% | 74.9395% |
| 静态杆/牌 i → OCC i-1 | 98.9673%（93有效时刻） | 92.1882%（72有效时刻） |
| 静态杆/牌 i → OCC i | 19.0100% | 18.9377% |
| 原始车辆点 i → 标注 i | 99.9943%（105有效时刻） | 99.9793%（65有效时刻） |

有效时刻表示该帧筛选后确有相应目标，不用无目标帧充当满分。每时刻和点/体素加权两种汇总及分路线记录均见 summary.json。原始点与OCC可能共享生成来源，因此它们适合做时间“指纹”，不适合作为独立感知精度评测。静态历史融合、体素边界与填充会导致不是100%；但是车辆/稀疏静态结构均强烈选择-1，同时原始点强烈选择同名标注，已经远强于仅根据文件从-1开始猜测。

具体修正方向：

| 对齐到标注100时刻 | 正确OCC文件 | 变换使用的标注位姿 |
|---|---:|---:|
| 历史 -0.6s | 93 | 94 |
| 历史 -0.4s | 95 | 96 |
| 历史 -0.2s | 97 | 98 |
| 当前 | 99 | 100 |
| 未来监督 +1s | 109 | 110 |
| 未来监督 +5s | 149 | 150 |

也就是：`occ_file = annotation_frame - 1`；位姿、状态、导航、未来轨迹的索引不减一。不能反向错改成OCC+1，也不能把所有模态一起平移。当前旧实现等效让OCC内容相对状态超前一帧，并用错误时刻位姿warp历史/未来OCC；按10Hz名义节拍约0.1s。绝对simulator timestamp仍未保存在这些NPZ中，因此“0.1s”来自采集频率，不是逐文件测得时间。

修正小测试：每条验证路线取一个现有窗口，共9个，输出到独立 corrected-smoke 目录。只重建历史/未来OCC，记录history_occ_ids、future_occ_ids、occ_file_offset=-1、diagnostic_only=True、preprocess_version=3。重新加载检查shape，并逐数组确认state/trajectory/frame/time_s/world2ego完全不变，正式原缓存SHA256也不变。详见 correction-smoke.json。这里只是旁路验证，正式代码、manifest和缓存仍为旧版，不能误称生产修复完成。

后续正式修复应限定为一个因素：建立独立occ_v3缓存和manifest，保留原训练/验证窗口列表以便配对比较，不趁机扩大网格/改目标/新增边界窗口；正确配对历史和未来OCC，记录每个源文件与位姿ID。先冻结权重做重配对前后诊断，但旧模型在错配分布上训练过，修正后指标不一定立刻改善，不能据此否定配对证据。随后再开独立训练run，不能使用旧断点“精确续训”跨越数据契约变化；若采用旧权重初始化，必须明确记录为新实验。测试集仍保留，不能拿它选偏移。

保护检查：原50个checkpoint/评估JSON哈希复核一致，训练暂停标记仍为true。新原始点只提取到本项目诊断目录；未上传数据或修改环境。两个原始点核查分别21.86s和15.19s；旁路窗口构造约2s。

复现：
```powershell
.\.venv-train\Scripts\python.exe scripts/probe_raw_lidar_alignment.py
.\.venv-train\Scripts\python.exe scripts/probe_raw_lidar_alignment.py --train-probe
.\.venv-train\Scripts\python.exe scripts/build_alignment_smoke.py
```

公开代码额外核查：[固定revision可视化loader](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/B2DVisualize/core/loader.py)同样直接拼接同名路径，不能作为正确同步的证明；[采集器](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/tools/data_collect.py)不包含该版OCC保存调用，精确生产代码来源仍待补。本轮结论以原始文件实测为依据。
