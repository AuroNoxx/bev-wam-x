# 长短历史小型数据契约验证

新增独立目录 `artifacts/causal-history-contract-mini`，16个train窗口：8个中段窗口、8个片段开头窗口。未覆盖现缓存，未读验证/测试，未训练模型。

输入历史：−5/−4/−3秒稀疏帧，加−2至0秒每0.2秒的11帧，共14时刻。只读取当前及过去位姿/原始OCC，依既有annotation i / OCC i−1配对，对齐至当前ego。每条actor文件仅含history、state、history_valid、age、geometry_valid、occupied_observed和历史索引/当前变换；未来OCC及专家轨迹写到单独labels文件。

实测16窗口共81个历史位置不可用，均填unknown255，history_valid=false、geometry_valid=false；不借未来或其他段补齐。全部16窗口末4帧与原occ_v3缓存逐像素一致。检查的历史自车相邻位姿最大位移0.808m/名义0.1秒，无>=8m跳变。边界遇缺失annotation/配对OCC或大跳变即停止跨越，当前含段首缺失实例。

这是数据存储/加载契约验证，不是新actor训练通过：已有actor尚未接入history_valid/age，在接入与因果测试完成之前不会拿这些padding序列训练。actor文件键不含future/trajectory/future_mask，标签被隔离；此前完整窗口actor已做过未来标签替换不变输出测试，但本轮没有谎称padding版本网络已验证。

geometry_valid仅是历史网格重投影落入源范围，occupied_observed是1..7占据类；0仍为未填充，不当free。二者不能互相替代。缺帧/位姿阈值只是初步保护，不能证明对象身份重用、所有teleport或源级未来融合已排除。名义索引时间不是原生timestamp。

脚本 `scripts/build_causal_history_contract_mini.py`，结果 `artifacts/causal-history-contract-mini/manifest.json`，各actor文件有SHA256。正式下一步应先接入显式mask/age、增加合成缺口/跳变负例和流式状态重置等价测试，再考虑小步拟合，不能直接上长训。
