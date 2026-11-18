# CARLA UE5 升级验证结果

2026-11-14：**CARLA 1.0 在本机 Windows 11 + RX7900XT 上通过基础运行验证**。尚未完成训练OCC等价性、稠密交通和学习策略闭环验收。

## 安装与版本

- 官方来源：https://downloads.carlasim.com/Windows/CARLA_1.0.zip
- ZIP大小13917765246字节，7-Zip完整测试与解压均为Everything is Ok（exit 0）。解压15832807011字节。
- 本地SHA256：E3BAA119E8BBF6CC7F15076CC9C9059447CB1BFBBD59634D1179469308D1CC49；这是本地完整性摘要，不冒充官方签名。
- 安装：runtime/CARLA-1.0/Carla-1.0-Win64-Shipping。
- 客户端：.venv-carla-ue5，复用现有项目Python3.13.17，包内cp313 wheel。没有新增Python版本，没有改训练环境/驱动。
- 服务端/客户端均为1.0-a1770de89。地图列表Town10HD_Opt、Town12、Town15、EmptyMap。
- 官方当前UE5文档指向1.0/UE5.8；通用GitHub latest仍曾指向0.9.16，升级不依赖该重定向。

## 实测

- DX12低画质GUI成功显示Town10和Town12。
- Town12地图有10422个spawn points，136650个5米间隔道路waypoints（含113920个非路口点）。这不是所有道路都适合高速的证明。
- Dodge Charger车辆，4轮物理参数可读取，执行短时油门/刹车/转向后转为CARLA自带Traffic Manager驾驶。
- 同步步长0.05秒，2400步，120秒仿真约120秒墙钟，约20Hz。
- RGB、语义相机、语义LiDAR各收到600次回调；图像480×270、5Hz，LiDAR16线/60米。RGB仅验证传感器，不进入本项目训练。
- 最高采样速度约115.7km/h，仅为模拟器车辆运动检查，不是学习模型高速能力。
- 系统可用内存最低14.40GiB；稳定阶段6次采样的专用GPU内存约6.05GiB，进程工作集约4.65–4.72GiB。不是全过程峰值；加载阶段一次采样约6.90GiB专用GPU内存。
- 静态道路/车辆包围盒API可调用；仅证明接口可用，不证明完整稠密OCC已生成。

机器可运行这版UE5的基础场景，但高密度交通、OCC在线转换和模型并行推理资源仍需测。未运行训练，没有修改全局设置。测试结束销毁探测车辆/传感器并恢复world settings；GUI服务器保留供查看。

## 已修复的验证脚本问题

1. UE5 CityObjectLabel无Vehicles分组，使用Car/Truck/Bus/Motorcycle/Bicycle分类。旧标注映射需要适配，不能照搬。
2. 2102是模拟器secondary端口，不可分配给Traffic Manager；TM独立使用8100。

初始错误记录在logs/carla-ue5/initial-api-mismatch.json与initial-tm-port-conflict.json；最终报告smoke-report.json，时序smoke-steps.jsonl，资源采样resource-samples.json。

## 启停与复现

```powershell
.\scripts\Start-CarlaUE5.ps1
.\.venv-carla-ue5\Scripts\python.exe -u scripts\validate_carla_ue5.py --seconds 120
.\scripts\Stop-CarlaUE5.ps1
```

Start脚本默认启动包内默认地图；本次先通过API load_world('Town12')再测试。客户端RPC2100，流2101，secondary2102，Traffic Manager8100。依赖锁见configs/requirements-carla-ue5-lock.txt。

旧0.9.15/环境暂未自动删除。公开数据支持离线训练，不保证跨引擎类别、几何、可见性及物理响应相同。接下来实现OCC规范适配、控制器校准与固定架构闭环。

## 来源

- https://carla-ue5.readthedocs.io/en/latest/start_quickstart/
- https://carla-ue5.readthedocs.io/en/latest/build_carla/
- https://github.com/carla-simulator/carla/releases/tag/1.0
