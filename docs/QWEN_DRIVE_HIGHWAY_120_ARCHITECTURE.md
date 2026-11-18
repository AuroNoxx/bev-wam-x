# v5：原生 Qwen-Drive + 动作条件 World，0–120 km/h

决策日期：2026-11-17。本文是用户授权目标与实施契约，不是已通过的驾驶能力声明。完整补充原文保存在 [V5_USER_SPEC_RAW.txt](V5_USER_SPEC_RAW.txt)，实施证据单列于 [V5_IMPLEMENTATION_STATUS.md](V5_IMPLEMENTATION_STATUS.md)。

本日最新性能改造授权取代“零额外视觉token合并”及“完整3D perception训练路径”：保留原生像素、原生ViT和全32层VLM；允许ViT输出2×2邻域池化、历史仅前视、当前保留全部已有相机、总token≤850；World改为纯2D解码器。它是有信息压缩的后训练改造，不宣称与官方原始多模态输入完全等价。

## 1. 隔离、硬件和框架

新主线 `v5-qwen-drive-world` 不修改旧任务源码、配置与正在使用的文件。用户进一步确认删除已完成旧运行权重，保留数据、日志与结果。清理清单在 artifacts/v5-qwen-drive-world。旧文档归档到 archive/pre-v5-20261117。

硬件为 Ryzen 7 7800X3D、RX 7900 XT 20GB、ROCm gfx1100。所有环境在项目内。使用 PyTorch Lightning、PEFT LoRA、TorchData StatefulDataLoader、本地 TensorBoard。训练不使用 INT4/NF4，不租卡、不上传、不调用付费 API。

训练精度为 BF16 autocast + FP32 可训练参数/优化器状态；冻结预训练参数可驻留 BF16。标准 Transformer attention 强制 SDPA MATH，包含 gradient checkpointing 重算；Flow ODE 与运动学积分显式 FP32。优先已验证的 PyTorch/ROCm 加速算子，不能以不兼容 CUDA 扩展或断梯度换速度。用户在重启后授权单卡微批=1、梯度累积=8、有效 batch=8；三个数同时记录，累积更新不等于物理 batch8 通过。

当前保留完整32层以提供planner所需8组真实KV，冻结layers[0:24]并关闭这些层checkpoint，只训练layers[24:32]的注意力投影LoRA并对后8层checkpoint。末8层含2个标准attention层和6个原生线性attention层，分别挂q/k/v/o和in_proj_qkv/out_proj适配器。前24层输出实测不需要梯度。截断为24层仅有6组KV，不是当前方案。

最新授权允许实测微批2/4、对应累积4/2，有效batch始终8。默认选择通过的最快档4×2；这只通过两次工程更新，不是长时间无OOM承诺。训练轮数初始上限5，报告3/5轮成本，不能由样本量断言3–5轮必然收敛。

仅 CARLA 同机闭环评测允许 VLM torchao W8A16；planner 与 World 保持 BF16/FP32。CARLA 的约7GB占用只是估计，真实峰值和推理延迟须实测。

## 2. 四层架构

1. **原生像素输入与授权token压缩**：当前帧保留全部已有相机，历史保留前视；每帧使用官方patchifier与原target_size。原生ViT合并后，再按每张图实际空间网格做2×2平均池化，奇数边缘保留；重建image token数量及MROPE位置，ViT仍接原patch网格。若文本加视觉超过850，再显式自适应平均池化分配视觉预算，不裁掉当前视角、不截断导航。8张896×512当前图仅额外2×2池化就有896视觉tokens，不能声称固定2×2必然满足850。物理FLU导航/状态、时间标签保持；160m、FOV与全速度域仍是采集/能力目标。
2. **因果基座**：官方 Qwen-Drive-1.0-4B，共享场景 H_scene；因果顺序为历史输入 → REASON → PLAN。REASON监督12类元动作，PLAN提供规划条件。新增标记是后训练扩展，不宣称是官方已有特殊token。未来标签不能写入prompt或KV。Qwen3.5包含原生线性注意力；它不是SDPA，需单独验证因果梯度。
3. **并行输出**：官方 planner-rl DiT/AdaLN专家扩展为控制量 Flow；官方 QwenDrivePerception 扩展为动作条件未来 World。两者共享原生视觉/语言信息，不能另造占用输入编码器。
4. **World Critic和执行**：训练用World占用/动力学/车道代价；推理5Hz采样K=4，选择符合碰撞阈值的最高奖励轨迹，20Hz Stanley或Pure Pursuit+纵向速度/加速度前馈PID追踪。不得加旧2m/s限速或1.2m横移锁，不增加额外守卫网络。

## 3. 控制量 Flow 和积分

保留原生50步×0.1s=5s（另可配置25步×0.2s，但不能混淆两者dt）。控制归一化 u_x=(a_x+2)/6、u_y=a_y/3.5；物理范围 a_x∈[-8,4]m/s²、a_y∈[-3.5,3.5]m/s²。单一模型连续覆盖全速域。

从真实v0和x=y=θ=0开始，在FP32递推：

```
v_next = clamp(v + ax*dt, 0, 35)
kappa = ay / max(v*v, 5*5)
theta_next = theta + v*kappa*dt
x_next = x + v_next*cos(theta_next)*dt
y_next = y + v_next*sin(theta_next)*dt
```

Flow采样用10步Euler，积分状态FP32，网络BF16 autocast。输出平滑性通过jerk损失学习；clamp本身不保证jerk小，也不保证真实轮胎不侧滑。

官方规划器原来学习的是归一化(x,y,heading)的**clean endpoint**，不是2维控制速度场。必须保留并加载官方预训练中间层，明确扩展输入/输出头；不能把原3维坐标权重改名后称为已预训练加速度模型。现成sample有no_grad，只能用于推理，不能直接用于GRPO反传。

## 4. 动作条件 World

输入H_scene、当前自车运动先验和候选轨迹τ_k；残差预测 Z_future=stopgrad(Z_now)+ΔZ。解码+1/+2/+3s动态占用概率。

统一当前固定FLU（x前、y左、z上），网格张量[B,time,Y,X]，X128格[-16,112)m、dx=1m；Y128格[-19.2,19.2)m、dy=0.3m。像素中心x=-16+(j+0.5)、y=-19.2+(i+0.5)*0.3。grid_sample按(x,y)归一化，align_corners=False。未来标签必须用真实位姿变换到当前坐标，附有效覆盖mask，unknown不等于free。

当前`world2d.py`复用官方perception的scale1语言1×1 Conv2d投影与LayerNorm2d、按物理坐标重采样BEV query。64个查询cross-attend H_scene，形成[256,8,8]当前latent；轨迹/v0/未来时刻调制残差，stop-gradient基底加ΔZ后，经4级2D残差上采样8→16→32→64→128，输出[B,3,128,128]。新增参数与丢弃分支在`migration`登记。World中无Conv3d；官方冻结ViT的时空patch embedding保持原样。

不再加载3D depth/UVTR/OCC decoder到训练GPU。新2D解码器未预训练；官方BEV范围外初始化不代表远距观测，112/160m能力需要真实监督和验证。后续EMA教师目标须使用相同2D latent形状与坐标，旧3D latent缓存不能沿用。当前尚未完成教师链路。

120km/h、5m/s²制动和0.2–0.5s延迟对应117.8–127.8m刹停距离，超过112m网格。112–160m RGB观测目标不能自动补足World碰撞评分；网格外支撑必须显式标注，不能零填充后当无风险。5s高速轨迹可达166.7m，World仅预测3s，后2s没有同等占用监督。保持用户网格目标，同时记录这些覆盖缺口，不能宣称已解决制动盲区。

## 5. 数据和标签

- 目标70% Bench2Drive复杂交互、30% Town04/Town06高速专项；目标巡航80/100/120km/h。数据速度覆盖以实测统计为准，不以目标速度当实际速度。
- 六类：straight_accelerate_brake、curved_lane_hold、following、cut_in_and_brake、lane_change、overtake_and_abort。NPC独立驾驶；同一路线/同初态反事实分支同split；test不用于选择checkpoint。
- 前视配置1920×1080、FOV80°，环视960×540、FOV100°仅为采集初始值；相机内外参、时间戳、姿态必须保存，适配官方perception输入不能丢失标定。
- SpeedBinBalancedSampler按0/20/40/60/80/100/120km/h分桶。逆频率权重只保证期望，不保证每批均衡，因此使用按桶配额的batch sampler；来源70/30在有限batch中按累计配额兑现，报告缺桶，不能伪造缺失高速样本。
- 纵向4类：未来0–3s平均ax < -3紧急制动；[-3,-0.5)减速；[-0.5,0.5]匀速；>0.5加速。横向3类：3s Δy≥1.2左、≤-1.2右、其余保持。编码intent=longitudinal*3+lateral。固定自车坐标Δy是用户指定代理标签，在弯道不等于真实变道，记录此限制。
- actor只取历史RGB/状态/导航；future轨迹/OCC/EMA图像仅在label分支读取。原±25.6m OCC不扩成真实远距标签。高速数据未采集前不能用低速日志宣称70/30覆盖。

## 6. Stage 1 SFT

冻结ViT和VLM主体，仅后8层注意力投影使用PEFT LoRA r16/alpha32/dropout0.05。学习率LoRA=5e-5、World=2e-4、planner调制/运动学头=1e-4。AdamW wd0.01、clip1.0、10%warmup+cosine；本机使用foreach，融合AdamW与Lightning BF16裁剪不兼容，不启用。正式SFT计划3–5轮；当前诊断更新只验证梯度与内存，不执行正式损失或调度。

```
L = 0.5*L_reason + 1.0*L_fm + 0.2*L_kin + 0.5*L_world
L_reason = CE(intent_logits, intent_label)
u_s = (1-s)*noise + s*u_gt; s ~ Uniform(0,1)
L_fm = MSE(v_theta(u_s,s,condition), u_gt-noise)
       + 0.2*SmoothL1(integrate(u_s+(1-s)*v_theta), trajectory_gt)
L_kin = Huber(diff(acceleration)/dt) + mean(ReLU(abs(ay)-3)^2)
L_world = sum_h [Focal(occ_logits_h, occ_gt_h; support_mask)
                + 0.1*MSE(delta_z_h, stopgrad(z_ema_future_h-z_now); support_mask)]
```

World反馈开启条件：验证集latent MSE与occupancy Brier均显著优于静态Persistence。Persistence仅复制当前预测，不训练额外模型；必须同mask、同坐标、同样本，对路线聚合改进做置信区间，不能把随机像素当独立样本。EMA只用于标签且无梯度。门控通过前不把World评估输出当已可靠裁判。

## 7. Stage 2 World-Guided GRPO

冻结World全部参数，只更新planner和VLM LoRA；group G=8，优势A=(R-mean(R))/(std(R)+1e-6)，奖励同场景组内归一化。常量奖励组优势为0，不引入额外对照训练。

需要实现真正连续Flow的策略概率/似然比或有文献依据的Flow-GRPO转移分布，保存old-policy log_prob、离散步长、噪声/概率定义和clip规则。确定性Euler生成轨迹没有可直接拿来用的token log_prob；把优势乘FM loss是加权回归，不得标成GRPO。官方仓库是推理发布，不提供本任务的完整RL训练器。该算法接口完成前不启动名义Stage2。

```
R = 1*progress - 10*collision - 5*ttc - 2*dynamics - 1*lane
collision = sum_h max_footprint P_occ(h, point), h=1/2/3s
ttc = max_t ReLU(2.5 - gap/max(v_ego-v_lead,0.1))^2
progress = mean_t(1-abs(v-v_target)/max(v_target,5)), only if TTC>3
dynamics = max_t ReLU(abs(ay)-3)^2 + max_t ReLU(abs(jerk)-8)^2
```

车体包络采用前保/后保/中心/四角，尺寸来自实际车型，不固定猜测。TTC需要当前/历史感知的障碍速度、车道关联和净间距，不能由单张二值占用凭空得到。lane权重原文未给定，初始配置取1.0，定义为超出可行驶区的footprint比例；需记录标注来源。动态World不涵盖静态障碍时，不能把动态碰撞指标解释为完整碰撞检测。

## 8. 双频闭环和必需验收

规划5Hz，K=4，按R选最高且collision<0.2；控制20Hz、tick0.05s。目标与真实端到端耗时分开报告，不能通过慢速同步仿真tick声称达到实时时限。

仅用户要求的动态介入：v>35或v<-0.05m/s、实际v²|κ|>4.5m/s²、净车头距<2m或有效TTC<1.2s。移除旧静态死锁锁，不另加无意义安全模块。无候选/过期轨迹处理必须明确记录，不悄悄调用旧规则驾驶器替代模型成绩。

验收顺序：官方组件严格加载与版本指纹 → 原生多模态/LoRA/FP32/后缀零梯度 → 真实数据时间、坐标和split → 一次真实batch及显存/耗时 → Lightning保存失败与连续恢复 → SFT → 规范要求的Persistence指标 → 正确GRPO → 6类高速场景和0–120km/h闭环。只做必要工程检查，不扩展消融组。

## 上游来源和已核验版本

- [官方源码](https://github.com/QwenLM/Qwen-Drive-1.0)，本地runtime/qwen-drive-official，实施需记录实际commit。
- [官方模型](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B)，固定revision `28484089a7cc8c335cf5089fb0745cf7c49b6eaa`；三个权重SHA256见下载记录。
- 官方接口参见本地 `src/qwen_drive/planning_expert.py`、`configuration_qwen_drive.py`、`src/qwen_drive_perception/modeling_perception.py`、`configuration_perception.py`。本规范的未来World/控制头/后训练是新增扩展，不能归为官方原生已有能力。
