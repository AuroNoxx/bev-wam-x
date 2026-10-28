import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path
    import sys

    # Ensure project root is in sys.path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import marimo as mo
    import numpy as np
    from PIL import Image

    from mini_wa_jepa.visualization import colorize_bev_4ch
    from mini_wa_jepa.visualization.reports import METRICS, load_stage_runs

    return (
        Image,
        METRICS,
        colorize_bev_4ch,
        json,
        load_stage_runs,
        mo,
        np,
        project_root,
        sys,
    )


@app.cell
def _(mo):
    mo.md("""
    # 🚗 mini-wa-jepa：交互式可视化与控制面板

    *轻量化 BEV 时空潜变量世界-动作模型工程系统*
    """)
    return


@app.cell
def _(json, mo, project_root):
    # Load Stage 1 report
    report_path = project_root / "artifacts/restart_stage1/egl_100/report.json"
    if report_path.exists():
        report_data = json.loads(report_path.read_text())
        episodes = report_data.get("episodes", [])
        fps_list = [ep.get("pipeline_fps", 0) for ep in episodes]
        avg_fps = report_data.get("pipeline_fps", 0)
        gate_status = "✅ 满足门槛 (≥120 FPS)" if avg_fps >= 120 else "❌ 未达标"
        stats_banner = mo.stat(
            value=f"{avg_fps:.1f} FPS",
            label="Stage 1 完整管线渲染帧率",
            caption=f"{gate_status} | 已记录 {len(episodes)} episodes | 含 reset {report_data.get('wall_fps', 0):.2f} FPS | 单 episode 最低 {min(fps_list, default=0):.2f} FPS",
        )
    else:
        stats_banner = mo.md("*未检测到 Stage 1 report.json*")
    stats_banner
    return


@app.cell
def _(mo, project_root):
    # Scan for available sample files
    samples_dir = project_root / "artifacts/restart_stage1/egl_100"
    npz_files = sorted(samples_dir.glob("seed_*.npz")) if samples_dir.exists() else []
    seed_options = {p.stem: p for p in npz_files}

    seed_select = mo.ui.dropdown(
        options=seed_options,
        value=list(seed_options.keys())[0] if seed_options else None,
        label="选择样本 Seed 进行 4 通道 BEV 查看:",
    )
    seed_select
    return (seed_select,)


@app.cell
def _(Image, colorize_bev_4ch, mo, np, seed_select):
    mo.stop(seed_select.value is None, mo.md("*请先选择样本*"))

    data = np.load(seed_select.value)
    bev = data["bev"]  # (4, 128, 128)
    rgb = data["native_rgb"] if "native_rgb" in data else None

    # Composite colorized BEV
    comp_rgb = colorize_bev_4ch(bev)

    # 4 individual channels
    ch0_img = Image.fromarray(bev[0]).resize((160, 160), Image.Resampling.NEAREST)
    ch1_img = Image.fromarray(bev[1]).resize((160, 160), Image.Resampling.NEAREST)
    ch2_img = Image.fromarray(bev[2]).resize((160, 160), Image.Resampling.NEAREST)
    ch3_img = Image.fromarray(bev[3]).resize((160, 160), Image.Resampling.NEAREST)

    comp_img = Image.fromarray(comp_rgb).resize((220, 220), Image.Resampling.NEAREST)
    cam_img = Image.fromarray(rgb).resize((220, 220), Image.Resampling.BILINEAR) if rgb is not None else None

    views = [
        mo.vstack([mo.md("**合成 4通道 BEV**"), comp_img]),
        mo.vstack([mo.md("**CH0: 可行驶区**"), ch0_img]),
        mo.vstack([mo.md("**CH1: 边界与静态障碍**"), ch1_img]),
        mo.vstack([mo.md("**CH2: 动态车辆**"), ch2_img]),
        mo.vstack([mo.md("**CH3: 视线掩码 (1=可见)**"), ch3_img]),
    ]
    if cam_img:
        views.insert(0, mo.vstack([mo.md("**原生顶视语义相机**"), cam_img]))

    mo.vstack([
        mo.md(f"### 样本详情：`{seed_select.selected_key}`"),
        mo.hstack(views, justify="start", gap=1.5),
    ])
    return


@app.cell
def _(mo):
    launch_rerun = mo.ui.run_button(label="🚀 打开 Rerun 独立桌面 App (时空多模态回放)")
    mo.vstack([
        mo.md("### 实时多模态交互回放 (Rerun)"),
        mo.md("点击下方按钮将 Stage 1 的 20 组 4通道 BEV 样本加载进 Rerun 视窗中，支持时间轴拖拽与遥测曲线查看："),
        launch_rerun,
    ])
    return (launch_rerun,)


@app.cell
def _(launch_rerun, mo, project_root, sys):
    import subprocess

    if launch_rerun.value:
        subprocess.Popen([sys.executable, "-m", "scripts.visualize_stage1_rerun", "--spawn"], cwd=project_root)
        msg = mo.md("✅ **Rerun 视窗已在后台唤起！请查看弹出的桌面应用。**")
    else:
        msg = mo.md("")
    msg
    return


@app.cell
def _(mo):
    stage_select = mo.ui.dropdown(options={f"阶段 {i}": i for i in (2, 3, 4, 5, 6, 7)}, value="阶段 7", label="阶段指标")
    refresh_reports = mo.ui.run_button(label="刷新阶段报告")
    mo.hstack([stage_select, refresh_reports], justify="start")
    return refresh_reports, stage_select


@app.cell
def _(load_stage_runs, mo, project_root, refresh_reports, stage_select):
    refresh_reports.value
    stage_rows, stage_sources, stage_errors = load_stage_runs(project_root, stage_select.value)
    mo.vstack([
        mo.md(f"### 阶段 {stage_select.value}：消融、开环与闭环实测指标"),
        mo.md("缺失指标留空；阶段 3 的表示学习任务不产生轨迹 ADE/FDE 时，不填零。"),
        mo.md("**第二版损失与整条轨迹消融为验证集结果**：按种子与温度分别比较原损失、米制路线/终点损失和整条路线系数生成；不能解释为测试集或闭环验收通过。")
        if any(row.get("evaluation_kind") in ("validation_loss_ablation", "validation_structured_ablation") for row in stage_rows) else mo.md(""),
        mo.ui.table(stage_rows, selection=None) if stage_rows else mo.md("**尚无新版阶段报告。** 旧版数据不作为新版验收结果。"),
        mo.md("\n\n".join(stage_errors)) if stage_errors else mo.md(""),
        mo.accordion({"报告来源与复现信息": mo.json(stage_sources)}) if stage_sources else mo.md(""),
    ])
    return (stage_rows,)


@app.cell
def _(METRICS, mo):
    metric_select = mo.ui.dropdown(options={label: key for key, label in METRICS.items()}, value="ADE (m)", label="比较指标")
    metric_select
    return (metric_select,)


@app.cell
def _(METRICS, metric_select, mo, stage_rows):
    _measured = [row for row in stage_rows if row[metric_select.value] is not None]
    if _measured:
        _metric_view = mo.ui.table([
            {"run": row["run_id"], "model": row["model"], "ablation": row["ablation"],
             "split": row["split"], "seed": row["seed"],
             **{key:row.get(key) for key in ('seed_kind','training_seed','evaluation_kind')},
             METRICS[metric_select.value]: row[metric_select.value]}
            for row in _measured], selection=None)
    else:
        _metric_view = mo.md("*所选指标尚未产出。*")
    _metric_view
    return


@app.cell
def _(mo):
    refresh_stage2_data = mo.ui.run_button(label="刷新阶段 2 数据与训练进度")
    mo.vstack([mo.md("## 阶段 2：当前四通道数据与基线训练"), refresh_stage2_data])
    return (refresh_stage2_data,)


@app.cell
def _(json, mo, project_root, refresh_stage2_data):
    refresh_stage2_data.value
    current_stage2_root = project_root / "outputs/data/stage2_v2_100k"
    _manifest = current_stage2_root / "manifest.json"
    _parts = sorted((project_root / "outputs/data/stage2_v2_parts").glob("*/manifest.json"))
    _manifests = [json.loads(path.read_text()) for path in _parts]
    _collected = sum(sum(ep["stored_frames"] for ep in item.get("episodes", [])) for item in _manifests)
    if _manifest.exists():
        _current = json.loads(_manifest.read_text())
        _collected = _current.get("stored_frames", _collected)
        _status = "已完成并校验" if _current.get("completed") else "处理中"
    else:
        _status = "采集中（两个隔离 EGL 进程）"
    _training_report = project_root / "artifacts/restart_stage2/baseline/report.json"
    _training = json.loads(_training_report.read_text()) if _training_report.exists() else None
    mo.vstack([
        mo.md(f"**当前数据：stage2_v2_100k** — {_status}；{_collected:,} / 120,000 帧（训练 100k，验证/测试各 10k）。"),
        mo.md("规格：4×128×128 CHW uint8 / 64m；10 个 0.2s 路点 `(x,y,v,heading)`。旧三通道训练数据已按用户要求清理。"),
        mo.md(f"基线训练：已完成 {len(_training.get('epochs', []))} / {_training['config']['epochs']} epochs；参数 {_training['parameter_count']:,}。")
        if _training else mo.md("基线训练：等待正式数据；冒烟训练不作为正式结果。"),
    ])
    return (current_stage2_root,)


@app.cell
def _(current_stage2_root, mo, refresh_stage2_data):
    refresh_stage2_data.value
    from dataset.storage import ShardReader
    stage2_reader = ShardReader(current_stage2_root / "train") if (current_stage2_root / "train/index.json").exists() else None
    stage2_frame = mo.ui.slider(start=0, stop=max(1, len(stage2_reader)-1) if stage2_reader else 1,
        value=len(stage2_reader)//2 if stage2_reader else 0, step=1, label="阶段 2 训练帧")
    stage2_frame if stage2_reader else mo.md("*合并完成后可在此查看正式训练帧与实际救车轨迹。*")
    return stage2_reader, stage2_frame


@app.cell
def _(Image, colorize_bev_4ch, mo, np, stage2_frame, stage2_reader):
    from PIL import ImageDraw
    if stage2_reader:
        _row = stage2_reader[int(stage2_frame.value)]
        _image = Image.fromarray(colorize_bev_4ch(_row["bev"])).resize((512,512),Image.Resampling.NEAREST)
        _draw = ImageDraw.Draw(_image)
        _xy = _row["future_trajectory"][_row["future_valid"], :2]
        _pixels = (63.5 - _xy[:, [1,0]] * 2 + .5) * 4
        if len(_pixels)>1: _draw.line([tuple(point) for point in _pixels], fill="orange", width=3)
        _view = mo.vstack([
            mo.md(f"阶段 2 / episode {int(_row['episode_id'])} / step {int(_row['step_id'])} / 速度 {float(_row['state'][3]):.2f} m/s / 扰动 {bool(_row['noise_active'])} / 恢复 {bool(_row['recovery_active'])} / 动作监督 {bool(_row['action_mask'])}"),
            _image,
        ])
    else: _view = mo.md("")
    _view
    return


@app.cell
def _(mo, project_root):
    _recordings = sorted([path for name in ("baseline", "baseline_eval", "expert_test", "replay")
        for path in (project_root / "artifacts/restart_stage2" / name).glob("**/*.rrd")])
    mo.vstack([
        mo.md("### Rerun 录制与训练审阅约定"),
        mo.md("训练保留 TensorBoard；阶段 3～5 定期保存开环轨迹（适用时）、真实/预测隐变量到 `.rrd`，并生成预测 vs 真值图。每阶段验收仍附视频。"),
        mo.ui.table([{"recording": str(path.relative_to(project_root)), "bytes": path.stat().st_size} for path in _recordings], selection=None)
        if _recordings else mo.md("*尚无 .rrd 录制*"),
    ])
    return


@app.cell
def _(json, mo, project_root):
    _stage3_paths = sorted((project_root / "artifacts/restart_stage3").glob("*/report.json"))
    _formal_paths = [path for path in _stage3_paths if json.loads(path.read_text()).get("max_batches") is None]
    stage3_run = mo.ui.dropdown(options={path.parent.name: path for path in _formal_paths},
        value=_formal_paths[-1].parent.name if _formal_paths else None, label="阶段 3 正式运行")
    refresh_stage3 = mo.ui.run_button(label="刷新阶段 3 训练与表示健康度")
    mo.vstack([mo.md("## 阶段 3：VICReg 内容有效秩与单步预测"),
               mo.hstack([stage3_run, refresh_stage3], justify="start")])
    return refresh_stage3, stage3_run


@app.cell
def _(json, mo, refresh_stage3, stage3_run):
    refresh_stage3.value
    stage3_report = json.loads(stage3_run.value.read_text()) if stage3_run.value else None
    if stage3_report:
        _epochs = stage3_report["epochs"]
        _status = "已完成，等待阶段验收" if stage3_report.get("completed") else (
            "异常停止，须审查失败现场" if stage3_report.get("error") else "训练中")
        _rows = [{"epoch": row["epoch"], "step": row["global_step"],
            "content rank / 256": row["validation"]["effective_rank_ratio"],
            "frame-pooled rank / 256": row["validation"]["pooled_effective_rank_ratio"],
            "channel std": row["validation"]["std_mean"],
            "prediction / copy MSE": row["validation"]["prediction_copy_ratio"],
            "peak allocated GiB": row["peak_allocated_gib"]} for row in _epochs]
        _stage3_view = mo.vstack([
            mo.md(f"**{_status}** — {len(_epochs)} / {stage3_report['config']['epochs']} epochs；"
                  f"编码器 {stage3_report['encoder_parameters']:,} / 总参数 {stage3_report['parameter_count']:,}。"),
            mo.md("内容有效秩按固定空间位置扣均值后，从原始 encoder 协方差计算；PCA/白化不用于抬高秩。"
                  "整帧池化秩另列。ADE/FDE 与闭环指标：**未产出（本阶段无轨迹头）**。"),
            mo.ui.table(_rows, selection=None) if _rows else mo.md("*首次完整验证尚在运行。*"),
            mo.json(stage3_report["gates"]) if stage3_report.get("gates") else mo.md("*验收门槛尚未汇总。*"),
            mo.accordion({"配置与来源": mo.json({key:stage3_report[key] for key in
                ("config", "git_commit", "source_tree_sha256", "goal_sha256", "rank_definition")})}),
        ])
    else:
        _stage3_view = mo.md("*尚无正式阶段 3 训练记录；冒烟结果不作为验收。*")
    _stage3_view
    return (stage3_report,)


@app.cell
def _(mo, stage3_report):
    if stage3_report and stage3_report["epochs"]:
        import matplotlib.pyplot as plt
        _ep = stage3_report["epochs"]
        _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.4))
        _x = [row["epoch"] for row in _ep]
        for _key, _label in [("effective_rank_ratio", "content"), ("pooled_effective_rank_ratio", "frame pooled")]:
            _axes[0].plot(_x, [row["validation"][_key] for row in _ep], marker=".", label=_label)
        _axes[0].axhline(.75, color="red", linestyle="--"); _axes[0].set_ylim(0, 1.05)
        _axes[0].set_ylabel("effective rank / 256"); _axes[0].legend(fontsize=8)
        _axes[1].plot(_x, [row["validation"]["std_mean"] for row in _ep], marker=".")
        _axes[1].set_ylabel("content std mean")
        _axes[2].plot(_x, [row["validation"]["prediction_copy_ratio"] for row in _ep], marker=".")
        _axes[2].axhline(1, color="gray", linestyle="--"); _axes[2].set_ylabel("prediction / copy MSE")
        for _ax in _axes: _ax.set_xlabel("epoch")
        _fig.tight_layout(); _health_view = mo.as_html(_fig); plt.close(_fig)
    else:
        _health_view = mo.md("")
    _health_view
    return


@app.cell
def _(mo, stage3_report):
    _snapshots = stage3_report.get("visualizations", []) if stage3_report else []
    _images = {f"step {entry['step']} / {path.rsplit('/', 1)[-1]}": path
               for entry in _snapshots for path in entry["paths"]}
    stage3_image = mo.ui.dropdown(options=_images, value=list(_images)[-1] if _images else None,
                                 label="阶段 3 周期预测/真实潜变量与 PCA")
    stage3_image
    return (stage3_image,)


@app.cell
def _(Image, mo, project_root, stage3_image):
    _latent_image = Image.open(project_root / stage3_image.value) if stage3_image.value else None
    mo.vstack([mo.md("潜变量热图是编码特征，**不是像素重建**。周期图中的全验证秩/PCA 参考来自最近完成的验证；预测/目标为该训练步实际输出。"),
               _latent_image]) if _latent_image else mo.md("")
    return


@app.cell
def _(json, mo, project_root):
    _stage4_paths = sorted((project_root / "artifacts/restart_stage4").glob("*/report.json"))
    _stage4_formal = [path for path in _stage4_paths if
        json.loads(path.read_text()).get("stage") == 4 and
        "epochs" in json.loads(path.read_text()) and json.loads(path.read_text()).get("max_batches") is None]
    stage4_run = mo.ui.dropdown(options={path.parent.name: path for path in _stage4_formal},
        value=_stage4_formal[-1].parent.name if _stage4_formal else None, label="阶段 4 正式训练")
    refresh_stage4 = mo.ui.run_button(label="刷新阶段 4 训练与验收")
    mo.vstack([mo.md("## 阶段 4：四帧世界/轨迹联合学习与动作对照"),
               mo.hstack([stage4_run, refresh_stage4], justify="start")])
    return refresh_stage4, stage4_run


@app.cell
def _(json, mo, refresh_stage4, stage4_run):
    refresh_stage4.value
    stage4_report = json.loads(stage4_run.value.read_text()) if stage4_run.value else None
    if stage4_report:
        _epochs4 = stage4_report["epochs"]
        _status4 = "训练已完成" if stage4_report.get("completed") else (
            "异常停止：请审查失败现场" if stage4_report.get("error") else "训练中")
        _progress4_path = stage4_run.value.parent / "progress.json"
        _progress4 = json.loads(_progress4_path.read_text()) if _progress4_path.exists() else None
        _rows4 = [{"epoch": row["epoch"], "step": row["global_step"],
            "val ADE m": row["validation"]["ade_m"], "val FDE m": row["validation"]["fde_m"],
            "content rank / 256": row["validation"]["effective_rank_ratio"],
            "pooled rank / 256": row["validation"]["pooled_effective_rank_ratio"],
            "world / copy MSE": row["validation"]["prediction_copy_ratio"],
            "peak allocated GiB": row["peak_allocated_gib"]} for row in _epochs4]
        _view4 = mo.vstack([
            mo.md(f"**{_status4}**：{len(_epochs4)} / {stage4_report['config']['epochs']} epochs；"
                  f"参数 {stage4_report['parameter_count']:,}；世界损失权重 {stage4_report['config']['latent_weight']}。"),
            mo.md("两组均保留本项目阶段 3 初始化、四帧历史、空间 token 与相同运动先验；"
                  "动作对照只在当前微调中关闭世界预测和 VICReg 损失。验证 ADE 选权重，测试不参与选型。"),
            mo.ui.table(_rows4, selection=None) if _rows4 else mo.md("*首次完整验证尚未完成。*"),
            mo.accordion({"最近训练批次": mo.json(_progress4)}) if _progress4 else mo.md(""),
            mo.accordion({"配置和来源": mo.json({key:stage4_report[key] for key in
                ("config", "git_commit", "source_tree_sha256", "goal_sha256", "initial_model_sha256")})}),
        ])
    else:
        _view4 = mo.md("*尚无正式阶段 4 训练记录。*")
    _view4
    return (stage4_report,)


@app.cell
def _(mo, stage4_report):
    if stage4_report and stage4_report["epochs"]:
        import matplotlib.pyplot as _plt4
        _fig4, _axes4 = _plt4.subplots(1, 3, figsize=(13, 3.4))
        _rows4 = stage4_report["epochs"]; _x4 = [row["epoch"] for row in _rows4]
        for _key4 in ("ade_m", "fde_m"):
            _axes4[0].plot(_x4, [row["validation"][_key4] for row in _rows4], marker=".", label=_key4)
        for _key4 in ("effective_rank_ratio", "pooled_effective_rank_ratio"):
            _axes4[1].plot(_x4, [row["validation"][_key4] for row in _rows4], marker=".", label=_key4)
        for _key4 in ("latent_mse", "copy_last_mse"):
            _axes4[2].plot(_x4, [row["validation"][_key4] for row in _rows4], marker=".", label=_key4)
        for _ax4 in _axes4: _ax4.set_xlabel("epoch"); _ax4.legend(fontsize=7)
        _axes4[0].set_ylabel("Validation trajectory error (m)")
        _axes4[1].set_ylabel("Effective rank / 256"); _axes4[2].set_ylabel("World MSE")
        _fig4.tight_layout(); _curves4 = mo.as_html(_fig4); _plt4.close(_fig4)
    else: _curves4 = mo.md("")
    _curves4
    return


@app.cell
def _(mo, stage4_report):
    _shots4 = stage4_report.get("visualizations", []) if stage4_report else []
    _images4 = {f"step {entry['step']} / {path.rsplit('/', 1)[-1]}": path
                for entry in _shots4 for path in entry["paths"]}
    stage4_image = mo.ui.dropdown(options=_images4, value=list(_images4)[-1] if _images4 else None,
                                 label="阶段 4 实际轨迹/潜变量预测与真值")
    stage4_image
    return (stage4_image,)


@app.cell
def _(Image, mo, project_root, stage4_image):
    _image4 = Image.open(project_root / stage4_image.value) if stage4_image.value else None
    mo.vstack([mo.md("上排：实际 BEV、2s 轨迹及速度；下排：实际预测/真实未来编码特征和误差。缺失未来使用掩码；热图不是像素重建。"),
               _image4]) if _image4 else mo.md("")
    return


@app.cell
def _(json, mo, project_root, refresh_stage4):
    refresh_stage4.value
    _test4_path = project_root / "artifacts/restart_stage4/test_eval/report.json"
    _accept4_path = project_root / "artifacts/restart_stage4/acceptance.json"
    _test4 = json.loads(_test4_path.read_text()) if _test4_path.exists() else None
    _accept4 = json.loads(_accept4_path.read_text()) if _accept4_path.exists() else None
    _comparison4 = []
    if _test4:
        for _name4, _metrics4 in _test4.get("models", {}).items():
            _comparison4.append({"model": _name4, "ADE m": _metrics4["ade_m"], "FDE m": _metrics4["fde_m"],
                "full horizon anchors": _metrics4["full_horizon_frames"]})
    _closed4 = []
    for _name4 in ("joint", "action_only"):
        _closed4_path = project_root / f"artifacts/restart_stage4/{_name4}_closed_loop/report.json"
        if _closed4_path.exists():
            _closed4_report = json.loads(_closed4_path.read_text())
            _m4 = _closed4_report.get("metrics", {})
            _p4 = _closed4_report.get("pipeline_ms_p50_p95_p99", [])
            _closed4.append({"model": _name4, "episodes": len(_closed4_report["episodes"]),
                "completed": _closed4_report["completed"], "route completion": _m4.get("route_completion"),
                "collision rate": _m4.get("collision_rate"), "out of road rate": _m4.get("out_of_road_rate"),
                "full pipeline P95 ms": _p4[1] if _p4 else None})
    mo.vstack([
        mo.md("### 阶段 4 固定测试、匹配对照和闭环验收"),
        mo.ui.table(_comparison4, selection=None) if _comparison4 else mo.md("*测试尚未产出；先完成两组训练。*"),
        mo.ui.table(_closed4, selection=None) if _closed4 else mo.md("*闭环尚未产出。*"),
        mo.json(_accept4) if _accept4 else mo.md("*阶段门槛尚未汇总。*"),
    ])
    return


@app.cell
def _(mo):
    refresh_stage5 = mo.ui.run_button(label="刷新阶段 5 压力测试实测结果")
    mo.vstack([mo.md("## 阶段 5：匹配队列感知退化测试"), refresh_stage5])
    return (refresh_stage5,)


@app.cell
def _(json, mo, project_root, refresh_stage5):
    refresh_stage5.value
    _open5_path = project_root / "artifacts/restart_stage5/open_loop/report.json"
    _suite5_path = project_root / "artifacts/restart_stage5/suite_progress.json"
    _accept5_path = project_root / "artifacts/restart_stage5/acceptance.json"
    _open5 = json.loads(_open5_path.read_text()) if _open5_path.exists() else None
    _suite5 = json.loads(_suite5_path.read_text()) if _suite5_path.exists() else None
    _table5 = []
    if _open5:
        for _model5, _run5 in _open5.get("models", {}).items():
            for _condition5, _case5 in _run5.get("conditions", {}).items():
                _table5.append({"model": _model5, "condition": _condition5, "ADE m": _case5["metrics"]["ade_m"],
                                "FDE m": _case5["metrics"]["fde_m"], "anchors": _case5["metrics"].get("full_horizon_frames"),
                                "latent MSE": _case5["metrics"].get("latent_mse")})
    _closed_table5 = []
    if _suite5:
        for _job5 in _suite5.get("jobs", []):
            if _job5["smoke"] or not _job5["completed"]:
                continue
            _r5 = json.loads((project_root / _job5["output"] / "report.json").read_text())
            _closed_table5.append({"model": _job5["model"], "condition": _job5["condition"], "episodes": len(_r5["episodes"]),
                **{_k5:_r5.get("metrics", {}).get(_k5) for _k5 in ("route_completion", "collision_rate", "out_of_road_rate", "success_rate")}})
    _gates5 = json.loads(_accept5_path.read_text()) if _accept5_path.exists() else None
    mo.vstack([
        mo.md("开环固定 9,600 测试锚点；闭环为预先固定的 6 个种子，共 126 episodes。"
              "表中只显示实际已完成结果；缺失不补零。中间分数不能当作最终科学结论。"),
        mo.md(f"开环完成：{bool(_open5 and _open5.get('completed'))}；闭环状态：{_suite5.get('phase') if _suite5 else '未启动'}"),
        mo.ui.table(_table5, selection=None) if _table5 else mo.md("开环尚无完整条件。"),
        mo.ui.table(_closed_table5, selection=None) if _closed_table5 else mo.md("闭环尚无完整条件。"),
        mo.json(_gates5) if _gates5 else mo.md("阶段 5 尚未完成交付审计。"),
    ])
    return


@app.cell
def _(Image, mo, project_root, refresh_stage5):
    refresh_stage5.value
    _plots5 = [project_root / "artifacts/restart_stage5" / _name5
               for _name5 in ("open_loop_ade_m.png", "open_loop_fde_m.png", "closed_loop_comparison.png")]
    mo.vstack([Image.open(_p5) for _p5 in _plots5 if _p5.exists()])
    return


@app.cell
def _(mo):
    refresh_stage6 = mo.ui.run_button(label="刷新阶段 6 动作条件实验")
    mo.vstack([mo.md("## 阶段 6：同一历史、不同驾驶方案的实际后果"), refresh_stage6])
    return (refresh_stage6,)


@app.cell
def _(Image, json, mo, project_root, refresh_stage6):
    refresh_stage6.value
    _root6 = project_root / "artifacts/restart_stage6"
    _result6 = json.loads((_root6 / "test_eval/report.json").read_text()) if (_root6 / "test_eval/report.json").exists() else None
    _status6 = json.loads((_root6 / "suite_status.json").read_text()) if (_root6 / "suite_status.json").exists() else {}
    _accept6 = json.loads((_root6 / "acceptance.json").read_text()) if (_root6 / "acceptance.json").exists() else None
    _rows6 = []
    if _result6:
        for _name6, _row6 in _result6.get("models", {}).items():
            _rows6.append({"模型": _name6, "训练种子": _row6["seed"], "潜变量误差": _row6["latent_mse"],
                "后果差分误差": _row6["delta_mse"], "候选后果对应准确率": _row6["retrieval_accuracy"],
                "错配命令误差": _row6["wrong_command_mse"]})
    _figures6 = sorted((_root6 / "test_eval/conditioned_20261019/samples").glob("epoch_*/case_0.png"))
    mo.vstack([
        mo.md("四条方案由实验指定，尚无模型选路器。虚线表示命令，实线表示仿真实际执行。"
              "潜变量不等于重建的未来道路；候选后果对应准确率也不等于驾驶成功率。"
              "本页为实验记录，chapter简化交互仍排在阶段6之后。"),
        mo.md(f"运行状态：{_status6.get('phase', '未启动')}"),
        mo.ui.table(_rows6, selection=None) if _rows6 else mo.md("正式测试尚未产出；不显示试跑成绩。"),
        Image.open(_figures6[0]) if _figures6 else mo.md("正式同起点分支对照图尚未产出。"),
        mo.accordion({"配对区间与复现噪声": mo.json({"paired": _result6.get("paired"), "noise": _result6.get("replay_noise")})}) if _result6 else mo.md(""),
        mo.json(_accept6) if _accept6 else mo.md("阶段6尚未完成验收。"),
    ])
    return


@app.cell
def _(mo, project_root):
    _aligned_paths = sorted((project_root / "artifacts/bev_wa_alignment").glob("stage3*/results.json"))
    aligned_stage3_run = mo.ui.dropdown(
        options={p.parent.name: p for p in _aligned_paths},
        value=_aligned_paths[-1].parent.name if _aligned_paths else None,
        label="Stage 3 aligned experiments (full validation)")
    aligned_stage3_refresh = mo.ui.run_button(label="刷新 Stage 3 aligned 结果")
    mo.hstack([aligned_stage3_run, aligned_stage3_refresh])
    return aligned_stage3_run, aligned_stage3_refresh


@app.cell
def _(aligned_stage3_run, aligned_stage3_refresh, json, mo, project_root):
    aligned_stage3_refresh.value
    _selected = aligned_stage3_run.value
    _comparison = []
    for _p in sorted((project_root / "artifacts/bev_wa_alignment").glob("stage3*/results.json")):
        _data = json.loads(_p.read_text())
        for _run in _data.get("training_runs", {}).values():
            _m = _run["final_metrics"]
            _comparison.append({"experiment": _p.parent.name, "seed": _run["seed"],
                                **{h: r["normalized_mse"] for h, r in _m["results_by_horizon"].items()},
                                "rank": _m["effective_rank"], "var/std_mean": _m["token_variance"],
                                "gate": _m["all_passed"], "ADE": None, "FDE": None, "closed_loop": None})
    aligned_stage3_data = json.loads(_selected.read_text()) if _selected else None
    mo.vstack([mo.md("### Stage 3 aligned ablations — final full-validation E_norm"),
               mo.md("E_norm ≤ 0.85 for all horizons; rank ≥ 16; std_mean ≥ 0.10. ADE/FDE/closed-loop remain missing at Stage 3."),
               mo.ui.table(_comparison, selection=None) if _comparison else mo.md("No results yet.")])
    return (aligned_stage3_data,)


@app.cell
def _(aligned_stage3_data, mo):
    _views = []
    if aligned_stage3_data:
        import matplotlib.pyplot as _plt_masked
        for _run in aligned_stage3_data["training_runs"].values():
            _hist = _run["history"]
            _fig, _ax = _plt_masked.subplots(1, 3, figsize=(13, 3.5))
            _x = [r["epoch"] for r in _hist]
            _ax[0].plot(_x, [r["train_flow_loss"] for r in _hist], label="flow")
            if all("mask_loss" in r for r in _hist):
                _ax[0].plot(_x, [r["mask_loss"] for r in _hist], label="mask")
            for _h in ("+0.5s", "+1.0s", "+1.5s", "+2.0s"):
                _ax[1].plot(_x, [r["norm_mse"][_h] for r in _hist], label=_h)
            _ax[1].axhline(.85, linestyle="--", color="red")
            _ax[2].plot(_x, [r["effective_rank"] for r in _hist], label="rank")
            for _a in _ax:
                _a.set_xlabel("epoch")
                _a.legend()
            _fig.suptitle(f"Seed {_run['seed']}: epoch metrics use first 32 validation batches")
            _fig.tight_layout()
            _views.append(mo.as_html(_fig))
            _plt_masked.close(_fig)
    mo.vstack(_views) if _views else mo.md("")
    return


if __name__ == "__main__":
    app.run()
