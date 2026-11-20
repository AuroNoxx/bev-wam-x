"""Native Windows launcher for local CARLA driving sessions."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import tkinter as tk
from tkinter import messagebox, ttk

from . import __version__


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/v6-dual-system/launcher"
SIMS = {
    "0.10.0": ("CARLA-0.10.0/Carla-0.10.0-Win64-Shipping", "CarlaUnreal", "Town10HD_Opt"),
    "0.9.15": ("CARLA-0.9.15/WindowsNoEditor", "CarlaUE4", "Town04"),
}
ROUTES = {"长路线巡航": "city", "无信号灯高速直道": "traffic", "高速弯道与右岔口": "traffic-right-fork"}


def installed_models():
    models={}
    for weights in sorted((ROOT/"models/carla-driver").glob("*/best.ckpt")):
        folder=weights.parent
        try:
            gate=json.loads((folder/"open-loop.json").read_bytes())
            graph=json.loads((folder/"graph.json").read_bytes())
            if gate["passed"] and graph["passed"] and gate["checkpoint_sha256"]==graph["sha256"]:
                name="v1 · 统一 BEV · 第 22 轮" if folder.name=="v6-unified-r22" else folder.name
                models[name]=folder
        except (OSError,ValueError,KeyError):
            continue
    return models


def installed_maps(version):
    directory, engine, preferred = SIMS[version]
    base = ROOT / "runtime" / directory
    maps = base / engine / "Content/Carla/Maps"
    if not (base / (engine + ".exe")).is_file():
        return []
    names = {f.stem for f in maps.rglob("*.umap")
             if re.fullmatch(r"Town\d+(?:HD)?(?:_Opt)?|Mine_01", f.stem)
             and (f.parent / "OpenDrive" / (f.stem + ".xodr")).is_file()}
    return sorted(names, key=lambda n: (n != preferred, n))


class Launcher:
    def __init__(self, root):
        self.root = root
        root.title(f"CARLA 驾驶启动器 · v{__version__}")
        root.resizable(False, False)
        root.option_add("*Font", ("Microsoft YaHei UI", 10))
        style = ttk.Style(root)
        style.configure("TLabel", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(12, 5))
        form = ttk.Frame(root, padding=(24, 18))
        form.grid()
        ttk.Label(form, text="驾驶演示", font=("Microsoft YaHei UI", 17, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        self.maps = {v: installed_maps(v) for v in SIMS}
        versions = [v for v, maps in self.maps.items() if maps]
        self.models = installed_models()
        self.fields = []
        def choice(row, title, values):
            ttk.Label(form, text=title).grid(row=row, column=0, sticky="w", padx=(0, 20))
            widget = ttk.Combobox(form, width=31, state="readonly", values=values)
            widget.grid(row=row, column=1, sticky="ew", pady=5)
            if values:widget.current(0)
            self.fields.append(widget)
            return widget

        self.version = choice(1, "CARLA 版本", versions)
        self.model = choice(2, "主驾模型", list(self.models))
        self.town = choice(3, "地图", [])
        self.route = choice(4, "路线", [])
        def number(row, title, value, limit, step):
            ttk.Label(form, text=title).grid(row=row, column=0, sticky="w")
            widget = ttk.Spinbox(form, from_=0, to=limit, increment=step, width=9)
            widget.set(value)
            widget.grid(row=row, column=1, sticky="w", pady=5)
            self.fields.append(widget)
            return widget
        self.npc = number(5, "NPC 车辆数", 10, 40, 1)
        self.speed = number(6, "巡航目标（km/h）", 40, 120, 10)
        self.version.bind("<<ComboboxSelected>>", self.set_version)
        self.town.bind("<<ComboboxSelected>>", self.set_routes)
        self.set_version()
        self.status = tk.StringVar(value="就绪")
        ttk.Separator(form).grid(row=7, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Label(form, textvariable=self.status, wraplength=395).grid(row=8, column=0, columnspan=2, sticky="w")
        actions = ttk.Frame(form)
        actions.grid(row=9, column=0, columnspan=2, sticky="e", pady=(14, 0))
        self.log_btn = ttk.Button(actions, text="查看日志", command=self.open_log, state="disabled")
        self.log_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(actions, text="停止", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        self.start_btn = ttk.Button(actions, text="开始驾驶", command=self.start)
        self.start_btn.pack(side="left", padx=4)
        self.process = None
        self.stopping = False
        self.close_after_stop = False
        root.protocol("WM_DELETE_WINDOW", self.close)

    def set_version(self, *_):
        maps = self.maps.get(self.version.get(), [])
        self.town.configure(values=maps)
        self.town.set(maps[0] if maps else "")
        self.set_routes()

    def set_routes(self, *_):
        town = self.town.get()
        routes = ["长路线巡航"]
        if self.version.get()=="0.9.15" and town=="Town04":
            routes += ["无信号灯高速直道", "高速弯道与右岔口"]
        self.route.configure(values=routes)
        self.route.set(routes[0])

    def open_log(self):
        os.startfile(self.session / "launcher.log")

    def start(self):
        if self.process is not None:return
        try:
            version, town = self.version.get(), self.town.get()
            if town not in self.maps.get(version, []) or self.model.get() not in self.models:
                raise ValueError("请选择已安装的版本、地图和模型")
            npc, speed = int(self.npc.get()), int(self.speed.get())
            if not 0<=npc<=40 or not 0<=speed<=120:
                raise ValueError("NPC 车辆数可选 0–40，巡航目标可选 0–120 km/h")
            ps = shutil.which("pwsh") or shutil.which("powershell")
            if not ps:raise OSError("找不到 PowerShell")
            CACHE.mkdir(parents=True, exist_ok=True)
            self.session = Path(tempfile.mkdtemp(prefix="drive-", dir=CACHE))
            args = [ps,"-NoProfile","-ExecutionPolicy","Bypass","-File",str(ROOT/"v6_dual_system/Start-FastPilot.ps1"),
                    "-SimulatorVersion",version,"-Town",town,"-Scenario",ROUTES[self.route.get()],
                    "-ModelDirectory",str(self.models[self.model.get()]),"-Mode","interactive","-NpcCount",str(npc),"-InitialSpeed",str(speed),
                    "-Seconds","3600","-Port","2350","-StopFile",str(self.session/"stop")]
            with (self.session/"launcher.log").open("wb") as log:
                self.process = subprocess.Popen(args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                                                stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        except (ValueError,OSError,KeyError) as exc:
            messagebox.showerror("无法启动", str(exc), parent=self.root)
            return
        self.stopping=False
        self.status.set("正在加载地图、车流和驾驶模型…")
        for field in self.fields:field.configure(state="disabled")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.log_btn.configure(state="normal")
        self.root.after(500,self.poll)

    def poll(self):
        code=self.process.poll()
        with (self.session/"launcher.log").open("rb") as log:
            log.seek(max(0,os.fstat(log.fileno()).st_size-16384))
            tail=log.read().decode("utf-8",errors="replace")
        if code is None:
            if not self.stopping and "Driving session:" in tail:
                count=re.search(r"Traffic Manager: (\d+) NPC",tail)
                traffic=f" · {count[1]} 辆 NPC" if count else ""
                self.status.set(f"驾驶中 · {self.town.get()}{traffic}")
            self.root.after(500,self.poll)
            return
        self.process=None
        if self.close_after_stop:
            self.root.destroy()
            return
        self.status.set("演示已停止" if self.stopping else "演示已结束" if code==0 else "启动或运行失败，请查看日志")
        for field in self.fields:field.configure(state="readonly" if isinstance(field,ttk.Combobox) else "normal")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def stop(self):
        if self.process is not None:
            (self.session/"stop").touch()
            self.stopping=True
            self.status.set("正在结束本次演示…")
            self.stop_btn.configure(state="disabled")

    def close(self):
        if self.process is None:self.root.destroy()
        else:
            self.close_after_stop=True
            self.stop()


if __name__=="__main__":
    root=tk.Tk()
    Launcher(root)
    root.mainloop()
