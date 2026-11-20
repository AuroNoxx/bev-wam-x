"""Measure the deployed process on held-out sensor packets without starting CARLA."""
import json
import subprocess
import sys
import tempfile
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

from .dataset import NavigationData
from .inference import RGB_BYTES, SIZE
from .sensors import ROOT,write_json


def receive(worker):
    for line in worker.stdout:
        if line.startswith("V1 "):return json.loads(line[3:])
    raise RuntimeError(f"Inference worker stopped ({worker.poll()})")


def main():
    folder=ROOT/"artifacts/v1-dual-system/navigation"
    ds=NavigationData("validation")
    slot=shared_memory.SharedMemory(create=True,size=SIZE)
    worker=monitor=None
    try:
        raw=np.ndarray((6,360,640),np.uint8,slot.buf)
        grid=np.ndarray((3,128,128),np.uint8,slot.buf,offset=RGB_BYTES)
        with tempfile.TemporaryDirectory(dir=ROOT/".cache/v1-dual-system") as temp:
            counters=Path(temp)/"gpu.jsonl"
            worker=subprocess.Popen([sys.executable,"-u","-m","v1_dual_system.inference",slot.name,str(folder/"best.ckpt")],
                cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
            pid=receive(worker)["pid"]
            monitor=subprocess.Popen(["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass","-File",
                str(ROOT/"v1_dual_system/watch_gpu.ps1"),str(pid),str(counters)],creationflags=subprocess.CREATE_NO_WINDOW)
            assert receive(worker)["graph"]
            ms=[];latency=[]
            for j,idx in enumerate(np.linspace(0,len(ds)-1,240,dtype=int)):
                rgb,bev,state,command,*_=ds[idx]
                before=time.perf_counter()
                raw[:]=rgb
                grid[:]=bev
                packet=dict(state=[float(state[0])],nav_anchors=(state[1:]*60).tolist(),target=float(command[3]),
                            dimensions=[4.8,1.9],intent=int(command[:3].argmax()),blind=False)
                worker.stdin.write(json.dumps(packet)+"\n")
                worker.stdin.flush()
                response=receive(worker)
                assert len(response["lane_pose"])==2
                elapsed=time.perf_counter()-before
                if j>=8:
                    ms.append(response["inference_ms"])
                    latency.append(elapsed*1000)
                time.sleep(max(0.,.05-elapsed))
            worker.stdin.write("null\n")
            worker.stdin.flush()
            worker.wait(timeout=15);monitor.wait(timeout=15)
            samples=[json.loads(s) for s in counters.read_text(encoding="utf-8-sig").splitlines()]
            peak=max(s["dedicated_bytes"] for s in samples if s["instances"])
            result=dict(timed_windows=len(ms),warmup=8,p95_inference_ms=float(np.percentile(ms,95)),
                        p95_roundtrip_ms=float(np.percentile(latency,95)),wddm_peak_dedicated_bytes=peak,
                        samples=sum(bool(s["instances"]) for s in samples),under_1p5GB=peak<1_500_000_000)
            write_json(folder/"worker-benchmark.json",result)
            print(json.dumps(result),flush=True)
    finally:
        if worker and worker.poll() is None:worker.kill();worker.wait()
        if monitor and monitor.poll() is None:monitor.terminate();monitor.wait()
        slot.close();slot.unlink()


if __name__=="__main__":main()
