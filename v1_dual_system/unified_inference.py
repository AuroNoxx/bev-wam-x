"""Validate, then instantiate a fixed-shape HIP graph in the inference process."""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path
import numpy as np
import torch
from .unified_data import ART
from .unified_gate import load
from .unified_dataset import UnifiedData
from .sensors import write_json


class Graph:
    def __init__(self,m):
        self.inputs=(torch.zeros(1,3,360,640,device="cuda",dtype=torch.uint8),
                     torch.zeros(1,3,128,128,device="cuda",dtype=torch.uint8),
                     torch.zeros(1,1,128,128,device="cuda",dtype=torch.uint8),
                     torch.zeros(1,device="cuda"),torch.zeros(1,3,device="cuda"),torch.zeros(1,3,device="cuda"))
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):m(*self.inputs)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.outputs=m(*self.inputs)
    def __call__(self,*values):
        for source,destination in zip(values,self.inputs):destination.copy_(source)
        self.graph.replay()
        return self.outputs


def verify():
    checkpoint = ART/"best.ckpt"
    offline = json.loads((ART/"open-loop.json").read_bytes())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if not offline["passed"] or offline["checkpoint_sha256"] != digest:
        raise ValueError("This checkpoint has not passed the offline gates")
    ds=UnifiedData("validation")
    m,step=load(checkpoint);m.cuda()
    durations=[];projection=[];diff=np.zeros(2)
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        g=Graph(m)
        for sample in np.linspace(0,len(ds)-1,68,dtype=int):
            row=ds[sample]
            args=tuple(torch.as_tensor(a,device="cuda")[None] for a in (*row[:5],row[-1]))
            ref=m(*args)
            start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record();m.ipm(args[0],args[-1]);end.record();end.synchronize()
            projection.append(start.elapsed_time(end))
            start.record();out=g(*args);end.record();end.synchronize()
            durations.append(start.elapsed_time(end))
            diff=np.maximum(diff,[float((a-b).abs().max()) for a,b in zip(ref,out)])
    p95=float(np.percentile(durations[4:],95));ipm=float(np.percentile(projection[4:],95))
    report=dict(step=step,sha256=digest,
                torch=torch.__version__,device=torch.cuda.get_device_name(),p95_ms=p95,ipm_p95_ms=ipm,
                max_difference=diff.tolist(),passed=bool(diff.max()<1e-4 and p95<20 and ipm<.3))
    write_json(ART/"graph.json",report);print(json.dumps(report))


def serve(slot,checkpoint,chase=False):
    protocol=os.fdopen(os.dup(sys.stdout.fileno()),"w",buffering=1)
    sys.stdout.flush();os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    def reply(v):print("V1 "+json.dumps(v),file=protocol,flush=True)
    reply({"pid":os.getpid()})
    started=time.perf_counter()
    path=Path(checkpoint);proof=json.loads((path.parent/"graph.json").read_bytes())
    if not proof["passed"] or proof["sha256"]!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError("Graph proof mismatch")
    m,_=load(path);m.cuda()
    shm=shared_memory.SharedMemory(name=slot)
    rgb=np.ndarray((1,3,360,640),np.uint8,shm.buf)
    lidar=np.ndarray((1,3,128,128),np.uint8,shm.buf,offset=rgb.nbytes)
    nav=np.ndarray((1,1,128,128),np.uint8,shm.buf,offset=rgb.nbytes+lidar.nbytes)
    occupancy=np.ndarray((5,128,128),np.uint8,shm.buf,offset=rgb.nbytes+lidar.nbytes+nav.nbytes)
    try:
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            g=Graph(m)
            reply(dict(ready=True,graph=True,startup_s=time.perf_counter()-started))
            for line in sys.stdin:
                req=json.loads(line)
                if req is None:break
                t=time.perf_counter()
                cmd=req["intent"]
                xy,logits=g(*(torch.as_tensor(a,device="cuda") for a in (rgb,lidar,nav)),
                            torch.tensor([req["state"][0]],dtype=torch.float32,device="cuda"),
                            torch.tensor([[cmd==0,cmd==1,cmd==2]],dtype=torch.float32,device="cuda"),
                            torch.tensor([req["plane"]],dtype=torch.float32,device="cuda"))
                occupancy[:]=(logits[0]>0).cpu().numpy()
                result=dict(xy=xy[0].cpu().tolist(),intent=cmd,veto=False,risk=False,footprint_score=None,
                    requested_footprint_score=None,inference_ms=(time.perf_counter()-t)*1000,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,grid_dy=.4)
                if chase:result["occupied_cells"]=np.argwhere(occupancy[0]).tolist()
                reply(result)
    finally:shm.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("slot",nargs="?");parser.add_argument("checkpoint",nargs="?")
    parser.add_argument("--chase-view",action="store_true")
    args=parser.parse_args()
    torch.backends.cudnn.benchmark=False;torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15)
    if args.slot:serve(args.slot,args.checkpoint,args.chase_view)
    else:verify()
