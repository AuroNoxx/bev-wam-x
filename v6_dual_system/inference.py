"""GPU worker for the stage-2 keyboard pilot, with a single shared observation slot."""
import argparse
import json
import os
from pathlib import Path
from multiprocessing import shared_memory
import sys
import time

import numpy as np
import torch

from v5_qwen_drive_world.objectives import footprint_collision
from .navigation import model_navigation_state
from .legacy_controller import TimedLaneCrossing
from .deployment import GraphDriver, fold_batch_norm, load_driver

RGB_BYTES=6*360*640
SIZE=RGB_BYTES+3*128*128


def main(name,ckpt,chase=False):
    # Keep native ROCm diagnostic output off the control protocol pipe.
    protocol=os.fdopen(os.dup(sys.stdout.fileno()),"w",buffering=1)
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    def reply(value):
        print("V6 "+json.dumps(value),file=protocol,flush=True)
    reply({"pid":os.getpid()})
    startup=time.perf_counter()
    torch.backends.cudnn.benchmark=False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.set_per_process_memory_fraction(.15)
    torch.set_num_threads(2)
    model,step,identity=load_driver(ckpt)
    report=Path(ckpt).with_name("inference-operators.json")
    folded=False
    if report.exists():
        result=json.loads(report.read_bytes())
        if result["deploy_folded"] and (result["step"],result["identity"],result["torch"],result["device"])==(
            step,identity,torch.__version__,torch.cuda.get_device_name()):
            fold_batch_norm(model)
            folded=True
    model.to(device="cuda",memory_format=torch.channels_last).eval()
    shm=shared_memory.SharedMemory(name=name)
    rgb=np.ndarray((1,6,360,640),np.uint8,shm.buf)
    bev=np.ndarray((1,3,128,128),np.uint8,shm.buf,offset=RGB_BYTES)
    support=torch.ones((1,3,128,128),device="cuda")
    try:
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            graph_file=Path(ckpt).with_name("graph.json")
            def forward(*obs):
                return model(*obs,return_lane=(chase or model.navigation) and hasattr(model,"lane_head"))
            engine=forward
            if graph_file.exists():
                proof=json.loads(graph_file.read_bytes())
                if proof.get("deploy_graph") and (proof["step"],proof["identity"],proof["torch"],proof["device"],proof["folded"])==(
                    step,identity,torch.__version__,torch.cuda.get_device_name(),folded):
                    engine=GraphDriver(forward,len(model.state_scale))
            if model.navigation and engine is forward:raise RuntimeError("Navigation deployment requires verified HIP Graph")
            for _ in range(2):
                outputs=engine(torch.zeros((1,6,360,640),dtype=torch.uint8,device="cuda"),
                      torch.zeros((1,3,128,128),dtype=torch.uint8,device="cuda"),
                      torch.zeros((1,len(model.state_scale)),device="cuda"),torch.tensor([[1.,0.,0.,0.]],device="cuda"))
                xy,logits=outputs[:2]
                delta=torch.diff(xy,dim=1,prepend=torch.zeros_like(xy[:,:1]))
                path=torch.cat((xy,torch.atan2(delta[:,:,1],delta[:,:,0]).unsqueeze(-1)),dim=-1)
                footprint_collision(logits[:,1:].float().sigmoid(),path,torch.tensor([[4.8,1.9]],device="cuda"),support)
                xy.float().cpu().tolist()
            torch.cuda.synchronize()
            reply({"ready":True,"graph":isinstance(engine,GraphDriver),"startup_s":time.perf_counter()-startup})
            crossing=TimedLaneCrossing()
            for line in sys.stdin:
                req=json.loads(line)
                if req is None:break
                start=time.perf_counter()
                image=torch.as_tensor(rgb,device="cuda")
                grid=torch.as_tensor(bev,device="cuda")
                model_state=(model_navigation_state(req["state"][0],req["nav_anchors"]).tolist() if model.navigation else req["state"])
                state=torch.tensor([model_state],device="cuda",dtype=torch.float32)
                dims=torch.tensor([req["dimensions"]],device="cuda",dtype=torch.float32)
                requested=req["intent"]
                intent=0 if req["blind"] else requested
                def plan(intent):
                    command=torch.tensor([[intent==0,intent==1,intent==2,req["target"]]],device="cuda",dtype=torch.float32)
                    outputs=engine(image,grid,state,command)
                    xy,logits=outputs[:2]
                    delta=torch.diff(xy,dim=1,prepend=torch.zeros_like(xy[:,:1]))
                    yaw=torch.atan2(delta[:,:,1],delta[:,:,0])
                    path=torch.cat((xy,yaw.unsqueeze(-1)),dim=-1)
                    risk,covered=footprint_collision(logits[:,1:].float().sigmoid(),path,dims,support)
                    score=float(risk) if bool(covered) else None
                    return outputs, score is None or score>.5,score
                outputs,risk,score=plan(intent)
                requested_score=score if not req["blind"] else None
                if intent and risk:
                    intent=0
                    outputs,risk,score=plan(0)
                veto=intent!=requested
                handoff=False
                if model.spatial:
                    offset=float(outputs[2][0,0])
                    if crossing.update(intent,offset,req.get("sim_s",0.)):
                        intent=0
                        outputs,risk,score=plan(0)
                        crossing.update(0,0.,req["sim_s"])
                        handoff=True
                xy,logits=outputs[:2]
                result=dict(xy=xy[0].float().cpu().tolist(),intent=intent,veto=veto,handoff=handoff,risk=risk,
                            footprint_score=score,requested_footprint_score=requested_score,
                            inference_ms=(time.perf_counter()-start)*1000,
                            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                if len(outputs)==3:
                    offset,angle10=outputs[2][0].float().tolist()
                    result["lane_pose"]=[offset,angle10/10.]
                if chase:
                    result["occupied_cells"]=(logits[0,0]>0).nonzero().cpu().tolist()
                reply(result)
    finally:
        shm.close()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("slot");p.add_argument("checkpoint")
    p.add_argument("--chase-view",action="store_true")
    args=p.parse_args();main(args.slot,args.checkpoint,args.chase_view)
