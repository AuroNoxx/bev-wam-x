"""Check complete batch-32 BF16 forward/backward before selecting a ROCm layout."""
import copy
import json
import statistics
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from .dataset import FastData
from .deployment import load_driver
from .legacy_fast_model import losses
from .sensors import ROOT, write_json


def main():
    torch.manual_seed(641)
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark=False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.set_per_process_memory_fraction(.8)
    folder=ROOT/"artifacts/v1-dual-system"
    ds=FastData("train",True)
    batch=[x.cuda() for x in default_collate([ds[i] for i in np.linspace(0,len(ds)-1,32,dtype=int)])]
    source,_,_=load_driver(folder/"best.ckpt")
    source.enable_metric_fusion()
    result={"identity":ds.identity,"device":torch.cuda.get_device_name(),"torch":torch.__version__}
    reference=None
    for name,layout in [("channels_last",torch.channels_last),("nchw",torch.contiguous_format)]:
        model=copy.deepcopy(source).to(device="cuda",memory_format=layout).train()
        model.layout=layout
        times=[]
        try:
            for iteration in range(9):
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                start=time.perf_counter()
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    xy,occ=model(*batch[:4])
                    loss=losses(xy,occ,*batch[4:],(batch[2][:,0],batch[3],torch.ones(3,device="cuda")))
                loss.backward()
                torch.cuda.synchronize()
                if iteration>=3:times.append((time.perf_counter()-start)*1000)
            grads=[p.grad.detach().float().cpu() for p in model.parameters()]
            if any(not g.isfinite().all() for g in grads):raise FloatingPointError("Nonfinite gradients")
            result[name+"_ms"]=statistics.median(times)
            if reference is None:
                reference=grads
            else:
                diff=sum((a-b).square().sum().item() for a,b in zip(grads,reference))
                norm=sum(a.square().sum().item() for a in reference)
                result["joint_gradient_relative_l2"]=(diff/max(norm,1e-12))**.5
            del grads
        except RuntimeError as error:
            result[name+"_error"]=str(error)
        finally:
            del model
            torch.cuda.empty_cache()
    if "channels_last_ms" not in result:raise RuntimeError(result)
    result["selected"]="channels_last"
    if result.get("joint_gradient_relative_l2",1)<.03 and result["nchw_ms"]<result["channels_last_ms"]:
        result["selected"]="nchw"
    out=folder/"corrective"
    out.mkdir(exist_ok=True)
    write_json(out/"layout.json",result)
    print(json.dumps(result))


if __name__=="__main__":main()
