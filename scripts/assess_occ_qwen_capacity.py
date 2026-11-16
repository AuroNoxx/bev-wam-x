"""Bounded capacity assessment, no optimizer or training updates."""
import json,time
from pathlib import Path
import torch,numpy as np
from torch.nn.attention import sdpa_kernel,SDPBackend
from occ_qwen_joint_flow import OccQwenJointFlow,load_text_backbone
R=Path(__file__).resolve().parents[1]
torch.set_num_threads(4);torch.manual_seed(71)
m=OccQwenJointFlow(load_text_backbone(R/'models/Qwen3-VL-2B-Instruct')).cuda()
manifest=json.loads((R/'data/occ_v3/manifest.json').read_text())
row=next(r for r in manifest['routes'] if r['split']=='train')
with np.load(R/'data/occ_v3'/row['samples'][0]) as f:
    data={k:torch.from_numpy(f[k].copy())[None].cuda() for k in ['history','state','trajectory','future']}
report={'gpu':torch.cuda.get_device_name(),'total_vram':torch.cuda.get_device_properties(0).total_memory,'optimizer_steps':0,'precision':'backbone BF16 frozen; adapters FP32; SDPA MATH; no gradient checkpointing','parts':{name:{'parameters':sum(p.numel() for p in mod.parameters()),'trainable':sum(p.numel() for p in mod.parameters() if p.requires_grad)} for name,mod in m.named_children()}}
with sdpa_kernel(SDPBackend.MATH):
    # One warmup, no update.
    loss,_=m.loss(**data);loss.backward();m.zero_grad(set_to_none=True);del loss
    torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    b={k:v.repeat(8,*([1]*(v.ndim-1))) for k,v in data.items()}
    start=time.perf_counter();loss,_=m.loss(**b);loss.backward();torch.cuda.synchronize()
    report['batch8_forward_backward']={'seconds':time.perf_counter()-start,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
    m.zero_grad(set_to_none=True);del loss,b
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();m.eval()
    start=time.perf_counter();out=m.sample(data['history'],data['state'],steps=8);torch.cuda.synchronize()
    report['batch1_8_flow_steps']={'seconds':time.perf_counter()-start,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'finite':all(bool(torch.isfinite(x).all()) for x in out.values())}
print(json.dumps(report,indent=2));(R/'artifacts/occ-qwen-capacity.json').write_text(json.dumps(report,indent=2))
