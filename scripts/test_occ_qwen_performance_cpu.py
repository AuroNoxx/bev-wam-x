"""CPU-only numerical check; no competing GPU work or production changes."""
import copy,json,time
from pathlib import Path
import torch
from occ_qwen_native_flow import NativeFrameEncoder
from occ_qwen_performance import channels_last_input

torch.set_num_threads(1);torch.manual_seed(71)
a=NativeFrameEncoder();b=copy.deepcopy(a)
torch.nn.utils.convert_conv2d_weight_memory_format(b,torch.channels_last);b.conv.register_forward_pre_hook(channels_last_input)
x=torch.randint(0,8,(1,4,20,128,128));ya=a(x);yb=b(x)
(ya.square().mean()).backward();(yb.square().mean()).backward()
out_diff=float((ya-yb).abs().max());grad_diff=max(float((p.grad-q.grad).abs().max()) for p,q in zip(a.parameters(),b.parameters()))
torch.testing.assert_close(ya,yb,rtol=1e-4,atol=1e-5)
for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-3,atol=1e-6)
report={'channels_last_cpu_fp32_output_max_abs':out_diff,'gradient_max_abs':grad_diff,'gpu_checked':False,'formal_training_touched':False}
# CPU fused availability is informative only, NOT a GPU correctness certificate.
p=torch.nn.Parameter(torch.randn(128));q=torch.nn.Parameter(p.detach().clone());o=torch.optim.AdamW([p],lr=1e-4);f=torch.optim.AdamW([q],lr=1e-4,fused=True)
try:
    for i in range(3):
        g=torch.randn_like(p);p.grad=g;q.grad=g.clone();o.step();f.step()
    torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-8);report['cpu_fused_adamw_max_abs']=float((p-q).abs().max())
except Exception as e:report['cpu_fused_adamw_unavailable']=repr(e)
path=Path('artifacts/occ-qwen-native-training/performance_cpu_validation.json');path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
