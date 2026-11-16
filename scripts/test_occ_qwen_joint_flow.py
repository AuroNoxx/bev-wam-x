"""Bounded functional/gradient smoke tests, no optimizer step or training."""
import argparse,json,time
from pathlib import Path
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from occ_qwen_joint_flow import OccQwenJointFlow,load_text_backbone,trajectory_to_velocity,velocity_to_trajectory

def main():
    p=argparse.ArgumentParser();p.add_argument('--pretrained',action='store_true');p.add_argument('--device',default='cpu');p.add_argument('--output');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(71);start=time.perf_counter()
    root=Path(__file__).resolve().parents[1]
    if a.pretrained:
        backbone=load_text_backbone(root/'models/Qwen3-VL-2B-Instruct',dtype=torch.bfloat16 if a.device=='cuda' else torch.float32)
    else:
        cfg=Qwen3VLTextConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,vocab_size=32,rope_scaling={'rope_type':'default','mrope_section':[2,3,3],'mrope_interleaved':True})
        cfg._attn_implementation='sdpa';backbone=Qwen3VLTextModel(cfg)
    model=OccQwenJointFlow(backbone).to(a.device).train()
    import numpy as np
    manifest=json.loads((root/'data/occ_v3/manifest.json').read_text())
    rows=[]
    for route in manifest['routes']:
        if route['split']=='train':
            with np.load(root/'data/occ_v3'/route['samples'][0]) as f:
                assert int(f['preprocess_version'])==3
                assert np.array_equal(f['history_occ_ids'],f['history_pose_ids']-1)
                assert np.array_equal(f['future_occ_ids'],f['future_pose_ids']-1)
                rows.append({k:torch.from_numpy(f[k].copy()) for k in ['history','state','trajectory','future']})
        if len(rows)==2:break
    b={k:torch.stack([r[k] for r in rows]).to(a.device) for k in rows[0]}
    torch.testing.assert_close(velocity_to_trajectory(trajectory_to_velocity(b['trajectory'],b['state']),b['state']),b['trajectory'][...,:2]*b['trajectory'].new_tensor([50.,10.]),atol=2e-5,rtol=1e-5)
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        loss,terms=model.loss(**b);loss.backward()
        assert torch.isfinite(loss)
        for module in [model.encoder,model.scene_in,model.action_in,model.world_in,model.action_out,model.world_out]:
            assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters()), type(module)
        assert all(p.grad is None for p in model.backbone.parameters())
        assert all(p.grad is None for p in model.teacher.parameters())
        old=next(model.teacher.parameters()).detach().clone()
        with torch.no_grad(): next(model.encoder.parameters()).add_(.001)
        model.update_teacher(.5)
        torch.testing.assert_close(next(model.teacher.parameters()),old+.0005)
        model.zero_grad(set_to_none=True)
        scene=model.encode_history(b['history'],b['state']).detach().requires_grad_()
        action=torch.randn(2,25,2,device=a.device,requires_grad=True)
        world=torch.randn(2,32,64,device=a.device,requires_grad=True)
        t=torch.full((2,1),.5,device=a.device)
        aa,ww,ss=model.velocity(scene,action,t,world,return_scene=True)
        # Each output must depend on the OTHER TWO token streams. In particular
        # scene tokens precede action/world: a causal mask would fail this test.
        checks={}
        for name,out,inputs in [('scene',ss,[action,world]),('action',aa,[scene,world]),('world',ww,[scene,action])]:
            g=torch.autograd.grad(out.square().mean(),inputs,retain_graph=True)
            checks[name]=[float(x.abs().sum()) for x in g]
            assert all(torch.isfinite(x).all() and x.abs().sum()>0 for x in g)
        del aa,ww,ss,scene,action,world
        model.eval()
        sample=model.sample(b['history'],b['state'],steps=1,generator=torch.Generator(device=a.device).manual_seed(9))
        assert sample['trajectory_xy'].shape==(2,25,2)
        assert all(torch.isfinite(v).all() for v in sample.values())
    report={'passed':True,'pretrained':a.pretrained,'device':a.device,'seconds':time.perf_counter()-start,'loss':float(loss.detach()),'terms':{k:float(v.detach()) for k,v in terms.items()},'cross_stream_gradient_l1':checks,'parameters':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'formal_training':False,'optimizer_steps':0,'data':'two train-only occ_v3 samples; alignment assertions passed','cuda_peak_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else None}
    text=json.dumps(report,indent=2);print(text)
    if a.output: Path(a.output).write_text(text)
if __name__=='__main__': main()

