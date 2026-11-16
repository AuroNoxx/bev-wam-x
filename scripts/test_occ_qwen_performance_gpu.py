"""Run only during a protected checkpoint switch: bounded GPU numerical gates."""
import argparse,copy,json,time,gc,psutil
from pathlib import Path
import torch
from occ_qwen_native_flow import NativeOccQwenJointFlow,load_text_backbone
from occ_qwen_performance import enable_channels_last
from occ_qwen_native_data import NativeDataset
from b2d_native_occ import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args()
    for proc in psutil.process_iter(['pid','cmdline']):
        cmd=' '.join(proc.info['cmdline'] or [])
        if proc.pid!=psutil.Process().pid and ('train_occ_qwen_native.py' in cmd or 'train_occ_qwen_native_optimized.py' in cmd):raise RuntimeError('Production trainer still running; no competing GPU test')
    torch.set_num_threads(4);cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    m=NativeOccQwenJointFlow(load_text_backbone(ROOT/'models/Qwen3-VL-2B-Instruct'))
    m.load_state_dict({k.removeprefix('model.'):v for k,v in cp['state_dict'].items()},strict=True);m.cuda().train()
    ds=NativeDataset('train');samples=[ds[0],ds[100]];batch={k:torch.stack([s[k] for s in samples]).cuda() for k in samples[0]}
    def gradients():
        m.zero_grad(set_to_none=True);torch.manual_seed(113)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=m.loss(**batch)
        loss.backward();torch.cuda.synchronize()
        return float(loss.detach()),[p.grad.detach().clone() for p in m.parameters() if p.requires_grad]
    baseline,basegrads=gradients();enable_channels_last(m);optimized,newgrads=gradients()
    norm=torch.sqrt(sum(g.float().square().sum() for g in basegrads));diff=torch.sqrt(sum((g.float()-h.float()).square().sum() for g,h in zip(basegrads,newgrads)))
    rel=float(diff/norm.clamp_min(1e-12));loss_rel=abs(optimized-baseline)/max(abs(baseline),1e-8)
    channels_pass=rel<=.05 and loss_rel<=.02
    # Isolate fused optimizer from channels_last differences by feeding identical baseline gradients.
    params=[p for p in m.parameters() if p.requires_grad]
    left=[torch.nn.Parameter(p.detach().clone()) for p in params];right=[torch.nn.Parameter(p.detach().clone()) for p in params]
    old=torch.optim.AdamW(left,lr=1e-4);fused=torch.optim.AdamW(right,lr=1e-4,fused=True)
    old.load_state_dict(copy.deepcopy(cp['optimizer_states'][0]));fused.load_state_dict(copy.deepcopy(cp['optimizer_states'][0]))
    for group in fused.param_groups:
        group.update(fused=True,foreach=False)
        for p in group['params']:
            if 'step' in fused.state[p]:fused.state[p]['step']=fused.state[p]['step'].to(p.device)
            for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                if key in fused.state[p]:fused.state[p][key]=torch.empty_like(p,memory_format=torch.preserve_format).copy_(fused.state[p][key])
    for p,q,g in zip(left,right,basegrads):p.grad=torch.empty_like(p).copy_(g);q.grad=torch.empty_like(q).copy_(g)
    torch.nn.utils.clip_grad_norm_(left,1.,error_if_nonfinite=True);torch.nn.utils.clip_grad_norm_(right,1.,error_if_nonfinite=True)
    old.step();fused.step();torch.cuda.synchronize()
    pd=max(float((p-q).abs().max()) for p,q in zip(left,right));sd=0.;state_ok=True
    for p,q in zip(left,right):
        for k,v in old.state[p].items():
            other=fused.state[q][k]
            if isinstance(v,torch.Tensor):
                v=v.cpu();other=other.cpu();sd=max(sd,float((v-other).abs().max()))
                state_ok=state_ok and (torch.equal(v,other) if k=='step' else torch.allclose(v,other,rtol=1e-5,atol=1e-8))
    report={'source_step':cp['global_step'],'test_only':True,'channels_last':{'pass':channels_pass,'gradient_relative_l2':rel,'loss_relative_change':loss_rel,'baseline_loss':baseline,'channels_last_loss':optimized,'thresholds':{'grad_rel':.05,'loss_rel':.02}},'fused_adamw':{'pass':pd<=1e-7 and state_ok,'parameter_max_abs_difference':pd,'optimizer_state_max_abs_difference':sd,'states_allclose':state_ok},'production_checkpoint_changed':False}
    out=ROOT/'artifacts/occ-qwen-native-training/performance_gpu_validation.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()

