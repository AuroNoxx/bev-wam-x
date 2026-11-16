"""Real teacher-loop SDPA backend comparison, same data and seeds, no checkpoints."""
import json,time,gc,torch
from contextlib import nullcontext
from torch.nn.attention import sdpa_kernel,SDPBackend
from probe_joint_occ_capacity import CapacityModel
from probe_teacher_history_matrix import idle
from train_joint_occ import MemorySamples,batch_to
from lightning_task import ROOT,configure_runtime
from stage1_objective import consistent_flow_loss

def main():
    idle();configure_runtime();out=ROOT/'artifacts/teacher-sdpa-loop-comparison';out.mkdir(exist_ok=False);cfg=json.loads((ROOT/'configs/joint-occ-v1.json').read_text());sel=json.loads((ROOT/cfg['selection']).read_text());results=[];reference=None
    names=sel['splits']['train'][:4];keys=['output.weight','temporal_gate.weight','transformer.layers.0.self_attn.in_proj_weight','predictor.0.weight']
    for backend in ['math','default']:
        idle();fullstart=time.perf_counter();torch.manual_seed(1701);batch=batch_to(MemorySamples(names).rows,'cuda');m=CapacityModel(768,15,12).cuda().train();opt=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=1e-5);times=[];losses=[];grad_first={};torch.cuda.reset_peak_memory_stats()
        for step in range(8):
            torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
            with (sdpa_kernel(SDPBackend.MATH) if backend=='math' else nullcontext()):
                with torch.autocast('cuda',dtype=torch.bfloat16):fm,_=consistent_flow_loss(m,batch,cfg['objective']);aux=m.auxiliary_loss(batch);loss=fm+.1*sum(aux)
                assert torch.isfinite(loss);loss.backward()
            if step==0:grad_first={k:dict(m.named_parameters())[k].grad.detach().cpu().clone() for k in keys}
            norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);assert torch.isfinite(norm);opt.step();m.update_teacher();torch.cuda.synchronize();times.append(time.perf_counter()-start);losses.append(float(loss.detach()))
        values={k:dict(m.named_parameters())[k].detach().cpu().clone() for k in keys};r={'backend':backend,'steps':8,'batch':4,'history_frames':4,'losses':losses,'hot_step_seconds':sum(times[2:])/len(times[2:]),'loop_seconds':sum(times),'end_to_end_seconds_including_load_init':time.perf_counter()-fullstart,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
        if reference is None:reference={'grads':grad_first,'params':values,'losses':losses}
        else:
            r['first_gradient_comparison']={k:{'max_abs':float((grad_first[k]-reference['grads'][k]).abs().max()),'relative_l2':float((grad_first[k]-reference['grads'][k]).norm()/reference['grads'][k].norm().clamp_min(1e-12))} for k in keys}
            r['final_selected_parameter_max_abs']={k:float((values[k]-reference['params'][k]).abs().max()) for k in keys};r['max_loss_abs_difference']=max(abs(a-b) for a,b in zip(losses,reference['losses']))
        results.append(r);print(json.dumps(r),flush=True);(out/'report.json').write_text(json.dumps({'non_ema_parameters':154628382,'dtype':'bf16 mixed','test_read':False,'inputs':names,'results':results,'scope':'8 update steps not convergence test; 4 selected gradient matrices, no full-gradient equivalence claim; no validation/checkpoint/CARLA timing'},indent=2));del m,opt,batch,fm,aux,loss,norm;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
