"""128-step tiny train-only fit diagnostic. Not a resumable formal run."""
import json,time,torch
from probe_joint_occ_capacity import CapacityModel
from lightning_task import ROOT,configure_runtime
from train_joint_occ import resource_guard,MemorySamples,batch_to
from stage1_objective import consistent_flow_loss

def main():
    resource_guard();configure_runtime();torch.manual_seed(1701)
    out=ROOT/'artifacts/joint-occ-teacher-150m-tinyfit';out.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'configs/joint-occ-v1.json').read_text());sel=json.loads((ROOT/cfg['selection']).read_text())
    names=sel['splits']['train'][:4];batch=batch_to(MemorySamples(names).rows,'cuda');m=CapacityModel(768,15,12).cuda()
    def evaluate():
        m.eval()
        with torch.random.fork_rng(devices=[0]),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            torch.manual_seed(8181);p=m.sample(batch,k=8,steps=16);target=batch['trajectory']
            err=((p[...,:2]-target[:,None,:,:2])*p.new_tensor([50.,10.])).norm(dim=-1)
            _,pred=m.forecast(batch['history'],batch['state']);future,valid,_=m.target_data(batch);y=m.teacher(future).float();den=valid.sum().clamp_min(1)
            latent=(((pred.float()-y).square().mean(-1))*valid).sum()/den
        return {'meanADE8_m':float(err.mean()),'minADE8_m':float(err.mean(-1).min(1).values.mean()),'per_window_meanADE8_m':err.mean((1,2)).tolist(),'latent_mse':float(latent)}
    before=evaluate();print('BEFORE '+json.dumps(before),flush=True);m.train();opt=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=1e-4)
    rows=[];torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for i in range(128):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            fm,_=consistent_flow_loss(m,batch,cfg['objective']);aux=m.auxiliary_loss(batch);loss=fm+.1*sum(aux)
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);assert torch.isfinite(norm)
        opt.step();m.update_teacher()
        if (i+1)%16==0:
            row={'step':i+1,'loss':float(loss.detach()),'fm':float(fm.detach()),'latent':float(aux[0].detach()),'variance':float(aux[1].detach()),'semantic':float(aux[2].detach())};rows.append(row);print(json.dumps(row),flush=True)
    torch.cuda.synchronize();wall=time.perf_counter()-start;after=evaluate()
    result={'parameters_non_ema':sum(p.numel() for n,p in m.named_parameters() if not n.startswith('teacher.')),'steps':128,'batch':4,'lr':1e-4,'before':before,'after':after,'train_wall_seconds':wall,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'loss_samples':rows,'train_files':names,'test_read':False,'checkpoint_saved':False,'scope':'Only four train samples; assesses ability to fit, not generalization or safety. Sampling seed identical before/after. Does not replace formal Lightning checkpoint training.'}
    (out/'report.json').write_text(json.dumps(result,indent=2));print('RESULT '+json.dumps(result),flush=True)
if __name__=='__main__':main()
