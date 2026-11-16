"""Train-only bounded contract tests and paired small-model fit; no weights."""
import json,time,gc
import numpy as np
import torch
from lightning_task import ROOT,configure_runtime
from probe_teacher_history_matrix import idle
from masked_history_joint import MaskedHistoryJoint
from stage1_objective import consistent_flow_loss

def main():
    idle();configure_runtime();out=ROOT/'artifacts/masked-history-joint-validation';out.mkdir(exist_ok=False)
    source=ROOT/'artifacts/causal-history-contract-mini';manifest=json.loads((source/'manifest.json').read_text())
    # Two mid-sequence and two boundary windows; same anchors for both models.
    ids=[0,1,8,9];rows=[]
    keys=['history','state','history_valid','history_age_seconds','geometry_valid']
    for i in ids:
        r=manifest['results'][i]
        with np.load(source/r['actor']) as a:row={k:torch.from_numpy(a[k].copy()) for k in keys}
        with np.load(source/r['labels']) as a:row.update({k:torch.from_numpy(a[k].copy()) for k in ['future','trajectory']})
        rows.append(row)
    b={k:torch.stack([r[k] for r in rows]).cuda() for k in rows[0]}
    torch.manual_seed(1701);m=MaskedHistoryJoint().cuda().eval();tests={}
    with torch.no_grad():
        original=m.context(b)
        poisoned={**b,'future':torch.zeros_like(b['future']),'trajectory':torch.randn_like(b['trajectory']),'future_mask':torch.ones(1,device='cuda')}
        tests['future_label_poison_invariant']=torch.equal(original,m.context(poisoned))
        corrupt={**b,'history':b['history'].clone()};corrupt['history'][~b['history_valid']]=3
        tests['invalid_frame_payload_invariant']=torch.equal(original,m.context(corrupt))
        zeros={**b,'history':torch.where(b['history']==0,255,b['history'])}
        tests['zero_not_free_invariant']=torch.equal(original,m.context(zeros))
        geo={**b,'history':torch.where(b['geometry_valid'],b['history'],4)}
        tests['outside_geometry_invariant']=torch.equal(original,m.context(geo))
        age={**b,'history_age_seconds':b['history_age_seconds']*1.5}
        tests['age_changes_output']=not torch.equal(original,m.context(age))
        invalid={**b,'history_valid':b['history_valid'].clone()};invalid['history_valid'][:,-1]=False
        try:m.context(invalid);tests['invalid_current_rejected']=False
        except ValueError:tests['invalid_current_rejected']=True
    assert all(tests.values()),tests
    del m;gc.collect();torch.cuda.empty_cache();cfg=json.loads((ROOT/'configs/joint-occ-v1.json').read_text());results=[]
    for history in ['short4','long14']:
        idle();torch.manual_seed(1701);m=MaskedHistoryJoint().cuda();batch=dict(b)
        if history=='short4':
            for k in ['history','history_valid','history_age_seconds','geometry_valid']:batch[k]=b[k][:,-4:]
        opt=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=1e-4)
        def evaluate():
            m.eval()
            with torch.random.fork_rng(devices=[0]),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                torch.manual_seed(8181);pred=m.sample(batch,k=4,steps=16)
                err=((pred[...,:2]-batch['trajectory'][:,None,:,:2])*pred.new_tensor([50.,10.])).norm(dim=-1)
                aux=m.auxiliary_loss(batch)
                return {'meanADE4_m':float(err.mean()),'minADE4_m':float(err.mean(-1).min(1).values.mean()),'latent_mse':float(aux[0])}
        before=evaluate();m.train();torch.manual_seed(444);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();losses=[]
        for step in range(256):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                fm,_=consistent_flow_loss(m,batch,cfg['objective']);aux=m.auxiliary_loss(batch);loss=fm+.1*sum(aux)
            assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn);opt.step();m.update_teacher()
            if (step+1)%64==0:losses.append({'step':step+1,'loss':float(loss.detach()),'fm':float(fm.detach())})
        torch.cuda.synchronize();wall=time.perf_counter()-start
        result={'history':history,'non_ema_parameters':sum(p.numel() for n,p in m.named_parameters() if not n.startswith('teacher.')),'before':before,'after':evaluate(),'steps':256,'batch':4,'lr':1e-4,'train_seconds':wall,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'loss_samples':losses}
        results.append(result);print(json.dumps(result),flush=True);del m,opt;gc.collect();torch.cuda.empty_cache()
    result={'tests':tests,'results':results,'selected_windows':[manifest['results'][i]['source'] for i in ids],'source_causality_certified':False,'timestamp_basis':'nominal index cadence, not measured timestamps','test_read':False,'checkpoint_saved':False,'scope':'Small width128/layers2/heads4 paired train-only diagnostic. Same anchors, init, optimizer RNG, steps and physical batch; different wall/compute. Not dense teacher quality acceptance, not safety certification. Unknown0/255 merged conservatively; no verified free-space supervision.'}
    (out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(tests),flush=True)
if __name__=='__main__':main()
