"""Bounded resource/causality probe, not training acceptance or driving validation."""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from torch import nn
from joint_occ_model import JointOccFlow
from lightning_task import ROOT,configure_runtime
from train_joint_occ import resource_guard,MemorySamples,batch_to
from stage1_objective import consistent_flow_loss

class CapacityModel(JointOccFlow):
    def __init__(self,width=768,layers=10,heads=12):
        super().__init__('fusion',width,2)
        layer=nn.TransformerDecoderLayer(width,heads,width*4,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.transformer=nn.TransformerDecoder(layer,layers)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--batch',type=int,default=1);p.add_argument('--sweep',action='store_true');p.add_argument('--width',type=int,default=768);p.add_argument('--layers',type=int,default=10);p.add_argument('--heads',type=int,default=12);p.add_argument('--out');a=p.parse_args()
    configure_runtime();torch.manual_seed(1701)
    m=CapacityModel(a.width,a.layers,a.heads);total=sum(p.numel() for p in m.parameters());teacher=sum(p.numel() for p in m.teacher.parameters())
    report={'width':a.width,'decoder_layers':a.layers,'heads':a.heads,'parameters_total':total,'parameters_teacher_only':teacher,'parameters_without_teacher':total-teacher,'physical_batch':a.batch,'steps':3,'test_split_read':False,'source_causality_certified':False,'orin_measured':False,'scope':'resource smoke, not fit/closed-loop acceptance'}
    print(json.dumps(report),flush=True)
    if not a.run:return
    resource_guard();assert torch.cuda.is_available();device='cuda'
    out=ROOT/(a.out or ('artifacts/joint-occ-capacity-120m-probe-b'+str(a.batch)+'-sweep' if a.sweep else 'artifacts/joint-occ-capacity-120m-probe'));out.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'configs/joint-occ-v1.json').read_text());sel=json.loads((ROOT/cfg['selection']).read_text())
    names=sel['splits']['train'][:a.batch];data=MemorySamples(names);batch=batch_to(data.rows,device)
    report['input_files']=names;report['input_sha256']=[hashlib.sha256((ROOT/'data/occ_v3'/n).read_bytes()).hexdigest() for n in names]
    m=m.to(device).eval();xt=torch.randn(a.batch,25,5,device=device);t=torch.full((a.batch,1),.3,device=device)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        y=m(batch,xt,t);poison={**batch,'future':torch.full_like(batch['future'],255),'trajectory':torch.randn_like(batch['trajectory'])}
        report['future_label_isolation']=torch.equal(y,m(poison,xt,t))
        h=m.history_features(batch['history']);changed=batch['history'].clone();changed[:,-1].fill_(255)
        report['history_causal']=torch.equal(h[:,:-1],m.history_features(changed)[:,:-1])
    assert report['future_label_isolation'] and report['history_causal']
    m.train();opt=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=1e-5)
    torch.cuda.reset_peak_memory_stats();rows=[]
    for step in range(3):
        torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            fm,_=consistent_flow_loss(m,batch,cfg['objective']);aux=m.auxiliary_loss(batch);loss=fm+.1*sum(aux)
        assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        opt.step();m.update_teacher();torch.cuda.synchronize()
        rows.append({'step':step+1,'loss':float(loss.detach()),'fm':float(fm.detach()),'aux':[float(v.detach()) for v in aux],'seconds':time.perf_counter()-start});print(json.dumps(rows[-1]),flush=True)
    report['training']=rows;report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30;report['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
    m.eval();batch={k:v[:1] for k,v in batch.items()};times=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(4):
            torch.cuda.synchronize();start=time.perf_counter();pred=m.sample(batch,k=8,steps=16);torch.cuda.synchronize()
            assert torch.isfinite(pred).all()
            if i:times.append((time.perf_counter()-start)*1000)
    report['model_only_batch1_k8_16steps_ms']=times
    if a.sweep:
        import numpy as np
        sweep=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for k in [1,4,8]:
                for nsteps in [4,8,16]:
                    samples=[]
                    for i in range(12):
                        torch.cuda.synchronize();start=time.perf_counter();pred=m.sample(batch,k=k,steps=nsteps);torch.cuda.synchronize()
                        assert torch.isfinite(pred).all()
                        if i>=2:samples.append((time.perf_counter()-start)*1000)
                    row={'k':k,'ode_steps':nsteps,'repetitions':len(samples),'p50_ms':float(np.percentile(samples,50)),'p95_ms':float(np.percentile(samples,95)),'p99_ms':float(np.percentile(samples,99)),'max_ms':max(samples),'raw_ms':samples}
                    sweep.append(row);print('SWEEP '+json.dumps(row),flush=True)
        report['latency_sweep']=sweep;report['tail_warning']='Only 10 timings per configuration; empirical interpolated percentiles, not reliable deployment tails; model-only AMD results.'
    report['nonzero_variance_loss']=any(x['aux'][1]>0 for x in rows)
    report['no_checkpoint_written']=True;report['warning']='Three optimizer steps and one training window; no quality conclusion. Optimizer state discarded intentionally; formal training must use verified checkpoint integration. Batch1 variance penalty has no eligible cross-example pairs; batch2 uses only two training examples and is not a fit result.'
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print('REPORT '+str(out/'report.json'),flush=True)
if __name__=='__main__':main()


