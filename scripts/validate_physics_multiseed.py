"""Bounded route-held-out diagnostic wholly inside the training partition."""
import json,time,gc,hashlib,statistics
from pathlib import Path
import torch
from lightning_task import ROOT,configure_runtime
from probe_teacher_history_matrix import idle
from train_joint_occ import MemorySamples,batch_to
from masked_history_joint import MaskedHistoryJoint
from stage1_objective import consistent_flow_loss,kinematic_residual

def main():
    idle();configure_runtime();out=ROOT/'artifacts/physics-multiseed-heldout';out.mkdir(exist_ok=False)
    sel=json.loads((ROOT/'artifacts/joint-occ-v1/audit/selection.json').read_text());names=sel['splits']['train'];assert all(Path(n).parts[1]=='train' for n in names)
    routes=sorted(set(Path(n).parts[2] for n in names));families={r:r.split('_Town')[0] for r in routes};held=set()
    for family in sorted(set(families.values())):
        group=[r for r in routes if families[r]==family]
        if len(group)>1:held.add(min(group,key=lambda r:hashlib.sha256(('heldout20261115'+r).encode()).hexdigest()))
    ds=MemorySamples(names)
    for row in ds.rows:
        row['history_valid']=torch.ones(4,dtype=torch.bool);row['history_age_seconds']=torch.tensor([.6,.4,.2,0.]);row['geometry_valid']=row['history']!=255
    train=[i for i,n in enumerate(names) if Path(n).parts[2] not in held];validation=[i for i,n in enumerate(names) if Path(n).parts[2] in held]
    assert set(Path(names[i]).parts[2] for i in train).isdisjoint(Path(names[i]).parts[2] for i in validation)
    def category(row):
        speed=float(row['state'][0]*30);end=float(row['trajectory'][-1,4]*30)
        if abs(speed)<.5 and float(row['trajectory'][:,4].abs().max()*30)<.5:return 'stationary'
        if speed-end>=1:return 'decelerating'
        return 'other'
    categories=[category(row) for row in ds.rows]
    selection={'train_files':[names[i] for i in train],'heldout_train_partition_files':[names[i] for i in validation],'heldout_routes':sorted(held),'category_counts':{s:{c:sum(categories[i]==c for i in inds) for c in ['stationary','decelerating','other']} for s,inds in [('train',train),('heldout',validation)]},'official_validation_read':False,'test_read':False,'caveat':'Held out from these newly initialized models only. These are training-partition samples that previous baseline studies may have used; not pristine final validation/test.'}
    (out/'selection.json').write_text(json.dumps(selection,indent=2));print('SELECTION '+json.dumps(selection['category_counts']),flush=True)
    cfg=json.loads((ROOT/'configs/joint-occ-v1.json').read_text());results=[];allstart=time.perf_counter()
    for seed in [1701,1702,1703]:
      for weight in [.1,.01]:
        idle();torch.manual_seed(seed);m=MaskedHistoryJoint().cuda();opt=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=5e-4);gen=torch.Generator().manual_seed(seed+777);order=torch.randperm(len(train),generator=gen).tolist();pos=0
        objective=json.loads(json.dumps(cfg['objective']));objective['weights']['kinematics']=weight
        def evaluate(indices):
            m.eval();records=[]
            for es in [8181,8182]:
              for start in range(0,len(indices),8):
                ix=indices[start:start+8];b=batch_to([ds.rows[i] for i in ix],'cuda')
                with torch.random.fork_rng(devices=[0]),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    torch.manual_seed(es+start);pred=m.sample(b,k=4,steps=16);delta=(pred[...,:2]-b['trajectory'][:,None,:,:2])*pred.new_tensor([50.,10.]);ade=delta.norm(dim=-1).mean((1,2));kin=kinematic_residual(pred.flatten(0,1),b['state'].repeat_interleave(4,0)).norm(dim=-1).reshape(len(ix),4,25).mean((1,2));forward=pred[...,0].mean((1,2))*50;speed=(pred[...,-1,4].mean(1)-b['trajectory'][:,-1,4])*30
                for j,i in enumerate(ix):records.append({'category':categories[i],'ADE_m':float(ade[j]),'kinematic_mps':float(kin[j]),'signed_forward_m':float(forward[j]),'final_speed_error_mps':float(speed[j])})
            return {c:{'n_records':len(rs),**{k:statistics.mean(x[k] for x in rs) for k in ['ADE_m','kinematic_mps','signed_forward_m','final_speed_error_mps']}} for c in ['all','stationary','decelerating','other'] if (rs:=[x for x in records if c=='all' or x['category']==c])}
        m.train();torch.manual_seed(seed+444);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter();trace=[];samples=0
        for step in range(384):
            if pos+8>len(order):order=torch.randperm(len(train),generator=gen).tolist();pos=0
            ix=[train[j] for j in order[pos:pos+8]];pos+=8;samples+=len(ix);b=batch_to([ds.rows[i] for i in ix],'cuda');opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                fm,parts=consistent_flow_loss(m,b,objective);aux=m.auxiliary_loss(b);loss=fm+.1*sum(aux)
            assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn);opt.step();m.update_teacher()
            if (step+1)%128==0:trace.append({'step':step+1,'loss':float(loss.detach())})
            if time.perf_counter()-allstart>600:raise RuntimeError('Total probe budget exceeded, no further runs')
        torch.cuda.synchronize();wall=time.perf_counter()-started;reserved=torch.cuda.max_memory_reserved()/2**30
        result={'seed':seed,'kinematics_weight':weight,'steps':384,'batch':8,'training_wall_seconds_including_batch_transfer':wall,'samples_per_second':samples/wall,'peak_reserved_gib':reserved,'train':evaluate(train),'heldout':evaluate(validation),'trace':trace}
        results.append(result);(out/f'seed{seed}-kin{weight}.json').write_text(json.dumps(result,indent=2));print('RESULT '+json.dumps(result),flush=True);del m,opt,b;gc.collect();torch.cuda.empty_cache()
    (out/'report.json').write_text(json.dumps({'selection':selection,'results':results,'test_read':False,'checkpoint_saved':False,'source_causality_certified':False,'scope':'Three training seeds, fixed route-disjoint heldout inside train partition; diagnostic only, no default changes.'},indent=2))
if __name__=='__main__':main()
