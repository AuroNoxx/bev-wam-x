"""Bounded train-only loss/sampler diagnosis. No baseline/checkpoint edits."""
import json,time,gc,gzip
import numpy as np
import torch
from torch.nn import functional as F
from lightning_task import ROOT,configure_runtime
from probe_teacher_history_matrix import idle
from masked_history_joint import MaskedHistoryJoint
from stage1_objective import consistent_flow_loss,kinematic_residual

def main():
    idle();configure_runtime();out=ROOT/'artifacts/masked-physics-scan';out.mkdir(exist_ok=False)
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

    for k in ['history','history_valid','history_age_seconds','geometry_valid']:b[k]=b[k][:,-4:]
    # Reconstruct target directly from original poses/speeds and compare cache.
    audit=[]
    from pathlib import Path
    for j,i in enumerate(ids):
        name=manifest['results'][i]['source'];idx=int(Path(name).stem);route=Path(name).parts[2];clip=ROOT/'data/public/bench2drive/clips'/route
        def anno(k):
            with gzip.open(clip/'anno'/f'{k:05d}.json.gz','rt') as f:return json.load(f)
        def pose(a):return np.array(next(x['world2ego'] for x in a['bounding_boxes'] if x['class']=='ego_vehicle'))
        a=anno(idx);p0=pose(a);target=[]
        for h in range(2,51,2):
            ah=anno(idx+h);rel=p0@np.linalg.inv(pose(ah));yaw=-np.arctan2(rel[1,0],rel[0,0]);target.append([rel[0,3]/50,-rel[1,3]/10,np.sin(yaw),np.cos(yaw),ah['speed']/30])
        diff=float(np.max(np.abs(np.asarray(target)-b['trajectory'][j].cpu().numpy())))
        assert diff<1e-5
        audit.append({'source':name,'raw_cache_max_abs':diff,'current_speed_kmh':float(b['state'][j,0]*108),'future_distance_m':float((b['trajectory'][j,-1,:2]*b['state'].new_tensor([50,10])).norm()),'target_kinematic_residual_mps':float(kinematic_residual(b['trajectory'][j:j+1],b['state'][j:j+1]).norm(dim=-1).mean())})
    cfg=json.loads((ROOT/'configs/joint-occ-v1.json').read_text());results=[]
    for mode in ['full','all_x0p1','kinematics_x0p1','ramp_all_x0p1']:
        idle();torch.manual_seed(1701);m=MaskedHistoryJoint().cuda();opt=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=5e-4)
        def evaluate(steps=16,seed=8181):
            m.eval()
            with torch.random.fork_rng(devices=[0]),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                torch.manual_seed(seed)
                pred=m.sample(b,k=4,steps=steps)
                delta=(pred[...,:2]-b['trajectory'][:,None,:,:2])*pred.new_tensor([50.,10.]);err=delta.norm(dim=-1)
                return {'meanADE_m':float(err.mean()),'per_window_ADE_m':err.mean((1,2)).tolist(),'x_MAE_m':float(delta[...,0].abs().mean()),'y_MAE_m':float(delta[...,1].abs().mean()),'kinematic_residual_mps':float(kinematic_residual(pred.flatten(0,1),b['state'].repeat_interleave(4,0)).norm(dim=-1).mean())}
        def gradient_audit():
            # Independent bridge RNG, restored afterward. Full trainable vector.
            audits=[];params=[p for p in m.parameters() if p.requires_grad]
            for seed in [9101,9102,9103]:
                with torch.random.fork_rng(devices=[0]),torch.autocast('cuda',dtype=torch.bfloat16):
                    torch.manual_seed(seed);_,parts=consistent_flow_loss(m,b,cfg['objective'])
                vectors={}
                for name,value in parts.items():
                    grads=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
                    vectors[name]=torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten() for p,g in zip(params,grads)])
                base=vectors['fm'];bn=base.norm();row={'seed':seed,'terms':{}}
                for name,vec in vectors.items():
                    norm=vec.norm();weight=cfg['objective']['weights'][name]
                    row['terms'][name]={'raw_loss':float(parts[name].detach()),'configured_weight':weight,'raw_gradient_norm':float(norm),'weighted_gradient_norm':float(norm*weight),'weighted_ratio_to_fm':float(norm*weight/bn.clamp_min(1e-12)),'cosine_to_fm':float(torch.dot(vec,base)/(norm*bn).clamp_min(1e-12))}
                audits.append(row)
            return audits
        gradients={'initial':gradient_audit()} if mode=='full' else {}
        before=evaluate();m.train();torch.manual_seed(444);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();rows=[]
        for step in range(512):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                combined,terms=consistent_flow_loss(m,b,cfg['objective'])
                scale=.1 if mode=='all_x0p1' else (min(1.,max(0.,(step-255)/256))*.1 if mode=='ramp_all_x0p1' else 1.)
                loss=terms['fm']
                for key in ['position','kinematics','unit_heading','negative_speed']:
                    factor=.1 if mode=='kinematics_x0p1' and key=='kinematics' else scale
                    loss=loss+cfg['objective']['weights'][key]*factor*terms[key]
                aux=m.auxiliary_loss(b);loss=loss+.1*sum(aux)
            assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn);opt.step();m.update_teacher()
            if (step+1)%128==0:
                rows.append({'step':step+1,'loss':float(loss.detach()),'evaluation':evaluate(),'terms':{k:float(v.detach()) for k,v in terms.items()},'world':None if aux is None else [float(v.detach()) for v in aux]});m.train()
        torch.cuda.synchronize();wall=time.perf_counter()-start
        if mode=='full':gradients['final']=gradient_audit()
        result={'gradients':gradients,'mode':mode,'steps':512,'batch':4,'lr':5e-4,'before':before,'after16':evaluate(),'after64':evaluate(64),'sampling_seeds':{str(seed):evaluate(16,seed) for seed in [8181,8182,8183]},'wall_including_periodic_evaluation_seconds':wall,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'trace':rows}
        results.append(result);print(json.dumps(result),flush=True);del m,opt;gc.collect();torch.cuda.empty_cache()
    r={'raw_target_audit':audit,'results':results,'test_read':False,'checkpoint_saved':False,'source_causality_certified':False,'limitations':'Four train anchors one seed, not generalization. Modes diagnose objectives/sampling, not equal compute. Every run includes world latent/variance/semantic. full uses configured physical weights; all_x0p1 scales all physics by0.1; kinematics_x0p1 scales only kinematics by0.1; ramp_all_x0p1 starts after256 updates and reaches0.1 on final update. Gradient audits are raw configured weights, not a claim about adaptive optimizer update sizes. Same init and bridge RNG. Three sampling seeds are not independent training seeds.'}
    (out/'report.json').write_text(json.dumps(r,indent=2))
if __name__=='__main__':main()
