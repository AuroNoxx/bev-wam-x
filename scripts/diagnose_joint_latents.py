"""Follow-up read-only representation checks on all 64 probe windows.

Adds fixed-location covariance and matched-support persistence. Does not change
training, pre-registered gates, or the original per-run diagnostic reports.
"""
import json
import torch
from torch.nn import functional as F
from lightning_task import ROOT, configure_runtime
from train_joint_occ import JointData, JointTask, digest, batch_to, resource_guard
from audit_joint_occ import write


def rank(z):
    # Remove location/time mean across examples before pooling covariance:
    # static coordinate patterns cannot raise this rank.
    x=(z-z.mean(0)).flatten(0,2).double()
    eigen=torch.linalg.eigvalsh(x.T@x).clamp_min(0)
    p=eigen/eigen.sum().clamp_min(1e-12)
    return float(torch.exp(-(p*p.clamp_min(1e-12).log()).sum())) if eigen.sum()>1e-12 else 0.


def main():
    resource_guard();configure_runtime();cfg=json.loads((ROOT/'configs/joint-occ-v1.json').read_text())
    selection=json.loads((ROOT/cfg['selection']).read_text());data=JointData(cfg,selection,1701);out={}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for seed in cfg['seeds']:
        for variant in ['auxiliary','fusion']:
            key=f'{variant}-seed{seed}';task=JointTask(variant,cfg,selection['manifest_sha'],digest(ROOT/cfg['selection']),seed).to(device).eval()
            task.load_state_dict(torch.load(ROOT/f'artifacts/joint-occ-v1/runs/{key}/latest.ckpt',map_location=device,weights_only=False)['state_dict'])
            model=task.model;collected={k:[] for k in ['pred','target','persistence','valid','dynamic','prob','fractions']}
            with torch.inference_mode(),torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):
                for start in range(0,64,8):
                    b=batch_to(data.datasets['validation'].rows[start:start+8],device)
                    _,pred=model.forecast(b['history'],b['state']);future,valid,fractions=model.target_data(b)
                    current=b['history'][:,-1:]
                    persistent=torch.where((current>0)&(current<8),current,255)
                    target=model.teacher(future).float();persist=model.teacher(persistent).float().expand_as(target)
                    current_dyn=F.adaptive_max_pool2d((current==4).any(2).float(),(8,20)).flatten(2).expand(-1,2,-1)>0
                    dynamic=(fractions[...,3]>0)|current_dyn
                    values={'pred':pred.float(),'target':target,'persistence':persist,'valid':valid,'dynamic':dynamic,
                            'prob':model.occ_head(pred).float().softmax(-1),'fractions':fractions}
                    for k,v in values.items():collected[k].append(v.cpu())
            c={k:torch.cat(v) for k,v in collected.items()};p=c['pred'];t=c['target'];valid=c['valid']
            def mse(pred,mask):return float(((pred-t).square().mean(-1)*mask).sum()/mask.sum().clamp_min(1))
            result={'windows':64,'pred_fixed_location_covariance_effective_rank':rank(p),'target_fixed_location_covariance_effective_rank':rank(t),
                'latent_width':64,'predicted_1s_2s_rms_difference':float((p[:,0]-p[:,1]).square().mean().sqrt()),
                'target_1s_2s_rms_difference':float((t[:,0]-t[:,1]).square().mean().sqrt()),
                'matched_target_mse':mse(p,valid),'shuffled_horizon_mse':mse(p.flip(1),valid),
                'matched_support_persistence_mse':mse(c['persistence'],valid),'dynamic':{}}
            for h in range(2):
                mask=valid[:,h]&c['dynamic'][:,h]
                residual=(p[:,h]-t[:,h]).square().mean(-1)
                # Compare conditional vehicle fraction to a zero-vehicle baseline;
                # sparse positives can otherwise make tiny MAE look impressive.
                truth=c['fractions'][:,h,:,3];guess=c['prob'][:,h,:,3]
                result['dynamic'][f'{h+1}s']={'eligible_tokens':int(mask.sum()),
                    'latent_mse':float((residual*mask).sum()/mask.sum().clamp_min(1)),
                    'vehicle_fraction_mae':float(((guess-truth).abs()*mask).sum()/mask.sum().clamp_min(1)),
                    'zero_vehicle_fraction_mae':float((truth*mask).sum()/mask.sum().clamp_min(1))}
            result['interpretation']='Nonconstant representation does not prove useful dynamics or absence of dimensional collapse; rank is relative to width and this small dataset.'
            out[key]=result
            del task,model
            if device=='cuda':torch.cuda.empty_cache()
    write(ROOT/'artifacts/joint-occ-v1/latent-followup.json',out);print(json.dumps(out),flush=True)


if __name__=='__main__':main()
