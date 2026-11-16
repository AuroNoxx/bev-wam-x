"""Causal inputs, gradients, EMA, mask semantics and real resume integration."""
import argparse, copy, json
from unittest.mock import patch
import numpy as np
import torch
import pytorch_lightning as pl
from lightning_task import ROOT, configure_runtime, VerifiedCheckpointIO
from joint_occ_model import JointOccFlow
from train_joint_occ import JointData, JointTask, Measure, make_trainer, batch_to, digest, resource_guard
from stage1_objective import consistent_flow_loss
from audit_joint_occ import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='artifacts/joint-occ-v1/tests');a=p.parse_args()
    resource_guard();configure_runtime();out=ROOT/a.out;out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'configs/joint-occ-v1.json').read_text());selection=json.loads((ROOT/cfg['selection']).read_text())
    data=JointData(cfg,selection,1701);device='cuda' if torch.cuda.is_available() else 'cpu'
    batch=batch_to(data.datasets['train'].rows[:8],device);checks={}
    for variant in ['no_occ','temporal_flow','auxiliary','fusion']:
        torch.manual_seed(1701);model=JointOccFlow(variant).to(device).eval()
        xt=torch.randn(8,25,5,device=device);t=torch.full((8,1),.3,device=device)
        original=model(batch,xt,t)
        poisoned={**batch,'future':torch.full_like(batch['future'],255),'trajectory':torch.randn_like(batch['trajectory']),
                  'world2ego':torch.randn(8,4,4,device=device),'future_mask':torch.rand(8,2,160,device=device)}
        assert torch.equal(original,model(poisoned,xt,t))
        actor={'history':batch['history'],'state':batch['state']}
        assert torch.equal(original,model(actor,xt,t))
        h=model.history_features(batch['history']);changed=batch['history'].clone();changed[:,-1].fill_(255)
        assert torch.equal(h[:,:-1],model.history_features(changed)[:,:-1])
        if variant=='no_occ':assert torch.equal(original,model({**actor,'history':changed},xt,t))
        model.zero_grad(set_to_none=True)
        fm,_=consistent_flow_loss(model,batch,cfg['objective']);fm.backward()
        predictor_grad=sum(float(p.grad.abs().sum()) for p in model.predictor.parameters() if p.grad is not None)
        assert (predictor_grad>0)==(variant=='fusion')
        if variant in ['auxiliary','fusion']:
            model.zero_grad(set_to_none=True);terms=model.auxiliary_loss(batch);sum(terms).backward()
            assert all(p.grad is None for p in model.teacher.parameters())
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.predictor.parameters())
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            empty={**batch,'future':torch.full_like(batch['future'],255)}
            assert all(float(x)==0 for x in model.auxiliary_loss(empty))
            before=next(model.teacher.parameters()).clone()
            with torch.no_grad():next(model.encoder.parameters()).add_(.1)
            model.update_teacher(.99)
            assert torch.allclose(next(model.teacher.parameters()),before+.001,atol=1e-6)
        checks[variant]={'future_labels_cannot_change_actor':True,'history_causal':True,'flow_predictor_gradient':predictor_grad,
                         'target_detached_and_aux_gradients':variant in ['auxiliary','fusion']}
    del model

    class Snapshot(pl.Callback):
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            if trainer.global_step==2:trainer.save_checkpoint(out/'split.ckpt')
    def run(resume=None):
        pl.seed_everything(1701,workers=True)
        d=JointData(cfg,selection,1701);task=JointTask('fusion',cfg,selection['manifest_sha'],digest(ROOT/cfg['selection']),1701)
        trainer=make_trainer(out,cfg,Measure(),max_steps=4,checkpoint=False,extra_callbacks=[] if resume else [Snapshot()])
        trainer.fit(task,datamodule=d,ckpt_path=str(resume) if resume else None,weights_only=False)
        return {'model':{k:v.detach().cpu().clone() for k,v in task.state_dict().items()},
                'optimizer':copy.deepcopy(trainer.optimizers[0].state_dict()),'scheduler':trainer.lr_scheduler_configs[0].scheduler.state_dict(),'step':trainer.global_step}
    full=run();resumed=run(out/'split.ckpt')
    delta=max(float((v-resumed['model'][k]).abs().max()) for k,v in full['model'].items())
    assert delta==0,(delta,'Model or EMA differs after resume')
    assert full['step']==resumed['step']==4 and full['scheduler']==resumed['scheduler']
    optdelta=0.
    for key,values in full['optimizer']['state'].items():
        for field,value in values.items():
            other=resumed['optimizer']['state'][key][field]
            if isinstance(value,torch.Tensor):
                d=float((value.cpu()-other.cpu()).abs().max());optdelta=max(optdelta,d)
                if field=='step':assert d==0
                else:assert torch.allclose(value.cpu(),other.cpu(),rtol=1e-5,atol=1e-8)
            else:assert value==other
    ck=torch.load(out/'split.ckpt',map_location='cpu',weights_only=False)
    assert ck['task_state']['settings']['selection_sha']==digest(ROOT/cfg['selection'])
    io=VerifiedCheckpointIO();io.save_checkpoint(ck,out/'atomic.ckpt');before=(out/'atomic.ckpt').read_bytes()
    for failure in ['replace','nonfinite']:
        try:
            if failure=='replace':
                with patch('lightning_task.os.replace',side_effect=OSError('simulated failure')):io.save_checkpoint(ck,out/'atomic.ckpt')
            else:
                bad=copy.deepcopy(ck);next(iter(bad['state_dict'].values())).flatten()[0]=float('nan');io.save_checkpoint(bad,out/'atomic.ckpt')
        except (OSError,RuntimeError):pass
        else:raise AssertionError('Save did not reject simulated failure')
        assert (out/'atomic.ckpt').read_bytes()==before
    report={'pass':True,'checks':checks,'resume_max_parameter_and_ema_difference':delta,'optimizer_max_difference':optdelta,
            'optimizer_steps_equal':True,'scheduler_equal':True,'resume_step':2,'final_step':4,
            'save_replace_failure_preserves_checkpoint':True,'nonfinite_save_preserves_checkpoint':True,
            'test_split_read':False,'device':device,'torch':torch.__version__,'hip':torch.version.hip,
            'source_causality_certified':False}
    write(out/'result.json',report);print('TEST_RESULT '+json.dumps(report),flush=True)


if __name__=='__main__':main()
