"""Exercise the new objective on real OCC and produce a resume-test fixture."""
import json
import torch
import pytorch_lightning as pl
from lightning_task import ROOT,configure_runtime,OccData,DrivingTask,VerifiedCheckpointIO


class GradientCheck(pl.Callback):
    def on_after_backward(self,trainer,module):
        self.gradients={}
        for name,parameters in [('occ_encoder',module.model.encoder.parameters()),
                                ('state_navigation',module.model.state.parameters()),
                                ('trajectory_output',module.model.output.parameters())]:
            grads=[p.grad for p in parameters if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads),name
            self.gradients[name]=sum(float(g.abs().sum()) for g in grads)
            assert self.gradients[name]>0,name


def main():
    configure_runtime();pl.seed_everything(20261114,workers=True)
    settings=json.loads((ROOT/'configs/lightning-stage1-v2.json').read_text())
    data=OccData('flow',settings);model=DrivingTask('flow',settings,data.sha,data.manifest['counts']['train'])
    baseline=torch.load(ROOT/'artifacts/lightning/flow/best.ckpt',map_location='cpu',weights_only=False)
    model.load_state_dict(baseline['state_dict']);before={k:v.clone() for k,v in model.state_dict().items()}
    check=GradientCheck();trainer=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_steps=2,max_epochs=60,
        callbacks=[check],plugins=[VerifiedCheckpointIO()],logger=False,gradient_clip_val=1.,limit_val_batches=0,
        num_sanity_val_steps=0,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False)
    trainer.fit(model,datamodule=data)
    change=max(float((v.cpu()-before[k]).abs().max()) for k,v in model.state_dict().items())
    assert change>0
    folder=ROOT/'artifacts/lightning/stage1-v2-contract';folder.mkdir(parents=True,exist_ok=True)
    trainer.save_checkpoint(folder/'initial.ckpt')
    report={'pass':True,'gradient_l1':check.gradients,'max_weight_update':change,'global_step':trainer.global_step}
    (folder/'gradient-check.json').write_text(json.dumps(report,indent=2));print('GRADIENT_CHECK '+json.dumps(report),flush=True)


if __name__=='__main__':main()
