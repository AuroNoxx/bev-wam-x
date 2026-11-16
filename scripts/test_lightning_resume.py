"""Real GPU mid-epoch restore and atomic failure protection integration checks."""
import json,copy,argparse
from unittest.mock import patch
import torch
import pytorch_lightning as pl
from lightning_task import ROOT,configure_runtime,OccData,DrivingTask,VerifiedCheckpointIO

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--settings',default='configs/lightning.json');parser.add_argument('--source',default='artifacts/lightning/flow/latest.ckpt');parser.add_argument('--out',default='artifacts/lightning/contract-test');args=parser.parse_args()
    configure_runtime();settings=json.loads((ROOT/args.settings).read_text());folder=ROOT/args.out;folder.mkdir(parents=True,exist_ok=True)
    source=ROOT/args.source;initial=torch.load(source,map_location='cpu',weights_only=False);start=initial['global_step']
    class Snapshot(pl.Callback):
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            if trainer.global_step==start+2:trainer.save_checkpoint(folder/'split.ckpt')
    def run(path,snapshot=False):
        pl.seed_everything(20261114,workers=True);data=OccData('flow',settings);model=DrivingTask('flow',settings,data.sha,data.manifest['counts']['train'])
        trainer=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_steps=start+4,max_epochs=60,
            logger=False,callbacks=[Snapshot()] if snapshot else [],plugins=[VerifiedCheckpointIO()],gradient_clip_val=1.,
            limit_val_batches=0,num_sanity_val_steps=0,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,benchmark=settings.get('conv_autotune',False))
        trainer.fit(model,datamodule=data,ckpt_path=str(path),weights_only=False)
        result={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'optimizer':copy.deepcopy(trainer.optimizers[0].state_dict()),
            'scheduler':trainer.lr_scheduler_configs[0].scheduler.state_dict(),'global_step':trainer.global_step}
        for v in result['optimizer']['state'].values():
            for k,t in v.items():
                if isinstance(t,torch.Tensor):v[k]=t.cpu()
        return result
    full=run(source,True);resumed=run(folder/'split.ckpt')
    delta=max(float((full['model'][k]-resumed['model'][k]).abs().max()) for k in full['model'])
    assert delta==0,(delta,'Interrupted training diverged')
    assert full['global_step']==resumed['global_step']==start+4
    assert full['scheduler']==resumed['scheduler']
    optimizer_max_difference=0.
    for i,state in full['optimizer']['state'].items():
        for k,v in state.items():
            other=resumed['optimizer']['state'][i][k]
            if isinstance(v,torch.Tensor):
                d=float((v-other).abs().max());optimizer_max_difference=max(optimizer_max_difference,d)
                if d:print('OPTIMIZER_DIFFERENCE',i,k,d,tuple(v.shape),flush=True)
                if k=='step':assert torch.equal(v,other),(i,k,d)
                else:assert torch.allclose(v,other,rtol=1e-5,atol=1e-8),(i,k,d)
            else:assert v==other
    path=folder/'atomic.ckpt';io=VerifiedCheckpointIO();io.save_checkpoint(initial,path);before=path.read_bytes()
    try:
        with patch('lightning_task.os.replace',side_effect=OSError('simulated replacement failure')):io.save_checkpoint(initial,path)
    except OSError:pass
    else:raise AssertionError('Failure simulation did not run')
    assert path.read_bytes()==before
    report={'pass':True,'max_parameter_difference':delta,'optimizer_allclose':True,'optimizer_max_absolute_difference':optimizer_max_difference,
        'optimizer_tolerance':{'rtol':1e-5,'atol':1e-8},'optimizer_step_exact':True,'scheduler_equal':True,'final_global_step':start+4,
        'resume_mid_epoch_step':start+2,'atomic_failure_preserves_old_checkpoint':True,'settings':settings}
    (folder/'result.json').write_text(json.dumps(report,indent=2));print('CONTRACT '+json.dumps(report),flush=True)

if __name__=='__main__':main()
