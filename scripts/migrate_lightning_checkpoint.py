"""One-time epoch-boundary import; subsequent resumes use native Lightning checkpoints."""
from pathlib import Path
import copy,json,math
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader,TensorDataset
from lightning_task import ROOT,VerifiedCheckpointIO,callbacks,configure_runtime

class Template(pl.LightningModule):
    def __init__(self):super().__init__();self.weight=torch.nn.Linear(1,1)
    def training_step(self,batch,batch_idx):return self.weight(batch[0]).square().mean()
    def configure_optimizers(self):return torch.optim.AdamW(self.parameters())

def main():
    configure_runtime();folder=ROOT/'artifacts/lightning/flow';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'latest.ckpt').exists():raise RuntimeError('Migration already exists; do not overwrite subsequent training')
    settings=json.loads((ROOT/'configs/lightning.json').read_text());manifest=json.loads((ROOT/'data/occ_v1/manifest.json').read_text())
    trainer=pl.Trainer(accelerator='cpu',max_epochs=1,logger=False,enable_checkpointing=False,enable_model_summary=False,enable_progress_bar=False)
    trainer.fit(Template(),DataLoader(TensorDataset(torch.ones(2,1)),batch_size=1))
    template_path=ROOT/'.cache/lightning-template/native.ckpt';trainer.save_checkpoint(template_path)
    template=torch.load(template_path,map_location='cpu',weights_only=False);report=[]
    for kind in ['best','latest']:
        source=ROOT/f'artifacts/v1/v1-flow/{kind}.pt';old=torch.load(source,map_location='cpu',weights_only=False)
        assert old['next_batch']==0 and not old['order'],'Migration requires a completed epoch'
        ep=old['epoch'];step=old['global_step'];batch=old['config']['flow']['batch_size'];batches=math.ceil(manifest['counts']['train']/batch)
        assert step==ep*batches
        new=copy.deepcopy(template);new.update(epoch=ep,global_step=step,state_dict={'model.'+k:v for k,v in old['model'].items()},
            optimizer_states=[old['optimizer']],lr_schedulers=[old['scheduler']])
        for group in new['optimizer_states'][0]['param_groups']:group['fused']=settings['fused_adamw'];group['foreach']=None
        if settings['channels_last']:
            for state in new['optimizer_states'][0]['state'].values():
                for key,value in state.items():
                    if isinstance(value,torch.Tensor) and value.ndim==4:state[key]=value.contiguous(memory_format=torch.channels_last)
        loop=new['loops']['fit_loop'];loop['epoch_loop.state_dict']['_batches_that_stepped']=step
        for scope in ['total','current']:
            loop['epoch_progress'][scope]={key:ep for key in loop['epoch_progress'][scope]}
            for key in ['epoch_loop.batch_progress','epoch_loop.scheduler_progress']:
                loop[key][scope]={k:step if scope=='total' else batches for k in loop[key][scope]}
            for kind_progress in ['step','zero_grad']:
                progress=loop['epoch_loop.automatic_optimization.optim_progress']['optimizer'][kind_progress]
                progress[scope]={k:step if scope=='total' else batches for k in progress[scope]}
        new['task_state']={'config':old['config'],'manifest_sha':old['manifest_sha'],'rng':old['rng'],'settings':settings,
            'migration_source':str(source),'resume_contract':'Imported at epoch boundary; new shuffle stream starts here; future resumes use StatefulDataLoader'}
        new['callbacks']={}
        for callback in callbacks(folder,'flow'):
            state=callback.state_dict()
            if isinstance(callback,pl.callbacks.EarlyStopping):state.update(wait_count=old['bad_epochs'],best_score=torch.tensor(old['best_loss']))
            if isinstance(callback,pl.callbacks.ModelCheckpoint):
                target=str(folder/('best.ckpt' if callback.monitor else 'latest.ckpt'))
                score=torch.tensor(old['best_loss'])
                state.update(best_model_path=target,dirpath=str(folder))
                if callback.monitor:state.update(best_model_score=score,current_score=score,best_k_models={target:score},kth_best_model_path=target,kth_value=score)
            if state:new['callbacks'][callback.state_key]=state
        VerifiedCheckpointIO().save_checkpoint(new,folder/f'{kind}.ckpt')
        check=torch.load(folder/f'{kind}.ckpt',map_location='cpu',weights_only=False)
        assert all(torch.equal(v,check['state_dict']['model.'+k]) for k,v in old['model'].items())
        assert check['lr_schedulers'][0]['last_epoch']==step
        report.append({'kind':kind,'epoch':ep,'global_step':step,'weights_identical':True,'optimizer_state_count':len(check['optimizer_states'][0]['state']),'scheduler_last_epoch':step,'best_loss':old['best_loss']})
    (folder/'migration.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
