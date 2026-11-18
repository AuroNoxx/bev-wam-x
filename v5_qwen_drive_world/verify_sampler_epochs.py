"""CPU framework contract: full epochs and epoch-end/mid-epoch restoration."""
from .runtime import ROOT,configure
configure()
import json,copy
from collections import Counter
import torch
import pytorch_lightning as pl
from torchdata.stateful_dataloader import StatefulDataLoader
from .data import SpeedBinBalancedSampler
from .checkpoint_io import VerifiedCheckpointIO
from .epoch_resume import boundary_sampler_state,restore_boundary_sampler

RECORDS=json.loads((ROOT/'data/v5-qwen-drive-world/manifest.json').read_text())['splits']['train']


def loader():
    return StatefulDataLoader(list(range(2000)),batch_sampler=SpeedBinBalancedSampler(RECORDS,1,2000,71),num_workers=0)


class Probe(pl.LightningModule):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(()));self.counts=Counter();self.ids=[]
    def training_step(self,batch,batch_idx):
        self.counts[self.current_epoch]+=len(batch);self.ids.extend(batch.tolist())
        return (self.weight*.01).square()
    def configure_optimizers(self):
        o=torch.optim.AdamW(self.parameters(),lr=.001)
        return {'optimizer':o,'lr_scheduler':{'scheduler':torch.optim.lr_scheduler.LambdaLR(o,lambda s:1),'interval':'step'}}
    def on_save_checkpoint(self,c):c['task_state']={'scope':'CPU sampler/Lightning contract only, no driving model'}


def trainer(epochs,callbacks=None):
    return pl.Trainer(accelerator='cpu',devices=1,max_epochs=epochs,accumulate_grad_batches=8,
                      logger=False,enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,
                      callbacks=callbacks or [],enable_checkpointing=bool(callbacks),plugins=[VerifiedCheckpointIO()])


def main():
    torch.set_num_threads(2)
    folder=ROOT/'artifacts/v5-qwen-drive-world/sampler-epoch-contract';folder.mkdir(exist_ok=True)
    a=loader();it=iter(a)
    for _ in range(81):next(it)
    state=copy.deepcopy(a.state_dict());expected=[next(it).tolist() for _ in range(16)]
    b=loader();b.load_state_dict(state);bi=iter(b);assert [next(bi).tolist() for _ in range(16)]==expected
    callback=pl.callbacks.ModelCheckpoint(dirpath=folder,filename='epoch-end',save_top_k=1,every_n_epochs=1,enable_version_counter=False)
    first=Probe();t=trainer(2,[callback]);t.fit(first,train_dataloaders=loader())
    assert dict(first.counts)=={0:2000,1:2000} and t.global_step==500
    checkpoint=torch.load(folder/'epoch-end.ckpt',map_location='cpu',weights_only=False)
    assert checkpoint['global_step']==500
    resumed_loader=restore_boundary_sampler(loader(),boundary_sampler_state(checkpoint))
    second=Probe();continued=trainer(3);continued.fit(second,train_dataloaders=resumed_loader,ckpt_path=folder/'epoch-end.ckpt',weights_only=False)
    assert dict(second.counts)=={2:2000} and continued.global_step==750,(second.counts,continued.global_step)
    reference=SpeedBinBalancedSampler(RECORDS,1,2000,71)
    full=[]
    for _ in range(3):full.extend(i for batch in reference for i in batch)
    assert first.ids+second.ids==full
    # Replay the exact loop metadata of the affected real checkpoint, with
    # only the tiny CPU probe model. This includes the historical empty epoch.
    actual=torch.load(ROOT/'artifacts/v5-qwen-drive-world/subset2000/latest.ckpt',map_location='cpu',weights_only=False,mmap=True)
    assert actual['global_step']==500
    checkpoint['loops']=copy.deepcopy(actual['loops']);checkpoint['epoch']=actual['epoch']
    state=boundary_sampler_state(actual);del actual
    replay_path=folder/'affected-loop.ckpt';VerifiedCheckpointIO().save_checkpoint(checkpoint,replay_path)
    replay=Probe();replay_trainer=trainer(4)
    replay_trainer.fit(replay,train_dataloaders=restore_boundary_sampler(loader(),state),ckpt_path=replay_path,weights_only=False)
    assert replay_trainer.global_step==750 and len(replay.ids)==2000,(replay.counts,replay_trainer.global_step)
    reference_actual=SpeedBinBalancedSampler(RECORDS,1,2000,71);reference_actual.load_state_dict(state)
    assert replay.ids==[i for batch in reference_actual for i in batch]
    report={'passed':True,'scope':'CPU-only framework/data contract, not a driving model or extra training group',
            'first_two_epoch_counts':dict(first.counts),'resumed_epoch_counts':dict(second.counts),'final_updates':continued.global_step,
            'sampler_draws':len(full),'source_counts':dict(Counter(RECORDS[i]['source'] for i in full)),
            'mid_epoch_next16_exact':True,'epoch_end_resume_order_exact':True,
            'actual_affected_loop_replay_counts':dict(replay.counts),'actual_affected_loop_final_updates':replay_trainer.global_step}
    (folder/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
