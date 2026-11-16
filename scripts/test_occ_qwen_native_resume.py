"""Bounded ENGINEERING TEST: 4 tiny-model updates + 2 resumed updates, not training a driving model."""
import copy,json,hashlib,time
from pathlib import Path
from unittest.mock import patch
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from occ_qwen_native_flow import NativeOccQwenJointFlow
from occ_qwen_native_data import NativeDataset,require_training_authorization
from train_occ_qwen_native import NativeTask,NativeData,settings
from lightning_task import ROOT,VerifiedCheckpointIO

def main():
    torch.set_num_threads(4)
    folder=ROOT/'artifacts/occ-qwen-native-engineering-test';folder.mkdir(exist_ok=True)
    cfg=settings();sha=hashlib.sha256((ROOT/cfg['manifest']).read_bytes()).hexdigest()
    ds=NativeDataset('train');rows=[ds[0],ds[100],ds[300],ds[500]]*2
    class Snapshot(pl.Callback):
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            if trainer.global_step==2:trainer.save_checkpoint(folder/'split.ckpt')
    def run(path=None):
        pl.seed_everything(71,workers=True,verbose=False)
        c=Qwen3VLTextConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,vocab_size=32,rope_scaling={'rope_type':'default','mrope_section':[2,3,3],'mrope_interleaved':True})
        c._attn_implementation='sdpa'
        task=NativeTask(NativeOccQwenJointFlow(Qwen3VLTextModel(c)),cfg,sha,contract_test=True)
        before={k:v.clone() for k,v in task.model.encoder.state_dict().items()}
        frozen={k:v.clone() for k,v in task.model.backbone.state_dict().items()}
        trainer=pl.Trainer(accelerator='cpu',precision='32-true',max_steps=4,max_epochs=3,logger=CSVLogger(str(folder/'logs'),name='resume' if path else 'full'),callbacks=[] if path else [Snapshot()],plugins=[VerifiedCheckpointIO()],gradient_clip_val=1.,limit_val_batches=0,num_sanity_val_steps=0,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=1)
        trainer.fit(task,datamodule=NativeData(cfg,rows),ckpt_path=str(path) if path else None,weights_only=False)
        trainer.save_checkpoint(folder/('resumed.ckpt' if path else 'full.ckpt'))
        assert any(not torch.equal(v,before[k]) for k,v in task.model.encoder.state_dict().items())
        assert all(torch.equal(v,frozen[k]) for k,v in task.model.backbone.state_dict().items())
        return copy.deepcopy(task.state_dict()),copy.deepcopy(trainer.optimizers[0].state_dict()),copy.deepcopy(trainer.lr_scheduler_configs[0].scheduler.state_dict()),trainer.global_step
    start=time.perf_counter();full=run();resumed=run(folder/'split.ckpt')
    delta=max(float((v-resumed[0][k]).abs().max()) for k,v in full[0].items());assert delta==0,delta
    assert full[2]==resumed[2] and full[3]==resumed[3]==4
    for i,state in full[1]['state'].items():
        for k,v in state.items():
            if isinstance(v,torch.Tensor):assert torch.equal(v,resumed[1]['state'][i][k]),(i,k)
            else:assert v==resumed[1]['state'][i][k]
    checkpoint=torch.load(folder/'full.ckpt',weights_only=False)
    io=VerifiedCheckpointIO();atomic=folder/'atomic.ckpt';io.save_checkpoint(checkpoint,atomic);old=atomic.read_bytes()
    try:
        with patch('lightning_task.os.replace',side_effect=OSError('test: replacement failure')):io.save_checkpoint(checkpoint,atomic)
    except OSError:pass
    else:raise AssertionError('failure injection did not execute')
    assert old==atomic.read_bytes()
    broken=copy.deepcopy(checkpoint);broken['state_dict'][next(iter(broken['state_dict']))].view(-1)[0]=float('nan')
    try:io.save_checkpoint(broken,atomic)
    except RuntimeError:pass
    else:raise AssertionError('nonfinite checkpoint accepted')
    assert old==atomic.read_bytes()
    try:require_training_authorization()
    except RuntimeError:guard=True
    else:raise AssertionError('formal training pause lost')
    report={'passed':True,'test_only':True,'formal_training':False,'backbone':'random tiny Qwen, NOT pretrained driving weights','updates_executed_total':6,'full_final_step':4,'resumed_from_step':2,'resumed_final_step':4,'model_max_diff':delta,'optimizer_exact':True,'scheduler_exact':True,'ema_in_state_exact':True,'atomic_replace_failure_preserved_old':True,'nonfinite_save_preserved_old':True,'pause_guard_active':guard,'seconds':time.perf_counter()-start,'checkpoint_bytes':(folder/'full.ckpt').stat().st_size,'logs':str(folder/'logs')}
    (folder/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
