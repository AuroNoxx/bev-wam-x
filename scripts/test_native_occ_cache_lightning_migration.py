"""CPU-only four-step tiny-model full Lightning legacy->cached migration test."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HIP_VISIBLE_DEVICES']='-1'
os.environ['ROCR_VISIBLE_DEVICES']='-1'
import copy,json,hashlib,time,argparse
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from b2d_native_occ import ROOT
from occ_qwen_native_flow import NativeOccQwenJointFlow
from occ_qwen_native_data import NativeDataset
from native_occ_cached_data import NativeCachedDataset,sha
from native_occ_exact_loader import ExactResumeDataLoader
from train_occ_qwen_native import NativeTask,NativeData,settings
from occ_qwen_performance import OptimizedNativeTask
from lightning_task import VerifiedCheckpointIO,random_state

class Data(NativeData):
    def train_dataloader(self):
        cls=ExactResumeDataLoader if getattr(self,'fix',False) else StatefulDataLoader
        extra={'allow_legacy_resume':True} if getattr(self,'fix',False) else {}
        return cls(self.dataset,batch_size=self.settings['batch_size'],shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(self.settings['seed']),**extra)
    def val_dataloader(self):return []

class Trace(pl.Callback):
    def __init__(self,folder,save):self.folder=folder;self.save=save;self.batches=[]
    def on_train_batch_start(self,trainer,module,batch,batch_idx):
        self.batches.append({k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in batch.items()})
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        if self.save and trainer.global_step==2:trainer.save_checkpoint(self.folder/'split.ckpt')

def exact(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--legacy-control',action='store_true');parser.add_argument('--fix-loader',action='store_true');parser.add_argument('--fixed-source',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);started=time.perf_counter();folder=ROOT/('artifacts/native-occ-cache-lightning-control' if args.legacy_control else 'artifacts/native-occ-cache-lightning-test');folder=folder.with_name(folder.name+'-fixed') if args.fix_loader else folder;folder=folder.with_name(folder.name+'-newstate') if args.fixed_source else folder;folder.mkdir(exist_ok=True)
    index=json.loads((ROOT/'artifacts/native-occ-cache-pilot/report.json').read_text())['index']
    cached=NativeCachedDataset(index,'train');raw=NativeDataset('train');raw.names=cached.names
    assert raw.names==cached.names
    cfg=settings();cfg={**cfg,'batch_size':1};manifest_sha=sha(ROOT/cfg['manifest']);assert manifest_sha==raw.manifest_sha256
    def run(resume=False):
        pl.seed_everything(71,workers=True,verbose=False)
        c=Qwen3VLTextConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,vocab_size=32,rope_scaling={'rope_type':'default','mrope_section':[2,3,3],'mrope_interleaved':True});c._attn_implementation='sdpa'
        model=NativeOccQwenJointFlow(Qwen3VLTextModel(c))
        if resume and not args.legacy_control:task=OptimizedNativeTask(model,cfg,manifest_sha,contract_test=True,performance={'channels_last':False,'fused_adamw':False,'workers':0})
        else:task=NativeTask(model,cfg,manifest_sha,contract_test=True)
        trace=Trace(folder,not resume)
        trainer=pl.Trainer(accelerator='cpu',precision='32-true',max_steps=4,max_epochs=3,logger=CSVLogger(folder/'logs',name='cached' if resume else 'raw'),callbacks=[trace],plugins=[VerifiedCheckpointIO()],gradient_clip_val=1.,limit_val_batches=0,num_sanity_val_steps=0,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=1)
        data=Data(cfg,cached if resume and not args.legacy_control else raw);data.fix=(resume or args.fixed_source) and args.fix_loader
        trainer.fit(task,datamodule=data,ckpt_path=str(folder/'split.ckpt') if resume else None,weights_only=False)
        trainer.save_checkpoint(folder/('cached-final.ckpt' if resume else 'raw-final.ckpt'))
        return {'model':copy.deepcopy(task.state_dict()),'optimizer':copy.deepcopy(trainer.optimizers[0].state_dict()),'scheduler':copy.deepcopy(trainer.lr_scheduler_configs[0].scheduler.state_dict()),'rng':random_state(),'step':trainer.global_step,'batches':trace.batches}
    full=run();resumed=run(True)
    cp=torch.load(folder/'split.ckpt',map_location='cpu',weights_only=False)
    delta=max(float((full['model'][k]-resumed['model'][k]).abs().max()) for k in full['model'])
    results={'test_only':True,'formal_training_modified':False,'gpu_used':False,'tiny_random_backbone':True,'total_optimizer_updates':6,'max_step_each_run':4,'checkpoint_step':cp['global_step'],'model_max_difference':delta,'model_exact':exact(full['model'],resumed['model']),'optimizer_exact':exact(full['optimizer'],resumed['optimizer']),'scheduler_exact':exact(full['scheduler'],resumed['scheduler']),'rng_exact':exact(full['rng'],resumed['rng']),'remaining_batch_sequence_exact':full['batches'][2:]==resumed['batches'],'ema_exact':all(torch.equal(v,resumed['model'][k]) for k,v in full['model'].items() if '.teacher.' in k),'settings_match':cp['task_state']['settings']==cfg,'manifest_match':cp['task_state']['manifest_sha']==manifest_sha,'dataset_order_exact':raw.names==cached.names,'cache_index':index,'seconds':time.perf_counter()-started,'full_batches':full['batches'],'resumed_batches':resumed['batches']}
    results['passed']=all(results[k] for k in ['model_exact','optimizer_exact','scheduler_exact','rng_exact','remaining_batch_sequence_exact','ema_exact','settings_match','manifest_match','dataset_order_exact']) and full['step']==resumed['step']==4
    (folder/'result.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2));assert results['passed']
if __name__=='__main__':main()




