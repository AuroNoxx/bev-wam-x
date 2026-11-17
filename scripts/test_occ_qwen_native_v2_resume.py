"""Bounded CPU-only v2 Lightning exact restart test across an epoch boundary."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['HIP_VISIBLE_DEVICES']='-1';os.environ['ROCR_VISIBLE_DEVICES']='-1'
import copy,hashlib,json,time
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from b2d_native_occ import ROOT
from native_occ_cached_data import NativeCachedDataset,sha
from native_occ_exact_loader import ExactResumeDataLoader
from occ_qwen_native_v2 import NativeOccQwenV2
from train_occ_qwen_native_v2 import NativeV2Task,config
from lightning_task import VerifiedCheckpointIO,random_state

def exact(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return a==b
class Data(pl.LightningDataModule):
 def __init__(self,ds,cfg):super().__init__();self.ds=ds;self.cfg=cfg
 def train_dataloader(self):return ExactResumeDataLoader(self.ds,batch_size=1,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(self.cfg['seed']))
class Trace(pl.Callback):
 def __init__(self,folder,save):self.folder=folder;self.save=save;self.batches=[]
 def on_train_batch_start(self,trainer,module,batch,batch_idx):self.batches.append({k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in batch.items()})
 def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
  if self.save and trainer.global_step==2:trainer.save_checkpoint(self.folder/'split.ckpt')

def main():
 torch.set_num_threads(1);started=time.perf_counter();folder=ROOT/'artifacts/occ-qwen-native-v2-cpu-resume-test';folder.mkdir(exist_ok=True)
 cfg={**config(),'batch_size':1};manifest_sha=sha(ROOT/cfg['manifest']);index=json.loads((ROOT/'artifacts/native-occ-cache-pilot/report.json').read_text())['index'];ds=NativeCachedDataset(index,'train');assert len(ds)==3
 def run(resume=False):
  pl.seed_everything(cfg['seed'],workers=True,verbose=False)
  c=Qwen3VLTextConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,vocab_size=32,rope_scaling={'rope_type':'default','mrope_section':[2,3,3],'mrope_interleaved':True});c._attn_implementation='sdpa'
  task=NativeV2Task(NativeOccQwenV2(Qwen3VLTextModel(c),decoder_width=64,decoder_layers=2,heads=4,accel_weight=cfg['accel_weight'],jerk_weight=cfg['jerk_weight']),cfg,manifest_sha,contract_test=True)
  trace=Trace(folder,not resume)
  trainer=pl.Trainer(accelerator='cpu',precision='32-true',max_steps=4,max_epochs=3,logger=CSVLogger(folder/'logs',name='resumed' if resume else 'full'),callbacks=[trace],plugins=[VerifiedCheckpointIO()],gradient_clip_val=1.,limit_val_batches=0,num_sanity_val_steps=0,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=1)
  trainer.fit(task,datamodule=Data(ds,cfg),ckpt_path=str(folder/'split.ckpt') if resume else None,weights_only=False)
  trainer.save_checkpoint(folder/('resumed.ckpt' if resume else 'full.ckpt'))
  return {'model':copy.deepcopy(task.state_dict()),'optimizer':copy.deepcopy(trainer.optimizers[0].state_dict()),'scheduler':copy.deepcopy(trainer.lr_scheduler_configs[0].scheduler.state_dict()),'rng':random_state(),'step':trainer.global_step,'epoch':trainer.current_epoch,'batches':trace.batches,'task':task}
 full=run();resumed=run(True);cp=torch.load(folder/'split.ckpt',map_location='cpu',weights_only=False)
 delta=max(float((v-resumed['model'][k]).abs().max()) for k,v in full['model'].items())
 bad=copy.deepcopy(cp);bad['task_state']['schema']='native-occ-qwen-lightning-v1'
 try:resumed['task'].on_load_checkpoint(bad)
 except RuntimeError:schema_rejected=True
 else:schema_rejected=False
 results={'test_only':True,'device':'cpu','tiny_random_backbone':True,'total_optimizer_updates':6,'full_final_step':full['step'],'resumed_final_step':resumed['step'],'checkpoint_step':cp['global_step'],'full_final_epoch_index':full['epoch'],'sample_count':3,'batch_size':1,'crossed_epoch_boundary':full['epoch']>=1,'model_max_difference':delta,'model_exact':exact(full['model'],resumed['model']),'optimizer_exact':exact(full['optimizer'],resumed['optimizer']),'scheduler_exact':exact(full['scheduler'],resumed['scheduler']),'rng_exact':exact(full['rng'],resumed['rng']),'ema_exact':all(torch.equal(v,resumed['model'][k]) for k,v in full['model'].items() if k.startswith('model.teacher.')),'remaining_batch_sequence_exact':full['batches'][2:]==resumed['batches'],'settings_exact':cp['task_state']['settings']==cfg,'manifest_exact':cp['task_state']['manifest_sha']==manifest_sha,'schema':cp['task_state']['schema'],'v1_schema_rejected':schema_rejected,'optimizer_group_names':[g['name'] for g in full['optimizer']['param_groups']],'base_lrs':full['scheduler']['base_lrs'],'final_lrs':[g['lr'] for g in full['optimizer']['param_groups']],'production_files_modified':False,'seconds':time.perf_counter()-started}
 results['lr_groups_correct']=results['optimizer_group_names']==['occ_encoder_projectors','action_world_adapters'] and results['base_lrs']==[cfg['encoder_lr'],cfg['lr']]
 results['passed']=all(results[k] for k in ['crossed_epoch_boundary','model_exact','optimizer_exact','scheduler_exact','rng_exact','ema_exact','remaining_batch_sequence_exact','settings_exact','manifest_exact','v1_schema_rejected','lr_groups_correct']) and results['schema']=='native-occ-qwen-v2-lightning-v1' and full['step']==resumed['step']==4
 (folder/'result.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2));assert results['passed']
if __name__=='__main__':main()
